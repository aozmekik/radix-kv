#include "radix_kv/simulator.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/hash_prefix_cache.hpp"
#include "radix_kv/null_prefix_cache.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/radix_prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/types.hpp"
#include "radix_kv/workload.hpp"

namespace radix_kv {

namespace {

constexpr std::uint32_t kNoRequest = std::numeric_limits<std::uint32_t>::max();
// Added (times the sample index) to the trace's output tokens so parallel samples diverge.
constexpr std::uint64_t kSampleStride = 7919;

struct RunningRequest {
  std::uint32_t request = 0;
  std::vector<Sequence> samples;
  NodeHandle lock;
  std::uint32_t generated = 0;
};

void validate(const Trace& trace, const SimulatorConfig& config) {
  if (config.num_blocks == 0 || config.block_size == 0 || config.max_running == 0 ||
      config.samples_per_request == 0) {
    throw std::invalid_argument("SimulatorConfig: every numeric field must be positive");
  }
  if (trace.vocab_size == 0 || trace.requests.size() >= kNoRequest) {
    throw std::invalid_argument("Trace: vocab_size must be positive and requests < 2^32 - 1");
  }
  for (const TraceRequest& request : trace.requests) {
    if (request.session >= trace.sessions.size() || request.prompt_tokens == 0 ||
        request.output_tokens == 0 ||
        std::uint64_t{request.prompt_tokens} + request.output_tokens >
            trace.sessions[request.session].size()) {
      throw std::invalid_argument(
          "Trace: every request needs a non-empty prompt and output inside its session");
    }
  }
}

/// One simulation run over a concrete cache type; see `simulate()` for the scheduling policy.
template <PrefixCache Cache>
class Simulation {
 public:
  Simulation(const Trace& trace, Strategy strategy, const SimulatorConfig& config,
             const Clock& clock)
      : trace_(&trace),
        config_(config),
        clock_(&clock),
        allocator_(config.num_blocks, config.block_size),
        cache_(allocator_),
        next_turn_(trace.requests.size(), kNoRequest),
        admitted_before_(trace.requests.size(), 0) {
    report_.strategy = strategy;
    // Chain every request to the next request of its session; the first turn of each session
    // waits from the start. Sessions may interleave in the trace.
    std::vector<std::uint32_t> last_turn(trace.sessions.size(), kNoRequest);
    for (std::uint32_t i = 0; i < trace.requests.size(); ++i) {
      std::uint32_t& last = last_turn[trace.requests[i].session];
      if (last == kNoRequest) {
        waiting_.push_back(i);
      } else {
        next_turn_[last] = i;
      }
      last = i;
    }
  }

  SimulationReport run() {
    const std::chrono::nanoseconds start = (*clock_)();
    while (report_.inconsistency.empty() && (!waiting_.empty() || !running_.empty())) {
      while (running_.size() < config_.max_running && !waiting_.empty() &&
             admit(waiting_.front())) {
        waiting_.pop_front();
      }
      if (running_.empty()) {
        // With nothing running every cached block is evictable, so a request that fits the pool
        // is always admissible; reaching this point would be a bug.
        check(waiting_.empty(), "a request that fits the pool could not be admitted while idle");
        break;
      }
      for (std::size_t i = 0; i < running_.size();) {
        if (decode(i)) {
          ++i;  // otherwise the request at `i` preempted itself and is gone
        }
      }
      const auto active = allocator_.num_used() - cache_.stats().evictable_blocks;
      report_.peak_blocks_active =
          std::max(report_.peak_blocks_active, static_cast<std::uint32_t>(active));
      finish_completed();
      ++report_.steps;
      if (config_.check_invariants) {
        if (auto problem = cache_.check_invariants()) {
          check(false, "cache invariant violated: " + *problem);
        }
      }
    }
    final_checks();
    report_.wall_time = (*clock_)() - start;
    return report_;
  }

 private:
  template <class Op>
  auto timed(Op&& op) {
    const std::chrono::nanoseconds start = (*clock_)();
    auto result = std::forward<Op>(op)();  // NOLINT(misc-const-correctness): returned by move
    report_.cache_time += (*clock_)() - start;
    ++report_.cache_ops;
    return result;
  }

  // Tries to admit `id`; false means it does not fit right now and must keep waiting.
  bool admit(std::uint32_t id) {
    const TraceRequest& request = trace_->requests[id];
    if (pool_demand(request) > config_.num_blocks) {
      reject_session(id);
      return true;
    }
    const std::span<const TokenId> prompt = trace_->prompt(request);
    // The last prompt token is always recomputed: the engine needs its logits.
    const PrefixMatch match =
        timed([&] { return cache_.match_prefix(prompt.first(prompt.size() - 1)); });
    check_ok(timed([&] { return cache_.lock(match.node); }));
    Sequence sequence(allocator_);
    check_ok(sequence.assign_prefix(prompt.first(match.num_tokens), match.blocks));
    if (!make_room(allocator_.blocks_for(prompt.size()) - match.blocks.size())) {
      sequence.release();
      check_ok(timed([&] { return cache_.unlock(match.node); }));
      return false;
    }
    check_ok(sequence.append(prompt.subspan(match.num_tokens)));
    if (admitted_before_[id] == 0) {
      admitted_before_[id] = 1;
      report_.prompt_tokens += prompt.size();
      report_.cached_tokens += match.num_tokens;
    } else {
      report_.recompute_prompt_tokens += prompt.size();
      report_.recompute_cached_tokens += match.num_tokens;
    }

    // Cache the prompt right after prefill so that requests admitted later can share it, and move
    // the lock to the inserted path. When nothing new was cached (a block-aligned prompt whose
    // last block is already cached as a duplicate) keep the original lock, which covers exactly
    // the blocks this request uses.
    NodeHandle lock = match.node;
    const Result<InsertResult> inserted =
        timed([&] { return cache_.insert(prompt, sequence.block_table()); });
    if (check_ok(inserted) && inserted->new_blocks > 0) {
      check_ok(timed([&] { return cache_.lock(inserted->node); }));
      check_ok(timed([&] { return cache_.unlock(match.node); }));
      lock = inserted->node;
    }

    RunningRequest running{.request = id, .samples = {}, .lock = lock, .generated = 0};
    running.samples.reserve(config_.samples_per_request);
    running.samples.push_back(std::move(sequence));
    for (std::uint32_t s = 1; s < config_.samples_per_request; ++s) {
      Result<Sequence> child = running.samples.front().fork();
      if (!check_ok(child)) {
        break;
      }
      running.samples.push_back(std::move(*child));
    }
    running_.push_back(std::move(running));
    return true;
  }

  // Makes at least `blocks` blocks free by evicting from the cache; false if impossible.
  bool make_room(std::size_t blocks) {
    const std::uint32_t free = allocator_.num_free();
    if (free >= blocks) {
      return true;
    }
    timed([&] { return cache_.evict(blocks - free); });
    return allocator_.num_free() >= blocks;
  }

  // Decodes one token for every sample of running_[index]; false if that request was preempted.
  bool decode(std::size_t index) {
    const std::size_t samples = running_[index].samples.size();
    for (std::size_t sample = 0; sample < samples; ++sample) {
      const TokenId token = sample_token(running_[index], sample);
      for (;;) {
        const Result<AppendResult> appended = running_[index].samples[sample].append(token);
        if (appended) {
          report_.cow_copies += appended->copy.has_value() ? 1U : 0U;
          break;
        }
        if (make_room(1)) {
          continue;
        }
        const std::size_t newest = running_.size() - 1;
        preempt(newest);
        if (newest == index) {
          return false;
        }
      }
    }
    report_.decode_tokens += samples;
    ++running_[index].generated;
    return true;
  }

  // Recompute-style preemption: free everything and queue the request again at the front.
  void preempt(std::size_t index) {
    RunningRequest victim = std::move(running_[index]);
    running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(index));
    for (Sequence& sample : victim.samples) {
      sample.release();
    }
    check_ok(timed([&] { return cache_.unlock(victim.lock); }));
    waiting_.push_front(victim.request);
    ++report_.preemptions;
  }

  void finish_completed() {
    std::size_t kept = 0;
    for (std::size_t i = 0; i < running_.size(); ++i) {
      RunningRequest& running = running_[i];
      if (running.generated >= trace_->requests[running.request].output_tokens) {
        finish(running);
      } else {
        if (kept != i) {
          running_[kept] = std::move(running);
        }
        ++kept;
      }
    }
    running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(kept), running_.end());
  }

  void finish(RunningRequest& running) {
    for (const Sequence& sample : running.samples) {
      check_ok(timed([&] { return cache_.insert(sample.tokens(), sample.block_table()); }));
    }
    check_ok(timed([&] { return cache_.unlock(running.lock); }));
    for (Sequence& sample : running.samples) {
      sample.release();
    }
    ++report_.requests_completed;
    if (next_turn_[running.request] != kNoRequest) {
      waiting_.push_back(next_turn_[running.request]);
    }
  }

  // A request larger than the pool can never run, and the later turns of its session continue a
  // conversation that never happened: reject them too.
  void reject_session(std::uint32_t id) {
    for (std::uint32_t request = id; request != kNoRequest; request = next_turn_[request]) {
      ++report_.requests_rejected;
    }
  }

  // Upper bound on the blocks a request holds at once: sample 0 holds the whole sequence, every
  // other sample shares only the full prompt blocks.
  [[nodiscard]] std::size_t pool_demand(const TraceRequest& request) const noexcept {
    const std::size_t total =
        allocator_.blocks_for(std::size_t{request.prompt_tokens} + request.output_tokens);
    const std::size_t own = total - (request.prompt_tokens / config_.block_size);
    return total + ((config_.samples_per_request - 1) * own);
  }

  [[nodiscard]] TokenId sample_token(const RunningRequest& running,
                                     std::size_t sample) const noexcept {
    const TokenId base = trace_->output(trace_->requests[running.request])[running.generated];
    return static_cast<TokenId>((base + (sample * kSampleStride)) % trace_->vocab_size);
  }

  void final_checks() {
    if (auto problem = cache_.check_invariants()) {
      check(false, "cache invariant violated: " + *problem);
    }
    const CacheStats stats = cache_.stats();
    check(stats.locked_blocks == 0, "cache blocks are still locked after the last request");
    check(allocator_.num_used() == stats.cached_blocks,
          "blocks leaked: some allocated blocks are held by no cache");
    report_.evicted_blocks = stats.evicted_blocks;
    report_.peak_blocks_used = allocator_.peak_used();
    cache_.evict(std::numeric_limits<std::size_t>::max());
    check(allocator_.num_used() == 0, "blocks are still allocated after draining the cache");
    report_.consistent = report_.inconsistency.empty();
  }

  void check(bool ok, std::string_view what) {
    if (!ok && report_.inconsistency.empty()) {
      report_.inconsistency = what;
    }
  }

  template <class R>
  bool check_ok(const R& result) {
    if (!result) {
      check(false, "unexpected error: " + std::string(to_string(result.error())));
    }
    return result.has_value();
  }

  const Trace* trace_;
  SimulatorConfig config_;
  const Clock* clock_;
  BlockAllocator allocator_;  // declared before everything that holds blocks
  Cache cache_;
  std::vector<std::uint32_t> next_turn_;
  std::vector<std::uint8_t> admitted_before_;  // per request: 1 after its first admission
  std::deque<std::uint32_t> waiting_;
  std::vector<RunningRequest> running_;
  SimulationReport report_;
};

template <PrefixCache Cache>
SimulationReport run_with(const Trace& trace, Strategy strategy, const SimulatorConfig& config,
                          const Clock& clock) {
  Simulation<Cache> simulation(trace, strategy, config, clock);
  return simulation.run();
}

}  // namespace

double SimulationReport::hit_rate() const noexcept {
  return prompt_tokens == 0
             ? 0.0
             : static_cast<double>(cached_tokens) / static_cast<double>(prompt_tokens);
}

double SimulationReport::cache_ops_per_second() const noexcept {
  const auto nanos = cache_time.count();
  return nanos <= 0 ? 0.0 : static_cast<double>(cache_ops) * 1e9 / static_cast<double>(nanos);
}

Clock steady_clock() {
  return [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch());
  };
}

SimulationReport simulate(const Trace& trace, Strategy strategy, const SimulatorConfig& config,
                          const Clock& clock) {
  validate(trace, config);
  if (!clock) {
    throw std::invalid_argument("simulate: clock must not be empty");
  }
  switch (strategy) {
    case Strategy::none:
      return run_with<NullPrefixCache>(trace, strategy, config, clock);
    case Strategy::hash:
      return run_with<HashPrefixCache>(trace, strategy, config, clock);
    case Strategy::radix:
      return run_with<RadixPrefixCache>(trace, strategy, config, clock);
  }
  throw std::invalid_argument("simulate: unknown strategy");
}

}  // namespace radix_kv
