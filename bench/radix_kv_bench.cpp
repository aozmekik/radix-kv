// Microbenchmarks of the allocator, sequences and both prefix caches, plus end-to-end simulations.
// All inputs are synthetic.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/hash_prefix_cache.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/radix_prefix_cache.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/simulator.hpp"
#include "radix_kv/types.hpp"
#include "radix_kv/workload.hpp"

namespace {

using radix_kv::BlockAllocator;
using radix_kv::BlockId;
using radix_kv::HashPrefixCache;
using radix_kv::PrefixMatch;
using radix_kv::RadixPrefixCache;
using radix_kv::Sequence;
using radix_kv::TokenId;

constexpr std::uint32_t kBlockSize = 16;

std::vector<TokenId> tokens_from(TokenId first, std::size_t count) {
  std::vector<TokenId> tokens(count);
  std::iota(tokens.begin(), tokens.end(), first);
  return tokens;
}

void bm_allocate_release(benchmark::State& state) {
  BlockAllocator allocator(1024, kBlockSize);
  for ([[maybe_unused]] const auto _ : state) {
    auto block = allocator.allocate();
    benchmark::DoNotOptimize(block);
    auto remaining = allocator.release(*block);
    benchmark::DoNotOptimize(remaining);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(bm_allocate_release);

// One decode step per iteration; the sequence is reset when it reaches 4096 tokens.
void bm_decode_append(benchmark::State& state) {
  BlockAllocator allocator(512, kBlockSize);
  Sequence sequence(allocator);
  TokenId token = 0;
  for ([[maybe_unused]] const auto _ : state) {
    if (sequence.size() == 4096) {
      sequence.release();
    }
    auto appended = sequence.append(token++);
    benchmark::DoNotOptimize(appended);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(bm_decode_append);

// Fork a 1000-token sequence and decode one token in the fork (copy-on-write of the last block).
void bm_fork_and_copy_on_write(benchmark::State& state) {
  BlockAllocator allocator(1024, kBlockSize);
  Sequence parent(allocator);
  if (!parent.append(tokens_from(0, 1000))) {
    state.SkipWithError("setup failed");
    return;
  }
  for ([[maybe_unused]] const auto _ : state) {
    auto child = parent.fork();
    auto appended = child->append(TokenId{7});
    benchmark::DoNotOptimize(appended);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(bm_fork_and_copy_on_write);

// Cached prefix of state.range(0) tokens; each iteration matches the whole prefix (a full hit).
template <class Cache>
void bm_match_hit(benchmark::State& state) {
  const auto prefix_tokens = static_cast<std::size_t>(state.range(0));
  BlockAllocator allocator(static_cast<std::uint32_t>(prefix_tokens / kBlockSize) + 8, kBlockSize);
  Cache cache(allocator);
  Sequence sequence(allocator);
  const std::vector<TokenId> tokens = tokens_from(1, prefix_tokens);
  if (!sequence.append(tokens) || !cache.insert(sequence.tokens(), sequence.block_table())) {
    state.SkipWithError("setup failed");
    return;
  }
  for ([[maybe_unused]] const auto _ : state) {
    PrefixMatch match = cache.match_prefix(tokens);
    benchmark::DoNotOptimize(match);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(prefix_tokens));
}
BENCHMARK(bm_match_hit<HashPrefixCache>)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK(bm_match_hit<RadixPrefixCache>)->Arg(256)->Arg(2048)->Arg(16384);

// 1024 cached requests share a 1024-token system prompt and differ afterwards; each iteration
// matches one of them (fan-out lookup below the shared prefix).
template <class Cache>
void bm_match_shared_prefix_fan_out(benchmark::State& state) {
  constexpr std::size_t kBranches = 1024;
  constexpr std::size_t kSystem = 1024;
  constexpr std::size_t kSuffix = 64;
  BlockAllocator allocator(
      static_cast<std::uint32_t>((kSystem + (kBranches * kSuffix)) / kBlockSize) + 8, kBlockSize);
  Cache cache(allocator);
  const std::vector<TokenId> system = tokens_from(1'000'000, kSystem);
  std::vector<std::vector<TokenId>> prompts;
  for (std::size_t b = 0; b < kBranches; ++b) {
    std::vector<TokenId> prompt = system;
    const std::vector<TokenId> suffix = tokens_from(static_cast<TokenId>(b * kSuffix), kSuffix);
    prompt.insert(prompt.end(), suffix.begin(), suffix.end());
    const PrefixMatch hit = cache.match_prefix(prompt);
    Sequence sequence(allocator);
    if (!sequence.assign_prefix(std::span<const TokenId>(prompt).first(hit.num_tokens),
                                hit.blocks) ||
        !sequence.append(std::span<const TokenId>(prompt).subspan(hit.num_tokens)) ||
        !cache.insert(sequence.tokens(), sequence.block_table())) {
      state.SkipWithError("setup failed");
      return;
    }
    prompts.push_back(std::move(prompt));
  }
  std::size_t next = 0;
  for ([[maybe_unused]] const auto _ : state) {
    PrefixMatch match = cache.match_prefix(prompts[next]);
    benchmark::DoNotOptimize(match);
    next = (next + 1) % kBranches;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(bm_match_shared_prefix_fan_out<HashPrefixCache>);
BENCHMARK(bm_match_shared_prefix_fan_out<RadixPrefixCache>);

// Insert a fresh state.range(0)-token sequence under a shared 512-token prefix, then evict it.
template <class Cache>
void bm_insert_then_evict(benchmark::State& state) {
  const auto suffix_tokens = static_cast<std::size_t>(state.range(0));
  BlockAllocator allocator(static_cast<std::uint32_t>((512 + suffix_tokens) / kBlockSize) + 8,
                           kBlockSize);
  Cache cache(allocator);
  Sequence shared(allocator);
  if (!shared.append(tokens_from(1, 512)) || !cache.insert(shared.tokens(), shared.block_table())) {
    state.SkipWithError("setup failed");
    return;
  }
  const PrefixMatch pinned = cache.match_prefix(shared.tokens());
  if (!cache.lock(pinned.node)) {  // keep the shared prefix cached across iterations
    state.SkipWithError("setup failed");
    return;
  }
  const std::vector<TokenId> suffix = tokens_from(1'000'000, suffix_tokens);
  for ([[maybe_unused]] const auto _ : state) {
    Sequence sequence = *shared.fork();
    auto appended = sequence.append(suffix);
    auto inserted = cache.insert(sequence.tokens(), sequence.block_table());
    sequence.release();
    std::size_t evicted = cache.evict(std::numeric_limits<std::size_t>::max());
    benchmark::DoNotOptimize(appended);
    benchmark::DoNotOptimize(inserted);
    benchmark::DoNotOptimize(evicted);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(bm_insert_then_evict<HashPrefixCache>)->Arg(256)->Arg(2048);
BENCHMARK(bm_insert_then_evict<RadixPrefixCache>)->Arg(256)->Arg(2048);

// End-to-end scheduler simulation of a synthetic few-shot trace (200 requests).
void bm_simulate(benchmark::State& state) {
  const auto strategy = static_cast<radix_kv::Strategy>(state.range(0));
  const radix_kv::Trace trace =
      radix_kv::generate_trace(radix_kv::FewShotWorkload{.requests = 200}, /*seed=*/1);
  const radix_kv::SimulatorConfig config{.num_blocks = 2048, .block_size = kBlockSize};
  for ([[maybe_unused]] const auto _ : state) {
    radix_kv::SimulationReport report = radix_kv::simulate(trace, strategy, config);
    benchmark::DoNotOptimize(report);
  }
  state.SetLabel(std::string(radix_kv::to_string(strategy)));
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(trace.requests.size()));
}
BENCHMARK(bm_simulate)
    ->Arg(static_cast<int>(radix_kv::Strategy::none))
    ->Arg(static_cast<int>(radix_kv::Strategy::hash))
    ->Arg(static_cast<int>(radix_kv::Strategy::radix))
    ->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
