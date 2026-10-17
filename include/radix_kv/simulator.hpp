#pragma once

/// @file
/// Trace-driven scheduler simulation comparing prefix-caching strategies.

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "radix_kv/workload.hpp"

namespace radix_kv {

/// Prefix-caching strategy used by a simulation.
enum class Strategy : std::uint8_t {
  none,   ///< `NullPrefixCache`: no prefix caching (baseline).
  hash,   ///< `HashPrefixCache`: vLLM-style chained block hashes.
  radix,  ///< `RadixPrefixCache`: SGLang-style radix tree.
};

/// `"none"`, `"hash"` or `"radix"`.
[[nodiscard]] constexpr std::string_view to_string(Strategy strategy) noexcept {
  switch (strategy) {
    case Strategy::none:
      return "none";
    case Strategy::hash:
      return "hash";
    case Strategy::radix:
      return "radix";
  }
  return "unknown";
}

/// Scheduler and memory parameters of a simulation.
struct SimulatorConfig {
  std::uint32_t num_blocks = 4096;        ///< KV-cache pool size in blocks.
  std::uint32_t block_size = 16;          ///< Tokens per block.
  std::uint32_t max_running = 32;         ///< Maximum concurrently running requests.
  std::uint32_t samples_per_request = 1;  ///< Parallel samples, via `fork()` + copy-on-write.
  bool check_invariants = false;          ///< Validate the cache after every step (slow).
};

/// Counters produced by `simulate()`. All token and block numbers come from the synthetic trace.
struct SimulationReport {
  Strategy strategy = Strategy::none;    ///< Strategy that produced the report.
  std::uint64_t requests_completed = 0;  ///< Requests that generated all their output tokens.
  std::uint64_t requests_rejected = 0;   ///< Larger than the whole pool, with later session turns.
  std::uint64_t preemptions = 0;         ///< Requests evicted from the running batch (recomputed).
  std::uint64_t steps = 0;               ///< Scheduler iterations (one decode token per sample).
  std::uint64_t prompt_tokens = 0;       ///< Prompt tokens looked up at first admission.
  std::uint64_t cached_tokens = 0;       ///< Of those, tokens served by the prefix cache.
  std::uint64_t recompute_prompt_tokens = 0;  ///< Prompt tokens looked up again after preemption.
  std::uint64_t recompute_cached_tokens = 0;  ///< Of those, tokens served by the prefix cache.
  std::uint64_t decode_tokens = 0;     ///< Tokens generated, including work lost to preemption.
  std::uint64_t evicted_blocks = 0;    ///< Blocks released by cache eviction.
  std::uint64_t cow_copies = 0;        ///< Copy-on-write block copies.
  std::uint32_t peak_blocks_used = 0;  ///< Allocator high-water mark, idle cached blocks included.
  std::uint32_t peak_blocks_active = 0;  ///< High-water mark of blocks not reclaimable by eviction.
  std::uint64_t cache_ops = 0;           ///< match/insert/lock/unlock/evict calls.
  std::chrono::nanoseconds cache_time{0};  ///< Time spent inside those calls.
  std::chrono::nanoseconds wall_time{0};   ///< Time of the whole simulation.
  bool consistent = false;                 ///< All end-of-run (and per-step) checks passed.
  std::string inconsistency;               ///< First failed check, if any.

  /// Prefix-cache hit rate over first admissions: `cached_tokens / prompt_tokens` (0 when nothing
  /// was looked up). Re-admissions after preemption are counted separately because they mostly hit
  /// their own prompt and would make smaller pools look better.
  [[nodiscard]] double hit_rate() const noexcept;

  /// Cache operations per second of time spent inside them (0 when no time was measured).
  [[nodiscard]] double cache_ops_per_second() const noexcept;
};

/// Monotonic time source, injectable so tests do not depend on wall-clock timing.
using Clock = std::function<std::chrono::nanoseconds()>;

/// `std::chrono::steady_clock`, the default `Clock`.
[[nodiscard]] Clock steady_clock();

/// Replays `trace` through a continuous-batching scheduler backed by a `BlockAllocator` and the
/// prefix cache selected by `strategy`.
///
/// Per step the scheduler admits waiting requests first-come first-served while fewer than
/// `max_running` run and their prompt fits (evicting from the cache when needed), then every
/// running sample decodes one token. A request that cannot get a block preempts the most recently
/// admitted request, which is re-queued at the front and recomputed later. The last prompt token
/// is never served from the cache (the engine needs its logits). Prompt blocks are inserted into
/// the cache right after prefill, the full sequence when the request finishes; requests lock the
/// cached path they use. Turn `k + 1` of a chat session is queued when turn `k` finishes.
///
/// After the run every block must be back in the pool once the cache is drained;
/// `SimulationReport::consistent` reports this and the other end-of-run checks.
/// @throws std::invalid_argument for a zero field in `config` or a malformed `trace`.
[[nodiscard]] SimulationReport simulate(const Trace& trace, Strategy strategy,
                                        const SimulatorConfig& config,
                                        const Clock& clock = steady_clock());

}  // namespace radix_kv
