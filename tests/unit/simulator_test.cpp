#include "radix_kv/simulator.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "radix_kv/workload.hpp"

namespace radix_kv {
namespace {

using std::chrono::microseconds;
using std::chrono::nanoseconds;

constexpr std::array kCachingStrategies{Strategy::hash, Strategy::radix};
constexpr std::array kAllStrategies{Strategy::none, Strategy::hash, Strategy::radix};

Trace small_chat(std::uint64_t seed = 1) {
  return generate_trace(ChatWorkload{.sessions = 24,
                                     .system_prompt_tokens = 100,
                                     .turns = LengthDistribution::uniform(1, 4),
                                     .user_tokens = LengthDistribution::uniform(5, 40),
                                     .output_tokens = LengthDistribution::uniform(1, 30)},
                        seed);
}

Trace small_few_shot(std::uint64_t seed = 2) {
  return generate_trace(FewShotWorkload{.requests = 80,
                                        .templates = 3,
                                        .template_tokens = LengthDistribution::uniform(60, 200),
                                        .template_zipf = 1.0,
                                        .question_tokens = LengthDistribution::uniform(3, 20),
                                        .output_tokens = LengthDistribution::uniform(1, 12)},
                        seed);
}

Trace small_rag(std::uint64_t seed = 3) {
  return generate_trace(RagWorkload{.requests = 80,
                                    .system_prompt_tokens = 40,
                                    .documents = 12,
                                    .document_tokens = LengthDistribution::uniform(20, 80),
                                    .documents_per_query = 2,
                                    .document_zipf = 1.2,
                                    .query_tokens = LengthDistribution::uniform(3, 10),
                                    .output_tokens = LengthDistribution::uniform(1, 12)},
                        seed);
}

// Advances by one microsecond per reading, so every timed section lasts exactly one tick.
Clock ticking_clock() {
  return [now = nanoseconds{0}]() mutable {
    now += microseconds{1};
    return now;
  };
}

void expect_same_counters(const SimulationReport& a, const SimulationReport& b) {
  EXPECT_EQ(a.requests_completed, b.requests_completed);
  EXPECT_EQ(a.requests_rejected, b.requests_rejected);
  EXPECT_EQ(a.preemptions, b.preemptions);
  EXPECT_EQ(a.steps, b.steps);
  EXPECT_EQ(a.prompt_tokens, b.prompt_tokens);
  EXPECT_EQ(a.cached_tokens, b.cached_tokens);
  EXPECT_EQ(a.recompute_prompt_tokens, b.recompute_prompt_tokens);
  EXPECT_EQ(a.recompute_cached_tokens, b.recompute_cached_tokens);
  EXPECT_EQ(a.decode_tokens, b.decode_tokens);
  EXPECT_EQ(a.evicted_blocks, b.evicted_blocks);
  EXPECT_EQ(a.cow_copies, b.cow_copies);
  EXPECT_EQ(a.peak_blocks_used, b.peak_blocks_used);
  EXPECT_EQ(a.peak_blocks_active, b.peak_blocks_active);
}

TEST(Simulate, RejectsInvalidInput) {
  const Trace trace = small_few_shot();
  EXPECT_THROW(static_cast<void>(simulate(trace, Strategy::hash, SimulatorConfig{.num_blocks = 0})),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(simulate(trace, Strategy::hash, SimulatorConfig{.max_running = 0})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(simulate(trace, Strategy::hash, SimulatorConfig{.samples_per_request = 0})),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(simulate(trace, Strategy::hash, SimulatorConfig{}, Clock{})),
               std::invalid_argument);

  Trace broken = trace;
  broken.requests.front().output_tokens = 1'000'000;
  EXPECT_THROW(static_cast<void>(simulate(broken, Strategy::hash, SimulatorConfig{})),
               std::invalid_argument);
  broken = trace;
  broken.requests.front().session = 1'000'000;
  EXPECT_THROW(static_cast<void>(simulate(broken, Strategy::hash, SimulatorConfig{})),
               std::invalid_argument);
}

TEST(Simulate, BaselineWithoutCachingNeverHits) {
  const Trace trace = small_few_shot();
  const SimulationReport report = simulate(trace, Strategy::none, SimulatorConfig{});
  EXPECT_TRUE(report.consistent) << report.inconsistency;
  EXPECT_EQ(report.requests_completed, trace.requests.size());
  EXPECT_EQ(report.prompt_tokens, trace.total_prompt_tokens());
  EXPECT_EQ(report.cached_tokens, 0U);
  EXPECT_DOUBLE_EQ(report.hit_rate(), 0.0);
}

TEST(Simulate, StrategiesAgreeExactlyWithoutMemoryPressure) {
  const SimulatorConfig roomy{.num_blocks = 1U << 14U, .block_size = 16, .max_running = 8};
  for (const Trace& trace : {small_chat(), small_few_shot(), small_rag()}) {
    const SimulationReport hash = simulate(trace, Strategy::hash, roomy);
    const SimulationReport radix = simulate(trace, Strategy::radix, roomy);
    ASSERT_TRUE(hash.consistent) << hash.inconsistency;
    ASSERT_TRUE(radix.consistent) << radix.inconsistency;
    EXPECT_EQ(hash.evicted_blocks, 0U);
    EXPECT_GT(hash.cached_tokens, 0U);
    expect_same_counters(hash, radix);
  }
}

TEST(Simulate, SharedPrefixesAreServedFromTheCache) {
  const Trace trace = small_few_shot();
  for (const Strategy strategy : kCachingStrategies) {
    const SimulationReport report = simulate(trace, strategy, SimulatorConfig{});
    EXPECT_TRUE(report.consistent) << report.inconsistency;
    EXPECT_GT(report.hit_rate(), 0.5) << to_string(strategy);
    EXPECT_LE(report.cached_tokens, report.prompt_tokens);
  }
}

TEST(Simulate, RandomPromptsShareNothing) {
  const Trace trace = generate_trace(RandomWorkload{.requests = 60}, 4);
  for (const Strategy strategy : kAllStrategies) {
    const SimulationReport report = simulate(trace, strategy, SimulatorConfig{});
    EXPECT_TRUE(report.consistent) << report.inconsistency;
    EXPECT_EQ(report.cached_tokens, 0U) << to_string(strategy);
  }
}

TEST(Simulate, MemoryPressureEvictsAndPreemptsButStaysConsistent) {
  const Trace trace = small_chat();
  const SimulatorConfig tight{.num_blocks = 64, .block_size = 8, .max_running = 8};
  for (const Strategy strategy : kAllStrategies) {
    const SimulationReport report = simulate(trace, strategy, tight);
    EXPECT_TRUE(report.consistent) << to_string(strategy) << ": " << report.inconsistency;
    EXPECT_EQ(report.requests_completed, trace.requests.size());  // the pool fits every request
    EXPECT_GT(report.preemptions, 0U) << to_string(strategy);
    EXPECT_GT(report.recompute_prompt_tokens, 0U) << to_string(strategy);
    EXPECT_EQ(report.prompt_tokens, trace.total_prompt_tokens());  // first admissions only
    EXPECT_LE(report.peak_blocks_used, tight.num_blocks);
    if (strategy != Strategy::none) {
      EXPECT_GT(report.evicted_blocks, 0U) << to_string(strategy);
    }
  }
}

TEST(Simulate, RequestsLargerThanThePoolAreRejectedWithTheirSession) {
  const Trace trace = small_chat();  // every prompt starts with a 100-token system prompt
  const SimulatorConfig tiny{.num_blocks = 4, .block_size = 16, .max_running = 4};
  const SimulationReport report = simulate(trace, Strategy::radix, tiny);
  EXPECT_TRUE(report.consistent) << report.inconsistency;
  EXPECT_EQ(report.requests_completed, 0U);
  EXPECT_EQ(report.requests_rejected, trace.requests.size());
  EXPECT_EQ(report.prompt_tokens, 0U);
}

TEST(Simulate, ParallelSamplesShareThePromptAndCopyOnWrite) {
  const Trace trace = small_few_shot();
  const SimulatorConfig sampled{.samples_per_request = 4};
  for (const Strategy strategy : kAllStrategies) {
    const SimulationReport report = simulate(trace, strategy, sampled);
    EXPECT_TRUE(report.consistent) << report.inconsistency;
    EXPECT_EQ(report.preemptions, 0U);
    EXPECT_EQ(report.decode_tokens, 4 * trace.total_output_tokens());
    EXPECT_GT(report.cow_copies, 0U);
  }
}

TEST(Simulate, InjectedClockMakesTimingDeterministic) {
  const SimulationReport report =
      simulate(small_few_shot(), Strategy::hash, SimulatorConfig{}, ticking_clock());
  ASSERT_GT(report.cache_ops, 0U);
  EXPECT_EQ(report.cache_time, microseconds{1} * report.cache_ops);
  EXPECT_DOUBLE_EQ(report.cache_ops_per_second(), 1e6);
  EXPECT_GT(report.wall_time, report.cache_time);
}

TEST(Simulate, PerStepInvariantChecksPass) {
  const SimulatorConfig checked{
      .num_blocks = 64, .block_size = 4, .max_running = 6, .check_invariants = true};
  for (const Trace& trace : {small_chat(5), small_rag(6)}) {
    for (const Strategy strategy : kCachingStrategies) {
      const SimulationReport report = simulate(trace, strategy, checked);
      EXPECT_TRUE(report.consistent) << to_string(strategy) << ": " << report.inconsistency;
    }
  }
}

TEST(Simulate, IsDeterministic) {
  const Trace trace = small_rag();
  const SimulatorConfig config{.num_blocks = 128, .block_size = 8};
  for (const Strategy strategy : kAllStrategies) {
    expect_same_counters(simulate(trace, strategy, config), simulate(trace, strategy, config));
  }
}

TEST(Simulate, IndependentSimulationsRunConcurrently) {
  const Trace trace = small_chat();
  const SimulatorConfig config{.num_blocks = 96, .block_size = 8, .max_running = 6};
  std::vector<SimulationReport> sequential;
  sequential.reserve(kAllStrategies.size());
  for (const Strategy strategy : kAllStrategies) {
    sequential.push_back(simulate(trace, strategy, config));
  }
  std::vector<SimulationReport> concurrent(kAllStrategies.size());
  std::vector<std::thread> workers;
  workers.reserve(kAllStrategies.size());
  for (std::size_t i = 0; i < kAllStrategies.size(); ++i) {
    workers.emplace_back([&, i] { concurrent[i] = simulate(trace, kAllStrategies[i], config); });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  for (std::size_t i = 0; i < kAllStrategies.size(); ++i) {
    EXPECT_TRUE(concurrent[i].consistent) << concurrent[i].inconsistency;
    expect_same_counters(sequential[i], concurrent[i]);
  }
}

TEST(SimulationReport, RatesOfAnEmptyReportAreZero) {
  const SimulationReport empty;
  EXPECT_DOUBLE_EQ(empty.hit_rate(), 0.0);
  EXPECT_DOUBLE_EQ(empty.cache_ops_per_second(), 0.0);
  EXPECT_EQ(to_string(Strategy::radix), "radix");
}

}  // namespace
}  // namespace radix_kv
