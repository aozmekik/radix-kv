// Randomized invariant tests with fixed seeds (see tests/support/op_driver.hpp for the properties).

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "radix_kv/hash_prefix_cache.hpp"
#include "radix_kv/radix_prefix_cache.hpp"

#include "support/op_driver.hpp"

namespace radix_kv {
namespace {

using test_support::DriverOptions;
using test_support::DriverStats;
using test_support::NaivePrefixCache;
using test_support::OpDriver;

constexpr std::uint32_t kSeeds = 40;
constexpr std::size_t kBytesPerRun = 1500;

// std::mt19937_64's output sequence is fixed by the standard, so these inputs are portable.
std::vector<std::uint8_t> random_bytes(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 engine(seed);
  std::vector<std::uint8_t> bytes(count);
  for (std::uint8_t& byte : bytes) {
    byte = static_cast<std::uint8_t>(engine() >> 56U);
  }
  return bytes;
}

struct Variant {
  const char* name;
  DriverOptions options;
};

constexpr std::array kVariants{
    Variant{.name = "tokens", .options = {.num_blocks = 40, .block_size = 1}},
    Variant{.name = "pairs", .options = {.num_blocks = 48, .block_size = 2}},
    Variant{.name = "quads", .options = {.num_blocks = 20, .block_size = 4}},
    Variant{.name = "tight_pool", .options = {.num_blocks = 12, .block_size = 2}},
    Variant{.name = "roomy_pool",
            .options = {.num_blocks = 160, .block_size = 2, .max_sequences = 12}},
    Variant{.name = "wide_vocab", .options = {.num_blocks = 48, .block_size = 2, .vocab_size = 40}},
    Variant{.name = "collisions",
            .options = {.num_blocks = 48, .block_size = 2, .hasher = &test_support::weak_hasher}},
    Variant{.name = "token_collisions",
            .options = {.num_blocks = 40, .block_size = 1, .hasher = &test_support::weak_hasher}},
};

class InvariantSuite : public ::testing::TestWithParam<Variant> {
 protected:
  template <class Cache>
  static void run_all_seeds() {
    DriverStats total;
    for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
      OpDriver<Cache> driver(GetParam().options);
      ASSERT_EQ(driver.run(random_bytes(seed, kBytesPerRun)), std::nullopt) << "seed " << seed;
      total += driver.stats();
    }
    // Guard against vacuous runs: the interesting paths must actually have been exercised.
    EXPECT_GT(total.matches_with_hits, 100U);
    EXPECT_GT(total.evicted_blocks, 10U);
    if (GetParam().options.block_size > 1) {  // with one token per block nothing is ever partial
      EXPECT_GT(total.copy_on_writes, 10U);
    }
    if (GetParam().options.num_blocks < 100) {
      EXPECT_GT(total.failed_appends, 0U);
    }
  }
};

TEST_P(InvariantSuite, HashPrefixCache) {
  run_all_seeds<HashPrefixCache>();
}

TEST_P(InvariantSuite, RadixPrefixCache) {
  run_all_seeds<RadixPrefixCache>();
}

INSTANTIATE_TEST_SUITE_P(Property, InvariantSuite, ::testing::ValuesIn(kVariants),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                           return std::string(info.param.name);
                         });

TEST(Property, NaiveReferenceModelIsSelfConsistent) {
  for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
    OpDriver<NaivePrefixCache> driver(
        DriverOptions{.num_blocks = 256, .block_size = 2, .allow_evict = false});
    ASSERT_EQ(driver.run(random_bytes(seed, kBytesPerRun)), std::nullopt) << "seed " << seed;
  }
}

// Without eviction, all three strategies cache exactly the same block-aligned prefixes, so every
// lookup must return the same number of tokens (block-aligned shared prefixes hit identically).
class DifferentialSuite : public ::testing::TestWithParam<std::uint32_t> {};

TEST_P(DifferentialSuite, StrategiesReturnIdenticalHitsWithoutEviction) {
  const DriverOptions options{
      .num_blocks = 256, .block_size = GetParam(), .max_sequences = 10, .allow_evict = false};
  for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
    const std::vector<std::uint8_t> input = random_bytes(1000 + seed, kBytesPerRun);
    OpDriver<HashPrefixCache> hash(options);
    OpDriver<RadixPrefixCache> radix(options);
    OpDriver<NaivePrefixCache> naive(options);
    ASSERT_EQ(hash.run(input), std::nullopt) << "seed " << seed;
    ASSERT_EQ(radix.run(input), std::nullopt) << "seed " << seed;
    ASSERT_EQ(naive.run(input), std::nullopt) << "seed " << seed;
    ASSERT_FALSE(naive.match_lengths().empty());
    EXPECT_EQ(hash.match_lengths(), naive.match_lengths()) << "seed " << seed;
    EXPECT_EQ(radix.match_lengths(), naive.match_lengths()) << "seed " << seed;
  }
}

INSTANTIATE_TEST_SUITE_P(Property, DifferentialSuite, ::testing::Values(1U, 2U, 4U),
                         [](const ::testing::TestParamInfo<std::uint32_t>& info) {
                           return "block_size_" + std::to_string(info.param);
                         });

}  // namespace
}  // namespace radix_kv
