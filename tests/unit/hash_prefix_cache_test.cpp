// Behaviour specific to the chained-hash strategy (one node per block, exact eviction).

#include "radix_kv/hash_prefix_cache.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {
namespace {

TEST(HashPrefixCache, RejectsANullHasher) {
  BlockAllocator allocator(4, 2);
  EXPECT_THROW(HashPrefixCache(allocator, nullptr), std::invalid_argument);
}

TEST(HashPrefixCache, EveryCachedBlockIsANode) {
  BlockAllocator allocator(16, 2);
  HashPrefixCache cache(allocator);
  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.append(std::vector<TokenId>{1, 2, 3, 4, 5, 6}));
  ASSERT_TRUE(cache.insert(sequence.tokens(), sequence.block_table()));
  const CacheStats stats = cache.stats();
  EXPECT_EQ(stats.cached_blocks, 3U);
  EXPECT_EQ(stats.nodes, 3U);
}

TEST(HashPrefixCache, EvictsExactlyTheRequestedBlocksDeepestFirst) {
  BlockAllocator allocator(16, 2);
  HashPrefixCache cache(allocator);
  {
    Sequence sequence(allocator);
    ASSERT_TRUE(sequence.append(std::vector<TokenId>{1, 2, 3, 4, 5, 6}));
    ASSERT_TRUE(cache.insert(sequence.tokens(), sequence.block_table()));
  }
  EXPECT_EQ(cache.evict(1), 1U);
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 2, 3, 4, 5, 6}).num_tokens, 4U);
  EXPECT_EQ(cache.evict(1), 1U);
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 2, 3, 4, 5, 6}).num_tokens, 2U);
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
}

TEST(HashPrefixCache, SharedAncestorOutlivesItsLeaves) {
  BlockAllocator allocator(16, 2);
  HashPrefixCache cache(allocator);
  {
    Sequence left(allocator);
    Sequence right(allocator);
    ASSERT_TRUE(left.append(std::vector<TokenId>{1, 1, 2, 2}));
    ASSERT_TRUE(right.append(std::vector<TokenId>{1, 1, 3, 3}));
    ASSERT_TRUE(cache.insert(left.tokens(), left.block_table()));
    ASSERT_TRUE(cache.insert(right.tokens(), right.block_table()));
  }
  EXPECT_EQ(cache.stats().cached_blocks, 3U);
  EXPECT_EQ(cache.evict(2), 2U);  // both leaves go before their shared parent
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 1, 2, 2}).num_tokens, 2U);
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 1, 3, 3}).num_tokens, 2U);
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
}

TEST(HashPrefixCache, ChainCollisionStopsCachingButStaysCorrect) {
  BlockAllocator allocator(16, 2);
  // Equal hashes for every block: only the first block of the first sequence can be indexed.
  HashPrefixCache cache(
      allocator, [](std::uint64_t, std::span<const TokenId>) noexcept { return std::uint64_t{7}; });
  Sequence first(allocator);
  Sequence second(allocator);
  ASSERT_TRUE(first.append(std::vector<TokenId>{1, 2, 3, 4}));
  ASSERT_TRUE(second.append(std::vector<TokenId>{5, 6, 7, 8}));
  EXPECT_EQ(cache.insert(first.tokens(), first.block_table())->new_blocks, 1U);
  EXPECT_EQ(cache.insert(second.tokens(), second.block_table())->new_blocks, 0U);
  EXPECT_EQ(cache.match_prefix(first.tokens()).num_tokens, 2U);
  EXPECT_EQ(cache.match_prefix(second.tokens()).num_tokens, 0U);
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
}

TEST(HashTokens, DependsOnSeedOrderAndLength) {
  const std::vector<TokenId> ab{1, 2};
  const std::vector<TokenId> ba{2, 1};
  const std::vector<TokenId> abc{1, 2, 0};
  EXPECT_EQ(hash_tokens(0, ab), hash_tokens(0, ab));
  EXPECT_NE(hash_tokens(0, ab), hash_tokens(1, ab));
  EXPECT_NE(hash_tokens(0, ab), hash_tokens(0, ba));
  EXPECT_NE(hash_tokens(0, ab), hash_tokens(0, abc));
  EXPECT_NE(hash_tokens(0, std::span<const TokenId>{}), hash_tokens(0, std::vector<TokenId>{0}));
}

}  // namespace
}  // namespace radix_kv
