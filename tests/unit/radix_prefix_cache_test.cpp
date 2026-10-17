// Behaviour specific to the radix tree (path compression, node splits, whole-leaf eviction).

#include "radix_kv/radix_prefix_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
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

class RadixPrefixCacheTest : public ::testing::Test {
 protected:
  // Caches `tokens` through a temporary sequence; afterwards only the tree holds the blocks.
  InsertResult cache_tokens(const std::vector<TokenId>& tokens) {
    Sequence sequence(allocator);
    EXPECT_TRUE(sequence.append(tokens));
    return cache.insert(sequence.tokens(), sequence.block_table()).value();
  }

  void expect_valid() const { EXPECT_EQ(cache.check_invariants(), std::nullopt); }

  BlockAllocator allocator{64, 2};
  RadixPrefixCache cache{allocator};
};

TEST(RadixPrefixCache, RejectsANullHasher) {
  BlockAllocator allocator(4, 2);
  EXPECT_THROW(RadixPrefixCache(allocator, nullptr), std::invalid_argument);
}

TEST_F(RadixPrefixCacheTest, OneInsertIsOneNode) {
  cache_tokens({1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_EQ(cache.stats().nodes, 1U);
  EXPECT_EQ(cache.stats().cached_blocks, 4U);
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, MatchEndingInsideANodeSplitsIt) {
  cache_tokens({1, 2, 3, 4, 5, 6});
  const PrefixMatch match = cache.match_prefix(std::vector<TokenId>{1, 2, 3, 4, 9, 9});
  EXPECT_EQ(match.num_tokens, 4U);
  EXPECT_EQ(cache.stats().nodes, 2U);
  ASSERT_TRUE(cache.lock(match.node));
  EXPECT_EQ(cache.stats().locked_blocks, 2U);  // exactly the matched prefix
  ASSERT_TRUE(cache.unlock(match.node));
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, DivergingInsertSplitsAndAddsASibling) {
  cache_tokens({1, 2, 3, 4, 5, 6});
  const InsertResult inserted = cache_tokens({1, 2, 3, 4, 7, 8});
  EXPECT_EQ(inserted.new_blocks, 1U);
  EXPECT_EQ(cache.stats().nodes, 3U);  // [1 2 3 4] -> {[5 6], [7 8]}
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 2, 3, 4, 7, 8}).num_tokens, 6U);
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 2, 3, 4, 5, 6}).num_tokens, 6U);
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, InsertingAPrefixOfANodeSplitsIt) {
  cache_tokens({1, 2, 3, 4, 5, 6});
  const InsertResult inserted = cache_tokens({1, 2});
  EXPECT_EQ(inserted.new_blocks, 0U);
  EXPECT_EQ(cache.stats().nodes, 2U);
  ASSERT_TRUE(cache.lock(inserted.node));
  EXPECT_EQ(cache.stats().locked_blocks, 1U);
  ASSERT_TRUE(cache.unlock(inserted.node));
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, EvictionRemovesWholeLeaves) {
  cache_tokens({1, 2, 3, 4, 5, 6, 7, 8});
  EXPECT_EQ(cache.evict(1), 4U);  // the only leaf holds four blocks
  EXPECT_EQ(cache.stats().cached_blocks, 0U);
  EXPECT_EQ(allocator.num_free(), allocator.num_blocks());
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, InnerNodeBecomesEvictableAfterItsChildren) {
  cache_tokens({1, 1, 2, 2});
  cache_tokens({1, 1, 3, 3});  // [1 1] -> {[2 2], [3 3]}
  EXPECT_EQ(cache.stats().nodes, 3U);
  EXPECT_EQ(cache.evict(2), 2U);  // both leaves first
  EXPECT_EQ(cache.stats().nodes, 1U);
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 1, 2, 2}).num_tokens, 2U);
  EXPECT_EQ(cache.evict(1), 1U);
  EXPECT_EQ(cache.stats().nodes, 0U);
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, LockedHandleSurvivesASplitOfItsNode) {
  const InsertResult whole = cache_tokens({1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(cache.lock(whole.node));
  static_cast<void>(cache.match_prefix(std::vector<TokenId>{1, 2, 9, 9}));  // splits the node
  EXPECT_EQ(cache.stats().nodes, 2U);
  EXPECT_TRUE(cache.contains(whole.node));
  EXPECT_EQ(cache.stats().locked_blocks, 3U);
  EXPECT_EQ(cache.evict(std::numeric_limits<std::size_t>::max()), 0U);
  ASSERT_TRUE(cache.unlock(whole.node));
  EXPECT_EQ(cache.stats().locked_blocks, 0U);
  EXPECT_EQ(cache.evict(std::numeric_limits<std::size_t>::max()), 3U);
  expect_valid();
}

TEST_F(RadixPrefixCacheTest, LockingADescendantPinsItsAncestors) {
  cache_tokens({1, 1, 2, 2});
  const InsertResult right = cache_tokens({1, 1, 3, 3});
  ASSERT_TRUE(cache.lock(right.node));
  EXPECT_EQ(cache.stats().locked_blocks, 2U);  // [3 3] and the shared [1 1]
  EXPECT_EQ(cache.evict(10), 1U);              // only [2 2] is evictable
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{1, 1, 3, 3}).num_tokens, 4U);
  ASSERT_TRUE(cache.unlock(right.node));
  expect_valid();
}

TEST(RadixPrefixCache, TokenGranularTreeWithBlockSizeOne) {
  BlockAllocator allocator(32, 1);
  RadixPrefixCache cache(allocator);
  Sequence first(allocator);
  Sequence second(allocator);
  ASSERT_TRUE(first.append(std::vector<TokenId>{5, 6, 7}));
  ASSERT_TRUE(second.append(std::vector<TokenId>{5, 6, 8, 9}));
  ASSERT_TRUE(cache.insert(first.tokens(), first.block_table()));
  EXPECT_EQ(cache.insert(second.tokens(), second.block_table())->new_blocks, 2U);
  EXPECT_EQ(cache.stats().nodes, 3U);  // [5 6] -> {[7], [8 9]}
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{5, 6, 8}).num_tokens, 3U);
  EXPECT_EQ(cache.match_prefix(std::vector<TokenId>{5, 9}).num_tokens, 1U);
  EXPECT_EQ(cache.stats().nodes, 5U);  // each match ended inside a node and split it
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
}

TEST(RadixPrefixCache, ChildKeyCollisionBlocksOnlyTheCollidingBranch) {
  BlockAllocator allocator(32, 2);
  RadixPrefixCache cache(
      allocator, [](std::uint64_t, std::span<const TokenId>) noexcept { return std::uint64_t{3}; });
  Sequence first(allocator);
  Sequence second(allocator);
  ASSERT_TRUE(first.append(std::vector<TokenId>{1, 2, 3, 4}));
  ASSERT_TRUE(second.append(std::vector<TokenId>{5, 6, 7, 8}));
  EXPECT_EQ(cache.insert(first.tokens(), first.block_table())->new_blocks, 2U);
  EXPECT_EQ(cache.insert(second.tokens(), second.block_table())->new_blocks, 0U);
  EXPECT_EQ(cache.match_prefix(second.tokens()).num_tokens, 0U);
  EXPECT_EQ(cache.match_prefix(first.tokens()).num_tokens, 4U);
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
}

}  // namespace
}  // namespace radix_kv
