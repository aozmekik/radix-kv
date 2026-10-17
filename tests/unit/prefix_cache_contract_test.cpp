// Behaviour every PrefixCache strategy must share, run against both implementations.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/hash_prefix_cache.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/radix_prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {
namespace {

using ::testing::ElementsAreArray;

static_assert(PrefixCache<HashPrefixCache>);
static_assert(PrefixCache<RadixPrefixCache>);

std::vector<BlockId> first_blocks(const Sequence& sequence, std::size_t count) {
  return {sequence.block_table().begin(),
          sequence.block_table().begin() + static_cast<std::ptrdiff_t>(count)};
}

template <class Cache>
class PrefixCacheContract : public ::testing::Test {
 protected:
  static constexpr std::uint32_t kBlockSize = 2;

  Sequence make_sequence(const std::vector<TokenId>& tokens) {
    Sequence sequence(allocator);
    EXPECT_TRUE(sequence.append(tokens));
    return sequence;
  }

  InsertResult insert(const Sequence& sequence) {
    return cache.insert(sequence.tokens(), sequence.block_table()).value();
  }

  PrefixMatch match(const std::vector<TokenId>& tokens) { return cache.match_prefix(tokens); }

  void expect_valid() const { EXPECT_EQ(cache.check_invariants(), std::nullopt); }

  BlockAllocator allocator{32, kBlockSize};
  Cache cache{allocator};
};

using CacheTypes = ::testing::Types<HashPrefixCache, RadixPrefixCache>;
TYPED_TEST_SUITE(PrefixCacheContract, CacheTypes);

TYPED_TEST(PrefixCacheContract, EmptyCacheMissesAtTheRoot) {
  const PrefixMatch match = this->match({1, 2, 3, 4});
  EXPECT_EQ(match.num_tokens, 0U);
  EXPECT_TRUE(match.blocks.empty());
  EXPECT_EQ(match.node, this->cache.root());
  EXPECT_TRUE(this->cache.contains(this->cache.root()));
  EXPECT_EQ(this->cache.stats().cached_blocks, 0U);
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, MatchesTheLongestCachedBlockAlignedPrefix) {
  const Sequence sequence = this->make_sequence({1, 2, 3, 4, 5});
  const InsertResult inserted = this->insert(sequence);
  EXPECT_EQ(inserted.new_blocks, 2U);  // the partial third block is never cached
  EXPECT_EQ(this->cache.stats().cached_blocks, 2U);

  const PrefixMatch longer = this->match({1, 2, 3, 4, 5, 6});
  EXPECT_EQ(longer.num_tokens, 4U);
  EXPECT_THAT(longer.blocks, ElementsAreArray(first_blocks(sequence, 2)));

  const PrefixMatch diverging = this->match({1, 2, 3, 9});
  EXPECT_EQ(diverging.num_tokens, 2U);
  EXPECT_THAT(diverging.blocks, ElementsAreArray(first_blocks(sequence, 1)));

  EXPECT_EQ(this->match({1}).num_tokens, 0U);  // shorter than a block
  EXPECT_EQ(this->match({2, 1, 3, 4}).num_tokens, 0U);
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, ReinsertingIsANoOp) {
  const Sequence sequence = this->make_sequence({1, 2, 3, 4});
  EXPECT_EQ(this->insert(sequence).new_blocks, 2U);
  EXPECT_EQ(this->insert(sequence).new_blocks, 0U);
  for (const BlockId block : sequence.block_table()) {
    EXPECT_EQ(this->allocator.ref_count(block), 2U);  // the sequence and the cache
    EXPECT_TRUE(this->cache.holds_block(block));
  }
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, DuplicateContentKeepsTheCachedBlock) {
  const Sequence first = this->make_sequence({1, 2, 3, 4});
  const Sequence duplicate = this->make_sequence({1, 2, 3, 4});
  EXPECT_EQ(this->insert(first).new_blocks, 2U);
  const InsertResult again = this->insert(duplicate);
  EXPECT_EQ(again.new_blocks, 0U);
  for (const BlockId block : duplicate.block_table()) {
    EXPECT_EQ(this->allocator.ref_count(block), 1U);
    EXPECT_FALSE(this->cache.holds_block(block));
  }
  EXPECT_THAT(this->match({1, 2, 3, 4}).blocks, ElementsAreArray(first.block_table()));
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, InsertExtendsACachedPrefix) {
  const Sequence base = this->make_sequence({1, 2, 3, 4});
  this->insert(base);
  const PrefixMatch hit = this->match({1, 2, 3, 4, 5, 6});
  Sequence extended(this->allocator);
  ASSERT_TRUE(extended.assign_prefix(std::vector<TokenId>{1, 2, 3, 4}, hit.blocks));
  ASSERT_TRUE(extended.append(std::vector<TokenId>{5, 6}));
  const InsertResult inserted = this->insert(extended);
  EXPECT_EQ(inserted.new_blocks, 1U);
  EXPECT_EQ(this->match({1, 2, 3, 4, 5, 6}).num_tokens, 6U);
  EXPECT_EQ(this->cache.stats().cached_blocks, 3U);
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, InsertValidatesArgumentsWithoutSideEffects) {
  const Sequence cached = this->make_sequence({7, 8});
  this->insert(cached);
  const BlockId held = *this->allocator.allocate();
  const BlockId free_block = 31;
  ASSERT_FALSE(this->allocator.is_allocated(free_block));
  const std::vector<TokenId> two_blocks{1, 2, 3, 4};
  const CacheStats before = this->cache.stats();

  const auto insert_error = [&](std::vector<TokenId> tokens, std::vector<BlockId> blocks) {
    const Result<InsertResult> result = this->cache.insert(tokens, blocks);
    return result ? std::optional<Errc>() : std::optional<Errc>(result.error());
  };
  EXPECT_EQ(insert_error(two_blocks, {held}), Errc::invalid_argument);  // too few blocks
  EXPECT_EQ(insert_error(two_blocks, {held, 99}), Errc::invalid_block);
  EXPECT_EQ(insert_error(two_blocks, {held, free_block}), Errc::block_not_allocated);
  EXPECT_EQ(insert_error(two_blocks, {held, held}), Errc::invalid_argument);  // duplicate id
  EXPECT_EQ(insert_error({5, 6}, {cached.block_table()[0]}), Errc::invalid_argument);

  const CacheStats after = this->cache.stats();
  EXPECT_EQ(after.cached_blocks, before.cached_blocks);
  EXPECT_EQ(after.nodes, before.nodes);
  EXPECT_EQ(this->allocator.ref_count(held), 1U);
  EXPECT_EQ(this->allocator.ref_count(cached.block_table()[0]), 2U);
  this->expect_valid();
  ASSERT_TRUE(this->allocator.release(held));
}

TYPED_TEST(PrefixCacheContract, LockedPathsSurviveEviction) {
  Sequence kept = this->make_sequence({1, 2, 3, 4});
  Sequence dropped = this->make_sequence({5, 6, 7, 8});
  this->insert(kept);
  this->insert(dropped);
  const PrefixMatch hit = this->match({1, 2, 3, 4});
  ASSERT_TRUE(this->cache.lock(hit.node));
  EXPECT_EQ(this->cache.stats().locked_blocks, 2U);
  EXPECT_EQ(this->cache.stats().evictable_blocks, 2U);
  kept.release();
  dropped.release();

  EXPECT_EQ(this->cache.evict(std::numeric_limits<std::size_t>::max()), 2U);
  EXPECT_TRUE(this->cache.contains(hit.node));
  EXPECT_EQ(this->match({1, 2, 3, 4}).num_tokens, 4U);
  EXPECT_EQ(this->match({5, 6, 7, 8}).num_tokens, 0U);
  this->expect_valid();

  ASSERT_TRUE(this->cache.unlock(hit.node));
  EXPECT_EQ(this->cache.evict(std::numeric_limits<std::size_t>::max()), 2U);
  EXPECT_EQ(this->allocator.num_free(), this->allocator.num_blocks());
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, UnlockRequiresALockThroughTheSameHandle) {
  const Sequence sequence = this->make_sequence({1, 2, 3, 4});
  const InsertResult inserted = this->insert(sequence);
  EXPECT_EQ(this->cache.unlock(this->cache.root()).error(), Errc::not_locked);
  ASSERT_TRUE(this->cache.lock(inserted.node));
  // The root is on the locked path, but no lock was taken through the root handle.
  EXPECT_EQ(this->cache.unlock(this->cache.root()).error(), Errc::not_locked);
  ASSERT_TRUE(this->cache.lock(inserted.node));
  ASSERT_TRUE(this->cache.unlock(inserted.node));
  ASSERT_TRUE(this->cache.unlock(inserted.node));
  EXPECT_EQ(this->cache.unlock(inserted.node).error(), Errc::not_locked);
  EXPECT_EQ(this->cache.stats().locked_blocks, 0U);
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, StaleAndForgedHandlesAreRejected) {
  Sequence sequence = this->make_sequence({1, 2, 3, 4});
  const InsertResult inserted = this->insert(sequence);
  sequence.release();
  this->cache.evict(std::numeric_limits<std::size_t>::max());
  EXPECT_FALSE(this->cache.contains(inserted.node));
  EXPECT_EQ(this->cache.lock(inserted.node).error(), Errc::invalid_handle);
  EXPECT_EQ(this->cache.unlock(inserted.node).error(), Errc::invalid_handle);
  EXPECT_EQ(this->cache.lock(NodeHandle{.index = 9999, .generation = 0}).error(),
            Errc::invalid_handle);
  const NodeHandle wrong_generation{.index = this->cache.root().index, .generation = 7};
  EXPECT_EQ(this->cache.lock(wrong_generation).error(), Errc::invalid_handle);

  // A slot reused for a new node does not revive the old handle.
  const Sequence reused = this->make_sequence({1, 2, 3, 4});
  const InsertResult fresh = this->insert(reused);
  EXPECT_FALSE(this->cache.contains(inserted.node));
  EXPECT_TRUE(this->cache.contains(fresh.node));
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, LockingTheRootPinsNothing) {
  Sequence sequence = this->make_sequence({1, 2});
  this->insert(sequence);
  sequence.release();
  ASSERT_TRUE(this->cache.lock(this->cache.root()));
  EXPECT_EQ(this->cache.stats().locked_blocks, 0U);
  EXPECT_EQ(this->cache.evict(1), 1U);
  ASSERT_TRUE(this->cache.unlock(this->cache.root()));
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, EvictsTheLeastRecentlyUsedPathFirst) {
  Sequence older = this->make_sequence({1, 2, 3, 4});
  Sequence newer = this->make_sequence({5, 6, 7, 8});
  this->insert(older);
  this->insert(newer);
  older.release();
  newer.release();
  static_cast<void>(this->match({1, 2, 3, 4}));  // `older` becomes the most recently used

  EXPECT_GE(this->cache.evict(1), 1U);
  EXPECT_EQ(this->match({1, 2, 3, 4}).num_tokens, 4U);
  EXPECT_LT(this->match({5, 6, 7, 8}).num_tokens, 4U);
  this->expect_valid();
}

TYPED_TEST(PrefixCacheContract, CountsEvictedBlocks) {
  EXPECT_EQ(this->cache.evict(5), 0U);
  Sequence sequence = this->make_sequence({1, 2, 3, 4, 5, 6});
  this->insert(sequence);
  sequence.release();
  EXPECT_EQ(this->cache.stats().evictable_blocks, 3U);
  EXPECT_EQ(this->cache.evict(std::numeric_limits<std::size_t>::max()), 3U);
  EXPECT_EQ(this->cache.stats().evicted_blocks, 3U);
  EXPECT_EQ(this->cache.stats().cached_blocks, 0U);
  EXPECT_EQ(this->cache.stats().nodes, 0U);
}

TYPED_TEST(PrefixCacheContract, DestructorReleasesCachedBlocks) {
  BlockAllocator allocator(8, 2);
  {
    TypeParam cache(allocator);
    Sequence sequence(allocator);
    ASSERT_TRUE(sequence.append(std::vector<TokenId>{1, 2, 3, 4}));
    ASSERT_TRUE(cache.insert(sequence.tokens(), sequence.block_table()));
    sequence.release();
    EXPECT_EQ(allocator.num_used(), 2U);
  }
  EXPECT_EQ(allocator.num_used(), 0U);
}

TYPED_TEST(PrefixCacheContract, CollidingHasherNeverReturnsWrongBlocks) {
  BlockAllocator allocator(64, 2);
  TypeParam cache(allocator, [](std::uint64_t, std::span<const TokenId>) noexcept {
    return std::uint64_t{42};
  });
  std::vector<Sequence> sequences;
  for (TokenId first = 1; first <= 5; ++first) {
    Sequence sequence(allocator);
    ASSERT_TRUE(sequence.append(std::vector<TokenId>{first, first, 9, 9, 8, 8}));
    ASSERT_TRUE(cache.insert(sequence.tokens(), sequence.block_table()));
    sequences.push_back(std::move(sequence));
  }
  for (const Sequence& sequence : sequences) {
    const PrefixMatch match = cache.match_prefix(sequence.tokens());
    // Distinct first blocks: any hit must come from this very sequence.
    EXPECT_THAT(match.blocks, ElementsAreArray(first_blocks(sequence, match.blocks.size())));
  }
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
}

}  // namespace
}  // namespace radix_kv
