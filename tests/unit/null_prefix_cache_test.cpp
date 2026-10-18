#include "radix_kv/null_prefix_cache.hpp"

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {
namespace {

static_assert(PrefixCache<NullPrefixCache>);

TEST(NullPrefixCache, NeverCachesAnything) {
  BlockAllocator allocator(8, 2);
  NullPrefixCache cache(allocator);
  EXPECT_EQ(&cache.allocator(), &allocator);
  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.append(std::vector<TokenId>{1, 2, 3, 4}));
  const Result<InsertResult> inserted = cache.insert(sequence.tokens(), sequence.block_table());
  ASSERT_TRUE(inserted);
  EXPECT_EQ(inserted->new_blocks, 0U);
  EXPECT_EQ(inserted->node, cache.root());
  EXPECT_EQ(cache.match_prefix(sequence.tokens()).num_tokens, 0U);
  EXPECT_FALSE(cache.holds_block(sequence.block_table()[0]));
  EXPECT_EQ(cache.evict(10), 0U);
  EXPECT_EQ(cache.stats().cached_blocks, 0U);
  EXPECT_EQ(cache.check_invariants(), std::nullopt);
  EXPECT_EQ(allocator.ref_count(sequence.block_table()[0]), 1U);  // no reference taken
}

TEST(NullPrefixCache, ValidatesInsertArguments) {
  BlockAllocator allocator(8, 2);
  const NullPrefixCache cache(allocator);
  const std::vector<TokenId> tokens{1, 2, 3, 4};
  EXPECT_EQ(cache.insert(tokens, std::vector<BlockId>{0}).error(), Errc::invalid_argument);
}

TEST(NullPrefixCache, CountsLocksOnTheRootOnly) {
  BlockAllocator allocator(8, 2);
  NullPrefixCache cache(allocator);
  const NodeHandle other{.index = 1, .generation = 0};
  EXPECT_TRUE(cache.contains(cache.root()));
  EXPECT_FALSE(cache.contains(other));
  EXPECT_EQ(cache.lock(other).error(), Errc::invalid_handle);
  EXPECT_EQ(cache.unlock(other).error(), Errc::invalid_handle);
  EXPECT_EQ(cache.unlock(cache.root()).error(), Errc::not_locked);
  ASSERT_TRUE(cache.lock(cache.root()));
  ASSERT_TRUE(cache.unlock(cache.root()));
  EXPECT_EQ(cache.unlock(cache.root()).error(), Errc::not_locked);
}

}  // namespace
}  // namespace radix_kv
