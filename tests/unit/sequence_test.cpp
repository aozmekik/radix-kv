#include "radix_kv/sequence.hpp"

#include <cstddef>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

std::vector<TokenId> iota_tokens(std::size_t count, TokenId first = 0) {
  std::vector<TokenId> tokens(count);
  std::iota(tokens.begin(), tokens.end(), first);
  return tokens;
}

TEST(Sequence, AppendAllocatesBlocksAsNeeded) {
  BlockAllocator allocator(8, 4);
  Sequence sequence(allocator);
  EXPECT_TRUE(sequence.empty());
  EXPECT_EQ(sequence.block_size(), 4U);
  EXPECT_EQ(&sequence.allocator(), &allocator);

  const Result<AppendResult> first = sequence.append(iota_tokens(3));
  ASSERT_TRUE(first);
  EXPECT_EQ(first->new_blocks, 1U);
  EXPECT_FALSE(first->copy.has_value());

  const Result<AppendResult> second = sequence.append(iota_tokens(6, 3));
  ASSERT_TRUE(second);
  EXPECT_EQ(second->new_blocks, 2U);  // fills block 0, then blocks 1 and 2
  EXPECT_EQ(sequence.size(), 9U);
  EXPECT_THAT(sequence.tokens(), ElementsAreArray(iota_tokens(9)));
  EXPECT_THAT(sequence.block_table(), ElementsAre(0U, 1U, 2U));
  EXPECT_EQ(sequence.num_full_blocks(), 2U);
  EXPECT_EQ(allocator.num_used(), 3U);
}

TEST(Sequence, SingleTokenAppendIsOneDecodeStep) {
  BlockAllocator allocator(2, 2);
  Sequence sequence(allocator);
  EXPECT_EQ(sequence.append(TokenId{7})->new_blocks, 1U);
  EXPECT_EQ(sequence.append(TokenId{8})->new_blocks, 0U);
  EXPECT_EQ(sequence.append(TokenId{9})->new_blocks, 1U);
  EXPECT_THAT(sequence.tokens(), ElementsAre(7U, 8U, 9U));
}

TEST(Sequence, EmptyAppendChangesNothing) {
  BlockAllocator allocator(2, 4);
  Sequence sequence(allocator);
  const Result<AppendResult> result = sequence.append(std::vector<TokenId>{});
  ASSERT_TRUE(result);
  EXPECT_EQ(result->new_blocks, 0U);
  EXPECT_EQ(allocator.num_used(), 0U);
}

TEST(Sequence, FailedAppendLeavesEverythingUnchanged) {
  BlockAllocator allocator(2, 4);
  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.append(iota_tokens(5)));     // two blocks, the second holds one token
  ASSERT_TRUE(sequence.append(iota_tokens(3, 5)));  // fills the second block
  const Result<AppendResult> overflow = sequence.append(iota_tokens(1, 8));
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error(), Errc::out_of_blocks);
  EXPECT_THAT(sequence.tokens(), ElementsAreArray(iota_tokens(8)));
  EXPECT_EQ(sequence.num_blocks(), 2U);
  EXPECT_EQ(allocator.num_free(), 0U);
}

TEST(Sequence, ForkSharesEveryBlock) {
  BlockAllocator allocator(8, 4);
  Sequence parent(allocator);
  ASSERT_TRUE(parent.append(iota_tokens(6)));
  Result<Sequence> child = parent.fork();
  ASSERT_TRUE(child);
  EXPECT_THAT(child->tokens(), ElementsAreArray(parent.tokens()));
  EXPECT_THAT(child->block_table(), ElementsAreArray(parent.block_table()));
  for (const BlockId block : parent.block_table()) {
    EXPECT_EQ(allocator.ref_count(block), 2U);
  }
  EXPECT_EQ(allocator.num_used(), 2U);
}

TEST(Sequence, AppendToSharedPartialBlockCopiesOnWrite) {
  BlockAllocator allocator(8, 4);
  Sequence parent(allocator);
  ASSERT_TRUE(parent.append(iota_tokens(6)));  // blocks 0 (full) and 1 (2 of 4 tokens)
  Sequence child = *parent.fork();

  const Result<AppendResult> appended = child.append(TokenId{100});
  ASSERT_TRUE(appended);
  EXPECT_EQ(appended->copy, std::optional<CopyOp>(CopyOp{.src = 1, .dst = 2}));
  EXPECT_EQ(appended->new_blocks, 0U);
  EXPECT_THAT(child.block_table(), ElementsAre(0U, 2U));
  EXPECT_THAT(parent.block_table(), ElementsAre(0U, 1U));
  EXPECT_EQ(allocator.ref_count(0), 2U);  // full blocks stay shared
  EXPECT_EQ(allocator.ref_count(1), 1U);
  EXPECT_EQ(allocator.ref_count(2), 1U);

  // The child now owns its last block: further appends write in place.
  const Result<AppendResult> again = child.append(TokenId{101});
  ASSERT_TRUE(again);
  EXPECT_FALSE(again->copy.has_value());
  // So does the parent, whose block is no longer shared.
  EXPECT_FALSE(parent.append(TokenId{200})->copy.has_value());
}

TEST(Sequence, AppendAfterFullSharedBlockNeedsNoCopy) {
  BlockAllocator allocator(8, 4);
  Sequence parent(allocator);
  ASSERT_TRUE(parent.append(iota_tokens(4)));
  Sequence child = *parent.fork();
  const Result<AppendResult> appended = child.append(TokenId{9});
  ASSERT_TRUE(appended);
  EXPECT_FALSE(appended->copy.has_value());
  EXPECT_EQ(appended->new_blocks, 1U);
  EXPECT_EQ(allocator.ref_count(0), 2U);
}

TEST(Sequence, CopyOnWriteNeedsAFreeBlock) {
  BlockAllocator allocator(2, 4);
  Sequence parent(allocator);
  ASSERT_TRUE(parent.append(iota_tokens(6)));
  Sequence child = *parent.fork();
  const Result<AppendResult> appended = child.append(TokenId{1});
  ASSERT_FALSE(appended);
  EXPECT_EQ(appended.error(), Errc::out_of_blocks);
  EXPECT_THAT(child.block_table(), ElementsAreArray(parent.block_table()));
  EXPECT_EQ(child.size(), 6U);
}

TEST(Sequence, TruncateReleasesTrailingBlocks) {
  BlockAllocator allocator(8, 4);
  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.append(iota_tokens(10)));
  ASSERT_TRUE(sequence.truncate(5));
  EXPECT_EQ(sequence.size(), 5U);
  EXPECT_THAT(sequence.block_table(), ElementsAre(0U, 1U));
  EXPECT_EQ(allocator.num_used(), 2U);
  ASSERT_TRUE(sequence.truncate(5));  // no-op
  ASSERT_TRUE(sequence.truncate(0));
  EXPECT_TRUE(sequence.empty());
  EXPECT_EQ(allocator.num_used(), 0U);
  EXPECT_EQ(sequence.truncate(1).error(), Errc::invalid_argument);
}

TEST(Sequence, TruncateIntoSharedBlockThenAppendCopies) {
  BlockAllocator allocator(8, 4);
  Sequence parent(allocator);
  ASSERT_TRUE(parent.append(iota_tokens(8)));
  Sequence child = *parent.fork();
  ASSERT_TRUE(child.truncate(6));  // block 1 is now partial for the child, still shared
  const Result<AppendResult> appended = child.append(TokenId{42});
  ASSERT_TRUE(appended);
  ASSERT_TRUE(appended->copy.has_value());
  EXPECT_EQ(appended->copy.value_or(CopyOp{}).src, 1U);
  EXPECT_THAT(parent.tokens(), ElementsAreArray(iota_tokens(8)));
}

TEST(Sequence, ReleaseAndDestructorReturnBlocks) {
  BlockAllocator allocator(4, 4);
  {
    Sequence scoped(allocator);
    ASSERT_TRUE(scoped.append(iota_tokens(12)));
    EXPECT_EQ(allocator.num_free(), 1U);
  }
  EXPECT_EQ(allocator.num_free(), 4U);

  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.append(iota_tokens(3)));
  sequence.release();
  EXPECT_TRUE(sequence.empty());
  EXPECT_EQ(sequence.num_blocks(), 0U);
  EXPECT_EQ(allocator.num_free(), 4U);
  sequence.release();  // idempotent
}

TEST(Sequence, MoveTransfersOwnership) {
  BlockAllocator allocator(4, 4);
  Sequence first(allocator);
  ASSERT_TRUE(first.append(iota_tokens(5)));
  Sequence moved(std::move(first));
  EXPECT_EQ(moved.size(), 5U);
  EXPECT_EQ(allocator.num_used(), 2U);

  Sequence other(allocator);
  ASSERT_TRUE(other.append(iota_tokens(4)));
  EXPECT_EQ(allocator.num_used(), 3U);
  other = std::move(moved);  // releases other's block first
  EXPECT_EQ(other.size(), 5U);
  EXPECT_EQ(allocator.num_used(), 2U);
}

TEST(Sequence, AssignPrefixRetainsCachedBlocks) {
  BlockAllocator allocator(4, 2);
  Sequence source(allocator);
  ASSERT_TRUE(source.append(std::vector<TokenId>{1, 2, 3, 4}));
  const std::vector<BlockId> blocks(source.block_table().begin(), source.block_table().end());

  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.assign_prefix(source.tokens(), blocks));
  EXPECT_THAT(sequence.tokens(), ElementsAre(1U, 2U, 3U, 4U));
  EXPECT_EQ(allocator.ref_count(blocks[0]), 2U);
  EXPECT_EQ(allocator.ref_count(blocks[1]), 2U);
}

TEST(Sequence, AssignPrefixValidatesAndRollsBack) {
  BlockAllocator allocator(4, 2);
  const BlockId held = *allocator.allocate();
  const std::vector<TokenId> two_blocks{1, 2, 3, 4};
  Sequence sequence(allocator);

  EXPECT_EQ(sequence.assign_prefix(two_blocks, std::vector<BlockId>{held}).error(),
            Errc::invalid_argument);  // 4 tokens need 2 blocks
  EXPECT_EQ(sequence.assign_prefix(std::vector<TokenId>{1, 2, 3}, std::vector<BlockId>{held, held})
                .error(),
            Errc::invalid_argument);  // not block-aligned
  // The second block is free: the first retain must be rolled back.
  EXPECT_EQ(sequence.assign_prefix(two_blocks, std::vector<BlockId>{held, 3}).error(),
            Errc::block_not_allocated);
  EXPECT_EQ(allocator.ref_count(held), 1U);
  EXPECT_TRUE(sequence.empty());

  ASSERT_TRUE(sequence.append(TokenId{9}));
  EXPECT_EQ(sequence.assign_prefix(std::vector<TokenId>{1, 2}, std::vector<BlockId>{held}).error(),
            Errc::invalid_argument);  // sequence not empty
}

}  // namespace
}  // namespace radix_kv
