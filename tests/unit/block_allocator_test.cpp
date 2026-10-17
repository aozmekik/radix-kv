#include "radix_kv/block_allocator.hpp"

#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {
namespace {

using ::testing::ElementsAre;

TEST(BlockAllocator, RejectsEmptyConfigurations) {
  EXPECT_THROW(BlockAllocator(0, 16), std::invalid_argument);
  EXPECT_THROW(BlockAllocator(8, 0), std::invalid_argument);
  EXPECT_THROW(BlockAllocator(kInvalidBlock, 16), std::invalid_argument);
}

TEST(BlockAllocator, StartsWithEveryBlockFree) {
  const BlockAllocator allocator(8, 16);
  EXPECT_EQ(allocator.num_blocks(), 8U);
  EXPECT_EQ(allocator.block_size(), 16U);
  EXPECT_EQ(allocator.num_free(), 8U);
  EXPECT_EQ(allocator.num_used(), 0U);
  EXPECT_EQ(allocator.peak_used(), 0U);
  for (BlockId block = 0; block < 8; ++block) {
    EXPECT_EQ(allocator.ref_count(block), 0U);
    EXPECT_FALSE(allocator.is_allocated(block));
  }
}

TEST(BlockAllocator, AllocatesInIdOrderAndReusesMostRecentlyFreed) {
  BlockAllocator allocator(4, 16);
  EXPECT_EQ(*allocator.allocate(), 0U);
  EXPECT_EQ(*allocator.allocate(), 1U);
  EXPECT_EQ(*allocator.allocate(), 2U);
  EXPECT_EQ(*allocator.release(1), 0U);
  EXPECT_EQ(*allocator.allocate(), 1U);  // LIFO free list
  EXPECT_EQ(allocator.ref_count(1), 1U);
}

TEST(BlockAllocator, ReportsExhaustionWithoutSideEffects) {
  BlockAllocator allocator(2, 16);
  ASSERT_TRUE(allocator.allocate());
  ASSERT_TRUE(allocator.allocate());
  const Result<BlockId> third = allocator.allocate();
  ASSERT_FALSE(third);
  EXPECT_EQ(third.error(), Errc::out_of_blocks);
  EXPECT_EQ(allocator.num_free(), 0U);
}

TEST(BlockAllocator, BatchAllocationIsAllOrNothing) {
  BlockAllocator allocator(4, 16);
  std::vector<BlockId> blocks{99};
  const Result<void> too_many = allocator.allocate(5, blocks);
  ASSERT_FALSE(too_many);
  EXPECT_EQ(too_many.error(), Errc::out_of_blocks);
  EXPECT_THAT(blocks, ElementsAre(99U));
  EXPECT_EQ(allocator.num_free(), 4U);

  ASSERT_TRUE(allocator.allocate(3, blocks));
  EXPECT_THAT(blocks, ElementsAre(99U, 0U, 1U, 2U));
  EXPECT_EQ(allocator.num_free(), 1U);
  ASSERT_TRUE(allocator.allocate(0, blocks));
  EXPECT_EQ(blocks.size(), 4U);
}

TEST(BlockAllocator, ReferenceCountingFreesOnLastRelease) {
  BlockAllocator allocator(2, 16);
  const BlockId block = *allocator.allocate();
  ASSERT_TRUE(allocator.retain(block));
  ASSERT_TRUE(allocator.retain(block));
  EXPECT_EQ(allocator.ref_count(block), 3U);
  EXPECT_EQ(*allocator.release(block), 2U);
  EXPECT_EQ(*allocator.release(block), 1U);
  EXPECT_EQ(allocator.num_free(), 1U);
  EXPECT_EQ(*allocator.release(block), 0U);
  EXPECT_EQ(allocator.num_free(), 2U);
}

TEST(BlockAllocator, RejectsInvalidAndFreeBlocks) {
  BlockAllocator allocator(2, 16);
  EXPECT_EQ(allocator.retain(2).error(), Errc::invalid_block);
  EXPECT_EQ(allocator.release(kInvalidBlock).error(), Errc::invalid_block);
  EXPECT_EQ(allocator.retain(0).error(), Errc::block_not_allocated);
  EXPECT_EQ(allocator.release(0).error(), Errc::block_not_allocated);
  EXPECT_EQ(allocator.ref_count(kInvalidBlock), 0U);

  const BlockId block = *allocator.allocate();
  ASSERT_TRUE(allocator.release(block));
  EXPECT_EQ(allocator.release(block).error(), Errc::block_not_allocated);  // double free
  EXPECT_EQ(allocator.num_free(), 2U);
}

TEST(BlockAllocator, TracksPeakUsage) {
  BlockAllocator allocator(4, 16);
  const BlockId a = *allocator.allocate();
  const BlockId b = *allocator.allocate();
  const BlockId c = *allocator.allocate();
  ASSERT_TRUE(allocator.release(a));
  ASSERT_TRUE(allocator.release(b));
  EXPECT_EQ(allocator.peak_used(), 3U);
  allocator.reset_peak();
  EXPECT_EQ(allocator.peak_used(), 1U);
  ASSERT_TRUE(allocator.release(c));
  EXPECT_EQ(allocator.peak_used(), 1U);
}

TEST(BlockAllocator, ComputesBlocksForTokenCounts) {
  const BlockAllocator allocator(4, 16);
  EXPECT_EQ(allocator.blocks_for(0), 0U);
  EXPECT_EQ(allocator.blocks_for(1), 1U);
  EXPECT_EQ(allocator.blocks_for(16), 1U);
  EXPECT_EQ(allocator.blocks_for(17), 2U);
}

}  // namespace
}  // namespace radix_kv
