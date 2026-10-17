#include "radix_kv/block_allocator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

BlockAllocator::BlockAllocator(std::uint32_t num_blocks, std::uint32_t block_size)
    : num_blocks_(num_blocks), block_size_(block_size), num_free_(num_blocks) {
  if (num_blocks == 0 || block_size == 0) {
    throw std::invalid_argument("BlockAllocator: num_blocks and block_size must be positive");
  }
  if (num_blocks == kInvalidBlock) {
    throw std::invalid_argument("BlockAllocator: num_blocks must be below kInvalidBlock");
  }
  ref_counts_.assign(num_blocks, 0);
  // Stack top is the end of the vector; push ids in reverse so block 0 is handed out first.
  free_.resize(num_blocks);
  std::ranges::generate(free_, [next = num_blocks]() mutable { return --next; });
}

Result<BlockId> BlockAllocator::allocate() noexcept {
  if (num_free_ == 0) {
    return fail(Errc::out_of_blocks);
  }
  const BlockId block = free_[--num_free_];
  ref_counts_[block] = 1;
  note_usage();
  return block;
}

Result<void> BlockAllocator::allocate(std::size_t count, std::vector<BlockId>& out) {
  if (count > num_free_) {
    return fail(Errc::out_of_blocks);
  }
  out.reserve(out.size() + count);  // the only step that can throw; nothing has changed yet
  for (std::size_t i = 0; i < count; ++i) {
    const BlockId block = free_[--num_free_];
    ref_counts_[block] = 1;
    out.push_back(block);
  }
  note_usage();
  return {};
}

Result<void> BlockAllocator::retain(BlockId block) noexcept {
  if (block >= num_blocks_) {
    return fail(Errc::invalid_block);
  }
  std::uint32_t& count = ref_counts_[block];
  if (count == 0) {
    return fail(Errc::block_not_allocated);
  }
  if (count == std::numeric_limits<std::uint32_t>::max()) {
    return fail(Errc::ref_count_overflow);
  }
  ++count;
  return {};
}

Result<std::uint32_t> BlockAllocator::release(BlockId block) noexcept {
  if (block >= num_blocks_) {
    return fail(Errc::invalid_block);
  }
  std::uint32_t& count = ref_counts_[block];
  if (count == 0) {
    return fail(Errc::block_not_allocated);
  }
  if (--count == 0) {
    free_[num_free_++] = block;  // a block is freed once per allocation: never overflows
  }
  return count;
}

std::uint32_t BlockAllocator::ref_count(BlockId block) const noexcept {
  return block < num_blocks_ ? ref_counts_[block] : 0;
}

void BlockAllocator::note_usage() noexcept {
  peak_used_ = std::max(peak_used_, num_used());
}

}  // namespace radix_kv
