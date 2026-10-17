#include "radix_kv/detail/block_token_store.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv::detail {

BlockTokenStore::BlockTokenStore(std::uint32_t num_blocks, std::uint32_t block_size)
    : block_size_(block_size),
      tokens_(static_cast<std::size_t>(num_blocks) * block_size),
      present_(num_blocks, 0),
      marks_(num_blocks, 0) {}

std::span<const TokenId> BlockTokenStore::tokens(BlockId block) const noexcept {
  return std::span<const TokenId>(tokens_).subspan(static_cast<std::size_t>(block) * block_size_,
                                                   block_size_);
}

bool BlockTokenStore::matches(BlockId block, std::span<const TokenId> tokens) const noexcept {
  return contains(block) && std::ranges::equal(this->tokens(block), tokens);
}

void BlockTokenStore::store(BlockId block, std::span<const TokenId> tokens) noexcept {
  std::ranges::copy(tokens, tokens_.begin() + static_cast<std::ptrdiff_t>(
                                                  static_cast<std::size_t>(block) * block_size_));
  if (present_[block] == 0) {
    present_[block] = 1;
    ++size_;
  }
}

void BlockTokenStore::erase(BlockId block) noexcept {
  if (present_[block] != 0) {
    present_[block] = 0;
    --size_;
  }
}

bool BlockTokenStore::contains(BlockId block) const noexcept {
  return block < present_.size() && present_[block] != 0;
}

Result<void> BlockTokenStore::validate_insert(const BlockAllocator& allocator,
                                              std::span<const BlockId> blocks,
                                              std::size_t first_new) {
  for (const BlockId block : blocks) {
    if (block >= allocator.num_blocks()) {
      return fail(Errc::invalid_block);
    }
    if (!allocator.is_allocated(block)) {
      return fail(Errc::block_not_allocated);
    }
  }
  if (first_new >= blocks.size()) {
    return {};
  }
  if (++epoch_ == 0) {  // wrapped: stale stamps could collide with the new epoch
    std::ranges::fill(marks_, 0U);
    epoch_ = 1;
  }
  for (const BlockId block : blocks.subspan(first_new)) {
    if (contains(block) || marks_[block] == epoch_) {
      return fail(Errc::invalid_argument);
    }
    if (allocator.ref_count(block) == std::numeric_limits<std::uint32_t>::max()) {
      return fail(Errc::ref_count_overflow);
    }
    marks_[block] = epoch_;
  }
  return {};
}

}  // namespace radix_kv::detail
