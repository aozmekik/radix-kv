#pragma once

/// @file
/// Implementation detail shared by the prefix caches. Not part of the public API.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv::detail {

/// Token ids of every cached block, in one flat array indexed by block id.
///
/// A block is cached at most once, so `num_blocks * block_size` slots are enough; storing the
/// tokens lets a cache confirm every hash hit by comparing token ids.
class BlockTokenStore {
 public:
  BlockTokenStore(std::uint32_t num_blocks, std::uint32_t block_size);

  /// The `block_size` tokens recorded for `block`. Precondition: `block < num_blocks`.
  [[nodiscard]] std::span<const TokenId> tokens(BlockId block) const noexcept;

  /// True when `block` is recorded and holds exactly `tokens`.
  [[nodiscard]] bool matches(BlockId block, std::span<const TokenId> tokens) const noexcept;

  /// Records `tokens` (exactly `block_size` of them) for `block`.
  void store(BlockId block, std::span<const TokenId> tokens) noexcept;

  /// Forgets `block`.
  void erase(BlockId block) noexcept;

  /// True when `block` is recorded.
  [[nodiscard]] bool contains(BlockId block) const noexcept;

  /// Number of recorded blocks.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  /// Checks the blocks an `insert()` was given before anything is mutated: every block must be
  /// allocated, and the blocks from `first_new` on (those the cache would start holding) must not
  /// be cached yet, must be pairwise distinct and must accept one more reference.
  [[nodiscard]] Result<void> validate_insert(const BlockAllocator& allocator,
                                             std::span<const BlockId> blocks,
                                             std::size_t first_new);

 private:
  std::uint32_t block_size_;
  std::size_t size_ = 0;
  std::vector<TokenId> tokens_;
  std::vector<std::uint8_t> present_;
  std::vector<std::uint32_t> marks_;  // epoch stamps for O(1) duplicate detection
  std::uint32_t epoch_ = 0;
};

}  // namespace radix_kv::detail
