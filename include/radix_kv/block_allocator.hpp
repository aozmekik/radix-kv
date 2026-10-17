#pragma once

/// @file
/// Fixed pool of reference-counted KV-cache blocks.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

/// A fixed pool of `num_blocks` KV-cache blocks of `block_size` tokens each, as in PagedAttention.
///
/// The allocator only manages block ids and reference counts; it never touches KV tensors. Every
/// holder of a block (a `Sequence`, a prefix cache) owns exactly one reference, so a block returns
/// to the free list exactly when its last holder releases it. Sharing a block between forked
/// sequences or between a sequence and a prefix cache is therefore just `retain()`.
///
/// All operations are O(1) (the batch `allocate()` is O(count)) and none of them throw: failures
/// are reported through `Result`. The free list is a LIFO stack, so recently freed blocks are
/// reused first and allocation order is deterministic.
///
/// Not thread-safe: like an inference engine's scheduler, use it from one thread or guard it
/// externally. Sequences and caches keep a pointer to their allocator, so it is neither copyable
/// nor movable and must outlive them.
class BlockAllocator {
 public:
  /// Creates a pool with every block free.
  /// @throws std::invalid_argument if `num_blocks` or `block_size` is zero, or if `num_blocks`
  ///         equals the `kInvalidBlock` sentinel.
  BlockAllocator(std::uint32_t num_blocks, std::uint32_t block_size);

  BlockAllocator(const BlockAllocator&) = delete;
  BlockAllocator& operator=(const BlockAllocator&) = delete;
  BlockAllocator(BlockAllocator&&) = delete;
  BlockAllocator& operator=(BlockAllocator&&) = delete;
  ~BlockAllocator() = default;

  /// Takes one free block and gives the caller its only reference.
  /// @return the block id, or `Errc::out_of_blocks` when the pool is exhausted.
  [[nodiscard]] Result<BlockId> allocate() noexcept;

  /// Takes `count` free blocks and appends their ids to `out`, or takes none at all.
  /// @return `Errc::out_of_blocks` (with `out` unchanged) when fewer than `count` blocks are free.
  /// @throws std::bad_alloc if `out` cannot grow; the pool is left unchanged in that case.
  [[nodiscard]] Result<void> allocate(std::size_t count, std::vector<BlockId>& out);

  /// Adds a reference to an allocated block.
  /// @return `Errc::invalid_block`, `Errc::block_not_allocated` or `Errc::ref_count_overflow`.
  [[nodiscard]] Result<void> retain(BlockId block) noexcept;

  /// Drops a reference; the block becomes free when its count reaches zero.
  /// @return the remaining reference count (0 means the block was freed), or
  ///         `Errc::invalid_block` / `Errc::block_not_allocated` (e.g. a double release).
  [[nodiscard]] Result<std::uint32_t> release(BlockId block) noexcept;

  /// Current reference count of `block`; 0 for free blocks and for ids outside the pool.
  [[nodiscard]] std::uint32_t ref_count(BlockId block) const noexcept;

  /// True when `block` is inside the pool and currently allocated.
  [[nodiscard]] bool is_allocated(BlockId block) const noexcept { return ref_count(block) > 0; }

  /// Number of blocks in the pool.
  [[nodiscard]] std::uint32_t num_blocks() const noexcept { return num_blocks_; }

  /// Tokens per block.
  [[nodiscard]] std::uint32_t block_size() const noexcept { return block_size_; }

  /// Blocks currently free.
  [[nodiscard]] std::uint32_t num_free() const noexcept { return num_free_; }

  /// Blocks currently allocated.
  [[nodiscard]] std::uint32_t num_used() const noexcept { return num_blocks_ - num_free(); }

  /// Highest `num_used()` observed since construction or the last `reset_peak()`.
  [[nodiscard]] std::uint32_t peak_used() const noexcept { return peak_used_; }

  /// Restarts peak tracking from the current usage.
  void reset_peak() noexcept { peak_used_ = num_used(); }

  /// Number of blocks needed to hold `num_tokens` tokens (rounded up).
  [[nodiscard]] std::size_t blocks_for(std::size_t num_tokens) const noexcept {
    return (num_tokens + block_size_ - 1) / block_size_;
  }

 private:
  void note_usage() noexcept;

  std::uint32_t num_blocks_;
  std::uint32_t block_size_;
  std::uint32_t peak_used_ = 0;
  std::uint32_t num_free_;
  std::vector<std::uint32_t> ref_counts_;  // indexed by BlockId
  std::vector<BlockId> free_;              // LIFO stack of free ids in free_[0, num_free_)
};

}  // namespace radix_kv
