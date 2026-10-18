#pragma once

/// @file
/// Vocabulary types shared by the allocator, sequences and prefix caches.

#include <cstdint>
#include <limits>

namespace radix_kv {

/// Index of a KV-cache block in a `BlockAllocator` pool, in `[0, num_blocks)`.
using BlockId = std::uint32_t;

/// A token id produced by the tokenizer. The library only compares token ids for equality.
using TokenId = std::uint32_t;

/// Sentinel for "no block".
inline constexpr BlockId kInvalidBlock = std::numeric_limits<BlockId>::max();

/// Copy-on-write instruction for the engine: copy the KV data of block `src` into block `dst`
/// before running the step that writes to `dst`.
struct CopyOp {
  BlockId src = kInvalidBlock;  ///< Shared block whose contents must be preserved.
  BlockId dst = kInvalidBlock;  ///< Freshly allocated private copy.

  friend constexpr bool operator==(const CopyOp&, const CopyOp&) = default;
};

/// Opaque reference to a prefix-cache node: a radix-tree node in `RadixPrefixCache` or a cached
/// block in `HashPrefixCache`.
///
/// Handles are checked on use: once the node is evicted its slot's generation changes and the
/// handle is rejected with `Errc::invalid_handle` instead of aliasing a newer node. A node cannot
/// be evicted while it is locked, so a handle stays valid for as long as it holds a lock. A handle
/// does not record which cache issued it: use it only with that cache, since another cache may
/// have a live node in the same slot.
struct NodeHandle {
  std::uint32_t index = 0;       ///< Slot of the node inside its cache.
  std::uint32_t generation = 0;  ///< Slot generation at the time the handle was issued.

  friend constexpr bool operator==(const NodeHandle&, const NodeHandle&) = default;
};

}  // namespace radix_kv
