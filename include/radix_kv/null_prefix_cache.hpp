#pragma once

/// @file
/// A prefix cache that never caches: the baseline without prefix caching.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

/// `PrefixCache` that caches nothing, so every lookup misses. Used as the no-caching baseline.
///
/// It still validates its arguments and counts locks on its only node (the root), so code driven
/// through it follows the same protocol as with the real caches.
class NullPrefixCache {
 public:
  /// Creates the cache; `allocator` only provides the block size for argument checks.
  explicit NullPrefixCache(BlockAllocator& allocator) noexcept
      : block_size_(allocator.block_size()) {}

  // The members below ignore the cache state but stay non-static to match the PrefixCache
  // interface of the real caches.
  // NOLINTBEGIN(readability-convert-member-functions-to-static)

  /// Always an empty match at the root.
  [[nodiscard]] PrefixMatch match_prefix(std::span<const TokenId> /*tokens*/) const {
    return PrefixMatch{};
  }

  /// Caches nothing. @return `Errc::invalid_argument` if `blocks` is shorter than the number of
  /// full blocks in `tokens`.
  [[nodiscard]] Result<InsertResult> insert(std::span<const TokenId> tokens,
                                            std::span<const BlockId> blocks) const noexcept {
    if (blocks.size() < tokens.size() / block_size_) {
      return fail(Errc::invalid_argument);
    }
    return InsertResult{};
  }

  /// Counts a lock on the root. @return `Errc::invalid_handle` for any other handle.
  [[nodiscard]] Result<void> lock(NodeHandle node) noexcept {
    if (node != root()) {
      return fail(Errc::invalid_handle);
    }
    if (locks_ == std::numeric_limits<std::uint32_t>::max()) {
      return fail(Errc::ref_count_overflow);
    }
    ++locks_;
    return {};
  }

  /// Releases a lock on the root. @return `Errc::invalid_handle` or `Errc::not_locked`.
  [[nodiscard]] Result<void> unlock(NodeHandle node) noexcept {
    if (node != root()) {
      return fail(Errc::invalid_handle);
    }
    if (locks_ == 0) {
      return fail(Errc::not_locked);
    }
    --locks_;
    return {};
  }

  /// Nothing to evict; always returns 0.
  std::size_t evict(std::size_t /*num_blocks*/) noexcept { return 0; }

  /// Handle of the root (the only node).
  [[nodiscard]] NodeHandle root() const noexcept { return NodeHandle{}; }

  /// True only for the root handle.
  [[nodiscard]] bool contains(NodeHandle node) const noexcept { return node == root(); }

  /// Always false.
  [[nodiscard]] bool holds_block(BlockId /*block*/) const noexcept { return false; }

  /// All zero.
  [[nodiscard]] CacheStats stats() const noexcept { return CacheStats{}; }

  /// Always `std::nullopt`.
  [[nodiscard]] std::optional<std::string> check_invariants() const { return std::nullopt; }

  // NOLINTEND(readability-convert-member-functions-to-static)

 private:
  std::uint32_t block_size_;
  std::uint32_t locks_ = 0;
};

}  // namespace radix_kv
