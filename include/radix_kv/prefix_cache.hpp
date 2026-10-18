#pragma once

/// @file
/// Types and the `PrefixCache` concept shared by the prefix-caching strategies.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

/// Result of `match_prefix()`: the longest cached, block-aligned prefix of the query.
struct PrefixMatch {
  /// Matched tokens; always a multiple of the block size.
  std::size_t num_tokens = 0;
  /// Cached blocks holding the matched tokens, in order. The cache keeps owning them: take your own
  /// references (`Sequence::assign_prefix()`) and `lock(node)` while a request uses them.
  std::vector<BlockId> blocks;
  /// Deepest matched node; the cache root when nothing matched.
  NodeHandle node;
};

/// Result of `insert()`.
struct InsertResult {
  /// Blocks the cache started holding (and took a reference to) during this call. Blocks of the
  /// inserted prefix that were already cached are not counted and the caller's copies are not
  /// retained.
  std::size_t new_blocks = 0;
  /// Deepest node covering the inserted prefix; lock it to protect the whole path from eviction.
  NodeHandle node;
};

/// Point-in-time counters of a prefix cache.
struct CacheStats {
  std::size_t cached_blocks = 0;     ///< Blocks currently held by the cache.
  std::size_t locked_blocks = 0;     ///< Cached blocks on a locked path (not evictable).
  std::size_t evictable_blocks = 0;  ///< `cached_blocks - locked_blocks`.
  std::size_t nodes = 0;             ///< Live nodes, excluding the root.
  std::uint64_t evicted_blocks = 0;  ///< Blocks released by `evict()` since construction.
};

/// Hash function over the tokens of one block, chained through `seed`.
///
/// Caches never trust a hash alone: every hit is confirmed by comparing token ids, so a weak or
/// colliding hasher can only lower the hit rate, never return wrong blocks. Tests inject one to
/// force collisions.
using BlockHasher = std::uint64_t (*)(std::uint64_t seed, std::span<const TokenId> tokens) noexcept;

/// Default `BlockHasher`: a fast, non-cryptographic 64-bit mix of `seed` and every token.
[[nodiscard]] std::uint64_t hash_tokens(std::uint64_t seed,
                                        std::span<const TokenId> tokens) noexcept;

/// Interface shared by `HashPrefixCache`, `RadixPrefixCache` and `NullPrefixCache`, used by the
/// simulator to run the same scheduler code over every strategy without virtual calls.
///
/// Contract, identical for every implementation:
/// - `match_prefix(tokens)` returns the longest cached prefix made of whole blocks and marks it
///   recently used. It takes no references and no locks.
/// - `insert(tokens, blocks)` caches the `tokens.size() / block_size` full blocks of `blocks`,
///   taking one allocator reference per newly cached block. It either succeeds or changes nothing.
/// - `lock(node)` pins the path from the root to `node`; locked blocks are never evicted.
///   `unlock(node)` undoes exactly one `lock(node)` taken through the same handle.
/// - `evict(n)` releases least-recently-used unlocked leaves until at least `n` blocks were
///   released or nothing evictable is left, and returns the number released.
/// - `allocator()` is the pool the cache was created over. Block ids passed to `insert` must be
///   blocks of that pool: an id is only checked against the pool, not against who allocated it.
template <class Cache>
concept PrefixCache =
    requires(Cache& cache, const Cache& view, std::span<const TokenId> tokens,
             std::span<const BlockId> blocks, NodeHandle node, std::size_t count, BlockId block) {
      { cache.match_prefix(tokens) } -> std::same_as<PrefixMatch>;
      { cache.insert(tokens, blocks) } -> std::same_as<Result<InsertResult>>;
      { cache.lock(node) } -> std::same_as<Result<void>>;
      { cache.unlock(node) } -> std::same_as<Result<void>>;
      { cache.evict(count) } -> std::same_as<std::size_t>;
      { view.root() } -> std::same_as<NodeHandle>;
      { view.contains(node) } -> std::same_as<bool>;
      { view.holds_block(block) } -> std::same_as<bool>;
      { view.stats() } -> std::same_as<CacheStats>;
      { view.check_invariants() } -> std::same_as<std::optional<std::string>>;
      { view.allocator() } -> std::same_as<BlockAllocator&>;
    };

}  // namespace radix_kv
