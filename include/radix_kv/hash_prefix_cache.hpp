#pragma once

/// @file
/// vLLM-style prefix cache: chained hashes of full blocks, a hash-to-block map and LRU eviction.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/detail/block_token_store.hpp"
#include "radix_kv/detail/lru_list.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

/// Prefix cache keyed by chained block hashes, in the style of vLLM's automatic prefix caching.
///
/// Block `i` of a sequence is identified by `h_i = hasher(h_{i-1}, tokens of block i)`, so equal
/// hashes mean equal prefixes up to and including that block. A query walks its blocks in order and
/// stops at the first one missing from the hash map. Each cached block is one node whose parent is
/// the previous block, which makes the cache an (uncompressed) trie with one node per block and a
/// hash index over it. Every hit is confirmed against the stored token ids and the parent link, so
/// a hash collision costs a cache miss, never a wrong block.
///
/// Eviction takes the least-recently-used unlocked block. Paths are marked used deepest block
/// first, so a block is always older than its ancestors and the eviction victim is always a leaf:
/// the cache never strands a cached block behind an evicted parent (checked by
/// `check_invariants()`).
///
/// Complexity: `match_prefix` and `insert` are O(L) hash computations and map operations for L
/// query blocks; `lock`, `unlock` and `match_prefix`'s LRU update are O(depth); `evict(n)` is
/// O(n). Not thread-safe; the allocator must outlive the cache.
class HashPrefixCache {
 public:
  /// Creates an empty cache over `allocator`'s pool. `hasher` is injectable for testing.
  explicit HashPrefixCache(BlockAllocator& allocator, BlockHasher hasher = &hash_tokens);

  HashPrefixCache(const HashPrefixCache&) = delete;
  HashPrefixCache& operator=(const HashPrefixCache&) = delete;
  HashPrefixCache(HashPrefixCache&&) = delete;
  HashPrefixCache& operator=(HashPrefixCache&&) = delete;
  /// Releases the cache's reference to every cached block. Blocks still used by sequences stay
  /// allocated until those sequences release them.
  ~HashPrefixCache();

  /// Longest cached prefix of `tokens` made of whole blocks; marks it most recently used.
  [[nodiscard]] PrefixMatch match_prefix(std::span<const TokenId> tokens);

  /// Caches the `tokens.size() / block_size` full blocks of `blocks` (block `i` holds tokens
  /// `[i * block_size, (i + 1) * block_size)`), taking a reference to each newly cached block.
  /// Positions already cached keep their block; the caller's duplicate is left alone.
  /// @return `Errc::invalid_argument` if `blocks` is too short, a block to cache is already cached
  ///         elsewhere or appears twice; `Errc::invalid_block` / `Errc::block_not_allocated` for
  ///         bad block ids. Nothing changes on error.
  [[nodiscard]] Result<InsertResult> insert(std::span<const TokenId> tokens,
                                            std::span<const BlockId> blocks);

  /// Pins every block on the path from the root to `node` against eviction.
  /// @return `Errc::invalid_handle` or `Errc::ref_count_overflow`.
  [[nodiscard]] Result<void> lock(NodeHandle node) noexcept;

  /// Releases one lock previously taken with `lock(node)`.
  /// @return `Errc::invalid_handle`, or `Errc::not_locked` if `node` itself holds no lock.
  [[nodiscard]] Result<void> unlock(NodeHandle node) noexcept;

  /// Evicts least-recently-used unlocked blocks until `num_blocks` were released or none is left.
  /// @return blocks released (at most `num_blocks`).
  std::size_t evict(std::size_t num_blocks) noexcept;

  /// Handle of the root, the empty prefix. Locking it is allowed and pins nothing.
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static): PrefixCache interface.
  [[nodiscard]] NodeHandle root() const noexcept { return NodeHandle{.index = 0, .generation = 0}; }

  /// True when `node` refers to a live node of this cache.
  [[nodiscard]] bool contains(NodeHandle node) const noexcept;

  /// True when the cache holds a reference to `block`.
  [[nodiscard]] bool holds_block(BlockId block) const noexcept { return store_.contains(block); }

  /// Current counters.
  [[nodiscard]] CacheStats stats() const noexcept;

  /// The pool this cache was created over; `insert` expects block ids of this pool.
  [[nodiscard]] BlockAllocator& allocator() const noexcept { return *allocator_; }

  /// Full structural check (O(cached blocks)); returns a description of the first violated
  /// invariant, or `std::nullopt`. Intended for tests and debugging.
  [[nodiscard]] std::optional<std::string> check_invariants() const;

 private:
  static constexpr std::uint32_t kNone = 0xffffffffU;
  static constexpr std::uint32_t kRootSlot = 0;

  // Slot `block + 1` holds the entry for `block`; slot 0 is the root.
  struct Entry {
    std::uint64_t hash = 0;
    BlockId block = kInvalidBlock;
    std::uint32_t parent = kNone;
    std::uint32_t generation = 0;
    std::uint32_t lock_ref = 0;      // locks on this entry or any descendant
    std::uint32_t direct_locks = 0;  // locks taken through a handle to this entry
    std::uint32_t children = 0;
    bool live = false;
  };

  [[nodiscard]] bool valid(NodeHandle node) const noexcept;
  [[nodiscard]] NodeHandle handle(std::uint32_t slot) const noexcept;
  void touch_path(std::uint32_t slot) noexcept;
  void evict_slot(std::uint32_t slot) noexcept;
  [[nodiscard]] std::optional<std::string> check_entry(std::uint32_t slot) const;
  [[nodiscard]] std::optional<std::string> check_lru() const;

  BlockAllocator* allocator_;
  BlockHasher hasher_;
  std::uint32_t block_size_;
  detail::BlockTokenStore store_;
  std::vector<Entry> entries_;
  std::unordered_map<std::uint64_t, std::uint32_t> index_;  // chained hash -> slot
  detail::LruList lru_;                                     // every unlocked entry
  std::size_t locked_blocks_ = 0;
  std::uint64_t evicted_blocks_ = 0;
};

}  // namespace radix_kv
