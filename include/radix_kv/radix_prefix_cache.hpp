#pragma once

/// @file
/// SGLang-style prefix cache: a radix tree over token sequences with LRU eviction of leaves.

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

/// Prefix cache organised as a radix tree, in the style of SGLang's RadixAttention.
///
/// Every edge is labelled with a run of whole blocks (the paged variant of the tree: with
/// `block_size == 1` it is a token-level radix tree), so a node holds one or more blocks and
/// children are found by their first block. Lookups compare token ids directly; a child index keyed
/// by a hash of the first block keeps fan-out lookups O(1) and is always confirmed by comparison.
///
/// `match_prefix` and `insert` split a node when the query diverges inside it, so the returned
/// handle covers exactly the matched or inserted prefix. Locks count running requests per path
/// (`lock_ref` in SGLang). `evict` removes whole least-recently-used unlocked leaves, as SGLang
/// does, and may therefore release more blocks than requested. Recency is kept in the same
/// intrusive LRU order as `HashPrefixCache` (paths are marked used deepest node first), so the
/// front of the order is always a leaf.
///
/// Complexity: `match_prefix` and `insert` are O(L) token comparisons for L query tokens plus one
/// child lookup per traversed node; `lock`/`unlock` are O(depth in nodes); `evict` is O(1) per
/// evicted leaf plus O(blocks released). Not thread-safe; the allocator must outlive the cache.
class RadixPrefixCache {
 public:
  /// Creates an empty tree over `allocator`'s pool. `hasher` (used only for the child index) is
  /// injectable for testing.
  explicit RadixPrefixCache(BlockAllocator& allocator, BlockHasher hasher = &hash_tokens);

  RadixPrefixCache(const RadixPrefixCache&) = delete;
  RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;
  RadixPrefixCache(RadixPrefixCache&&) = delete;
  RadixPrefixCache& operator=(RadixPrefixCache&&) = delete;
  /// Releases the tree's reference to every cached block.
  ~RadixPrefixCache();

  /// Longest cached prefix of `tokens` made of whole blocks. Splits the last node when the match
  /// ends inside it (the tree's shape changes, what is cached does not) and marks the path most
  /// recently used.
  /// @throws std::bad_alloc if a split cannot allocate its node; the tree is unchanged then.
  [[nodiscard]] PrefixMatch match_prefix(std::span<const TokenId> tokens);

  /// Caches the `tokens.size() / block_size` full blocks of `blocks`, taking a reference to each
  /// newly cached block; positions already cached keep their block. May split one node.
  /// @return `Errc::invalid_argument` if `blocks` is too short, a block to cache is already cached
  ///         elsewhere or appears twice; `Errc::invalid_block` / `Errc::block_not_allocated` for
  ///         bad block ids. Nothing changes on error.
  /// @throws std::bad_alloc if a node cannot be allocated; a split made for the insert is undone,
  ///         so nothing changes.
  [[nodiscard]] Result<InsertResult> insert(std::span<const TokenId> tokens,
                                            std::span<const BlockId> blocks);

  /// Pins the path from the root to `node` against eviction.
  /// @return `Errc::invalid_handle` or `Errc::ref_count_overflow`.
  [[nodiscard]] Result<void> lock(NodeHandle node) noexcept;

  /// Releases one lock previously taken with `lock(node)`. Still valid after `node` was split: the
  /// handle keeps referring to the node that holds the end of the locked prefix.
  /// @return `Errc::invalid_handle`, or `Errc::not_locked` if `node` itself holds no lock.
  [[nodiscard]] Result<void> unlock(NodeHandle node) noexcept;

  /// Evicts least-recently-used unlocked leaves until at least `num_blocks` blocks were released or
  /// nothing is evictable.
  /// @return blocks released; can exceed `num_blocks` because leaves are evicted whole.
  std::size_t evict(std::size_t num_blocks) noexcept;

  /// Handle of the root, the empty prefix. Locking it is allowed and pins nothing.
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static): PrefixCache interface.
  [[nodiscard]] NodeHandle root() const noexcept { return NodeHandle{.index = 0, .generation = 0}; }

  /// True when `node` refers to a live node of this tree.
  [[nodiscard]] bool contains(NodeHandle node) const noexcept;

  /// True when the tree holds a reference to `block`.
  [[nodiscard]] bool holds_block(BlockId block) const noexcept { return store_.contains(block); }

  /// Current counters.
  [[nodiscard]] CacheStats stats() const noexcept;

  /// The pool this tree was created over; `insert` expects block ids of this pool.
  [[nodiscard]] BlockAllocator& allocator() const noexcept { return *allocator_; }

  /// Full structural check (O(cached blocks)); returns a description of the first violated
  /// invariant, or `std::nullopt`. Intended for tests and debugging.
  [[nodiscard]] std::optional<std::string> check_invariants() const;

 private:
  static constexpr std::uint32_t kNone = 0xffffffffU;
  static constexpr std::uint32_t kRootNode = 0;

  struct Node {
    std::vector<BlockId> blocks;  // edge label: the tokens of these blocks, in order
    std::unordered_map<std::uint64_t, std::uint32_t> children;  // first-block key -> node
    std::uint64_t key = 0;  // first-block key under which the parent indexes this node
    std::uint32_t parent = kNone;
    std::uint32_t generation = 0;
    std::uint32_t lock_ref = 0;      // locks on this node or any descendant
    std::uint32_t direct_locks = 0;  // locks taken through a handle to this node
    bool live = false;
  };

  // Read-only plan computed by the first phase of insert().
  struct InsertPlan {
    std::uint32_t parent = kRootNode;   // deepest node matched completely
    std::uint32_t split_child = kNone;  // child of `parent` the inserted prefix ends or diverges in
    std::size_t split_at = 0;           // leading blocks of `split_child` that match
    std::size_t next = 0;               // first block of the insert that is not cached yet
    bool blocked = false;               // the uncached rest collides with an existing child key
  };

  [[nodiscard]] bool valid(NodeHandle node) const noexcept;
  [[nodiscard]] NodeHandle handle(std::uint32_t node) const noexcept;
  [[nodiscard]] std::uint64_t child_key(std::span<const TokenId> block_tokens) const noexcept;
  [[nodiscard]] std::span<const TokenId> block_tokens(std::span<const TokenId> tokens,
                                                      std::size_t index) const noexcept;
  [[nodiscard]] std::size_t common_blocks(std::uint32_t node, std::span<const TokenId> tokens,
                                          std::size_t first,
                                          std::size_t full_blocks) const noexcept;
  [[nodiscard]] InsertPlan plan_insert(std::span<const TokenId> tokens,
                                       std::size_t full_blocks) const noexcept;
  std::uint32_t new_node();
  void free_node(std::uint32_t node) noexcept;
  std::uint32_t split(std::uint32_t node, std::size_t prefix_blocks);
  void unsplit(std::uint32_t upper) noexcept;
  std::uint32_t add_leaf(std::uint32_t parent, std::span<const TokenId> tokens,
                         std::span<const BlockId> blocks, std::size_t first);
  void touch(std::uint32_t node) noexcept;
  void remove_leaf(std::uint32_t node) noexcept;
  [[nodiscard]] std::optional<std::string> check_node(std::uint32_t node,
                                                      std::uint64_t& child_locks) const;
  [[nodiscard]] std::optional<std::string> check_lru() const;

  BlockAllocator* allocator_;
  BlockHasher hasher_;
  std::uint32_t block_size_;
  detail::BlockTokenStore store_;
  std::vector<Node> nodes_;                // index 0 is the root
  std::vector<std::uint32_t> free_nodes_;  // recycled slots in free_nodes_[0, num_free_nodes_)
  std::size_t num_free_nodes_ = 0;
  detail::LruList lru_;  // every unlocked node
  std::size_t live_nodes_ = 0;
  std::size_t locked_blocks_ = 0;
  std::uint64_t evicted_blocks_ = 0;
};

}  // namespace radix_kv
