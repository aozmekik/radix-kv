#include "radix_kv/radix_prefix_cache.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/detail/lru_list.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

#include "check.hpp"

namespace radix_kv {

namespace {

// Seed of the child-index hash (a different constant than the hash cache's root hash).
constexpr std::uint64_t kChildSeed = 0xbb67ae8584caa73bULL;

std::ptrdiff_t offset(std::size_t count) noexcept {
  return static_cast<std::ptrdiff_t>(count);
}

}  // namespace

RadixPrefixCache::RadixPrefixCache(BlockAllocator& allocator, BlockHasher hasher)
    : allocator_(&allocator),
      hasher_(hasher),
      block_size_(allocator.block_size()),
      store_(allocator.num_blocks(), allocator.block_size()),
      nodes_(1),
      lru_(1) {
  if (hasher_ == nullptr) {
    throw std::invalid_argument("RadixPrefixCache: hasher must not be null");
  }
  nodes_[kRootNode].live = true;
}

RadixPrefixCache::~RadixPrefixCache() {
  for (const Node& node : nodes_) {
    if (node.live) {
      for (const BlockId block : node.blocks) {
        detail::expect_ok(allocator_->release(block));
      }
    }
  }
}

PrefixMatch RadixPrefixCache::match_prefix(std::span<const TokenId> tokens) {
  PrefixMatch match;
  const std::size_t full_blocks = tokens.size() / block_size_;
  std::uint32_t node = kRootNode;
  std::size_t matched = 0;
  while (matched < full_blocks) {
    const auto& children = nodes_[node].children;
    const auto found = children.find(child_key(block_tokens(tokens, matched)));
    if (found == children.end()) {
      break;
    }
    const std::uint32_t child = found->second;
    const std::size_t common = common_blocks(child, tokens, matched, full_blocks);
    if (common == 0) {
      break;  // child-key collision: that child starts with different tokens
    }
    const std::vector<BlockId>& edge = nodes_[child].blocks;
    match.blocks.insert(match.blocks.end(), edge.begin(), edge.begin() + offset(common));
    matched += common;
    if (common < edge.size()) {
      node = split(child, common);  // the match ends inside `child`
      break;
    }
    node = child;
  }
  touch(node);
  match.num_tokens = matched * block_size_;
  match.node = handle(node);
  return match;
}

Result<InsertResult> RadixPrefixCache::insert(std::span<const TokenId> tokens,
                                              std::span<const BlockId> blocks) {
  const std::size_t full_blocks = tokens.size() / block_size_;
  if (blocks.size() < full_blocks) {
    return fail(Errc::invalid_argument);
  }
  const std::span<const BlockId> to_cache = blocks.first(full_blocks);
  const InsertPlan plan = plan_insert(tokens, full_blocks);
  if (const Result<void> valid = store_.validate_insert(*allocator_, to_cache, plan.next); !valid) {
    return fail(valid.error());
  }

  std::uint32_t node = plan.parent;
  if (plan.split_child != kNone) {
    node = split(plan.split_child, plan.split_at);
  }
  std::size_t added = 0;
  if (!plan.blocked && plan.next < full_blocks) {
    node = add_leaf(node, tokens, to_cache, plan.next);
    added = full_blocks - plan.next;
  }
  touch(node);
  return InsertResult{.new_blocks = added, .node = handle(node)};
}

Result<void> RadixPrefixCache::lock(NodeHandle node) noexcept {
  if (!valid(node)) {
    return fail(Errc::invalid_handle);
  }
  // The root's count is the total number of locks, an upper bound for every other count.
  if (nodes_[kRootNode].lock_ref == std::numeric_limits<std::uint32_t>::max()) {
    return fail(Errc::ref_count_overflow);
  }
  ++nodes_[node.index].direct_locks;
  for (std::uint32_t n = node.index; n != kNone; n = nodes_[n].parent) {
    if (nodes_[n].lock_ref++ == 0 && n != kRootNode) {
      locked_blocks_ += nodes_[n].blocks.size();
      lru_.remove(n);
    }
  }
  return {};
}

Result<void> RadixPrefixCache::unlock(NodeHandle node) noexcept {
  if (!valid(node)) {
    return fail(Errc::invalid_handle);
  }
  Node& target = nodes_[node.index];
  if (target.direct_locks == 0) {
    return fail(Errc::not_locked);
  }
  --target.direct_locks;
  // Walking upwards appends the newly unlocked nodes deepest first (children before parents).
  for (std::uint32_t n = node.index; n != kNone; n = nodes_[n].parent) {
    if (--nodes_[n].lock_ref == 0 && n != kRootNode) {
      locked_blocks_ -= nodes_[n].blocks.size();
      lru_.push_back(n);
    }
  }
  return {};
}

std::size_t RadixPrefixCache::evict(std::size_t num_blocks) noexcept {
  std::size_t released = 0;
  while (released < num_blocks && !lru_.empty()) {
    const std::uint32_t victim = lru_.front();
    released += nodes_[victim].blocks.size();
    remove_leaf(victim);
  }
  evicted_blocks_ += released;
  return released;
}

bool RadixPrefixCache::contains(NodeHandle node) const noexcept {
  return valid(node);
}

CacheStats RadixPrefixCache::stats() const noexcept {
  return CacheStats{
      .cached_blocks = store_.size(),
      .locked_blocks = locked_blocks_,
      .evictable_blocks = store_.size() - locked_blocks_,
      .nodes = live_nodes_,
      .evicted_blocks = evicted_blocks_,
  };
}

bool RadixPrefixCache::valid(NodeHandle node) const noexcept {
  return node.index < nodes_.size() && nodes_[node.index].live &&
         nodes_[node.index].generation == node.generation;
}

NodeHandle RadixPrefixCache::handle(std::uint32_t node) const noexcept {
  return NodeHandle{.index = node, .generation = nodes_[node].generation};
}

std::uint64_t RadixPrefixCache::child_key(std::span<const TokenId> block_tokens) const noexcept {
  return hasher_(kChildSeed, block_tokens);
}

std::span<const TokenId> RadixPrefixCache::block_tokens(std::span<const TokenId> tokens,
                                                        std::size_t index) const noexcept {
  return tokens.subspan(index * block_size_, block_size_);
}

std::size_t RadixPrefixCache::common_blocks(std::uint32_t node, std::span<const TokenId> tokens,
                                            std::size_t first,
                                            std::size_t full_blocks) const noexcept {
  const std::vector<BlockId>& edge = nodes_[node].blocks;
  std::size_t count = 0;
  while (count < edge.size() && first + count < full_blocks &&
         store_.matches(edge[count], block_tokens(tokens, first + count))) {
    ++count;
  }
  return count;
}

RadixPrefixCache::InsertPlan RadixPrefixCache::plan_insert(std::span<const TokenId> tokens,
                                                           std::size_t full_blocks) const noexcept {
  InsertPlan plan;
  while (plan.next < full_blocks) {
    const auto& children = nodes_[plan.parent].children;
    const auto found = children.find(child_key(block_tokens(tokens, plan.next)));
    if (found == children.end()) {
      return plan;  // the rest becomes a new leaf under plan.parent
    }
    const std::uint32_t child = found->second;
    const std::size_t common = common_blocks(child, tokens, plan.next, full_blocks);
    if (common == 0) {
      plan.blocked = true;  // child-key collision with different tokens
      return plan;
    }
    plan.next += common;
    if (common < nodes_[child].blocks.size()) {
      plan.split_child = child;
      plan.split_at = common;
      // After the split the rest becomes a sibling of the child's tail; their first blocks differ
      // but their keys could still collide.
      if (plan.next < full_blocks) {
        const std::uint64_t tail_key = child_key(store_.tokens(nodes_[child].blocks[common]));
        plan.blocked = child_key(block_tokens(tokens, plan.next)) == tail_key;
      }
      return plan;
    }
    plan.parent = child;
  }
  return plan;
}

std::uint32_t RadixPrefixCache::new_node() {
  std::uint32_t index = 0;
  if (num_free_nodes_ == 0) {
    index = static_cast<std::uint32_t>(nodes_.size());
    // Grow every side table first so that freeing a node later never allocates.
    lru_.grow(nodes_.size() + 1);
    free_nodes_.resize(nodes_.size() + 1);
    nodes_.emplace_back();
  } else {
    index = free_nodes_[--num_free_nodes_];
  }
  nodes_[index].live = true;
  ++live_nodes_;
  return index;
}

void RadixPrefixCache::free_node(std::uint32_t node) noexcept {
  Node& slot = nodes_[node];
  slot.blocks.clear();
  slot.children.clear();
  slot.key = 0;
  slot.parent = kNone;
  slot.lock_ref = 0;
  slot.direct_locks = 0;
  slot.live = false;
  ++slot.generation;                      // invalidates outstanding handles to this slot
  free_nodes_[num_free_nodes_++] = node;  // sized in new_node(): never overflows
  --live_nodes_;
}

std::uint32_t RadixPrefixCache::split(std::uint32_t node, std::size_t prefix_blocks) {
  assert(prefix_blocks > 0 && prefix_blocks < nodes_[node].blocks.size());
  // Allocate everything up front so that std::bad_alloc leaves the tree untouched.
  const std::vector<BlockId>& edge = nodes_[node].blocks;
  std::vector<BlockId> upper_blocks(edge.begin(), edge.begin() + offset(prefix_blocks));
  const std::uint64_t lower_key = child_key(store_.tokens(edge[prefix_blocks]));
  std::unordered_map<std::uint64_t, std::uint32_t> upper_children;
  upper_children.emplace(lower_key, node);
  const std::uint32_t upper = new_node();

  Node& top = nodes_[upper];
  Node& bottom = nodes_[node];
  top.blocks = std::move(upper_blocks);
  top.children = std::move(upper_children);
  top.key = bottom.key;
  top.parent = bottom.parent;
  top.lock_ref = bottom.lock_ref;  // every lock below `top` is a lock on `bottom` or below it
  bottom.blocks.erase(bottom.blocks.begin(), bottom.blocks.begin() + offset(prefix_blocks));
  bottom.key = lower_key;
  bottom.parent = upper;
  nodes_[top.parent].children.find(top.key)->second = upper;
  if (lru_.contains(node)) {
    lru_.insert_after(node, upper);  // same recency, and still in front of the old parent
  }
  return upper;
}

std::uint32_t RadixPrefixCache::add_leaf(std::uint32_t parent, std::span<const TokenId> tokens,
                                         std::span<const BlockId> blocks, std::size_t first) {
  std::vector<BlockId> edge(blocks.begin() + offset(first), blocks.end());
  const std::uint64_t key = child_key(block_tokens(tokens, first));
  const std::uint32_t leaf = new_node();
  try {
    [[maybe_unused]] const bool inserted = nodes_[parent].children.emplace(key, leaf).second;
    assert(inserted && "plan_insert() rules out key collisions");
  } catch (...) {
    free_node(leaf);
    throw;
  }
  Node& node = nodes_[leaf];
  node.parent = parent;
  node.key = key;
  node.blocks = std::move(edge);
  for (std::size_t i = 0; i < node.blocks.size(); ++i) {
    detail::expect_ok(allocator_->retain(node.blocks[i]));
    store_.store(node.blocks[i], block_tokens(tokens, first + i));
  }
  return leaf;  // unlocked: insert() adds it to the LRU order through touch()
}

void RadixPrefixCache::touch(std::uint32_t node) noexcept {
  // Move the unlocked part of the path to the most-recently-used end, deepest node first. A
  // locked node has only locked ancestors, so the walk can stop there.
  for (std::uint32_t n = node; n != kRootNode && nodes_[n].lock_ref == 0; n = nodes_[n].parent) {
    lru_.move_to_back(n);
  }
}

void RadixPrefixCache::remove_leaf(std::uint32_t node) noexcept {
  const Node& leaf = nodes_[node];
  // The LRU order keeps children in front of their parents, so the victim is always a leaf.
  assert(node != kRootNode && leaf.live && leaf.lock_ref == 0 && leaf.children.empty());
  lru_.remove(node);
  nodes_[leaf.parent].children.erase(leaf.key);
  for (const BlockId block : leaf.blocks) {
    store_.erase(block);
    detail::expect_ok(allocator_->release(block));
  }
  free_node(node);
}

std::optional<std::string> RadixPrefixCache::check_node(std::uint32_t node,
                                                        std::uint64_t& child_locks) const {
  const Node& current = nodes_[node];
  const std::string where = "node " + std::to_string(node) + ": ";
  child_locks = 0;
  for (const auto& [key, child] : current.children) {
    if (child >= nodes_.size() || !nodes_[child].live || child == kRootNode) {
      return where + "child is not a live node";
    }
    const Node& next = nodes_[child];
    if (next.parent != node || next.key != key || next.blocks.empty()) {
      return where + "child link, key or edge is inconsistent";
    }
    if (child_key(store_.tokens(next.blocks.front())) != key) {
      return where + "child key does not match the child's first block";
    }
    child_locks += next.lock_ref;
  }
  if (current.lock_ref != current.direct_locks + child_locks) {
    return where + "lock count != direct locks + child locks";
  }
  if (node == kRootNode) {
    return std::nullopt;
  }
  for (const BlockId block : current.blocks) {
    if (!store_.contains(block) || !allocator_->is_allocated(block)) {
      return where + "edge block " + std::to_string(block) + " is not cached or not allocated";
    }
  }
  if (lru_.contains(node) != (current.lock_ref == 0)) {
    return where + "LRU membership does not match the lock state";
  }
  return std::nullopt;
}

std::optional<std::string> RadixPrefixCache::check_invariants() const {
  const Node& root = nodes_[kRootNode];
  if (!root.live || root.parent != kNone || !root.blocks.empty() || lru_.contains(kRootNode)) {
    return std::string("root node is malformed");
  }
  std::vector<std::uint8_t> seen(nodes_.size(), 0);
  std::vector<std::uint32_t> stack{kRootNode};
  std::size_t reached = 0;
  std::size_t blocks = 0;
  std::size_t locked = 0;
  while (!stack.empty()) {
    const std::uint32_t node = stack.back();
    stack.pop_back();
    if (seen[node] != 0) {
      return "node " + std::to_string(node) + ": reached twice (not a tree)";
    }
    seen[node] = 1;
    std::uint64_t child_locks = 0;
    if (auto problem = check_node(node, child_locks)) {
      return problem;
    }
    for (const auto& entry : nodes_[node].children) {
      stack.push_back(entry.second);
    }
    if (node != kRootNode) {
      ++reached;
      blocks += nodes_[node].blocks.size();
      locked += nodes_[node].lock_ref > 0 ? nodes_[node].blocks.size() : 0;
    }
  }
  std::size_t live = 0;
  for (std::size_t n = 1; n < nodes_.size(); ++n) {
    live += nodes_[n].live ? 1U : 0U;
  }
  if (live != reached || live != live_nodes_) {
    return std::string("live nodes are unreachable or the node counter is wrong");
  }
  // Equal totals also rule out a block appearing in two nodes: the store counts distinct blocks.
  if (blocks != store_.size()) {
    return std::string("tree and token store disagree on the cached blocks");
  }
  if (locked != locked_blocks_) {
    return std::string("locked block counter is wrong");
  }
  return check_lru();
}

std::optional<std::string> RadixPrefixCache::check_lru() const {
  std::vector<std::size_t> position(nodes_.size(), 0);
  std::size_t count = 0;
  std::uint32_t previous = detail::LruList::kNil;
  for (std::uint32_t n = lru_.front(); n != detail::LruList::kNil; n = lru_.next(n)) {
    if (n >= nodes_.size() || !nodes_[n].live || lru_.prev(n) != previous ||
        ++count > nodes_.size()) {
      return std::string("LRU list links are broken");
    }
    position[n] = count;
    previous = n;
  }
  if (previous != lru_.back() || count != lru_.size()) {
    return std::string("LRU list size or tail is wrong");
  }
  // Every unlocked node must be older than its (unlocked) parent: eviction then only takes leaves.
  for (std::uint32_t n = 1; n < nodes_.size(); ++n) {
    const std::uint32_t parent = nodes_[n].parent;
    if (lru_.contains(n) && parent != kRootNode && lru_.contains(parent) &&
        position[n] >= position[parent]) {
      return "node " + std::to_string(n) + ": not older than its parent in the LRU order";
    }
  }
  return std::nullopt;
}

}  // namespace radix_kv
