#include "radix_kv/hash_prefix_cache.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/detail/lru_list.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

#include "check.hpp"

namespace radix_kv {

namespace {

// Hash of the empty prefix; the first block's hash is hasher(kRootHash, tokens of block 0).
constexpr std::uint64_t kRootHash = 0x6a09e667f3bcc908ULL;

std::span<const TokenId> block_tokens(std::span<const TokenId> tokens, std::size_t index,
                                      std::size_t block_size) noexcept {
  return tokens.subspan(index * block_size, block_size);
}

}  // namespace

HashPrefixCache::HashPrefixCache(BlockAllocator& allocator, BlockHasher hasher)
    : allocator_(&allocator),
      hasher_(hasher),
      block_size_(allocator.block_size()),
      store_(allocator.num_blocks(), allocator.block_size()),
      entries_(static_cast<std::size_t>(allocator.num_blocks()) + 1),
      lru_(entries_.size()) {
  if (hasher_ == nullptr) {
    throw std::invalid_argument("HashPrefixCache: hasher must not be null");
  }
  Entry& root = entries_[kRootSlot];
  root.hash = kRootHash;
  root.live = true;
}

HashPrefixCache::~HashPrefixCache() {
  for (std::size_t slot = 1; slot < entries_.size(); ++slot) {
    if (entries_[slot].live) {
      detail::expect_ok(allocator_->release(entries_[slot].block));
    }
  }
}

PrefixMatch HashPrefixCache::match_prefix(std::span<const TokenId> tokens) {
  PrefixMatch match;
  const std::size_t full_blocks = tokens.size() / block_size_;
  std::uint32_t parent = kRootSlot;
  std::uint64_t hash = kRootHash;
  for (std::size_t i = 0; i < full_blocks; ++i) {
    const std::span<const TokenId> chunk = block_tokens(tokens, i, block_size_);
    hash = hasher_(hash, chunk);
    const auto found = index_.find(hash);
    if (found == index_.end()) {
      break;
    }
    const Entry& entry = entries_[found->second];
    if (entry.parent != parent || !store_.matches(entry.block, chunk)) {
      break;  // hash collision: a different prefix owns this hash
    }
    match.blocks.push_back(entry.block);
    parent = found->second;
  }
  touch_path(parent);
  match.num_tokens = match.blocks.size() * block_size_;
  match.node = handle(parent);
  return match;
}

Result<InsertResult> HashPrefixCache::insert(std::span<const TokenId> tokens,
                                             std::span<const BlockId> blocks) {
  const std::size_t full_blocks = tokens.size() / block_size_;
  if (blocks.size() < full_blocks) {
    return fail(Errc::invalid_argument);
  }
  const std::span<const BlockId> to_cache = blocks.first(full_blocks);

  // Phase 1 (read-only): find how much of the prefix is already cached, then validate the rest.
  std::uint32_t parent = kRootSlot;
  std::uint64_t hash = kRootHash;
  std::size_t next = 0;
  for (; next < full_blocks; ++next) {
    const std::span<const TokenId> chunk = block_tokens(tokens, next, block_size_);
    const std::uint64_t candidate = hasher_(hash, chunk);
    const auto found = index_.find(candidate);
    if (found == index_.end() || entries_[found->second].parent != parent ||
        !store_.matches(entries_[found->second].block, chunk)) {
      break;
    }
    parent = found->second;
    hash = candidate;
  }
  if (const Result<void> valid = store_.validate_insert(*allocator_, to_cache, next); !valid) {
    return fail(valid.error());
  }

  // Phase 2: cache the remaining blocks. A hash owned by a different prefix (a collision) ends
  // the chain early: everything cached so far stays consistent, the rest is simply not cached.
  std::size_t added = 0;
  try {
    for (; next < full_blocks; ++next) {
      const std::span<const TokenId> chunk = block_tokens(tokens, next, block_size_);
      hash = hasher_(hash, chunk);
      const BlockId block = to_cache[next];
      const std::uint32_t slot = block + 1;
      if (!index_.try_emplace(hash, slot).second) {  // the only step that can throw
        break;
      }
      Entry& entry = entries_[slot];
      entry.hash = hash;
      entry.block = block;
      entry.parent = parent;
      entry.lock_ref = 0;
      entry.direct_locks = 0;
      entry.children = 0;
      entry.live = true;
      ++entries_[parent].children;
      detail::expect_ok(allocator_->retain(block));
      store_.store(block, chunk);
      parent = slot;
      ++added;
    }
  } catch (...) {
    touch_path(parent);  // keep the blocks cached so far evictable, then report the failure
    throw;
  }
  touch_path(parent);
  return InsertResult{.new_blocks = added, .node = handle(parent)};
}

Result<void> HashPrefixCache::lock(NodeHandle node) noexcept {
  if (!valid(node)) {
    return fail(Errc::invalid_handle);
  }
  // The root's count is the total number of locks, an upper bound for every other count.
  if (entries_[kRootSlot].lock_ref == std::numeric_limits<std::uint32_t>::max()) {
    return fail(Errc::ref_count_overflow);
  }
  ++entries_[node.index].direct_locks;
  for (std::uint32_t slot = node.index; slot != kNone; slot = entries_[slot].parent) {
    if (entries_[slot].lock_ref++ == 0 && slot != kRootSlot) {
      ++locked_blocks_;
      lru_.remove(slot);
    }
  }
  return {};
}

Result<void> HashPrefixCache::unlock(NodeHandle node) noexcept {
  if (!valid(node)) {
    return fail(Errc::invalid_handle);
  }
  Entry& target = entries_[node.index];
  if (target.direct_locks == 0) {
    return fail(Errc::not_locked);
  }
  --target.direct_locks;
  // Walking upwards appends the newly unlocked entries deepest first, so every entry stays in
  // front of its parent in the LRU order.
  for (std::uint32_t slot = node.index; slot != kNone; slot = entries_[slot].parent) {
    if (--entries_[slot].lock_ref == 0 && slot != kRootSlot) {
      --locked_blocks_;
      lru_.push_back(slot);
    }
  }
  return {};
}

std::size_t HashPrefixCache::evict(std::size_t num_blocks) noexcept {
  std::size_t released = 0;
  while (released < num_blocks && !lru_.empty()) {
    evict_slot(lru_.front());
    ++released;
  }
  evicted_blocks_ += released;
  return released;
}

bool HashPrefixCache::contains(NodeHandle node) const noexcept {
  return valid(node);
}

CacheStats HashPrefixCache::stats() const noexcept {
  return CacheStats{
      .cached_blocks = store_.size(),
      .locked_blocks = locked_blocks_,
      .evictable_blocks = store_.size() - locked_blocks_,
      .nodes = store_.size(),
      .evicted_blocks = evicted_blocks_,
  };
}

bool HashPrefixCache::valid(NodeHandle node) const noexcept {
  return node.index < entries_.size() && entries_[node.index].live &&
         entries_[node.index].generation == node.generation;
}

NodeHandle HashPrefixCache::handle(std::uint32_t slot) const noexcept {
  return NodeHandle{.index = slot, .generation = entries_[slot].generation};
}

void HashPrefixCache::touch_path(std::uint32_t slot) noexcept {
  // Move the unlocked part of the path to the most-recently-used end, deepest entry first. A
  // locked entry has only locked ancestors (they count its locks), so the walk can stop there.
  for (std::uint32_t s = slot; s != kRootSlot && entries_[s].lock_ref == 0;
       s = entries_[s].parent) {
    lru_.move_to_back(s);
  }
}

void HashPrefixCache::evict_slot(std::uint32_t slot) noexcept {
  Entry& entry = entries_[slot];
  // The LRU order keeps children in front of their parents, so the victim is always a leaf.
  assert(entry.live && entry.lock_ref == 0 && entry.children == 0);
  lru_.remove(slot);
  index_.erase(entry.hash);
  --entries_[entry.parent].children;
  store_.erase(entry.block);
  detail::expect_ok(allocator_->release(entry.block));
  entry.live = false;
  ++entry.generation;  // invalidates outstanding handles to this slot
  entry.parent = kNone;
  entry.block = kInvalidBlock;
}

std::optional<std::string> HashPrefixCache::check_invariants() const {
  const Entry& root = entries_[kRootSlot];
  if (!root.live || root.parent != kNone || root.hash != kRootHash) {
    return "root entry is malformed";
  }
  std::vector<std::uint32_t> children(entries_.size(), 0);
  std::vector<std::uint64_t> child_locks(entries_.size(), 0);
  std::size_t live = 0;
  std::size_t locked = 0;
  for (std::uint32_t slot = 1; slot < entries_.size(); ++slot) {
    if (auto problem = check_entry(slot)) {
      return problem;
    }
    const Entry& entry = entries_[slot];
    if (entry.live) {
      ++live;
      ++children[entry.parent];
      child_locks[entry.parent] += entry.lock_ref;
      locked += entry.lock_ref > 0 ? 1U : 0U;
    }
  }
  for (std::uint32_t slot = 0; slot < entries_.size(); ++slot) {
    const Entry& entry = entries_[slot];
    if (!entry.live) {
      continue;
    }
    if (entry.children != children[slot]) {
      return "entry " + std::to_string(slot) + ": wrong child count";
    }
    if (entry.lock_ref != entry.direct_locks + child_locks[slot]) {
      return "entry " + std::to_string(slot) + ": lock count != direct locks + child locks";
    }
  }
  if (live != index_.size() || live != store_.size()) {
    return std::string("entries, hash index and token store disagree on the cached block count");
  }
  if (locked != locked_blocks_) {
    return std::string("locked block counter is wrong");
  }
  return check_lru();
}

std::optional<std::string> HashPrefixCache::check_entry(std::uint32_t slot) const {
  const Entry& entry = entries_[slot];
  const std::string where = "entry " + std::to_string(slot) + ": ";
  if (!entry.live) {
    if (lru_.contains(slot) || store_.contains(slot - 1)) {
      return where + "dead slot is still referenced";
    }
    return std::nullopt;
  }
  if (entry.block != slot - 1 || !store_.contains(entry.block)) {
    return where + "does not own its block";
  }
  if (!allocator_->is_allocated(entry.block)) {
    return where + "cached block is free in the allocator";
  }
  if (entry.parent >= entries_.size() || !entries_[entry.parent].live) {
    return where + "parent is not live";
  }
  if (entry.hash != hasher_(entries_[entry.parent].hash, store_.tokens(entry.block))) {
    return where + "hash does not chain from the parent hash and the block tokens";
  }
  const auto found = index_.find(entry.hash);
  if (found == index_.end() || found->second != slot) {
    return where + "missing from the hash index";
  }
  if (lru_.contains(slot) != (entry.lock_ref == 0)) {
    return where + "LRU membership does not match the lock state";
  }
  return std::nullopt;
}

std::optional<std::string> HashPrefixCache::check_lru() const {
  // Every unlocked entry must be older than its (unlocked) parent: eviction then only takes leaves.
  std::vector<std::size_t> position(entries_.size(), 0);
  std::size_t count = 0;
  std::uint32_t previous = detail::LruList::kNil;
  for (std::uint32_t slot = lru_.front(); slot != detail::LruList::kNil; slot = lru_.next(slot)) {
    if (slot >= entries_.size() || lru_.prev(slot) != previous || ++count > entries_.size()) {
      return std::string("LRU list links are broken");
    }
    position[slot] = count;
    previous = slot;
  }
  if (previous != lru_.back() || count != lru_.size() || count != store_.size() - locked_blocks_) {
    return std::string("LRU list size or tail is wrong");
  }
  for (std::uint32_t slot = 1; slot < entries_.size(); ++slot) {
    const std::uint32_t parent = entries_[slot].parent;
    if (lru_.contains(slot) && parent != kRootSlot && lru_.contains(parent) &&
        position[slot] >= position[parent]) {
      return "entry " + std::to_string(slot) + ": not older than its parent in the LRU order";
    }
  }
  return std::nullopt;
}

}  // namespace radix_kv
