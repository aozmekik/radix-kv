#pragma once

/// @file
/// Implementation detail shared by the prefix caches. Not part of the public API.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace radix_kv::detail {

/// Recency order over slot indices `[0, capacity)`, as a doubly linked list stored in a vector.
///
/// Every operation except `grow()` is O(1) and allocation-free. The front is the least recently
/// used slot. Both prefix caches keep every evictable node in such a list and move a whole path to
/// the back deepest node first, which keeps each node in front of its parent.
class LruList {
 public:
  /// Marks "no slot".
  static constexpr std::uint32_t kNil = 0xffffffffU;

  /// A list able to hold slots `[0, capacity)`, initially empty.
  explicit LruList(std::size_t capacity = 0) : links_(capacity) {}

  /// Makes room for slots `[0, capacity)`. May throw `std::bad_alloc`; the list is unchanged then.
  void grow(std::size_t capacity) {
    if (capacity > links_.size()) {
      links_.resize(capacity);
    }
  }

  /// True when `slot` is in the list.
  [[nodiscard]] bool contains(std::uint32_t slot) const noexcept {
    return slot < links_.size() && links_[slot].linked;
  }

  /// True when the list is empty.
  [[nodiscard]] bool empty() const noexcept { return head_ == kNil; }

  /// Number of slots in the list.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  /// Least recently used slot, or `kNil`.
  [[nodiscard]] std::uint32_t front() const noexcept { return head_; }

  /// Most recently used slot, or `kNil`.
  [[nodiscard]] std::uint32_t back() const noexcept { return tail_; }

  /// The slot used just after `slot`, or `kNil`. Precondition: `contains(slot)`.
  [[nodiscard]] std::uint32_t next(std::uint32_t slot) const noexcept { return links_[slot].next; }

  /// The slot used just before `slot`, or `kNil`. Precondition: `contains(slot)`.
  [[nodiscard]] std::uint32_t prev(std::uint32_t slot) const noexcept { return links_[slot].prev; }

  /// Appends `slot` as the most recently used. Precondition: `!contains(slot)`.
  void push_back(std::uint32_t slot) noexcept {
    assert(slot < links_.size() && !links_[slot].linked);
    Links& links = links_[slot];
    links.prev = tail_;
    links.next = kNil;
    links.linked = true;
    if (tail_ != kNil) {
      links_[tail_].next = slot;
    } else {
      head_ = slot;
    }
    tail_ = slot;
    ++size_;
  }

  /// Inserts `slot` right after `anchor`. Preconditions: `contains(anchor)`, `!contains(slot)`.
  void insert_after(std::uint32_t anchor, std::uint32_t slot) noexcept {
    assert(contains(anchor) && slot < links_.size() && !links_[slot].linked);
    Links& links = links_[slot];
    const std::uint32_t after = links_[anchor].next;
    links.prev = anchor;
    links.next = after;
    links.linked = true;
    links_[anchor].next = slot;
    if (after != kNil) {
      links_[after].prev = slot;
    } else {
      tail_ = slot;
    }
    ++size_;
  }

  /// Unlinks `slot`. Precondition: `contains(slot)`.
  void remove(std::uint32_t slot) noexcept {
    assert(contains(slot));
    Links& links = links_[slot];
    if (links.prev != kNil) {
      links_[links.prev].next = links.next;
    } else {
      head_ = links.next;
    }
    if (links.next != kNil) {
      links_[links.next].prev = links.prev;
    } else {
      tail_ = links.prev;
    }
    links = Links{};
    --size_;
  }

  /// Makes `slot` the most recently used, inserting it if needed.
  void move_to_back(std::uint32_t slot) noexcept {
    if (contains(slot)) {
      remove(slot);
    }
    push_back(slot);
  }

 private:
  struct Links {
    std::uint32_t prev = kNil;
    std::uint32_t next = kNil;
    bool linked = false;
  };

  std::vector<Links> links_;
  std::uint32_t head_ = kNil;
  std::uint32_t tail_ = kNil;
  std::size_t size_ = 0;
};

}  // namespace radix_kv::detail
