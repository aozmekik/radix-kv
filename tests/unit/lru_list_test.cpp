#include "radix_kv/detail/lru_list.hpp"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace radix_kv::detail {
namespace {

using ::testing::ElementsAre;

std::vector<std::uint32_t> order(const LruList& list) {
  std::vector<std::uint32_t> slots;
  for (std::uint32_t slot = list.front(); slot != LruList::kNil; slot = list.next(slot)) {
    slots.push_back(slot);
  }
  return slots;
}

TEST(LruList, StartsEmpty) {
  const LruList list(4);
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.size(), 0U);
  EXPECT_EQ(list.front(), LruList::kNil);
  EXPECT_EQ(list.back(), LruList::kNil);
  EXPECT_FALSE(list.contains(0));
  // Out of range is simply absent. `volatile` keeps GCC from constant-folding the index into a
  // (guarded, unreachable) element access and reporting a false -Warray-bounds positive.
  const volatile std::uint32_t past_the_end = 99;
  const volatile std::uint32_t nil = LruList::kNil;
  EXPECT_FALSE(list.contains(past_the_end));
  EXPECT_FALSE(list.contains(nil));
}

TEST(LruList, PushBackKeepsInsertionOrder) {
  LruList list(4);
  list.push_back(2);
  list.push_back(0);
  list.push_back(3);
  EXPECT_THAT(order(list), ElementsAre(2U, 0U, 3U));
  EXPECT_EQ(list.front(), 2U);
  EXPECT_EQ(list.back(), 3U);
  EXPECT_EQ(list.size(), 3U);
  EXPECT_EQ(list.prev(0), 2U);
}

TEST(LruList, RemovesFromAnyPosition) {
  LruList list(5);
  for (std::uint32_t slot = 0; slot < 5; ++slot) {
    list.push_back(slot);
  }
  list.remove(2);  // middle
  list.remove(0);  // head
  list.remove(4);  // tail
  EXPECT_THAT(order(list), ElementsAre(1U, 3U));
  EXPECT_EQ(list.front(), 1U);
  EXPECT_EQ(list.back(), 3U);
  EXPECT_FALSE(list.contains(2));
  list.remove(1);
  list.remove(3);
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.back(), LruList::kNil);
}

TEST(LruList, InsertAfterAndMoveToBack) {
  LruList list(5);
  list.push_back(0);
  list.push_back(1);
  list.insert_after(0, 4);  // middle
  list.insert_after(1, 3);  // after the tail
  EXPECT_THAT(order(list), ElementsAre(0U, 4U, 1U, 3U));
  EXPECT_EQ(list.back(), 3U);
  list.move_to_back(0);
  list.move_to_back(2);  // not yet present: inserted
  EXPECT_THAT(order(list), ElementsAre(4U, 1U, 3U, 0U, 2U));
  EXPECT_EQ(list.size(), 5U);
}

TEST(LruList, GrowKeepsContents) {
  LruList list(1);
  list.push_back(0);
  list.grow(3);
  list.push_back(2);
  list.grow(2);  // never shrinks
  EXPECT_THAT(order(list), ElementsAre(0U, 2U));
}

}  // namespace
}  // namespace radix_kv::detail
