// Exception safety under memory exhaustion: every allocation an operation makes is failed in turn,
// and after each std::bad_alloc the cache, the sequence and the pool must be exactly as before.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <optional>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/hash_prefix_cache.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/radix_prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/types.hpp"

#include "fault_injection/failing_allocator.hpp"

namespace radix_kv {
namespace {

using test_support::FailingAllocations;

// Runs `operation` with allocation number n and every later one failing, for n = 0, 1, 2, ...,
// until it runs without a failure. After every failed attempt (with allocations working again)
// `expect_unchanged` checks the state. Returns the number of allocations the operation makes.
template <class Operation, class Check>
std::size_t fail_each_allocation(const Operation& operation, const Check& expect_unchanged) {
  for (std::size_t allowed = 0;; ++allowed) {
    bool failed = false;
    {
      const FailingAllocations failing(allowed);
      try {
        operation();
      } catch (const std::bad_alloc&) {
        failed = true;
      }
    }
    if (!failed) {
      return allowed;
    }
    expect_unchanged();
    if (::testing::Test::HasFailure()) {
      ADD_FAILURE() << "state changed after allocation " << allowed << " failed";
      return allowed;
    }
  }
}

std::vector<std::uint32_t> ref_counts(const BlockAllocator& allocator) {
  std::vector<std::uint32_t> counts(allocator.num_blocks());
  for (BlockId block = 0; block < allocator.num_blocks(); ++block) {
    counts[block] = allocator.ref_count(block);
  }
  return counts;
}

template <class Cache>
std::vector<bool> held_blocks(const Cache& cache, const BlockAllocator& allocator) {
  std::vector<bool> held(allocator.num_blocks());
  for (BlockId block = 0; block < allocator.num_blocks(); ++block) {
    held[block] = cache.holds_block(block);
  }
  return held;
}

// Everything observable about a cache and its pool, except node handles: a failed attempt may
// recycle node slots, so a later handle's generation can differ.
template <class Cache>
auto observe(const Cache& cache, const BlockAllocator& allocator) {
  const CacheStats stats = cache.stats();
  return std::tuple(stats.cached_blocks, stats.locked_blocks, stats.evictable_blocks, stats.nodes,
                    stats.evicted_blocks, allocator.num_free(), ref_counts(allocator),
                    held_blocks(cache, allocator));
}

std::vector<TokenId> tokens_of(std::initializer_list<TokenId> tokens) {
  return tokens;
}

// A cache with an unlocked four-block prompt and a locked two-block prompt still held by a
// sequence. Two instances built by this fixture are identical, including their LRU order.
template <class Cache>
struct Scenario {
  static constexpr std::uint32_t kBlockSize = 2;

  Scenario() {
    Sequence first(allocator);
    EXPECT_TRUE(first.append(tokens_of({1, 2, 3, 4, 5, 6, 7, 8})));
    EXPECT_TRUE(cache.insert(first.tokens(), first.block_table()));
    EXPECT_TRUE(running.append(tokens_of({20, 21, 22, 23, 24})));
    const Result<InsertResult> inserted = cache.insert(running.tokens(), running.block_table());
    EXPECT_TRUE(inserted);
    lock = inserted->node;
    EXPECT_TRUE(cache.lock(lock));
  }

  // Evicts one entry at a time and records what is left, which exposes the LRU order.
  std::vector<std::vector<bool>> drain() {
    std::vector<std::vector<bool>> steps;
    EXPECT_TRUE(cache.unlock(lock));
    while (cache.evict(1) > 0) {
      steps.push_back(held_blocks(cache, allocator));
    }
    return steps;
  }

  BlockAllocator allocator{32, kBlockSize};
  Cache cache{allocator};
  Sequence running{allocator};
  NodeHandle lock;
};

template <class Cache>
class AllocationFailure : public ::testing::Test {};

using CacheTypes = ::testing::Types<HashPrefixCache, RadixPrefixCache>;
TYPED_TEST_SUITE(AllocationFailure, CacheTypes);

TYPED_TEST(AllocationFailure, InsertIsAllOrNothing) {
  Scenario<TypeParam> tested;
  Scenario<TypeParam> reference;
  // Shares the first two blocks with the cached prompt, then diverges: the radix tree splits a
  // node and adds a leaf, the hash cache adds three entries.
  Sequence diverging(tested.allocator);
  ASSERT_TRUE(diverging.append(tokens_of({1, 2, 3, 4, 9, 9, 10, 10, 11, 11})));
  Sequence same(reference.allocator);
  ASSERT_TRUE(same.append(diverging.tokens()));
  const auto before = observe(tested.cache, tested.allocator);
  ASSERT_EQ(before, observe(reference.cache, reference.allocator));

  std::size_t new_blocks = 0;
  const std::size_t allocations = fail_each_allocation(
      [&] {
        const Result<InsertResult> inserted =
            tested.cache.insert(diverging.tokens(), diverging.block_table());
        ASSERT_TRUE(inserted);
        new_blocks = inserted->new_blocks;
      },
      [&] {
        EXPECT_EQ(observe(tested.cache, tested.allocator), before);
        EXPECT_EQ(tested.cache.check_invariants(), std::nullopt);
      });
  EXPECT_GT(allocations, 0U);
  EXPECT_EQ(new_blocks, 3U);

  ASSERT_TRUE(reference.cache.insert(same.tokens(), same.block_table()));
  EXPECT_EQ(observe(tested.cache, tested.allocator), observe(reference.cache, reference.allocator));
  EXPECT_EQ(tested.drain(), reference.drain());
}

TYPED_TEST(AllocationFailure, MatchPrefixChangesNothingWhenItFails) {
  Scenario<TypeParam> tested;
  Scenario<TypeParam> reference;
  const std::vector<TokenId> query = tokens_of({1, 2, 3, 4, 5, 6, 0});  // ends inside an edge
  const auto before = observe(tested.cache, tested.allocator);

  std::size_t matched = 0;
  const std::size_t allocations =
      fail_each_allocation([&] { matched = tested.cache.match_prefix(query).num_tokens; },
                           [&] {
                             EXPECT_EQ(observe(tested.cache, tested.allocator), before);
                             EXPECT_EQ(tested.cache.check_invariants(), std::nullopt);
                           });
  EXPECT_GT(allocations, 0U);
  EXPECT_EQ(matched, 6U);

  EXPECT_EQ(reference.cache.match_prefix(query).num_tokens, 6U);
  EXPECT_EQ(tested.drain(), reference.drain());
}

TEST(AllocationFailure, SequenceOperationsAreAllOrNothing) {
  BlockAllocator allocator(32, 4);
  Sequence sequence(allocator);
  ASSERT_TRUE(sequence.append(tokens_of({1, 2, 3, 4, 5, 6})));
  const auto state = [&] {
    return std::tuple(
        std::vector<TokenId>(sequence.tokens().begin(), sequence.tokens().end()),
        std::vector<BlockId>(sequence.block_table().begin(), sequence.block_table().end()),
        ref_counts(allocator));
  };
  const auto before = state();
  const auto expect_unchanged = [&] { EXPECT_EQ(state(), before); };

  // fork(): the child is dropped at the end of every attempt.
  EXPECT_GT(fail_each_allocation([&] { ASSERT_TRUE(sequence.fork()); }, expect_unchanged), 0U);
  EXPECT_EQ(state(), before);

  // assign_prefix() into a new sequence that shares the first (full) block.
  const std::vector<TokenId> prefix(sequence.tokens().begin(), sequence.tokens().begin() + 4);
  const std::vector<BlockId> blocks{sequence.block_table().front()};
  EXPECT_GT(fail_each_allocation(
                [&] {
                  Sequence seeded(allocator);
                  ASSERT_TRUE(seeded.assign_prefix(prefix, blocks));
                },
                expect_unchanged),
            0U);

  // append() across several blocks, into a partial last block shared with a fork (copy-on-write).
  const Result<Sequence> fork = sequence.fork();
  ASSERT_TRUE(fork);
  const auto shared = state();
  const std::vector<TokenId> more(20, 7);
  EXPECT_GT(fail_each_allocation([&] { ASSERT_TRUE(sequence.append(more)); },
                                 [&] { EXPECT_EQ(state(), shared); }),
            0U);
  EXPECT_EQ(sequence.size(), 26U);
}

}  // namespace
}  // namespace radix_kv
