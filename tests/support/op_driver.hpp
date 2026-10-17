#pragma once

// Randomized operation driver shared by the property tests and the libFuzzer target.
//
// It interprets a byte string as a sequence of scheduler-like operations on one allocator, one
// prefix cache and a handful of sequences, and checks after every operation that:
//   - every allocator reference count equals the number of holders (sequences + cache) of the
//   block,
//     and the free count matches;
//   - the cache's structural invariants hold (check_invariants());
//   - every lock still held refers to a live node (locked nodes are never evicted);
//   - matched blocks hold exactly the queried tokens, and no holder ever sees a block's tokens
//     change underneath it (copy-on-write and cache immutability).
// At the end it releases everything and checks that every block is free.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/sequence.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv::test_support {

/// A deliberately terrible hasher (16 distinct values) that forces hash and child-key collisions.
inline std::uint64_t weak_hasher(std::uint64_t seed, std::span<const TokenId> tokens) noexcept {
  return hash_tokens(seed, tokens) & 0xfU;
}

/// Reference cache for differential tests: a map from every cached block-aligned prefix to its
/// block. Never evicts; locks are counted but pin nothing (nothing is ever evicted anyway).
class NaivePrefixCache {
 public:
  explicit NaivePrefixCache(BlockAllocator& allocator, BlockHasher /*unused*/ = nullptr)
      : allocator_(&allocator) {}
  NaivePrefixCache(const NaivePrefixCache&) = delete;
  NaivePrefixCache& operator=(const NaivePrefixCache&) = delete;
  NaivePrefixCache(NaivePrefixCache&&) = delete;
  NaivePrefixCache& operator=(NaivePrefixCache&&) = delete;
  ~NaivePrefixCache() {
    for (const auto& entry : prefixes_) {
      static_cast<void>(allocator_->release(entry.second));
    }
  }

  PrefixMatch match_prefix(std::span<const TokenId> tokens) {
    PrefixMatch match;
    const std::size_t block_size = allocator_->block_size();
    std::vector<TokenId> key;
    for (std::size_t i = 0; (i + 1) * block_size <= tokens.size(); ++i) {
      const auto chunk = tokens.subspan(i * block_size, block_size);
      key.insert(key.end(), chunk.begin(), chunk.end());
      const auto found = prefixes_.find(key);
      if (found == prefixes_.end()) {
        break;
      }
      match.blocks.push_back(found->second);
    }
    match.num_tokens = match.blocks.size() * block_size;
    return match;
  }

  Result<InsertResult> insert(std::span<const TokenId> tokens, std::span<const BlockId> blocks) {
    const std::size_t block_size = allocator_->block_size();
    const std::size_t full = tokens.size() / block_size;
    if (blocks.size() < full) {
      return fail(Errc::invalid_argument);
    }
    InsertResult result;
    std::vector<TokenId> key;
    for (std::size_t i = 0; i < full; ++i) {
      const auto chunk = tokens.subspan(i * block_size, block_size);
      key.insert(key.end(), chunk.begin(), chunk.end());
      if (prefixes_.contains(key)) {
        continue;
      }
      if (const Result<void> retained = allocator_->retain(blocks[i]); !retained) {
        return fail(retained.error());
      }
      prefixes_.emplace(key, blocks[i]);
      held_.insert(blocks[i]);
      ++result.new_blocks;
    }
    return result;
  }

  // Interface members that ignore the (absent) tree; non-static to match the real caches.
  // NOLINTBEGIN(readability-convert-member-functions-to-static)
  Result<void> lock(NodeHandle /*node*/) noexcept {
    ++locks_;
    return {};
  }
  Result<void> unlock(NodeHandle /*node*/) noexcept {
    if (locks_ == 0) {
      return fail(Errc::not_locked);
    }
    --locks_;
    return {};
  }
  std::size_t evict(std::size_t /*num_blocks*/) noexcept { return 0; }
  [[nodiscard]] NodeHandle root() const noexcept { return NodeHandle{}; }
  [[nodiscard]] bool contains(NodeHandle /*node*/) const noexcept { return true; }
  [[nodiscard]] bool holds_block(BlockId block) const noexcept { return held_.contains(block); }
  [[nodiscard]] CacheStats stats() const noexcept {
    return CacheStats{.cached_blocks = held_.size(),
                      .locked_blocks = 0,
                      .evictable_blocks = held_.size(),
                      .nodes = held_.size(),
                      .evicted_blocks = 0};
  }
  [[nodiscard]] std::optional<std::string> check_invariants() const { return std::nullopt; }
  // NOLINTEND(readability-convert-member-functions-to-static)

 private:
  BlockAllocator* allocator_;
  std::map<std::vector<TokenId>, BlockId> prefixes_;
  std::set<BlockId> held_;
  std::uint32_t locks_ = 0;
};

static_assert(PrefixCache<NaivePrefixCache>);

/// Parameters of a driver run.
struct DriverOptions {
  std::uint32_t num_blocks = 48;
  std::uint32_t block_size = 2;
  std::uint32_t vocab_size = 3;  // tiny alphabet: random prompts share prefixes all the time
  std::size_t max_sequences = 8;
  bool allow_evict = true;
  BlockHasher hasher = &hash_tokens;
};

/// What a driver run exercised; lets tests assert that a property was not checked vacuously.
struct DriverStats {
  std::size_t operations = 0;
  std::size_t matches_with_hits = 0;
  std::size_t copy_on_writes = 0;
  std::size_t failed_appends = 0;
  std::size_t evicted_blocks = 0;

  DriverStats& operator+=(const DriverStats& other) noexcept {
    operations += other.operations;
    matches_with_hits += other.matches_with_hits;
    copy_on_writes += other.copy_on_writes;
    failed_appends += other.failed_appends;
    evicted_blocks += other.evicted_blocks;
    return *this;
  }
};

template <PrefixCache Cache>
class OpDriver {
 public:
  explicit OpDriver(const DriverOptions& options)
      : options_(options),
        allocator_(options.num_blocks, options.block_size),
        cache_(allocator_, options.hasher),
        shadow_(options.num_blocks),
        holders_(options.num_blocks),
        views_(static_cast<std::size_t>(options.num_blocks) * options.block_size),
        view_lengths_(options.num_blocks) {}

  /// Interprets `input`; returns the first violated property, or std::nullopt.
  std::optional<std::string> run(std::span<const std::uint8_t> input) {
    input_ = input;
    position_ = 0;
    while (position_ < input_.size()) {
      ++stats_.operations;
      if (auto problem = step()) {
        return problem;
      }
      if (auto problem = verify()) {
        return problem;
      }
    }
    return teardown();
  }

  /// Coverage counters of the run so far.
  [[nodiscard]] const DriverStats& stats() const noexcept { return stats_; }

  /// Token count of every match performed, in order (for differential tests).
  [[nodiscard]] const std::vector<std::size_t>& match_lengths() const noexcept {
    return match_lengths_;
  }

 private:
  struct Tracked {
    Sequence sequence;
    std::optional<NodeHandle> lock;
  };

  std::uint8_t byte() noexcept { return position_ < input_.size() ? input_[position_++] : 0; }

  std::size_t pick(std::size_t count) noexcept { return count == 0 ? 0 : byte() % count; }

  std::vector<TokenId> random_tokens(std::size_t max_length) {
    std::vector<TokenId> tokens(1 + pick(max_length));
    for (TokenId& token : tokens) {
      token = byte() % options_.vocab_size;
    }
    return tokens;
  }

  // Half of the prompts extend a prefix of an earlier sequence, the way requests share system
  // prompts and conversation history; the rest are fresh.
  std::vector<TokenId> prompt() {
    if (history_.empty() || byte() % 2 == 0) {
      return random_tokens(16);
    }
    const std::vector<TokenId>& base = history_[pick(history_.size())];
    std::vector<TokenId> tokens(base.begin(),
                                base.begin() + static_cast<std::ptrdiff_t>(pick(base.size() + 1)));
    const std::vector<TokenId> suffix = random_tokens(8);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    return tokens;
  }

  void remember(const Sequence& sequence) {
    constexpr std::size_t kHistory = 8;
    if (sequence.empty()) {
      return;
    }
    if (history_.size() == kHistory) {
      history_.erase(history_.begin());
    }
    history_.emplace_back(sequence.tokens().begin(), sequence.tokens().end());
  }

  std::optional<std::string> step() {
    switch (byte() % 9) {
      case 0:
        return start_sequence();
      case 1:
        return append_tokens();
      case 2:
        return fork_sequence();
      case 3:
        return truncate_sequence();
      case 4:
        return insert_sequence(/*finish=*/false);
      case 5:
        return insert_sequence(/*finish=*/true);
      case 6:
        return evict_some();
      case 7:
        return query_only();
      default:
        return extra_lock();
    }
  }

  std::optional<std::string> checked_match(std::span<const TokenId> tokens, PrefixMatch& match) {
    match = cache_.match_prefix(tokens);
    match_lengths_.push_back(match.num_tokens);
    stats_.matches_with_hits += match.num_tokens > 0 ? 1U : 0U;
    const std::size_t block_size = options_.block_size;
    if (match.num_tokens % block_size != 0 || match.num_tokens > tokens.size() ||
        match.blocks.size() * block_size != match.num_tokens) {
      return "match_prefix returned a malformed match";
    }
    for (std::size_t i = 0; i < match.blocks.size(); ++i) {
      const BlockId block = match.blocks[i];
      const auto expected = tokens.subspan(i * block_size, block_size);
      if (!cache_.holds_block(block) || !std::ranges::equal(shadow_[block], expected)) {
        return "match_prefix returned a block that does not hold the queried tokens";
      }
    }
    return std::nullopt;
  }

  // Appends `tokens`, evicting once if allowed; false if there was no room (nothing changed).
  bool append_with_eviction(Sequence& sequence, std::span<const TokenId> tokens) {
    const std::size_t before = sequence.size();
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (const Result<AppendResult> appended = sequence.append(tokens)) {
        stats_.copy_on_writes += appended->copy.has_value() ? 1U : 0U;
        return true;
      }
      if (sequence.size() != before || !options_.allow_evict) {
        break;
      }
      stats_.evicted_blocks += cache_.evict(tokens.size() / options_.block_size + 2);
    }
    ++stats_.failed_appends;
    return false;
  }

  std::optional<std::string> start_sequence() {
    if (sequences_.size() >= options_.max_sequences) {
      return std::nullopt;
    }
    const std::vector<TokenId> prompt = this->prompt();
    PrefixMatch match;
    if (auto problem = checked_match(prompt, match)) {
      return problem;
    }
    if (!cache_.lock(match.node)) {
      return "lock() failed on a handle returned by match_prefix()";
    }
    Sequence sequence(allocator_);
    const auto prefix = std::span<const TokenId>(prompt).first(match.num_tokens);
    if (!sequence.assign_prefix(prefix, match.blocks)) {
      return "assign_prefix() failed on a fresh match";
    }
    if (!append_with_eviction(sequence,
                              std::span<const TokenId>(prompt).subspan(match.num_tokens))) {
      sequence.release();
      return cache_.unlock(match.node) ? std::nullopt
                                       : std::optional<std::string>("unlock() failed");
    }
    sequences_.push_back(Tracked{.sequence = std::move(sequence), .lock = match.node});
    return std::nullopt;
  }

  std::optional<std::string> append_tokens() {
    if (sequences_.empty()) {
      return std::nullopt;
    }
    Tracked& tracked = sequences_[pick(sequences_.size())];
    const std::size_t before = tracked.sequence.size();
    const std::vector<TokenId> tokens = random_tokens(5);
    if (!append_with_eviction(tracked.sequence, tokens) && tracked.sequence.size() != before) {
      return "a failed append() changed the sequence";
    }
    return std::nullopt;
  }

  std::optional<std::string> fork_sequence() {
    if (sequences_.empty() || sequences_.size() >= options_.max_sequences) {
      return std::nullopt;
    }
    const std::size_t parent = pick(sequences_.size());
    Result<Sequence> child = sequences_[parent].sequence.fork();
    if (!child) {
      return "fork() failed";
    }
    std::optional<NodeHandle> lock = sequences_[parent].lock;
    if (lock && !cache_.lock(*lock)) {
      return "re-locking the parent's handle for a fork failed";
    }
    sequences_.push_back(Tracked{.sequence = std::move(*child), .lock = lock});
    return std::nullopt;
  }

  std::optional<std::string> truncate_sequence() {
    if (sequences_.empty()) {
      return std::nullopt;
    }
    Sequence& sequence = sequences_[pick(sequences_.size())].sequence;
    if (!sequence.truncate(pick(sequence.size() + 1))) {
      return "truncate() within bounds failed";
    }
    return std::nullopt;
  }

  std::optional<std::string> insert_sequence(bool finish) {
    if (sequences_.empty()) {
      return std::nullopt;
    }
    const std::size_t index = pick(sequences_.size());
    Tracked& tracked = sequences_[index];
    const Result<InsertResult> inserted =
        cache_.insert(tracked.sequence.tokens(), tracked.sequence.block_table());
    if (!inserted) {
      return "insert() of a live sequence failed: " + std::string(to_string(inserted.error()));
    }
    record_cached(tracked.sequence);
    remember(tracked.sequence);
    if (!finish) {
      // Move the sequence's lock to the inserted path, as a scheduler does after prefill.
      if (!cache_.lock(inserted->node)) {
        return "lock() failed on a handle returned by insert()";
      }
      if (tracked.lock && !cache_.unlock(*tracked.lock)) {
        return "unlock() of a held lock failed";
      }
      tracked.lock = inserted->node;
      return std::nullopt;
    }
    if (tracked.lock && !cache_.unlock(*tracked.lock)) {
      return "unlock() of a held lock failed";
    }
    sequences_.erase(sequences_.begin() + static_cast<std::ptrdiff_t>(index));
    return std::nullopt;
  }

  std::optional<std::string> evict_some() {
    if (!options_.allow_evict) {
      return std::nullopt;
    }
    const CacheStats before = cache_.stats();
    const std::size_t requested = pick(6);
    const std::size_t released = cache_.evict(requested);
    stats_.evicted_blocks += released;
    const CacheStats after = cache_.stats();
    if (after.cached_blocks + released != before.cached_blocks ||
        after.evicted_blocks != before.evicted_blocks + released ||
        after.locked_blocks != before.locked_blocks) {
      return "evict() counters are inconsistent";
    }
    if (released < requested && after.evictable_blocks != 0) {
      return "evict() stopped early although evictable blocks remain";
    }
    return std::nullopt;
  }

  std::optional<std::string> query_only() {
    PrefixMatch match;
    return checked_match(prompt(), match);
  }

  std::optional<std::string> extra_lock() {
    if (!extra_locks_.empty() && (byte() % 2 == 0)) {
      const NodeHandle node = extra_locks_.back();
      extra_locks_.pop_back();
      return cache_.unlock(node) ? std::nullopt
                                 : std::optional<std::string>("unlock() of an extra lock failed");
    }
    PrefixMatch match;
    if (auto problem = checked_match(prompt(), match)) {
      return problem;
    }
    if (!cache_.lock(match.node)) {
      return "lock() failed on a fresh match";
    }
    extra_locks_.push_back(match.node);
    return std::nullopt;
  }

  void record_cached(const Sequence& sequence) {
    const std::size_t block_size = options_.block_size;
    for (std::size_t i = 0; i < sequence.num_full_blocks(); ++i) {
      const BlockId block = sequence.block_table()[i];
      if (cache_.holds_block(block) && shadow_[block].empty()) {
        const auto chunk = sequence.tokens().subspan(i * block_size, block_size);
        shadow_[block].assign(chunk.begin(), chunk.end());
      }
    }
  }

  std::optional<std::string> verify() {
    if (auto problem = collect_views()) {
      return problem;
    }
    if (auto problem = verify_blocks()) {
      return problem;
    }
    for (const Tracked& tracked : sequences_) {
      if (tracked.lock && !cache_.contains(*tracked.lock)) {
        return std::string("a locked node was evicted");
      }
    }
    for (const NodeHandle& node : extra_locks_) {
      if (!cache_.contains(node)) {
        return std::string("a locked node was evicted");
      }
    }
    if (auto problem = cache_.check_invariants()) {
      return "cache invariant violated: " + *problem;
    }
    return std::nullopt;
  }

  // Counts the holders of every block and checks that they all see the same tokens in it.
  std::optional<std::string> collect_views() {
    const std::size_t block_size = options_.block_size;
    std::ranges::fill(holders_, 0U);
    std::ranges::fill(view_lengths_, 0U);
    for (const Tracked& tracked : sequences_) {
      const Sequence& sequence = tracked.sequence;
      if (sequence.num_blocks() != allocator_.blocks_for(sequence.size())) {
        return std::string("block table size does not match the token count");
      }
      for (std::size_t i = 0; i < sequence.num_blocks(); ++i) {
        const BlockId block = sequence.block_table()[i];
        ++holders_[block];
        // Every holder's view of a block must agree with every other holder's on their overlap.
        const std::size_t end = std::min(sequence.size(), (i + 1) * block_size);
        const auto view = sequence.tokens().subspan(i * block_size, end - (i * block_size));
        const auto known = std::span<TokenId>(views_).subspan(block * block_size, block_size);
        const std::size_t common = std::min<std::size_t>(view_lengths_[block], view.size());
        if (!std::equal(view.begin(), view.begin() + static_cast<std::ptrdiff_t>(common),
                        known.begin())) {
          return "two holders of block " + std::to_string(block) + " see different tokens";
        }
        if (view.size() > view_lengths_[block]) {
          std::ranges::copy(view, known.begin());
          view_lengths_[block] = static_cast<std::uint32_t>(view.size());
        }
      }
    }
    return std::nullopt;
  }

  // Checks reference counts against the holders and cache immutability against the shadow copy.
  std::optional<std::string> verify_blocks() {
    const std::size_t block_size = options_.block_size;
    std::uint32_t free = 0;
    for (BlockId block = 0; block < options_.num_blocks; ++block) {
      const bool cached = cache_.holds_block(block);
      if (!cached) {
        shadow_[block].clear();
      } else if (view_lengths_[block] > 0) {
        // A cached block is immutable: every holder's view must be a prefix of the cached tokens.
        const auto known =
            std::span<const TokenId>(views_).subspan(block * block_size, view_lengths_[block]);
        if (shadow_[block].size() != block_size ||
            !std::equal(known.begin(), known.end(), shadow_[block].begin())) {
          return "cached block " + std::to_string(block) + " was modified";
        }
      }
      const std::uint32_t expected = holders_[block] + (cached ? 1U : 0U);
      if (allocator_.ref_count(block) != expected) {
        return "block " + std::to_string(block) + " has ref count " +
               std::to_string(allocator_.ref_count(block)) + ", expected " +
               std::to_string(expected);
      }
      free += expected == 0 ? 1U : 0U;
    }
    if (allocator_.num_free() != free) {
      return std::string("allocator free count does not match the reference model");
    }
    return std::nullopt;
  }

  std::optional<std::string> teardown() {
    for (Tracked& tracked : sequences_) {
      if (tracked.lock && !cache_.unlock(*tracked.lock)) {
        return std::string("unlock() at teardown failed");
      }
      tracked.sequence.release();
    }
    sequences_.clear();
    for (const NodeHandle& node : extra_locks_) {
      if (!cache_.unlock(node)) {
        return std::string("unlock() of an extra lock at teardown failed");
      }
    }
    extra_locks_.clear();
    if (cache_.stats().locked_blocks != 0) {
      return std::string("blocks remain locked after every lock was released");
    }
    if (allocator_.num_used() != cache_.stats().cached_blocks) {
      return std::string("blocks leaked: allocated blocks not held by the cache");
    }
    if (options_.allow_evict) {
      cache_.evict(std::numeric_limits<std::size_t>::max());
      if (allocator_.num_free() != allocator_.num_blocks()) {
        return std::string("blocks remain allocated after draining the cache");
      }
    }
    return cache_.check_invariants();
  }

  DriverOptions options_;
  BlockAllocator allocator_;
  Cache cache_;
  std::vector<std::vector<TokenId>> shadow_;  // tokens of each cached block, recorded at insert
  std::vector<std::uint32_t> holders_;        // scratch for verify(): sequences per block
  std::vector<TokenId> views_;                // scratch: longest view of each block
  std::vector<std::uint32_t> view_lengths_;   // scratch: length of that view
  std::vector<Tracked> sequences_;
  std::vector<std::vector<TokenId>> history_;  // recently inserted token sequences
  std::vector<NodeHandle> extra_locks_;
  std::vector<std::size_t> match_lengths_;
  DriverStats stats_;
  std::span<const std::uint8_t> input_;
  std::size_t position_ = 0;
};

}  // namespace radix_kv::test_support
