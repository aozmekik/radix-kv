#pragma once

/// @file
/// A sequence's tokens and its block table, with copy-on-write forking.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

/// What `Sequence::append()` did to the block table.
struct AppendResult {
  /// Blocks newly appended to the block table (not counting a copy-on-write replacement).
  std::uint32_t new_blocks = 0;
  /// Set when the last block was shared and had to be copied before writing to it. The engine must
  /// copy the KV data of `copy->src` into `copy->dst` before running the step.
  std::optional<CopyOp> copy;
};

/// The token ids of one sequence and its block table: block `i` holds tokens
/// `[i * block_size, (i + 1) * block_size)`.
///
/// A sequence owns one allocator reference per block in its table and releases them on
/// destruction (RAII). `fork()` shares every block with a new sequence; a shared block is never
/// written: before appending into a shared, partially filled last block, `append()` allocates a
/// private copy and reports a `CopyOp` (copy-on-write, as used for parallel sampling and beam
/// search in vLLM). Full blocks are immutable, which is also what makes them safe to share with a
/// prefix cache.
///
/// Mutating operations either succeed completely or leave the sequence and the pool unchanged.
/// Move-only; the allocator must outlive the sequence.
class Sequence {
 public:
  /// Creates an empty sequence drawing blocks from `allocator`.
  explicit Sequence(BlockAllocator& allocator) noexcept : allocator_(&allocator) {}

  Sequence(const Sequence&) = delete;
  Sequence& operator=(const Sequence&) = delete;
  /// Takes over `other`'s tokens and blocks; `other` becomes empty.
  Sequence(Sequence&& other) noexcept;
  /// Releases this sequence's blocks, then takes over `other`'s.
  Sequence& operator=(Sequence&& other) noexcept;
  /// Releases every block.
  ~Sequence();

  /// Appends `tokens`, allocating blocks as needed. `tokens` may view this sequence's own tokens
  /// (for example `seq.append(seq.tokens())`); such input is copied first.
  /// @return what changed, or `Errc::out_of_blocks` with nothing changed.
  /// @throws std::bad_alloc if the token or block vectors cannot grow (nothing changes).
  [[nodiscard]] Result<AppendResult> append(std::span<const TokenId> tokens);

  /// Appends a single token (one decode step).
  [[nodiscard]] Result<AppendResult> append(TokenId token);

  /// Returns a new sequence with the same tokens that shares every block with this one.
  /// @return the fork, or `Errc::ref_count_overflow` with nothing changed.
  [[nodiscard]] Result<Sequence> fork() const;

  /// Keeps the first `num_tokens` tokens and releases blocks that no longer hold any of them.
  /// A kept, shared last block is copied on the next `append()`, never modified in place.
  /// @return `Errc::invalid_argument` if `num_tokens > size()`.
  [[nodiscard]] Result<void> truncate(std::size_t num_tokens) noexcept;

  /// Releases every block and clears the tokens.
  void release() noexcept;

  /// Seeds an empty sequence with full blocks obtained from a prefix cache, taking one reference
  /// to each. `tokens.size()` must equal `blocks.size() * block_size()`.
  /// @return `Errc::invalid_argument` (sequence not empty or size mismatch), or the error of the
  ///         failing `retain()`; nothing changes on error.
  [[nodiscard]] Result<void> assign_prefix(std::span<const TokenId> tokens,
                                           std::span<const BlockId> blocks);

  /// All token ids, in order.
  [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

  /// The block table: block ids in sequence order.
  [[nodiscard]] std::span<const BlockId> block_table() const noexcept { return blocks_; }

  /// Number of tokens.
  [[nodiscard]] std::size_t size() const noexcept { return tokens_.size(); }

  /// True when the sequence holds no tokens (and therefore no blocks).
  [[nodiscard]] bool empty() const noexcept { return tokens_.empty(); }

  /// Number of blocks in the block table.
  [[nodiscard]] std::size_t num_blocks() const noexcept { return blocks_.size(); }

  /// Number of completely filled blocks; only these can be inserted into a prefix cache.
  [[nodiscard]] std::size_t num_full_blocks() const noexcept {
    return tokens_.size() / allocator_->block_size();
  }

  /// Tokens per block of the underlying pool.
  [[nodiscard]] std::uint32_t block_size() const noexcept { return allocator_->block_size(); }

  /// The allocator this sequence draws from.
  [[nodiscard]] BlockAllocator& allocator() const noexcept { return *allocator_; }

 private:
  BlockAllocator* allocator_;
  std::vector<TokenId> tokens_;
  std::vector<BlockId> blocks_;
};

}  // namespace radix_kv
