#include "radix_kv/sequence.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "radix_kv/block_allocator.hpp"
#include "radix_kv/result.hpp"
#include "radix_kv/types.hpp"

#include "check.hpp"

namespace radix_kv {

Sequence::Sequence(Sequence&& other) noexcept
    : allocator_(other.allocator_),
      tokens_(std::exchange(other.tokens_, {})),
      blocks_(std::exchange(other.blocks_, {})) {}

Sequence& Sequence::operator=(Sequence&& other) noexcept {
  if (this != &other) {
    release();
    allocator_ = other.allocator_;
    tokens_ = std::exchange(other.tokens_, {});
    blocks_ = std::exchange(other.blocks_, {});
  }
  return *this;
}

Sequence::~Sequence() {
  release();
}

Result<AppendResult> Sequence::append(std::span<const TokenId> tokens) {
  AppendResult result;
  if (tokens.empty()) {
    return result;
  }
  const std::size_t block_size = allocator_->block_size();
  const std::size_t used_in_last = tokens_.size() % block_size;
  const bool partial_last = used_in_last != 0;
  // Writing into a block that another holder can see would corrupt it for them: copy first.
  const bool needs_copy = partial_last && allocator_->ref_count(blocks_.back()) > 1;
  const std::size_t room_in_last = partial_last ? block_size - used_in_last : 0;
  const std::size_t overflow = tokens.size() > room_in_last ? tokens.size() - room_in_last : 0;
  const std::size_t new_blocks = (overflow + block_size - 1) / block_size;
  const std::size_t needed = new_blocks + (needs_copy ? 1U : 0U);
  if (needed > allocator_->num_free()) {
    return fail(Errc::out_of_blocks);
  }

  // The only steps that can throw; nothing has changed yet.
  tokens_.reserve(tokens_.size() + tokens.size());
  blocks_.reserve(blocks_.size() + new_blocks);

  // From here on nothing fails: `needed` blocks are free and capacity is reserved.
  if (needs_copy) {
    const BlockId src = blocks_.back();
    const Result<BlockId> dst = allocator_->allocate();
    detail::expect_ok(dst);
    detail::expect_ok(allocator_->release(src));  // the other holders keep it alive
    blocks_.back() = *dst;
    result.copy = CopyOp{.src = src, .dst = *dst};
  }
  detail::expect_ok(allocator_->allocate(new_blocks, blocks_));
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  result.new_blocks = static_cast<std::uint32_t>(new_blocks);
  return result;
}

Result<AppendResult> Sequence::append(TokenId token) {
  return append(std::span<const TokenId>(&token, 1));
}

Result<Sequence> Sequence::fork() const {
  Sequence child(*allocator_);
  child.tokens_ = tokens_;
  child.blocks_.reserve(blocks_.size());
  for (const BlockId block : blocks_) {
    if (const Result<void> retained = allocator_->retain(block); !retained) {
      return fail(retained.error());  // `child` releases the references taken so far
    }
    child.blocks_.push_back(block);
  }
  return child;
}

Result<void> Sequence::truncate(std::size_t num_tokens) noexcept {
  if (num_tokens > tokens_.size()) {
    return fail(Errc::invalid_argument);
  }
  const std::size_t keep = allocator_->blocks_for(num_tokens);
  for (std::size_t i = keep; i < blocks_.size(); ++i) {
    detail::expect_ok(allocator_->release(blocks_[i]));
  }
  blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(keep), blocks_.end());
  tokens_.erase(tokens_.begin() + static_cast<std::ptrdiff_t>(num_tokens), tokens_.end());
  return {};
}

void Sequence::release() noexcept {
  for (const BlockId block : blocks_) {
    detail::expect_ok(allocator_->release(block));
  }
  blocks_.clear();
  tokens_.clear();
}

Result<void> Sequence::assign_prefix(std::span<const TokenId> tokens,
                                     std::span<const BlockId> blocks) {
  const std::size_t block_size = allocator_->block_size();
  if (!tokens_.empty() || tokens.size() % block_size != 0 ||
      tokens.size() / block_size != blocks.size()) {
    return fail(Errc::invalid_argument);
  }
  tokens_.reserve(tokens.size());
  blocks_.reserve(blocks.size());
  for (const BlockId block : blocks) {
    if (const Result<void> retained = allocator_->retain(block); !retained) {
      release();
      return fail(retained.error());
    }
    blocks_.push_back(block);
  }
  tokens_.assign(tokens.begin(), tokens.end());
  return {};
}

}  // namespace radix_kv
