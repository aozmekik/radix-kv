#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "radix_kv/prefix_cache.hpp"
#include "radix_kv/types.hpp"

namespace radix_kv {

namespace {

// Constants and structure follow MurmurHash3's x64 body and finalizer (Austin Appleby, public
// domain), adapted to mix two 32-bit token ids per 64-bit lane.
constexpr std::uint64_t kC1 = 0x87c37b91114253d5ULL;
constexpr std::uint64_t kC2 = 0x4cf5ad432745937fULL;
constexpr std::uint64_t kLengthMul = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kRoundAdd = 0x52dce729ULL;
constexpr std::uint64_t kRoundMul = 5;
constexpr int kLaneRot = 31;
constexpr int kStateRot = 27;

constexpr std::uint64_t scramble(std::uint64_t lane) noexcept {
  return std::rotl(lane * kC1, kLaneRot) * kC2;
}

constexpr std::uint64_t finalize(std::uint64_t h) noexcept {
  h ^= h >> 33U;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33U;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33U;
  return h;
}

}  // namespace

std::uint64_t hash_tokens(std::uint64_t seed, std::span<const TokenId> tokens) noexcept {
  std::uint64_t h = seed ^ (static_cast<std::uint64_t>(tokens.size()) * kLengthMul);
  const std::size_t pairs = tokens.size() / 2;
  for (std::size_t i = 0; i < pairs; ++i) {
    const std::uint64_t lane =
        (static_cast<std::uint64_t>(tokens[2 * i]) << 32U) | tokens[(2 * i) + 1];
    h ^= scramble(lane);
    h = (std::rotl(h, kStateRot) * kRoundMul) + kRoundAdd;
  }
  if (tokens.size() % 2 != 0) {
    h ^= scramble(tokens.back());
  }
  return finalize(h);
}

}  // namespace radix_kv
