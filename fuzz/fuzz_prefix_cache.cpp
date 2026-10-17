// libFuzzer target: drives both prefix caches through the same randomized operation sequences as
// the property tests (tests/support/op_driver.hpp) and checks every invariant after each step.
// Without eviction it also checks that both caches agree with the naive reference model.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "radix_kv/hash_prefix_cache.hpp"
#include "radix_kv/radix_prefix_cache.hpp"

#include "support/op_driver.hpp"

namespace {

using radix_kv::HashPrefixCache;
using radix_kv::RadixPrefixCache;
using radix_kv::test_support::DriverOptions;
using radix_kv::test_support::NaivePrefixCache;
using radix_kv::test_support::OpDriver;

[[noreturn]] void report(const char* cache, const std::string& problem) {
  std::fputs(cache, stderr);
  std::fputs(": ", stderr);
  std::fputs(problem.c_str(), stderr);
  std::fputs("\n", stderr);
  std::abort();
}

template <class Cache>
std::vector<std::size_t> run(const char* name, const DriverOptions& options,
                             std::span<const std::uint8_t> input) {
  OpDriver<Cache> driver(options);
  if (auto problem = driver.run(input)) {
    report(name, *problem);
  }
  return driver.match_lengths();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::span<const std::uint8_t> bytes(data, size);
  if (bytes.size() < 2) {
    return 0;
  }
  // The first two bytes pick the configuration, the rest are operations.
  const std::uint8_t shape = bytes[0];
  const std::uint8_t flags = bytes[1];
  const DriverOptions options{
      .num_blocks = 4U + (shape % 61U),
      .block_size = 1U + ((shape >> 6U) % 4U),
      .vocab_size = 2U + (flags % 6U),
      .max_sequences = 8,
      .allow_evict = (flags & 0x40U) == 0,
      .hasher =
          (flags & 0x80U) != 0 ? &radix_kv::test_support::weak_hasher : &radix_kv::hash_tokens,
  };
  const std::span<const std::uint8_t> ops = bytes.subspan(2);
  const auto hash = run<HashPrefixCache>("HashPrefixCache", options, ops);
  const auto radix = run<RadixPrefixCache>("RadixPrefixCache", options, ops);
  if (!options.allow_evict && options.hasher == &radix_kv::hash_tokens) {
    const auto naive = run<NaivePrefixCache>("NaivePrefixCache", options, ops);
    if (hash != naive || radix != naive) {
      report("differential", "caches disagree with the reference model on a match length");
    }
  }
  return 0;
}
