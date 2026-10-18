// Minimal consumer: exits 0 when the package works end to end.

#include <iostream>
#include <string>
#include <vector>

#include "radix_kv/radix_kv.hpp"

// The installed version header must be usable in preprocessor conditionals.
#if RADIX_KV_VERSION_MAJOR != 0 || RADIX_KV_VERSION_MINOR < 1
#error "this consumer expects radix-kv 0.1 or a later 0.x release"
#endif

int main() {
  const std::string macro_version = std::to_string(RADIX_KV_VERSION_MAJOR) + "." +
                                    std::to_string(RADIX_KV_VERSION_MINOR) + "." +
                                    std::to_string(RADIX_KV_VERSION_PATCH);
  if (macro_version != radix_kv::kVersion) {
    std::cout << "version macros (" << macro_version << ") disagree with kVersion\n";
    return 1;
  }

  radix_kv::BlockAllocator pool(16, 4);
  radix_kv::RadixPrefixCache cache(pool);
  radix_kv::Sequence sequence(pool);
  const std::vector<radix_kv::TokenId> tokens{1, 2, 3, 4, 5};
  if (!sequence.append(tokens) || !cache.insert(sequence.tokens(), sequence.block_table())) {
    return 1;
  }
  const bool hit = cache.match_prefix(tokens).num_tokens == 4;
  std::cout << "radix-kv " << radix_kv::kVersion << (hit ? ": ok\n" : ": unexpected miss\n");
  return hit ? 0 : 1;
}
