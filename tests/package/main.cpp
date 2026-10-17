// Minimal consumer: exits 0 when the package works end to end.

#include <iostream>
#include <vector>

#include "radix_kv/radix_kv.hpp"

int main() {
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
