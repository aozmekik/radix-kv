// A tour of the core API: a block pool, a radix prefix cache shared by two requests, and a forked
// sequence that triggers copy-on-write.

#include <exception>
#include <iostream>
#include <span>
#include <vector>

#include "radix_kv/radix_kv.hpp"

namespace {

using radix_kv::TokenId;

int fail_with(radix_kv::Errc error) {
  std::cerr << "error: " << radix_kv::describe(error) << '\n';
  return 1;
}

int run() {
  radix_kv::BlockAllocator pool(/*num_blocks=*/64, /*block_size=*/4);
  radix_kv::RadixPrefixCache cache(pool);
  const std::vector<TokenId> system_prompt{101, 102, 103, 104, 105, 106, 107, 108};

  // Request 1: prefill, then cache its full blocks.
  std::vector<TokenId> first_prompt = system_prompt;
  first_prompt.insert(first_prompt.end(), {1, 2, 3});
  radix_kv::Sequence first(pool);
  if (const auto appended = first.append(first_prompt); !appended) {
    return fail_with(appended.error());
  }
  const auto inserted = cache.insert(first.tokens(), first.block_table());
  if (!inserted) {
    return fail_with(inserted.error());
  }
  std::cout << "request 1: " << first.size() << " tokens in " << first.num_blocks() << " blocks, "
            << inserted->new_blocks << " blocks cached\n";

  // Request 2 starts with the same system prompt. The last prompt token is never taken from the
  // cache because the engine needs its logits.
  std::vector<TokenId> second_prompt = system_prompt;
  second_prompt.insert(second_prompt.end(), {7, 7, 7, 7, 7});
  const std::span<const TokenId> prompt(second_prompt);
  const radix_kv::PrefixMatch hit = cache.match_prefix(prompt.first(prompt.size() - 1));
  if (const auto locked = cache.lock(hit.node); !locked) {  // pin the shared blocks while in use
    return fail_with(locked.error());
  }
  radix_kv::Sequence second(pool);
  if (const auto adopted = second.assign_prefix(prompt.first(hit.num_tokens), hit.blocks);
      !adopted) {
    return fail_with(adopted.error());
  }
  if (const auto appended = second.append(prompt.subspan(hit.num_tokens)); !appended) {
    return fail_with(appended.error());
  }
  std::cout << "request 2: reused " << hit.num_tokens << " of " << prompt.size()
            << " prompt tokens from the cache\n";

  // Parallel sampling: the fork shares every block until it writes to the shared last block.
  auto sample = second.fork();
  if (!sample) {
    return fail_with(sample.error());
  }
  const auto decoded = sample->append(TokenId{42});
  if (!decoded) {
    return fail_with(decoded.error());
  }
  if (const auto& copy = decoded->copy) {
    std::cout << "copy-on-write: copy block " << copy->src << " into block " << copy->dst
              << " before the decode step\n";
  }

  if (const auto unlocked = cache.unlock(hit.node); !unlocked) {
    return fail_with(unlocked.error());
  }
  std::cout << "pool: " << pool.num_used() << " blocks used, " << pool.num_free()
            << " free; cache holds " << cache.stats().cached_blocks << " blocks\n";
  return 0;
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
