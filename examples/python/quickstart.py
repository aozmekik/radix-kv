"""Two requests sharing a system prompt through a radix prefix cache, plus a copy-on-write fork."""

import radix_kv as rk


def main() -> None:
    pool = rk.BlockAllocator(num_blocks=64, block_size=4)
    cache = rk.RadixPrefixCache(pool)
    system_prompt = [101, 102, 103, 104, 105, 106, 107, 108]

    # Request 1: prefill and cache its full blocks.
    first = rk.Sequence(pool)
    first.extend([*system_prompt, 1, 2, 3])
    print(f"request 1 cached {cache.insert(first).new_blocks} blocks")

    # Request 2 reuses the shared prefix; the last prompt token is always recomputed.
    prompt = [*system_prompt, 7, 7, 7, 7, 7]
    hit = cache.match_prefix(prompt[:-1])
    cache.lock(hit.node)  # pin the shared blocks while the request runs
    second = rk.Sequence(pool)
    second.assign_prefix(prompt[: hit.num_tokens], hit.blocks)
    second.extend(prompt[hit.num_tokens :])
    print(f"request 2 reused {hit.num_tokens} of {len(prompt)} prompt tokens")

    # Parallel sampling: the fork shares every block until it writes to the shared last block.
    sample = second.fork()
    decoded = sample.append(42)
    if decoded.copy is not None:
        print(f"copy-on-write: copy block {decoded.copy.src} into block {decoded.copy.dst}")

    cache.unlock(hit.node)
    print(f"pool: {pool.num_used} used, {pool.num_free} free")


if __name__ == "__main__":
    main()
