"""Tests for the prefix cache bindings, run against both strategies."""

import pytest

import radix_kv as rk

CacheType = type[rk.HashPrefixCache] | type[rk.RadixPrefixCache]
CACHES: list[CacheType] = [rk.HashPrefixCache, rk.RadixPrefixCache]


def cached_sequence(
    pool: rk.BlockAllocator, cache: rk.HashPrefixCache | rk.RadixPrefixCache, tokens: list[int]
) -> rk.Sequence:
    seq = rk.Sequence(pool)
    seq.extend(tokens)
    cache.insert(seq)
    return seq


@pytest.mark.parametrize("cache_type", CACHES)
def test_insert_then_match(cache_type: CacheType) -> None:
    pool = rk.BlockAllocator(16, 2)
    cache = cache_type(pool)
    seq = cached_sequence(pool, cache, [1, 2, 3, 4, 5])
    match = cache.match_prefix([1, 2, 3, 4, 9])
    assert match.num_tokens == 4
    assert match.blocks == seq.block_table[:2]
    assert cache.contains(match.node)
    assert cache.holds_block(seq.block_table[0])
    stats = cache.stats
    assert (stats.cached_blocks, stats.locked_blocks, stats.evictable_blocks) == (2, 0, 2)
    assert cache.check_invariants() is None


@pytest.mark.parametrize("cache_type", CACHES)
def test_insert_accepts_tokens_and_blocks(cache_type: CacheType) -> None:
    pool = rk.BlockAllocator(16, 2)
    cache = cache_type(pool)
    seq = rk.Sequence(pool)
    seq.extend([7, 7, 8, 8])
    inserted = cache.insert(seq.tokens, seq.block_table)
    assert inserted.new_blocks == 2
    assert cache.insert(seq.tokens, seq.block_table).new_blocks == 0
    with pytest.raises(ValueError, match="invalid_argument"):
        cache.insert([1, 2, 3, 4], [seq.block_table[0]])  # too few blocks


@pytest.mark.parametrize("cache_type", CACHES)
def test_locks_protect_paths_from_eviction(cache_type: CacheType) -> None:
    pool = rk.BlockAllocator(16, 2)
    cache = cache_type(pool)
    cached_sequence(pool, cache, [1, 1, 2, 2]).release()
    cached_sequence(pool, cache, [3, 3, 4, 4]).release()
    hit = cache.match_prefix([1, 1, 2, 2])
    cache.lock(hit.node)
    assert cache.evict(100) == 2
    assert cache.match_prefix([1, 1, 2, 2]).num_tokens == 4
    cache.unlock(hit.node)
    with pytest.raises(rk.InvalidHandleError, match="not_locked"):
        cache.unlock(hit.node)
    assert cache.evict(100) == 2
    assert cache.stats.evicted_blocks == 4
    assert pool.num_free == pool.num_blocks


@pytest.mark.parametrize("cache_type", CACHES)
def test_stale_handles_are_rejected(cache_type: CacheType) -> None:
    pool = rk.BlockAllocator(16, 2)
    cache = cache_type(pool)
    seq = cached_sequence(pool, cache, [5, 6])
    node = cache.match_prefix([5, 6]).node
    seq.release()
    cache.evict(10)
    assert not cache.contains(node)
    with pytest.raises(rk.InvalidHandleError, match="invalid_handle"):
        cache.lock(node)
    assert cache.root == cache.root
    assert hash(cache.root) == hash(cache.root)


@pytest.mark.parametrize("cache_type", CACHES)
def test_insert_rejects_a_sequence_of_another_allocator(cache_type: CacheType) -> None:
    pool = rk.BlockAllocator(8, 2)
    other = rk.BlockAllocator(8, 2)
    cache = cache_type(pool)
    assert cache.allocator is pool
    owner = rk.Sequence(pool)
    owner.extend([9, 9, 9, 9])  # holds blocks 0 and 1 of pool
    stranger = rk.Sequence(other)
    stranger.extend([1, 2, 3, 4])  # holds blocks 0 and 1 of the other pool
    with pytest.raises(ValueError, match="different BlockAllocator"):
        cache.insert(stranger)
    assert cache.stats.cached_blocks == 0
    assert pool.ref_count(0) == pool.ref_count(1) == 1
    assert cache.match_prefix([1, 2, 3, 4]).num_tokens == 0


def test_result_types_have_readable_reprs() -> None:
    pool = rk.BlockAllocator(8, 2)
    cache = rk.RadixPrefixCache(pool)
    seq = cached_sequence(pool, cache, [1, 2, 3, 4])
    inserted = cache.insert(seq)
    assert repr(inserted) == f"InsertResult(new_blocks=0, node={inserted.node!r})"
    match = cache.match_prefix([1, 2, 3])
    block = seq.block_table[0]
    assert repr(match) == f"PrefixMatch(num_tokens=2, blocks=[{block}], node={match.node!r})"
    assert repr(cache.stats) == (
        "CacheStats(cached_blocks=2, locked_blocks=0, evictable_blocks=2, nodes=2, "
        "evicted_blocks=0)"
    )


def test_strategies_agree_on_block_aligned_prefixes() -> None:
    prompts = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 8], [1, 2, 9, 9], [5, 5, 5]]
    lengths: dict[str, list[int]] = {}
    for cache_type in CACHES:
        pool = rk.BlockAllocator(64, 2)
        cache = cache_type(pool)
        for prompt in prompts:
            cached_sequence(pool, cache, prompt)
        lengths[cache_type.__name__] = [
            cache.match_prefix(query).num_tokens
            for query in ([1, 2, 3, 4, 7], [1, 2, 9, 9, 9], [5, 5, 5, 5], [2, 2], [1, 2, 3])
        ]
    assert lengths["HashPrefixCache"] == lengths["RadixPrefixCache"] == [4, 4, 2, 0, 2]


def test_caches_keep_their_allocator_alive() -> None:
    def make_cache() -> rk.RadixPrefixCache:
        return rk.RadixPrefixCache(rk.BlockAllocator(4, 2))

    cache = make_cache()
    assert cache.match_prefix([1, 2]).num_tokens == 0
