# ADR 0004: One LRU order for both caches; the radix tree evicts whole leaves

- Status: accepted
- Date: 2026-10-18

## Context

Both caches must evict least-recently-used entries that no running request uses, and must never
evict an entry that still has cached descendants: for the hash cache such descendants would become
unreachable (a lookup stops at the first missing block), for the radix tree they would be detached.
To compare the two data structures fairly, both should use the same notion of recency.

SGLang's radix cache evicts unlocked leaves by last-access time, collecting them into a heap when
eviction runs. vLLM frees a finished request's blocks tail first, so the free queue tends to hold
descendants ahead of their ancestors.

## Options considered

1. **Timestamps plus an ordered set or heap of evictable leaves.** Exact, but O(log n) per update,
   an allocation per entry, and a heap rebuild if done SGLang's way.
2. **Intrusive list of evictable leaves only.** O(1), but when a node becomes a leaf (its last child
   was evicted) it would have to be inserted in the middle of the list at its true recency.
3. **Intrusive list of every unlocked node, updated deepest node first.** Whenever a path is used
   (match, insert, unlock), its unlocked suffix is moved to the most-recently-used end starting
   from the deepest node, so every node stays in front of its parent.

## Decision

Option 3, implemented once in `detail::LruList` and used by both caches. A node is in the list
exactly when it is unlocked. Because children always precede their parents, the front of the list
is always a leaf, and eviction is O(1) per victim without allocating. Splitting a radix node inserts
the new upper half right after the lower half, which preserves the order. `check_invariants()`
verifies the ordering, and the property tests and the fuzzer call it after every operation.

The hash cache evicts exactly one block per victim. The radix tree evicts a whole leaf, as SGLang
does, so `evict(n)` may release more than `n` blocks.

## Consequences

- O(1) eviction per victim, no allocation on the eviction path, and no orphaned descendants.
- Both strategies see the same recency, so hit-rate differences under memory pressure come from
  eviction granularity alone. In the synthetic simulations the radix tree evicts more blocks and
  hits slightly less when the pool is small, and the two agree exactly when nothing is evicted.
- Splitting a long leaf to evict only its tail would remove the over-eviction; it is on the
  roadmap rather than the default because it changes SGLang's behaviour.
