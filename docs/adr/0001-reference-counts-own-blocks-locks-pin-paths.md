# ADR 0001: Reference counts own blocks, cache locks only pin paths

- Status: accepted
- Date: 2026-10-18

## Context

A KV-cache block can be held by several parties at once: the sequence that computed it, forks of
that sequence (parallel sampling, beam search) and a prefix cache that keeps it for future requests.
The two systems this library models answer "who owns a block?" differently:

- vLLM's block pool (v1 engine) keeps a reference count per block. A cached block whose count
  drops to zero stays in the free queue with its hash, so allocation and eviction are the same
  operation.
- SGLang's radix tree owns the KV indices of everything it caches. Running requests do not hold
  references; they increment a `lock_ref` on their tree path so that eviction skips it.

Both strategies have to run on one allocator, with the same `Sequence` type, behind one interface,
so that the simulator compares data structures rather than ownership models. Copy-on-write also
needs to know whether a block is shared, whoever shares it.

## Options considered

1. **Allocator-integrated cache (vLLM style).** The allocator knows block hashes and treats idle
   cached blocks as free. Allocation can evict transparently, but the allocator becomes specific to
   the hash strategy and the radix tree cannot reuse it.
2. **Cache owns cached blocks (SGLang style).** Sequences borrow blocks from the tree under a lock.
   Forking then needs per-request bookkeeping of which blocks are borrowed and which are private,
   and a missing lock lets eviction free a block that is still in use.
3. **One reference per holder, plus path locks.** Every holder, sequence or cache, owns exactly one
   allocator reference. A cache takes a reference when it starts holding a block and drops it on
   eviction. Locks are a cache-level pin: locked paths are never evicted.

## Decision

Option 3. The allocator is a plain reference-counted pool (`BlockAllocator`). `Sequence` takes a
reference for every block in its table (including blocks adopted from a cache through
`assign_prefix`). Both caches take one reference per cached block and count locks per path, with a
separate count of locks taken through each handle so that `unlock` cannot release a lock it never
took.

## Consequences

- Memory safety does not depend on lock discipline: a forgotten lock can make an eviction release
  nothing (the sequence still holds the block), but it can never free a block that is in use.
- Copy-on-write is a single question, `ref_count(last_block) > 1`, no matter whether the other
  holder is a fork or a cache.
- The allocator stays small and strategy-agnostic, and both caches satisfy one `PrefixCache`
  concept, so the simulator runs identical scheduler code over them.
- Idle cached blocks are not "free": a scheduler must call `evict()` when allocation fails. The
  real headroom is `num_free() + stats().evictable_blocks`. vLLM folds this into allocation; here
  it is an explicit call.
- Each cached block costs one extra reference-count increment and decrement over its lifetime.
- Invariants are easy to state and test: every block's reference count equals the number of
  sequences holding it plus one if a cache holds it (checked after every step of the property
  tests and the fuzzer).
