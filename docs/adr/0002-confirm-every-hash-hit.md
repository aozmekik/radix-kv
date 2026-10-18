# ADR 0002: Confirm every hash hit against the stored tokens

- Status: accepted
- Date: 2026-10-18

## Context

The hash strategy identifies block `i` of a prompt by a chained hash
`h_i = hash(h_{i-1}, tokens of block i)` and looks it up in a hash-to-block map. The radix tree
also uses a hash, of a child's first block, to find children without scanning them. If two
different prefixes produce the same 64-bit hash and the lookup trusts it, a request is served KV
data computed for other tokens: the engine silently produces wrong output.

With a good 64-bit hash an accidental collision is very unlikely, but not impossible, and a
non-cryptographic hash can be attacked: in a multi-tenant server, a crafted prompt could make
another tenant's request reuse the attacker's blocks.

## Options considered

1. **Trust a fast 64-bit hash.** Cheapest. Correctness is probabilistic and adversarially weak.
2. **Trust a cryptographic hash (for example SHA-256).** Collisions become infeasible, but every
   block of every prompt pays for a cryptographic hash on the scheduling path.
3. **Keep a fast hash as an index and confirm every hit.** Store the token ids of each cached block
   and compare them, together with the parent link, before accepting a hit.

## Decision

Option 3. `detail::BlockTokenStore` keeps the tokens of every cached block in one flat array
indexed by block id (a block is cached at most once, so `num_blocks * block_size` slots suffice).
A hash-cache hit must match the block's tokens and its parent entry; a radix child must match its
first block's tokens. When two prefixes really do collide, the second is simply not cached.

The hash function is injectable (`BlockHasher`), which lets the tests force collisions with a
16-valued hasher and check that matches stay exact.

## Consequences

- No false hits by construction, whatever the hash; a collision costs a cache miss.
- Hash flooding can at worst lower the hit rate. It cannot corrupt another request's output.
- Memory: 4 bytes per token slot of the pool, allocated up front (4096 blocks of 16 tokens is
  256 KiB). Both strategies pay the same, which keeps the comparison fair.
- CPU: one `block_size`-token comparison per hit block, small next to computing the hash itself.
- The default hasher (MurmurHash3-style mixing of two token ids per 64-bit lane) only has to be
  fast and well distributed, not collision-resistant.
