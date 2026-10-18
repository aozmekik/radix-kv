# radix-kv

The memory-management core of an LLM inference server's KV cache in C++20, with Python bindings:
a PagedAttention-style block allocator with copy-on-write, and two prefix caches (vLLM-style
chained block hashes and an SGLang-style radix tree) that you can compare in a trace-driven
scheduler simulation.

[![CI](https://github.com/aozmekik/radix-kv/actions/workflows/ci.yml/badge.svg)](https://github.com/aozmekik/radix-kv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Status:** early, v0.1.0. The API may change. Not yet published to PyPI or any package
> registry. There is no GPU code and there are no tensors: the library manages block ids and
> metadata, the way an engine's scheduler does.

## Why

An LLM server keeps the attention keys and values (the KV cache) of every running request in
accelerator memory. PagedAttention ([Kwon et al., SOSP 2023](#references)) stores that cache in
fixed-size blocks, so memory does not fragment and sequences can share blocks. Prefix caching goes
further and reuses the blocks of a shared prompt prefix (a system prompt, few-shot examples, an
earlier chat turn) across requests. vLLM finds reusable blocks through chained hashes of full
blocks; SGLang ([Zheng et al., NeurIPS 2024](#references)) keeps a radix tree of token sequences.

radix-kv implements both strategies on the same allocator and the same ownership model, with the
invariants spelled out and tested, so the two can be studied and compared without a GPU, and the
pieces can be reused in a scheduler.

## Features

- **`BlockAllocator`**: fixed pool of `N` blocks of `B` tokens, O(1) allocate, retain and release,
  reference counts, an all-or-nothing batch allocate and peak tracking.
- **`Sequence`**: tokens plus block table with RAII release; `fork()` shares every block, and an
  append into a shared, partially filled block copies it first and reports a `CopyOp` (src, dst)
  for the engine (parallel sampling, beam search). `truncate()` and `assign_prefix()` included.
- **`HashPrefixCache`**: chained 64-bit block hashes, a hash-to-block index and LRU eviction of
  unlocked blocks.
- **`RadixPrefixCache`**: radix tree over whole blocks with `match_prefix`, `insert`,
  `lock`/`unlock` for running requests, node splitting and LRU eviction of unlocked leaves.
- **No false hits**: every hash hit is confirmed against the stored token ids.
- **Errors without exceptions** on hot paths: `Result<T>` is `std::expected` in C++23 and a
  compatible fallback in C++20.
- **Simulator**: deterministic synthetic workloads (multi-turn chat, few-shot, RAG, uniform
  random) and a continuous-batching scheduler that reports hit rate, evictions, preemptions, peak
  blocks, copy-on-write copies and cache operations per second for each strategy.
- **Python bindings** (nanobind, scikit-build-core) with type stubs; `simulate()` releases the GIL.
- **Tests**: GoogleTest unit tests, randomized invariant and differential tests with fixed seeds,
  a libFuzzer target, sanitizer presets, Google Benchmark microbenchmarks.

## Install

### Python

Not yet published to PyPI. Install from GitHub (this compiles the C++ extension, so a C++20
compiler is needed; pip fetches CMake and Ninja if they are missing):

```sh
pip install "git+https://github.com/aozmekik/radix-kv"
```

The CI workflow builds and tests the package on Python 3.11 to 3.14, on Linux and macOS.

### C++ with CMake

```cmake
include(FetchContent)
FetchContent_Declare(
  radix_kv
  GIT_REPOSITORY https://github.com/aozmekik/radix-kv
  GIT_TAG main)  # or a commit SHA
FetchContent_MakeAvailable(radix_kv)

target_link_libraries(your_target PRIVATE radix_kv::radix_kv)
```

As a subproject it builds only the library: no tests, tools or `-Werror`. To install it instead:

```sh
cmake --preset release && cmake --build --preset release
cmake --install build/release --prefix /your/prefix   # then find_package(radix_kv CONFIG REQUIRED)
```

The install also provides the `radix-kv-sim` command-line tool.

## Quick start

C++ ([full example](examples/cpp/quickstart.cpp)):

```cpp
#include "radix_kv/radix_kv.hpp"

radix_kv::BlockAllocator pool(/*num_blocks=*/1024, /*block_size=*/16);
radix_kv::RadixPrefixCache cache(pool);

// A new request: reuse the cached prefix (the last prompt token is always recomputed).
const radix_kv::PrefixMatch hit = cache.match_prefix(prompt.first(prompt.size() - 1));
if (!cache.lock(hit.node)) { /* stale handle */ }
radix_kv::Sequence seq(pool);
if (!seq.assign_prefix(prompt.first(hit.num_tokens), hit.blocks)) { /* ... */ }
if (auto appended = seq.append(prompt.subspan(hit.num_tokens)); !appended) {
  cache.evict(/*num_blocks=*/64);  // appended.error() == radix_kv::Errc::out_of_blocks
}

// Parallel sampling: forks share every block until one of them writes.
auto sample = seq.fork();
if (auto step = sample->append(next_token); step && step->copy) {
  // The engine copies the KV data of block step->copy->src into step->copy->dst.
}

// Done: cache the full blocks, release the lock and the sequence's references.
static_cast<void>(cache.insert(seq.tokens(), seq.block_table()));
static_cast<void>(cache.unlock(hit.node));
```

Python ([full example](examples/python/quickstart.py)):

```python
import radix_kv as rk

pool = rk.BlockAllocator(num_blocks=1024, block_size=16)
cache = rk.RadixPrefixCache(pool)

hit = cache.match_prefix(prompt[:-1])
cache.lock(hit.node)
seq = rk.Sequence(pool)
seq.assign_prefix(prompt[: hit.num_tokens], hit.blocks)
seq.extend(prompt[hit.num_tokens :])  # raises rk.OutOfBlocksError when full

trace = rk.generate_trace(rk.ChatWorkload(sessions=100), seed=1)
for strategy in (rk.Strategy.none, rk.Strategy.hash, rk.Strategy.radix):
    report = rk.simulate(trace, strategy, rk.SimulatorConfig(num_blocks=4096))
    print(strategy.name, f"{report.hit_rate:.1%}", report.evicted_blocks)
```

Command line:

```sh
radix-kv-sim --workload chat --blocks 1024,4096 --strategy all      # markdown table
radix-kv-sim --workload rag --samples 4 --format csv                 # parallel sampling, CSV
```

## API overview

| Type or function | Role |
|---|---|
| `BlockAllocator` | Pool of reference-counted blocks: `allocate`, `retain`, `release`, `ref_count`, `num_free` |
| `Sequence` | Tokens and block table: `append`, `fork`, `truncate`, `release`, `assign_prefix` |
| `HashPrefixCache`, `RadixPrefixCache` | `match_prefix`, `insert`, `lock`, `unlock`, `evict`, `stats`, `check_invariants` |
| `NullPrefixCache` | Caches nothing; the no-caching baseline |
| `PrefixCache` (concept) | The interface shared by the three caches |
| `Result<T>`, `Errc` | Value-or-error results; `to_string(Errc)` and `describe(Errc)` |
| `generate_trace`, `Trace` | Synthetic workloads, or your own token streams |
| `simulate`, `SimulatorConfig`, `SimulationReport` | Scheduler simulation over a trace |

Every public item has a doc comment in [include/radix_kv](include/radix_kv); the Python API mirrors
it (see the shipped `_core.pyi` stubs).

## How it works

### Ownership: reference counts plus path locks

Every holder of a block, a sequence or a cache, owns exactly one allocator reference, so a block is
free exactly when nobody holds it. Caches additionally count locks per path: running requests lock
the cached prefix they use, and eviction only takes unlocked entries. A forgotten lock can make an
eviction release nothing, but it can never free a block that is in use
([ADR 0001](docs/adr/0001-reference-counts-own-blocks-locks-pin-paths.md)).

This is the request lifecycle the simulator drives (and a scheduler would):

```mermaid
flowchart LR
  A[match_prefix<br/>prompt minus last token] --> B[lock matched path]
  B --> C[assign_prefix<br/>+ append rest of prompt]
  C -->|out_of_blocks| E[evict, else wait]
  C --> D[insert prompt<br/>move lock to it]
  D --> F[decode: append one token<br/>per sample, copy-on-write]
  F -->|out_of_blocks| G[evict, else preempt<br/>newest request]
  F --> H[finish: insert, unlock,<br/>release sequences]
```

### Two strategies, one trie

Chained hashes make each cached block a node whose parent is the previous block: the hash cache is
an uncompressed trie with one node per block and a hash index over it. The radix tree is the same
trie with chains of single children compressed into one edge. For two cached prompts sharing two
system-prompt blocks:

```mermaid
flowchart TB
  subgraph hash["HashPrefixCache: one node per block"]
    h0((root)) --> h1[S1] --> h2[S2]
    h2 --> h3[X]
    h2 --> h4[Y]
  end
  subgraph radix["RadixPrefixCache: one node per run of blocks"]
    r0((root)) --> r1["S1 S2"]
    r1 --> r2[X]
    r1 --> r3[Y]
  end
```

Both cache exactly the same block-aligned prefixes, so without eviction they return identical hits;
the property tests check this against a naive reference model after every operation. They differ in
cost (the hash cache hashes every block of a query; the tree compares tokens along a few edges) and
in eviction granularity (one block versus a whole leaf).

### Design notes and trade-offs

- **Hits are verified.** Each cache stores the tokens of its blocks and compares them before
  accepting a hash hit, so a collision costs a miss, never wrong KV data
  ([ADR 0002](docs/adr/0002-confirm-every-hash-hit.md)).
- **`std::expected` with a C++20 fallback**; no exceptions on hot paths
  ([ADR 0003](docs/adr/0003-result-type-with-std-expected-fallback.md)).
- **One LRU order for both caches.** Paths are marked used deepest node first, so children always
  precede parents and the eviction victim is always a leaf; the radix tree evicts whole leaves as
  SGLang does ([ADR 0004](docs/adr/0004-shared-lru-order-and-whole-leaf-eviction.md)).
- **Strong guarantee.** Mutations either succeed or change nothing (`insert` validates every block
  before touching the cache); handles carry a generation, so a stale handle is rejected instead of
  aliasing a newer node.
- **Scheduler policy (simulator).** First-come first-served admission, one decode token per running
  sample per step, prompt cached right after prefill so later requests can share it, recompute
  preemption of the most recently admitted request, and the last prompt token always recomputed
  because the engine needs its logits.
- **Deterministic traces.** The generator uses its own xoshiro256** PRNG and integer sampling, not
  the implementation-defined `<random>` distributions; a golden-fingerprint test pins the output of
  an integer-only trace on every platform.

## Simulation results (synthetic traces)

These are results of the simulator on **synthetic traces** with uniformly random token ids and
illustrative length distributions, not measurements of a real model or real traffic.

```sh
cmake --workflow --preset release
./build/release/tools/radix-kv-sim --blocks 1024,4096,16384
```

Workload defaults as in [workload.hpp](include/radix_kv/workload.hpp) (200 chat sessions with a
shared 1024-token system prompt; 1000 few-shot, RAG and random requests), seed 1, 16-token blocks,
at most 32 running requests. The hit rate counts first admissions only; recomputation after
preemption is reported separately. Hit rates, evictions and preemptions do not depend on timing
and repeat exactly on the same platform; the default workloads draw log-normal lengths and Zipf
popularity through `<cmath>`, so another C library may produce slightly different traces. Run on
an Apple M1 Pro (8 cores), 16 GB, macOS 26, Apple Clang 21.0.0, Release build.

| Workload | Pool (blocks) | Hit rate hash | Hit rate radix | Evicted blocks hash | Evicted blocks radix | Preemptions |
|---|--:|--:|--:|--:|--:|--:|
| chat | 1024 | 65.6% | 65.6% | 33,281 | 35,960 | 162 |
| chat | 4096 | 68.0% | 68.0% | 27,485 | 27,567 | 0 |
| chat | 16384 | 93.5% | 93.5% | 0 | 0 | 0 |
| few-shot | 1024 | 89.0% | 86.2% | 14,534 | 18,985 | 207 |
| few-shot | 4096 | 94.5% | 94.5% | 4,847 | 4,849 | 0 |
| few-shot | 16384 | 94.5% | 94.5% | 0 | 0 | 0 |
| rag | 1024 | 22.1% | 21.8% | 110,202 | 116,395 | 118 |
| rag | 4096 | 30.6% | 30.4% | 95,463 | 95,748 | 0 |
| rag | 16384 | 39.7% | 39.7% | 70,930 | 70,955 | 0 |
| random | 1024 to 16384 | 0.0% | 0.0% | 58,338 to 74,574 | 58,343 to 80,732 | 0 to 186 |

Blocks held by running requests at their peak, at 16384 blocks (`peak active`, identical for both
caching strategies): chat 3,130 versus 5,114 without prefix caching, few-shot 1,198 versus 4,031,
RAG 3,853 versus 4,766, random 3,023 versus 3,023.

What the numbers show:

- With room to spare the two strategies agree exactly, as designed (and as the tests require).
- Under memory pressure the radix tree evicts more blocks and loses a little hit rate, because it
  evicts whole leaves where the hash cache evicts single blocks.
- RAG reuse is bounded by document order: only documents identical from the start of the prompt
  are a shared prefix. Prefix caching cannot reuse a popular document retrieved second.
- Sharing prefix blocks among running requests shrinks their working set by up to 70% here
  (few-shot), on top of the compute saved by the hits.

Cache cost at 4096 blocks, median of three runs of
`./build/release/tools/radix-kv-sim --blocks 4096 --format csv` on the same machine. Every call is
timed with `steady_clock`, so timer overhead is included, and the machine was heavily loaded by
other jobs (1-minute load average 22 to 27), so treat these as rough:

| Workload | Cache calls hash | Cache calls radix | Time in cache hash | Time in cache radix | Calls/s hash | Calls/s radix |
|---|--:|--:|--:|--:|--:|--:|
| chat (767 requests) | 11,349 | 6,144 | 11.2 ms | 2.4 ms | 1.01M | 2.52M |
| few-shot (1000) | 9,665 | 7,978 | 14.5 ms | 2.7 ms | 0.67M | 3.01M |
| rag (1000) | 14,967 | 7,966 | 21.2 ms | 3.6 ms | 0.71M | 2.19M |
| random (1000) | 16,230 | 7,886 | 12.4 ms | 2.1 ms | 1.31M | 3.79M |

The two strategies do not receive the same calls: the hash cache evicts exactly the number of
blocks asked for, so the scheduler calls `evict()` again for nearly every new block once the pool
is full, while a whole-leaf eviction in the radix tree frees room for many steps. Time in cache is
the fairer comparison. `HashPrefixCache` also recomputes the chained hashes of a prompt on every
call, while an engine such as vLLM computes each request's block hashes once and reuses them, so
this comparison overstates the hash strategy's CPU cost.

## Benchmarks

```sh
cmake --workflow --preset release
./build/release/bench/radix_kv_bench --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true --benchmark_min_time=0.2s
```

Apple M1 Pro (8 cores), 16 GB, macOS 26, Apple Clang 21.0.0, Release (`-O3`), Google Benchmark
1.10.0. Wall-time medians of 5 repetitions. Other jobs were running on the machine (1-minute load
average about 27 before and after the run), so treat the numbers as indicative: the coefficient of variation was
below 6% for every row except the radix-tree simulation (12%). All inputs are synthetic.

| Benchmark | `HashPrefixCache` | `RadixPrefixCache` |
|---|--:|--:|
| Match a cached 256-token prefix (full hit) | 568 ns | 129 ns |
| Match a cached 2,048-token prefix | 4,411 ns | 598 ns |
| Match a cached 16,384-token prefix | 37,161 ns | 4,404 ns |
| Match one of 1,024 prompts sharing a 1,024-token prefix | 2,628 ns | 468 ns |
| Fork a cached 512-token prefix, append 256 tokens, insert, evict them | 2,440 ns | 833 ns |
| The same with 2,048 appended tokens | 12,602 ns | 2,182 ns |
| Simulate 200 synthetic few-shot requests (2,048 blocks) | 8.79 ms | 6.27 ms |

| Benchmark | Median |
|---|--:|
| `BlockAllocator` allocate + release | 4.76 ns |
| `Sequence::append` of one token (decode step) | 10.4 ns |
| `fork()` of a 1,000-token sequence + copy-on-write append | 453 ns |
| Simulate the same 200 requests without prefix caching | 4.80 ms |

The radix tree's lookups are faster because they compare token ids along a few edges, while the
hash cache computes one chained hash and one hash-map lookup per block of the query.

## Testing

- `ctest --preset dev` runs 130 tests: 124 GoogleTest cases plus CLI and example smoke tests.
  `pytest` runs 32 Python tests (Python 3.14.5, pytest 9.1.1).
- Randomized invariant tests (fixed seeds, 8 configurations including tiny pools and a deliberately
  colliding hasher) check after every operation that reference counts equal the holders, that
  locked nodes are never evicted, that matched blocks hold the queried tokens, that no holder sees a
  block change underneath it, that the tree is structurally valid after every split, insert and
  eviction, and that every block is free once all sequences end. Differential tests check that
  both caches return the same hits as a naive reference model.
- `cmake --workflow --preset asan-ubsan` and `--preset tsan` pass (Apple Clang 21.0.0, macOS 26,
  Apple M1 Pro). The ThreadSanitizer run includes a test that runs independent simulations in
  parallel threads; pytest does the same through the bindings, with the GIL released.
- libFuzzer, same driver: `fuzz_prefix_cache -max_total_time=120 -max_len=4096` ran 215,968
  inputs in 121 s without a failure (Homebrew clang 23.1.3, AddressSanitizer and UBSan enabled,
  Apple M1 Pro under load).
- `./scripts/coverage.sh`: 92.9% of lines, 91.1% of regions and 84.1% of branches of `src/` and
  `include/` are covered by the C++ suite (Apple Clang 21.0.0 source-based coverage).
- `./scripts/lint.sh` runs clang-format and clang-tidy (strict configuration, warnings as errors)
  over every C++ file; ruff and `mypy --strict` check the Python code.

## Limitations

- A simulation, not an engine: no kernels and no timing model for prefill or decode; a step is one
  token per running sample. All quoted hit rates come from synthetic traces.
- The data structures are not thread-safe, like a scheduler loop; independent instances can be
  used from different threads.
- The radix tree evicts whole leaves and does not re-merge single-child nodes after eviction.
- Only full blocks are cached, one block per distinct prefix; a duplicate computed by another
  request is not cached.
- Each cache keeps a token store of `num_blocks * block_size * 4` bytes.
- The Python API copies token lists (no zero-copy NumPy input). No prebuilt wheels; Windows is not
  tested.

## Roadmap

- Accept precomputed per-request block hashes in `HashPrefixCache`, as vLLM does.
- Evict only the tail of a long radix leaf (split, then evict) as an option.
- Simulator: a prefill/decode cost model, swap-based preemption, cache-aware scheduling (for
  example longest prefix first), and a trace file format for replaying real tokenized logs.
- Zero-copy NumPy inputs and published wheels.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the build, test, lint and benchmark commands. Design
decisions are recorded in [docs/adr](docs/adr).

## References

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
  Gonzalez, Hao Zhang, Ion Stoica. "Efficient Memory Management for Large Language Model Serving
  with PagedAttention." SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao,
  Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng. "SGLang: Efficient
  Execution of Structured Language Model Programs." NeurIPS 2024.
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- vLLM documentation on automatic prefix caching, and the SGLang radix cache, for the behaviour
  modelled here. This is an independent implementation from the papers and public documentation;
  no code was copied.
- MurmurHash3 by Austin Appleby (public domain) for the structure of the block hash; xoshiro256**
  by David Blackman and Sebastiano Vigna for the trace generator.

## License

[MIT](LICENSE)

Built by [Ahmed Semih Özmekik](https://github.com/aozmekik) at [Fergani](https://fergani.tr).
