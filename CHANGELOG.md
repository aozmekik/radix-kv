# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0 any minor release may
change the API.

## [Unreleased]

## [0.1.0] - 2026-10-18

### Added

- `BlockAllocator`: fixed pool of reference-counted KV-cache blocks with O(1) allocate, retain
  and release, an all-or-nothing batch allocate and peak tracking.
- `Sequence`: token ids plus block table with RAII release, `fork()`, copy-on-write of a shared
  partial last block (reported as a `CopyOp`), `truncate()` and `assign_prefix()`.
- `HashPrefixCache`: vLLM-style prefix caching with chained block hashes, a hash index and LRU
  eviction of unlocked blocks.
- `RadixPrefixCache`: SGLang-style radix tree over blocks with node splitting, path locks and LRU
  eviction of whole unlocked leaves.
- `insert` on both caches is all-or-nothing, also when an allocation throws `std::bad_alloc`.
- `NullPrefixCache` baseline and the `PrefixCache` concept shared by all three, including
  `allocator()`.
- `Result<T>` (`std::expected` in C++23, a fallback in C++20) with a link-time check that the
  library and the code using it were compiled in the same mode.
- Deterministic synthetic workloads (multi-turn chat, few-shot, RAG, random) and `simulate()`, a
  continuous-batching scheduler simulation reporting hit rate, evictions, preemptions, rejected
  requests, peak blocks, copy-on-write copies and time spent in the cache.
- `radix-kv-sim` command-line tool (markdown or CSV), C++ and Python examples.
- Python bindings (nanobind, scikit-build-core) with generated type stubs and readable reprs;
  `cache.insert(sequence)` rejects a sequence that uses another allocator.
- GoogleTest unit, randomized property and differential tests; an allocation-failure suite; a
  libFuzzer target, also replayed on fixed inputs in every build; Google Benchmark suite; CMake
  presets for sanitizers, coverage, fuzzing and C++23.

[Unreleased]: https://github.com/aozmekik/radix-kv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aozmekik/radix-kv/releases/tag/v0.1.0
