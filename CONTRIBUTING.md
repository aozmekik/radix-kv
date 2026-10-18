# Contributing

Thanks for taking a look. Issues and pull requests are welcome; for larger changes please open an
issue first so the design can be discussed (see the ADRs in [docs/adr](docs/adr)).

## Requirements

- CMake 3.25 or newer and Ninja
- A C++20 compiler. The CI workflow is configured for GCC 14 and Clang 18 on Ubuntu 24.04 and Apple
  Clang on macOS 15
- Python 3.11 to 3.14 for the bindings
- For linting: `pip install --group lint` installs the pinned clang-format and clang-tidy, plus ruff
  and mypy

## Build and test (C++)

```sh
cmake --workflow --preset dev          # Debug: configure, build, run every test
cmake --workflow --preset release      # Release, including a benchmark smoke run
cmake --workflow --preset cxx23        # C++23 mode (Result is std::expected)
cmake --workflow --preset asan-ubsan   # AddressSanitizer + UndefinedBehaviorSanitizer
cmake --workflow --preset tsan         # ThreadSanitizer
```

The presets write to `build/<preset>` and use 4 parallel jobs. Run a subset of the GoogleTest
suite with `ctest --preset dev -R Property` or `./build/dev/tests/radix_kv_tests --gtest_filter=...`.
Besides the main suite, `ctest` runs the allocation-failure suite (`radix_kv_fault_tests`, not built
under sanitizers because it replaces the global `operator new`), a replay of the fuzz harness on
fixed inputs, the package consumer, the command-line checks and the examples.

Code that includes `radix_kv/result.hpp` must be compiled in the same C++ mode as the library
(C++20 or C++23); a mix fails to link by design (see
[ADR 0003](docs/adr/0003-result-type-with-std-expected-fallback.md)).

## Python package

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install . --group dev   # builds the extension with scikit-build-core
.venv/bin/python -m pytest
.venv/bin/python -m mypy                        # strict, over tests/python and examples/python
```

## Lint and format

```sh
PATH="$PWD/.venv/bin:$PATH" RUN_CLANG_TIDY=run-clang-tidy.py PYTHON=.venv/bin/python ./scripts/lint.sh
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/clang-format -i $(git ls-files '*.cpp' '*.hpp' '*.hpp.in')   # apply the C++ style
```

`scripts/lint.sh` checks the formatting of every C++ file and runs clang-tidy, with warnings as
errors, over every translation unit of the build: library, tests (the fuzz harness is compiled
through its replay driver, and the package consumer in-tree), tools, examples, benchmarks and
bindings. Every disabled check is justified in [.clang-tidy](.clang-tidy) or in the `.clang-tidy`
of the directory it applies to. The configuration is clean with the pinned PyPI tools (clang-format
23.1.3, clang-tidy 22.1.8) and with a Homebrew LLVM 23 toolchain:

```sh
LLVM=/opt/homebrew/opt/llvm/bin
CLANG_FORMAT=$LLVM/clang-format CLANG_TIDY=$LLVM/clang-tidy RUN_CLANG_TIDY=$LLVM/run-clang-tidy \
  PYTHON=.venv/bin/python BUILD_DIR=build/lint-llvm ./scripts/lint.sh
```

## Benchmarks, fuzzing and coverage

```sh
cmake --workflow --preset release
./build/release/bench/radix_kv_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
./build/release/tools/radix-kv-sim --blocks 1024,4096,16384

CXX=/path/to/clang++ cmake --preset fuzz && cmake --build --preset fuzz   # needs libFuzzer
./build/fuzz/fuzz/fuzz_prefix_cache -max_total_time=60 -max_len=4096
./build/dev/tests/radix_kv_fuzz_replay crash-<hash>   # replay a crash input in a normal build

./scripts/coverage.sh   # Clang source-based coverage of src/ and include/
```

Apple Clang does not ship libFuzzer; Homebrew LLVM or Clang on Linux does. The fuzz preset keeps the
library's asserts on and makes UBSan abort on its first report, so undefined behaviour fails a run.

## Conventions

- C++: leading return types, `snake_case` functions and variables, `CamelCase` types,
  `trailing_underscore_` members, `kConstants`. Public items carry `///` doc comments.
- Fallible operations return `Result<T>`; nothing on a hot path throws.
- Every behavioural change comes with tests; invariant-heavy code should also be covered by the
  randomized driver in `tests/support/op_driver.hpp`.
- Numbers quoted in the README must come from a command you ran; state the command, hardware and
  toolchain next to them.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
