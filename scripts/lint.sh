#!/usr/bin/env bash
# Format check plus clang-tidy over every C++ source (library, tests, tools, examples, benchmarks,
# fuzzer and Python bindings). Tool binaries can be overridden through the environment:
#   CLANG_FORMAT, CLANG_TIDY, RUN_CLANG_TIDY, PYTHON, BUILD_DIR, JOBS
set -euo pipefail

cd "$(dirname "$0")/.."
CLANG_FORMAT=${CLANG_FORMAT:-clang-format}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy}
PYTHON=${PYTHON:-python3}
BUILD_DIR=${BUILD_DIR:-build/lint}
JOBS=${JOBS:-4}

echo "== clang-format ($("$CLANG_FORMAT" --version))"
git ls-files '*.cpp' '*.hpp' '*.hpp.in' | xargs "$CLANG_FORMAT" --dry-run --Werror

# An absolute path keeps a virtual environment's interpreter (and its nanobind) selected.
python_path=$(command -v "$PYTHON")
python_path="$(cd "$(dirname "$python_path")" && pwd)/$(basename "$python_path")"

echo "== configure $BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DRADIX_KV_BUILD_BENCHMARKS=ON -DRADIX_KV_BUILD_PYTHON=ON -DPython_EXECUTABLE="$python_path" \
  > /dev/null

extra_args=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Standalone clang-tidy builds do not know the Xcode SDK location.
  extra_args+=("-extra-arg=-isysroot$(xcrun --show-sdk-path)")
fi

echo "== clang-tidy ($("$CLANG_TIDY" --version | grep -i version | head -n 1))"
"$RUN_CLANG_TIDY" -p "$BUILD_DIR" -clang-tidy-binary "$(command -v "$CLANG_TIDY")" -quiet \
  -j "$JOBS" "${extra_args[@]}" "$(pwd)/(src|tests|tools|examples|bench|fuzz|python/src)/.*\.cpp$"
