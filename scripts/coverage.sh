#!/usr/bin/env bash
# Line and region coverage of the library (src/ and include/) by the C++ test suite, using Clang
# source-based coverage. Requires Clang; set LLVM_PROFDATA / LLVM_COV to matching LLVM tools
# (on macOS the defaults use the ones shipped with Xcode through xcrun).
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == "Darwin" ]]; then
  LLVM_PROFDATA=${LLVM_PROFDATA:-"xcrun llvm-profdata"}
  LLVM_COV=${LLVM_COV:-"xcrun llvm-cov"}
else
  LLVM_PROFDATA=${LLVM_PROFDATA:-llvm-profdata}
  LLVM_COV=${LLVM_COV:-llvm-cov}
fi
BUILD_DIR=build/coverage
PROFILE_DIR="$BUILD_DIR/profiles"

cmake --preset coverage > /dev/null
cmake --build --preset coverage
rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"
LLVM_PROFILE_FILE="$(pwd)/$PROFILE_DIR/%p-%m.profraw" \
  ctest --test-dir "$BUILD_DIR" -j 4 --output-on-failure > /dev/null
$LLVM_PROFDATA merge -sparse "$PROFILE_DIR"/*.profraw -o "$BUILD_DIR/radix_kv.profdata"
$LLVM_COV report "$BUILD_DIR/tests/radix_kv_tests" -object "$BUILD_DIR/tests/radix_kv_fault_tests" \
  -instr-profile="$BUILD_DIR/radix_kv.profdata" \
  -ignore-filename-regex='(_deps|tests|tools|examples|bench|fuzz|generated)/' \
  "$(pwd)/src" "$(pwd)/include"
