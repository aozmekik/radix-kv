// Runs the libFuzzer entry point of fuzz/fuzz_prefix_cache.cpp without libFuzzer, so that every
// build compiles, lints and exercises the harness: on the files named on the command line, or else
// on a fixed set of pseudo-random inputs.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <vector>

// Defined in fuzz/fuzz_prefix_cache.cpp; libFuzzer fixes the name.
extern "C" int LLVMFuzzerTestOneInput(  // NOLINT(readability-identifier-naming)
    const std::uint8_t* data, std::size_t size);

namespace {

constexpr int kGeneratedInputs = 300;
constexpr std::uint64_t kMaxInputBytes = 1024;

int replay_files(std::span<char* const> paths) {
  for (const char* path : paths) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "fuzz replay: cannot read " << path << '\n';
      return 1;
    }
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>()};
    const std::vector<std::uint8_t> input(bytes.begin(), bytes.end());
    LLVMFuzzerTestOneInput(input.data(), input.size());
  }
  std::cout << "fuzz replay: ran " << paths.size() << " files\n";
  return 0;
}

int replay_generated() {
  // The raw output of std::mt19937_64 is fully specified, so these inputs are the same everywhere.
  // A constant seed on purpose: the same inputs on every run.
  // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
  std::mt19937_64 rng(20261018);
  for (int i = 0; i < kGeneratedInputs; ++i) {
    std::vector<std::uint8_t> input(rng() % kMaxInputBytes);
    for (std::uint8_t& byte : input) {
      byte = static_cast<std::uint8_t>(rng());
    }
    LLVMFuzzerTestOneInput(input.data(), input.size());
  }
  std::cout << "fuzz replay: ran " << kGeneratedInputs << " generated inputs\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span<char* const> args(argv, static_cast<std::size_t>(argc));
    return args.size() > 1 ? replay_files(args.subspan(1)) : replay_generated();
  } catch (const std::exception& error) {
    std::cerr << "fuzz replay: " << error.what() << '\n';
    return 1;
  }
}
