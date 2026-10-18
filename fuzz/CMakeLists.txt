# libFuzzer target. tests/CMakeLists.txt also builds the same harness with a plain main()
# (radix_kv_fuzz_replay), so every compiler compiles, lints and runs it.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "RADIX_KV_BUILD_FUZZERS needs a Clang that ships libFuzzer "
                      "(Apple Clang does not; use e.g. Homebrew LLVM or clang on Linux)")
endif()

# UBSan must abort instead of printing and carrying on, or undefined behaviour cannot fail a run.
set(fuzz_flags -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all)
add_executable(fuzz_prefix_cache fuzz_prefix_cache.cpp)
target_include_directories(fuzz_prefix_cache PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(fuzz_prefix_cache PRIVATE radix_kv::radix_kv)
target_compile_options(fuzz_prefix_cache PRIVATE ${fuzz_flags})
target_link_options(fuzz_prefix_cache PRIVATE ${fuzz_flags})
radix_kv_set_warnings(fuzz_prefix_cache)
