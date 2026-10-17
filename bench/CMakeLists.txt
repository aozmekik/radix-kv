add_executable(radix_kv_bench radix_kv_bench.cpp)
target_link_libraries(radix_kv_bench PRIVATE radix_kv::radix_kv benchmark::benchmark)
radix_kv_set_warnings(radix_kv_bench)

if(RADIX_KV_BUILD_TESTS)
  # Smoke run so the benchmarks keep compiling and running; timings are not checked.
  add_test(NAME bench.smoke COMMAND radix_kv_bench --benchmark_min_time=0.001s
                                    --benchmark_filter=match|allocate)
endif()
