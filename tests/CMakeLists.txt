include(GoogleTest)

add_executable(
  radix_kv_tests
  unit/block_allocator_test.cpp
  unit/hash_prefix_cache_test.cpp
  unit/lru_list_test.cpp
  unit/prefix_cache_contract_test.cpp
  unit/property_test.cpp
  unit/radix_prefix_cache_test.cpp
  unit/result_test.cpp
  unit/sequence_test.cpp
  unit/simulator_test.cpp
  unit/workload_test.cpp)
target_include_directories(radix_kv_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radix_kv_tests PRIVATE radix_kv::radix_kv GTest::gmock_main)
radix_kv_set_warnings(radix_kv_tests)

# PRE_TEST discovery lists the tests when ctest runs, so cross-compiled or sanitizer builds work.
gtest_discover_tests(radix_kv_tests DISCOVERY_MODE PRE_TEST)
