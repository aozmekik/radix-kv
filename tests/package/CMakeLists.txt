# Consumer project used by CI to check that radix-kv can be used as an installed CMake package
# (find_package) and as a subproject (FetchContent / add_subdirectory).
cmake_minimum_required(VERSION 3.25)
project(radix_kv_consumer LANGUAGES CXX)

set(RADIX_KV_SOURCE_DIR "" CACHE PATH "Use this radix-kv checkout through FetchContent")
if(RADIX_KV_SOURCE_DIR)
  include(FetchContent)
  FetchContent_Declare(radix_kv SOURCE_DIR ${RADIX_KV_SOURCE_DIR})
  FetchContent_MakeAvailable(radix_kv)
else()
  find_package(radix_kv 0.1 CONFIG REQUIRED)
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE radix_kv::radix_kv)
