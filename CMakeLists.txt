cmake_minimum_required(VERSION 3.25)

project(
  radix_kv
  VERSION 0.1.0
  DESCRIPTION "KV-cache block manager and prefix caches (chained hashes and radix tree) for LLM serving"
  HOMEPAGE_URL "https://github.com/aozmekik/radix-kv"
  LANGUAGES CXX)

option(RADIX_KV_BUILD_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})
option(RADIX_KV_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ${PROJECT_IS_TOP_LEVEL})
option(RADIX_KV_INSTALL "Generate install rules and a CMake package" ${PROJECT_IS_TOP_LEVEL})

if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

include(cmake/CompilerWarnings.cmake)

configure_file(include/radix_kv/version.hpp.in
               ${PROJECT_BINARY_DIR}/generated/include/radix_kv/version.hpp @ONLY)

add_library(radix_kv)
add_library(radix_kv::radix_kv ALIAS radix_kv)
target_sources(
  radix_kv
  PRIVATE src/block_allocator.cpp
          src/block_token_store.cpp
          src/hash.cpp
          src/hash_prefix_cache.cpp
          src/radix_prefix_cache.cpp
          src/sequence.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         include
         ${PROJECT_BINARY_DIR}/generated/include
         FILES
         include/radix_kv/block_allocator.hpp
         include/radix_kv/detail/block_token_store.hpp
         include/radix_kv/detail/lru_list.hpp
         include/radix_kv/hash_prefix_cache.hpp
         include/radix_kv/null_prefix_cache.hpp
         include/radix_kv/prefix_cache.hpp
         include/radix_kv/radix_prefix_cache.hpp
         include/radix_kv/result.hpp
         include/radix_kv/sequence.hpp
         include/radix_kv/types.hpp
         ${PROJECT_BINARY_DIR}/generated/include/radix_kv/version.hpp)
target_compile_features(radix_kv PUBLIC cxx_std_20)
set_target_properties(radix_kv PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION
                                                                   ${PROJECT_VERSION_MAJOR})
radix_kv_set_warnings(radix_kv)

if(RADIX_KV_BUILD_TESTS)
  enable_testing()
  include(cmake/Dependencies.cmake)
  radix_kv_fetch_googletest()
  add_subdirectory(tests)
endif()

if(RADIX_KV_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(
    TARGETS radix_kv
    EXPORT radix_kvTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(
    EXPORT radix_kvTargets
    NAMESPACE radix_kv::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radix_kv)
  configure_package_config_file(
    cmake/radix_kvConfig.cmake.in ${PROJECT_BINARY_DIR}/radix_kvConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radix_kv)
  # 0.x releases may break the API in any minor version.
  write_basic_package_version_file(${PROJECT_BINARY_DIR}/radix_kvConfigVersion.cmake
                                   COMPATIBILITY SameMinorVersion)
  install(FILES ${PROJECT_BINARY_DIR}/radix_kvConfig.cmake
                ${PROJECT_BINARY_DIR}/radix_kvConfigVersion.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/radix_kv)
endif()
