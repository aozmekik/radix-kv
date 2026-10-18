#include "radix_kv/version.hpp"

#include <string>

#include <gtest/gtest.h>

// The macros exist for preprocessor conditionals. A configure_file() placeholder left
// unsubstituted (such as `@PROJECT_VERSION_MAJOR @`) turns these lines into compile errors.
#if !defined(RADIX_KV_VERSION_MAJOR) || !defined(RADIX_KV_VERSION_MINOR) || \
    !defined(RADIX_KV_VERSION_PATCH)
#error "radix_kv/version.hpp must define the RADIX_KV_VERSION_* macros"
#endif
#if RADIX_KV_VERSION_MAJOR < 0 || RADIX_KV_VERSION_MINOR < 0 || RADIX_KV_VERSION_PATCH < 0
#error "the RADIX_KV_VERSION_* macros must be non-negative integers"
#endif

namespace radix_kv {
namespace {

TEST(Version, MacrosSpellTheVersionString) {
  const std::string expected = std::to_string(RADIX_KV_VERSION_MAJOR) + "." +
                               std::to_string(RADIX_KV_VERSION_MINOR) + "." +
                               std::to_string(RADIX_KV_VERSION_PATCH);
  EXPECT_EQ(kVersion, expected);
}

}  // namespace
}  // namespace radix_kv
