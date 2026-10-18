#include "radix_kv/result.hpp"

namespace radix_kv::detail {

// See the declarations in result.hpp: only the function of the mode the library is built in exists.
#if RADIX_KV_STD_EXPECTED
int built_with_std_expected_result() noexcept {
  return 1;
}
#else
int built_with_fallback_result() noexcept {
  return 0;
}
#endif

}  // namespace radix_kv::detail
