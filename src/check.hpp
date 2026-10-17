#pragma once

// Internal helpers for operations whose success is guaranteed by an earlier check.

#include <cassert>

namespace radix_kv::detail {

/// Consumes a `Result` that cannot hold an error because its preconditions were verified first.
/// Debug builds assert; release builds ignore the value.
template <class R>
constexpr void expect_ok(const R& result) noexcept {
  assert(result.has_value() && "operation guaranteed by an earlier check failed");
  static_cast<void>(result);
}

}  // namespace radix_kv::detail
