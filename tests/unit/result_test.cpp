#include "radix_kv/result.hpp"

#include <array>
#include <memory>
#include <set>
#include <string_view>
#include <type_traits>
#include <utility>
#include <version>  // IWYU pragma: keep (feature-test macros)

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#include <expected>
#endif

#include <gtest/gtest.h>

namespace radix_kv {
namespace {

Result<int> parse_positive(int value) {
  if (value <= 0) {
    return fail(Errc::invalid_argument);
  }
  return value;
}

Result<void> require_even(int value) {
  if (value % 2 != 0) {
    return fail(Errc::invalid_argument);
  }
  return {};
}

TEST(Result, IsStdExpectedExactlyWhenTheStandardLibraryHasIt) {
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
  static_assert(std::is_same_v<Result<int>, std::expected<int, Errc>>);
  EXPECT_EQ(RADIX_KV_STD_EXPECTED, 1);
#else
  static_assert(!std::is_same_v<Result<int>, int>);
  EXPECT_EQ(RADIX_KV_STD_EXPECTED, 0);
#endif
}

TEST(Result, HoldsValue) {
  const Result<int> result = parse_positive(7);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result.value(), 7);
  EXPECT_EQ(*result, 7);
  EXPECT_EQ(result.value_or(0), 7);
}

TEST(Result, HoldsError) {
  const Result<int> result = parse_positive(-1);
  ASSERT_FALSE(result.has_value());
  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(result.error(), Errc::invalid_argument);
  EXPECT_EQ(result.value_or(42), 42);
  EXPECT_ANY_THROW(static_cast<void>(result.value()));
}

TEST(Result, VoidResult) {
  const Result<void> ok = require_even(2);
  EXPECT_TRUE(ok.has_value());
  EXPECT_NO_THROW(ok.value());

  const Result<void> bad = require_even(3);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), Errc::invalid_argument);
  EXPECT_ANY_THROW(bad.value());
}

TEST(Result, SupportsMoveOnlyValues) {
  Result<std::unique_ptr<int>> result = std::make_unique<int>(5);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(**result, 5);
  const std::unique_ptr<int> taken = std::move(result).value();
  EXPECT_EQ(*taken, 5);
}

TEST(Result, ArrowAccessesMembers) {
  struct Pair {
    int first;
    int second;
  };
  const Result<Pair> result = Pair{.first = 1, .second = 2};
  EXPECT_EQ(result->second, 2);
}

TEST(Errc, EveryCodeHasDistinctNameAndDescription) {
  constexpr std::array kAll{Errc::out_of_blocks,       Errc::invalid_block,
                            Errc::block_not_allocated, Errc::ref_count_overflow,
                            Errc::invalid_handle,      Errc::not_locked,
                            Errc::invalid_argument};
  std::set<std::string_view> names;
  std::set<std::string_view> descriptions;
  for (const Errc code : kAll) {
    EXPECT_NE(to_string(code), "unknown");
    EXPECT_NE(describe(code), "unknown error");
    names.insert(to_string(code));
    descriptions.insert(describe(code));
  }
  EXPECT_EQ(names.size(), kAll.size());
  EXPECT_EQ(descriptions.size(), kAll.size());
  EXPECT_EQ(to_string(Errc::out_of_blocks), "out_of_blocks");
}

}  // namespace
}  // namespace radix_kv
