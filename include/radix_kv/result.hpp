#pragma once

/// @file
/// Error codes and the `Result<T>` type returned by every fallible operation.
///
/// `Result<T>` is `std::expected<T, Errc>` when the standard library provides it (C++23) and a
/// small compatible fallback otherwise, so the library builds as C++20. Only the subset shared by
/// both is part of the API: `has_value()`, `operator bool`, `value()`, `error()`, `operator*`,
/// `operator->` and `value_or()`. Errors are plain enumerators: creating or inspecting one never
/// allocates or throws.

#include <cstdint>
#include <string_view>
#include <version>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#include <expected>
/// 1 when `Result` is `std::expected`, 0 when it is the C++20 fallback.
#define RADIX_KV_STD_EXPECTED 1  // NOLINT(cppcoreguidelines-macro-usage): used in #if
#else
#include <exception>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#define RADIX_KV_STD_EXPECTED 0  // NOLINT(cppcoreguidelines-macro-usage): used in #if
#endif

namespace radix_kv {

/// Reasons an operation can fail.
enum class Errc : std::uint8_t {
  out_of_blocks = 1,    ///< The pool has too few free blocks for the request.
  invalid_block,        ///< A block id is outside the pool.
  block_not_allocated,  ///< The block is free (reference count zero), e.g. a double release.
  ref_count_overflow,   ///< A reference or lock count would exceed its 32-bit range.
  invalid_handle,       ///< A node handle is stale (its node was evicted) or was never issued.
  not_locked,           ///< `unlock()` on a node that holds no lock taken through that handle.
  invalid_argument,     ///< A documented precondition on the arguments does not hold.
};

/// Returns the enumerator name, e.g. `"out_of_blocks"`.
[[nodiscard]] constexpr std::string_view to_string(Errc error) noexcept {
  switch (error) {
    case Errc::out_of_blocks:
      return "out_of_blocks";
    case Errc::invalid_block:
      return "invalid_block";
    case Errc::block_not_allocated:
      return "block_not_allocated";
    case Errc::ref_count_overflow:
      return "ref_count_overflow";
    case Errc::invalid_handle:
      return "invalid_handle";
    case Errc::not_locked:
      return "not_locked";
    case Errc::invalid_argument:
      return "invalid_argument";
  }
  return "unknown";
}

/// Returns a one-sentence description of the error, suitable for logs and exception messages.
[[nodiscard]] constexpr std::string_view describe(Errc error) noexcept {
  switch (error) {
    case Errc::out_of_blocks:
      return "not enough free blocks in the pool";
    case Errc::invalid_block:
      return "block id is outside the pool";
    case Errc::block_not_allocated:
      return "block is not allocated (reference count is zero)";
    case Errc::ref_count_overflow:
      return "reference or lock count overflow";
    case Errc::invalid_handle:
      return "node handle is stale or was never issued by this cache";
    case Errc::not_locked:
      return "node is not locked through this handle";
    case Errc::invalid_argument:
      return "invalid argument";
  }
  return "unknown error";
}

#if RADIX_KV_STD_EXPECTED

/// Value-or-error result of a fallible operation.
template <class T>
using Result = std::expected<T, Errc>;

/// The error alternative of a `Result`; construct it with `fail()`.
using Unexpected = std::unexpected<Errc>;

#else

/// The error alternative of a `Result`; construct it with `fail()`.
class Unexpected {
 public:
  constexpr explicit Unexpected(Errc error) noexcept : error_(error) {}

  /// The wrapped error code.
  [[nodiscard]] constexpr Errc error() const noexcept { return error_; }

 private:
  Errc error_;
};

/// Thrown by `Result::value()` when the result holds an error (mirrors `std::bad_expected_access`).
class BadResultAccess : public std::exception {
 public:
  explicit BadResultAccess(Errc error) noexcept : error_(error) {}

  [[nodiscard]] const char* what() const noexcept override {
    return "radix_kv::Result::value() called on an error result";
  }

  /// The error the result held.
  [[nodiscard]] Errc error() const noexcept { return error_; }

 private:
  Errc error_;
};

/// Value-or-error result of a fallible operation (C++20 fallback for `std::expected<T, Errc>`).
template <class T>
class Result {
 public:
  using value_type = T;
  using error_type = Errc;

  /// Holds `value`.
  template <class U = T>
    requires(std::is_constructible_v<T, U&&> && !std::is_same_v<std::remove_cvref_t<U>, Result> &&
             !std::is_same_v<std::remove_cvref_t<U>, Unexpected>)
  // NOLINTNEXTLINE(*-explicit-constructor,*-explicit-conversions): mirrors std::expected
  constexpr Result(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>)
      : storage_(std::in_place_index<0>, std::forward<U>(value)) {}

  /// Holds `error.error()`.
  // NOLINTNEXTLINE(*-explicit-constructor,*-explicit-conversions): mirrors std::expected
  constexpr Result(Unexpected error) noexcept : storage_(std::in_place_index<1>, error.error()) {}

  /// True when the result holds a value.
  [[nodiscard]] constexpr bool has_value() const noexcept { return storage_.index() == 0; }
  constexpr explicit operator bool() const noexcept { return has_value(); }

  /// The value; throws `BadResultAccess` when the result holds an error.
  [[nodiscard]] constexpr T& value() & {
    check();
    return *get();
  }
  [[nodiscard]] constexpr const T& value() const& {
    check();
    return *get();
  }
  [[nodiscard]] constexpr T&& value() && {
    check();
    return std::move(*get());
  }

  /// The value without a check. Precondition: `has_value()`.
  [[nodiscard]] constexpr T& operator*() & noexcept { return *get(); }
  [[nodiscard]] constexpr const T& operator*() const& noexcept { return *get(); }
  [[nodiscard]] constexpr T&& operator*() && noexcept { return std::move(*get()); }
  [[nodiscard]] constexpr T* operator->() noexcept { return get(); }
  [[nodiscard]] constexpr const T* operator->() const noexcept { return get(); }

  /// The error. Precondition: `!has_value()`.
  [[nodiscard]] constexpr Errc error() const noexcept {
    const Errc* code = std::get_if<1>(&storage_);
    return code != nullptr ? *code : Errc::invalid_argument;
  }

  /// The value, or `fallback` when the result holds an error.
  template <class U>
  [[nodiscard]] constexpr T value_or(U&& fallback) const& {
    return has_value() ? **this : static_cast<T>(std::forward<U>(fallback));
  }
  template <class U>
  [[nodiscard]] constexpr T value_or(U&& fallback) && {
    return has_value() ? std::move(**this) : static_cast<T>(std::forward<U>(fallback));
  }

 private:
  constexpr void check() const {
    if (!has_value()) {
      throw BadResultAccess(error());
    }
  }

  // Unchecked access is a precondition violation; terminate instead of dereferencing null.
  [[nodiscard]] constexpr T* get() noexcept {
    T* value = std::get_if<0>(&storage_);
    if (value == nullptr) {
      std::terminate();
    }
    return value;
  }
  [[nodiscard]] constexpr const T* get() const noexcept {
    const T* value = std::get_if<0>(&storage_);
    if (value == nullptr) {
      std::terminate();
    }
    return value;
  }

  std::variant<T, Errc> storage_;
};

/// Success-or-error result of a fallible operation that returns nothing.
template <>
class Result<void> {
 public:
  using value_type = void;
  using error_type = Errc;

  /// Success.
  constexpr Result() noexcept = default;

  /// Holds `error.error()`.
  // NOLINTNEXTLINE(*-explicit-constructor,*-explicit-conversions): mirrors std::expected
  constexpr Result(Unexpected error) noexcept : error_(error.error()) {}

  /// True on success.
  [[nodiscard]] constexpr bool has_value() const noexcept { return !error_.has_value(); }
  constexpr explicit operator bool() const noexcept { return has_value(); }

  /// Throws `BadResultAccess` when the result holds an error.
  constexpr void value() const {
    if (error_.has_value()) {
      throw BadResultAccess(*error_);
    }
  }

  /// No-op accessor kept for parity with `std::expected<void, E>`.
  constexpr void operator*() const noexcept {}

  /// The error. Precondition: `!has_value()`.
  [[nodiscard]] constexpr Errc error() const noexcept {
    return error_.value_or(Errc::invalid_argument);
  }

 private:
  std::optional<Errc> error_;
};

#endif

/// Makes the error alternative of a `Result`: `return fail(Errc::out_of_blocks);`.
[[nodiscard]] constexpr Unexpected fail(Errc error) noexcept {
  return Unexpected(error);
}

}  // namespace radix_kv
