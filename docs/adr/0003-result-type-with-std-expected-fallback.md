# ADR 0003: `Result<T>` is `std::expected` when available, with a C++20 fallback

- Status: accepted
- Date: 2026-10-18

## Context

Allocation, append and lock operations run on every scheduler step. Running out of blocks is not
exceptional there: it is the signal to evict or preempt. The library has to build as C++20, while
`std::expected` arrived in C++23.

## Options considered

1. **Exceptions.** Idiomatic for rare failures, but `out_of_blocks` is an expected outcome on a hot
   path, and exception cost and control flow there are hard to reason about.
2. **Error codes with out-parameters or `std::optional`.** No allocation, but either loses the error
   reason (`optional`) or makes call sites clumsy.
3. **Always use a custom result type.** Works everywhere, but ignores the standard type that C++23
   users already know.
4. **`std::expected` when the standard library has it, a compatible fallback otherwise.**

## Decision

Option 4. `radix_kv::Result<T>` is an alias of `std::expected<T, Errc>` when `__cpp_lib_expected`
is defined and a small class with the same interface otherwise; `fail(Errc)` creates the error in
both modes. `Errc` is a one-byte enum with `to_string()` and `describe()`. Only the members both
implementations share are used: `has_value`, `operator bool`, `value`, `error`, `operator*`,
`operator->` and `value_or`.

Exceptions remain for programming errors outside the hot path: constructors throw
`std::invalid_argument` for impossible configurations, and `std::bad_alloc` propagates.

## Consequences

- No allocation or exception on any fallible hot-path operation.
- The same code compiles as C++20 and C++23; CI builds both and a test asserts which
  implementation is active.
- `value()` on an error throws `std::bad_expected_access<Errc>` in C++23 mode and
  `radix_kv::BadResultAccess` in C++20 mode. Code should test the result instead of relying on
  either exception.
- Monadic operations (`and_then`, `transform`) are only available in C++23 mode and are therefore
  not used by the library.
- `Result` is a different type in the two modes, so the library and the code using it must be
  compiled in the same mode; otherwise functions returning `Result` would be called with the wrong
  layout. `result.hpp` turns such a mix into a link error: every translation unit references
  `radix_kv::detail::built_with_std_expected_result()` or `built_with_fallback_result()`, and the
  library defines only the one of its own mode. The check costs one trivial static initializer.
