#pragma once

// Fault injection for the allocation-failure tests. failing_allocator.cpp replaces the global
// operator new; while a FailingAllocations guard is alive, a fixed number of allocations succeed
// and every later one throws std::bad_alloc.

#include <cstddef>

namespace radix_kv::test_support {

/// Lets `allowed` more allocations succeed, then makes every allocation throw until destroyed.
/// Not reentrant: one guard at a time.
class FailingAllocations {
 public:
  explicit FailingAllocations(std::size_t allowed) noexcept;
  FailingAllocations(const FailingAllocations&) = delete;
  FailingAllocations& operator=(const FailingAllocations&) = delete;
  FailingAllocations(FailingAllocations&&) = delete;
  FailingAllocations& operator=(FailingAllocations&&) = delete;
  ~FailingAllocations();
};

}  // namespace radix_kv::test_support
