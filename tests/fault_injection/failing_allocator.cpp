#include "failing_allocator.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

// Allocations still allowed to succeed; negative while no guard is alive.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): process-wide by nature
std::atomic<std::ptrdiff_t> allocation_budget{-1};

}  // namespace

namespace radix_kv::test_support {

FailingAllocations::FailingAllocations(std::size_t allowed) noexcept {
  allocation_budget.store(static_cast<std::ptrdiff_t>(allowed));
}

FailingAllocations::~FailingAllocations() {
  allocation_budget.store(-1);
}

}  // namespace radix_kv::test_support

// Replacements of the global allocation functions ([new.delete.single]). The array and nothrow
// forms call these by default.
// NOLINTBEGIN(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
void* operator new(std::size_t size) {
  const std::ptrdiff_t budget = allocation_budget.load();
  if (budget == 0) {
    throw std::bad_alloc();
  }
  if (budget > 0) {
    allocation_budget.store(budget - 1);
  }
  void* memory = std::malloc(size == 0 ? 1 : size);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

void operator delete(void* memory) noexcept {
  std::free(memory);
}

void operator delete(void* memory, std::size_t /*size*/) noexcept {
  std::free(memory);
}
// NOLINTEND(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
