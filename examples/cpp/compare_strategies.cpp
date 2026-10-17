// Replays one synthetic few-shot trace through every prefix-caching strategy.

#include <exception>
#include <iomanip>
#include <iostream>

#include "radix_kv/simulator.hpp"
#include "radix_kv/workload.hpp"

namespace {

int run() {
  const radix_kv::Trace trace =
      radix_kv::generate_trace(radix_kv::FewShotWorkload{.requests = 200}, /*seed=*/7);
  const radix_kv::SimulatorConfig config{.num_blocks = 1024, .block_size = 16, .max_running = 16};
  std::cout << "synthetic few-shot trace: " << trace.requests.size() << " requests, "
            << trace.total_prompt_tokens() << " prompt tokens\n";
  bool consistent = true;
  for (const radix_kv::Strategy strategy :
       {radix_kv::Strategy::none, radix_kv::Strategy::hash, radix_kv::Strategy::radix}) {
    const radix_kv::SimulationReport report = radix_kv::simulate(trace, strategy, config);
    std::cout << std::left << std::setw(6) << radix_kv::to_string(strategy) << std::right
              << "hit rate " << std::fixed << std::setprecision(1) << std::setw(5)
              << 100.0 * report.hit_rate() << "%  evicted " << std::setw(6) << report.evicted_blocks
              << " blocks  preemptions " << report.preemptions << '\n';
    consistent = consistent && report.consistent;
  }
  return consistent ? 0 : 1;
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
