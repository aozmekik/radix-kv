// radix-kv-sim: replays synthetic traces through the scheduler simulation and prints a comparison
// of the prefix-caching strategies.

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <optional>
#include <ratio>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "radix_kv/simulator.hpp"
#include "radix_kv/version.hpp"
#include "radix_kv/workload.hpp"

namespace {

using radix_kv::SimulationReport;
using radix_kv::SimulatorConfig;
using radix_kv::Strategy;
using radix_kv::Workload;

constexpr std::string_view kUsage = R"(Usage: radix-kv-sim [options]

Replays synthetic request traces through a continuous-batching scheduler simulation and compares
prefix-caching strategies. All numbers come from synthetic traces.

Options:
  --workload NAME      chat, few-shot, rag, random or all            (default: all)
  --strategy NAME      none, hash, radix or all                      (default: all)
  --blocks N[,N...]    KV-cache pool sizes in blocks                 (default: 2048)
  --block-size N       tokens per block                              (default: 16)
  --max-running N      maximum concurrently running requests         (default: 32)
  --samples N          parallel samples per request (fork + CoW)     (default: 1)
  --requests N         requests per workload (sessions for chat)     (default: per workload)
  --seed N             trace seed                                    (default: 1)
  --format FORMAT      markdown or csv                               (default: markdown)
  --version            print the version and exit
  --help               print this help and exit
)";

struct Options {
  std::vector<std::string> workloads{"chat", "few-shot", "rag", "random"};
  std::vector<Strategy> strategies{Strategy::none, Strategy::hash, Strategy::radix};
  std::vector<std::uint32_t> pool_sizes{2048};
  SimulatorConfig config{.num_blocks = 2048, .block_size = 16, .max_running = 32};
  std::optional<std::uint32_t> requests;
  std::uint64_t seed = 1;
  bool csv = false;
};

class UsageError : public std::exception {
 public:
  explicit UsageError(std::string message) : message_(std::move(message)) {}
  [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

template <class Int>
Int parse_number(std::string_view text, std::string_view flag) {
  Int value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw UsageError(std::format("{}: expected a non-negative integer, got '{}'", flag, text));
  }
  return value;
}

std::uint32_t parse_positive(std::string_view text, std::string_view flag) {
  const auto value = parse_number<std::uint32_t>(text, flag);
  if (value == 0) {
    throw UsageError(std::format("{}: must be positive", flag));
  }
  return value;
}

std::vector<std::uint32_t> parse_list(std::string_view text, std::string_view flag) {
  std::vector<std::uint32_t> values;
  while (!text.empty()) {
    const std::size_t comma = text.find(',');
    values.push_back(parse_positive(text.substr(0, comma), flag));
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
  }
  if (values.empty()) {
    throw UsageError(std::format("{}: expected at least one value", flag));
  }
  return values;
}

std::vector<Strategy> parse_strategies(std::string_view name) {
  if (name == "all") {
    return {Strategy::none, Strategy::hash, Strategy::radix};
  }
  for (const Strategy strategy : {Strategy::none, Strategy::hash, Strategy::radix}) {
    if (name == radix_kv::to_string(strategy)) {
      return {strategy};
    }
  }
  throw UsageError(std::format("--strategy: unknown strategy '{}'", name));
}

std::vector<std::string> parse_workloads(std::string_view name) {
  if (name == "all") {
    return {"chat", "few-shot", "rag", "random"};
  }
  if (name == "chat" || name == "few-shot" || name == "rag" || name == "random") {
    return {std::string(name)};
  }
  throw UsageError(std::format("--workload: unknown workload '{}'", name));
}

// Returns std::nullopt when the program should exit successfully (--help, --version).
std::optional<Options> parse(std::span<char*> args) {
  Options options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    if (flag == "--help" || flag == "-h") {
      std::cout << kUsage;
      return std::nullopt;
    }
    if (flag == "--version") {
      std::cout << "radix-kv-sim " << radix_kv::kVersion << '\n';
      return std::nullopt;
    }
    if (i + 1 >= args.size()) {
      throw UsageError(std::format("{}: missing value (see --help)", flag));
    }
    const std::string_view value = args[++i];
    if (flag == "--workload") {
      options.workloads = parse_workloads(value);
    } else if (flag == "--strategy") {
      options.strategies = parse_strategies(value);
    } else if (flag == "--blocks") {
      options.pool_sizes = parse_list(value, flag);
    } else if (flag == "--block-size") {
      options.config.block_size = parse_positive(value, flag);
    } else if (flag == "--max-running") {
      options.config.max_running = parse_positive(value, flag);
    } else if (flag == "--samples") {
      options.config.samples_per_request = parse_positive(value, flag);
    } else if (flag == "--requests") {
      options.requests = parse_positive(value, flag);
    } else if (flag == "--seed") {
      options.seed = parse_number<std::uint64_t>(value, flag);
    } else if (flag == "--format") {
      if (value != "markdown" && value != "csv") {
        throw UsageError(std::format("--format: expected markdown or csv, got '{}'", value));
      }
      options.csv = value == "csv";
    } else {
      throw UsageError(std::format("unknown option '{}' (see --help)", flag));
    }
  }
  return options;
}

Workload make_workload(std::string_view name, std::optional<std::uint32_t> requests) {
  if (name == "chat") {
    radix_kv::ChatWorkload chat;
    chat.sessions = requests.value_or(chat.sessions);
    return chat;
  }
  if (name == "few-shot") {
    radix_kv::FewShotWorkload few_shot;
    few_shot.requests = requests.value_or(few_shot.requests);
    return few_shot;
  }
  if (name == "rag") {
    radix_kv::RagWorkload rag;
    rag.requests = requests.value_or(rag.requests);
    return rag;
  }
  radix_kv::RandomWorkload random;
  random.requests = requests.value_or(random.requests);
  return random;
}

std::string format_rate(double per_second) {
  if (per_second >= 1e6) {
    return std::format("{:.2f}M", per_second / 1e6);
  }
  return std::format("{:.0f}k", per_second / 1e3);
}

void print_header(bool csv) {
  if (csv) {
    std::cout << "workload,strategy,blocks,requests,hit_rate,prompt_tokens,cached_tokens,"
                 "recompute_prompt_tokens,recompute_cached_tokens,evicted_blocks,preemptions,"
                 "peak_blocks_used,peak_blocks_active,cow_copies,cache_ops,cache_ops_per_s,"
                 "wall_ms,consistent\n";
    return;
  }
  std::cout << "| workload | strategy | blocks | hit rate | evicted blocks | preemptions | "
               "peak used | peak active | cache ops/s | wall ms |\n"
               "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n";
}

void print_row(bool csv, std::string_view workload, std::uint32_t blocks,
               const SimulationReport& report) {
  using std::chrono::duration;
  const double wall_ms = duration<double, std::milli>(report.wall_time).count();
  const std::string_view strategy = radix_kv::to_string(report.strategy);
  if (csv) {
    std::cout << std::format("{},{},{},{},{:.6f},{},{},{},{},{},{},{},{},{},{},{:.0f},{:.3f},{}\n",
                             workload, strategy, blocks, report.requests_completed,
                             report.hit_rate(), report.prompt_tokens, report.cached_tokens,
                             report.recompute_prompt_tokens, report.recompute_cached_tokens,
                             report.evicted_blocks, report.preemptions, report.peak_blocks_used,
                             report.peak_blocks_active, report.cow_copies, report.cache_ops,
                             report.cache_ops_per_second(), wall_ms, report.consistent);
    return;
  }
  std::cout << std::format(
      "| {} | {} | {} | {:.1f}% | {} | {} | {} | {} | {} | {:.1f} |\n", workload, strategy, blocks,
      100.0 * report.hit_rate(), report.evicted_blocks, report.preemptions, report.peak_blocks_used,
      report.peak_blocks_active, format_rate(report.cache_ops_per_second()), wall_ms);
}

int run(const Options& options) {
  print_header(options.csv);
  bool all_consistent = true;
  for (const std::string& name : options.workloads) {
    const radix_kv::Trace trace =
        radix_kv::generate_trace(make_workload(name, options.requests), options.seed);
    for (const std::uint32_t blocks : options.pool_sizes) {
      SimulatorConfig config = options.config;
      config.num_blocks = blocks;
      for (const Strategy strategy : options.strategies) {
        const SimulationReport report = radix_kv::simulate(trace, strategy, config);
        print_row(options.csv, name, blocks, report);
        if (!report.consistent) {
          std::cerr << "error: " << name << " / " << radix_kv::to_string(strategy) << ": "
                    << report.inconsistency << '\n';
          all_consistent = false;
        }
      }
    }
  }
  return all_consistent ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::optional<Options> options =
        parse(std::span<char*>(argv, static_cast<std::size_t>(argc)));
    return options ? run(*options) : 0;
  } catch (const UsageError& error) {
    std::cerr << "radix-kv-sim: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "radix-kv-sim: " << error.what() << '\n';
    return 1;
  }
}
