// radix-kv-sim: replays synthetic traces through the scheduler simulation and prints a comparison
// of the prefix-caching strategies.

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <ratio>
#include <span>
#include <sstream>
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

Requests larger than the whole pool are rejected (with the later turns of their session) and
reported in the "rejected" column; hit rates cover the completed requests only.

Exit status: 0 on success, 1 if a simulation failed its consistency checks or completed no
request, 2 for invalid options.
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
    throw UsageError(std::string(flag) + ": expected a non-negative integer, got '" +
                     std::string(text) + "'");
  }
  return value;
}

std::uint32_t parse_positive(std::string_view text, std::string_view flag) {
  const auto value = parse_number<std::uint32_t>(text, flag);
  if (value == 0) {
    throw UsageError(std::string(flag) + ": must be positive");
  }
  return value;
}

// Parses "N[,N...]"; an empty element (as in "1024," or ",") is an error.
std::vector<std::uint32_t> parse_list(std::string_view text, std::string_view flag) {
  std::vector<std::uint32_t> values;
  for (;;) {
    const std::size_t comma = text.find(',');
    values.push_back(parse_positive(text.substr(0, comma), flag));
    if (comma == std::string_view::npos) {
      return values;
    }
    text.remove_prefix(comma + 1);
  }
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
  throw UsageError("--strategy: unknown strategy '" + std::string(name) + "'");
}

std::vector<std::string> parse_workloads(std::string_view name) {
  if (name == "all") {
    return {"chat", "few-shot", "rag", "random"};
  }
  if (name == "chat" || name == "few-shot" || name == "rag" || name == "random") {
    return {std::string(name)};
  }
  throw UsageError("--workload: unknown workload '" + std::string(name) + "'");
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
      throw UsageError(std::string(flag) + ": missing value (see --help)");
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
        throw UsageError("--format: expected markdown or csv, got '" + std::string(value) + "'");
      }
      options.csv = value == "csv";
    } else {
      throw UsageError("unknown option '" + std::string(flag) + "' (see --help)");
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

// Fixed-point formatting through iostreams: formatting doubles with <format> needs macOS 13.3+.
std::string fixed(double value, int precision) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string format_rate(double per_second) {
  return per_second >= 1e6 ? fixed(per_second / 1e6, 2) + "M" : fixed(per_second / 1e3, 0) + "k";
}

void print_header(bool csv) {
  if (csv) {
    std::cout << "workload,strategy,blocks,completed,rejected,hit_rate,prompt_tokens,cached_tokens,"
                 "recompute_prompt_tokens,recompute_cached_tokens,evicted_blocks,preemptions,"
                 "peak_blocks_used,peak_blocks_active,cow_copies,cache_ops,cache_ms,"
                 "cache_ops_per_s,wall_ms,consistent\n";
    return;
  }
  std::cout
      << "| workload | strategy | blocks | rejected | hit rate | evicted blocks | preemptions | "
         "peak used | peak active | cache ms | cache ops/s | wall ms |\n"
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n";
}

void print_row(bool csv, std::string_view workload, std::uint32_t blocks,
               const SimulationReport& report) {
  using std::chrono::duration;
  const double wall_ms = duration<double, std::milli>(report.wall_time).count();
  const double cache_ms = duration<double, std::milli>(report.cache_time).count();
  const std::string_view strategy = radix_kv::to_string(report.strategy);
  if (csv) {
    std::cout << workload << ',' << strategy << ',' << blocks << ',' << report.requests_completed
              << ',' << report.requests_rejected << ',' << fixed(report.hit_rate(), 6) << ','
              << report.prompt_tokens << ',' << report.cached_tokens << ','
              << report.recompute_prompt_tokens << ',' << report.recompute_cached_tokens << ','
              << report.evicted_blocks << ',' << report.preemptions << ','
              << report.peak_blocks_used << ',' << report.peak_blocks_active << ','
              << report.cow_copies << ',' << report.cache_ops << ',' << fixed(cache_ms, 3) << ','
              << fixed(report.cache_ops_per_second(), 0) << ',' << fixed(wall_ms, 3) << ','
              << (report.consistent ? "true" : "false") << '\n';
    return;
  }
  std::cout << "| " << workload << " | " << strategy << " | " << blocks << " | "
            << report.requests_rejected << " | " << fixed(100.0 * report.hit_rate(), 1) << "% | "
            << report.evicted_blocks << " | " << report.preemptions << " | "
            << report.peak_blocks_used << " | " << report.peak_blocks_active << " | "
            << fixed(cache_ms, 1) << " | " << format_rate(report.cache_ops_per_second()) << " | "
            << fixed(wall_ms, 1) << " |\n";
}

int run(const Options& options) {
  print_header(options.csv);
  bool ok = true;
  for (const std::string& name : options.workloads) {
    const radix_kv::Trace trace =
        radix_kv::generate_trace(make_workload(name, options.requests), options.seed);
    for (const std::uint32_t blocks : options.pool_sizes) {
      SimulatorConfig config = options.config;
      config.num_blocks = blocks;
      for (const Strategy strategy : options.strategies) {
        const SimulationReport report = radix_kv::simulate(trace, strategy, config);
        print_row(options.csv, name, blocks, report);
        const std::string where = name + " / " + std::string(radix_kv::to_string(strategy)) +
                                  " / " + std::to_string(blocks) + " blocks: ";
        if (!report.consistent) {
          std::cerr << "error: " << where << report.inconsistency << '\n';
          ok = false;
        }
        if (report.requests_completed == 0) {
          std::cerr << "error: " << where << "every request was rejected (larger than the pool)\n";
        } else if (report.requests_rejected > 0) {
          std::cerr << "warning: " << where << report.requests_rejected << " of "
                    << trace.requests.size()
                    << " requests rejected (larger than the pool); the hit rate covers the "
                    << report.requests_completed << " completed requests only\n";
        }
        ok = ok && report.requests_completed > 0;
      }
    }
  }
  return ok ? 0 : 1;
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
