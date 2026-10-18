#pragma once

/// @file
/// Synthetic request traces for the simulator.
///
/// The traces are synthetic: token ids are uniformly random and the length distributions are
/// illustrative defaults, not fitted to any production traffic. What they do model is the
/// structure that decides prefix-cache behaviour: which parts of a prompt are shared, by how many
/// requests, and how far apart those requests arrive.

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "radix_kv/types.hpp"

namespace radix_kv {

/// Distribution of a positive length (tokens, turns). Construct it with one of the factories.
class LengthDistribution {
 public:
  /// Shape of the distribution.
  enum class Kind : std::uint8_t {
    fixed,      ///< Always `min()` (== `max()`).
    uniform,    ///< Uniform over the integers `[min(), max()]`.
    lognormal,  ///< `median * exp(sigma * Z)` with Z standard normal, rounded, clamped.
  };

  /// Always `value`. @throws std::invalid_argument if `value == 0`.
  static LengthDistribution fixed(std::uint32_t value);

  /// Uniform over `[min, max]`. @throws std::invalid_argument unless `1 <= min <= max`.
  static LengthDistribution uniform(std::uint32_t min, std::uint32_t max);

  /// Log-normal with the given median and log-space standard deviation, clamped to `[min, max]`.
  /// @throws std::invalid_argument unless `1 <= min <= max`, `median > 0` and `sigma >= 0`
  ///         (both finite).
  static LengthDistribution lognormal(double median, double sigma, std::uint32_t min,
                                      std::uint32_t max);

  /// Shape of the distribution.
  [[nodiscard]] Kind kind() const noexcept { return kind_; }
  /// Smallest value the distribution produces.
  [[nodiscard]] std::uint32_t min() const noexcept { return min_; }
  /// Largest value the distribution produces.
  [[nodiscard]] std::uint32_t max() const noexcept { return max_; }
  /// Median of the log-normal kind; 0 otherwise.
  [[nodiscard]] double median() const noexcept { return median_; }
  /// Log-space standard deviation of the log-normal kind; 0 otherwise.
  [[nodiscard]] double sigma() const noexcept { return sigma_; }

  friend bool operator==(const LengthDistribution&, const LengthDistribution&) = default;

 private:
  LengthDistribution(Kind kind, std::uint32_t min, std::uint32_t max, double median,
                     double sigma) noexcept
      : kind_(kind), min_(min), max_(max), median_(median), sigma_(sigma) {}

  Kind kind_;
  std::uint32_t min_;
  std::uint32_t max_;
  double median_;
  double sigma_;
};

/// Multi-turn chat: every session starts with the same system prompt; turn `k + 1` re-sends the
/// whole conversation so far (system prompt, earlier user turns and model outputs) plus a new user
/// message.
struct ChatWorkload {
  std::uint32_t sessions = 200;               ///< Number of conversations.
  std::uint32_t system_prompt_tokens = 1024;  ///< Shared by every session; may be 0.
  /// Turns (requests) per session.
  LengthDistribution turns = LengthDistribution::uniform(2, 6);
  /// Tokens of each new user message.
  LengthDistribution user_tokens = LengthDistribution::lognormal(64, 0.8, 8, 512);
  /// Tokens the model generates per turn.
  LengthDistribution output_tokens = LengthDistribution::lognormal(128, 0.7, 8, 1024);
};

/// Few-shot prompting: a small set of long templates (instructions plus examples), each followed by
/// a short question. Template popularity follows a Zipf law.
struct FewShotWorkload {
  std::uint32_t requests = 1000;  ///< Number of requests.
  std::uint32_t templates = 8;    ///< Distinct few-shot templates.
  /// Tokens of each template.
  LengthDistribution template_tokens = LengthDistribution::uniform(1024, 3072);
  double template_zipf = 1.0;  ///< Zipf exponent of template popularity; 0 means uniform.
  /// Tokens of the question appended to the template.
  LengthDistribution question_tokens = LengthDistribution::uniform(16, 128);
  /// Tokens the model generates.
  LengthDistribution output_tokens = LengthDistribution::lognormal(48, 0.6, 4, 256);
};

/// Retrieval-augmented generation: a shared system prompt, then `documents_per_query` distinct
/// documents drawn from a Zipf-popular pool (in retrieval order), then the user query. Only the
/// part of the document list that is identical from the start can be reused as a prefix.
struct RagWorkload {
  std::uint32_t requests = 1000;             ///< Number of requests.
  std::uint32_t system_prompt_tokens = 256;  ///< Shared by every request; may be 0.
  std::uint32_t documents = 200;             ///< Size of the document pool.
  /// Tokens of each document.
  LengthDistribution document_tokens = LengthDistribution::uniform(256, 1024);
  std::uint32_t documents_per_query = 3;  ///< Between 1 and `documents`.
  double document_zipf = 1.1;             ///< Zipf exponent of document popularity; 0 = uniform.
  /// Tokens of the user query after the documents.
  LengthDistribution query_tokens = LengthDistribution::uniform(16, 64);
  /// Tokens the model generates.
  LengthDistribution output_tokens = LengthDistribution::lognormal(96, 0.6, 8, 512);
};

/// Independent random prompts: no shared prefixes at all (the control workload).
struct RandomWorkload {
  std::uint32_t requests = 1000;  ///< Number of requests.
  /// Tokens of each (random) prompt.
  LengthDistribution prompt_tokens = LengthDistribution::uniform(64, 2048);
  /// Tokens the model generates.
  LengthDistribution output_tokens = LengthDistribution::uniform(16, 256);
};

/// Any of the synthetic workloads.
using Workload = std::variant<ChatWorkload, FewShotWorkload, RagWorkload, RandomWorkload>;

/// One request of a trace. Its prompt and output are consecutive ranges of its session's tokens.
struct TraceRequest {
  std::uint32_t session = 0;        ///< Index into `Trace::sessions`.
  std::uint32_t prompt_tokens = 0;  ///< The prompt is the first `prompt_tokens` session tokens.
  std::uint32_t output_tokens = 0;  ///< The output is the next `output_tokens` session tokens.

  friend bool operator==(const TraceRequest&, const TraceRequest&) = default;
};

/// A generated workload: token streams plus the requests that read them.
///
/// Requests of one session are consecutive and in turn order; the simulator releases turn `k + 1`
/// only after turn `k` finished, the way a chat client waits for the answer.
struct Trace {
  std::vector<std::vector<TokenId>> sessions;  ///< Token stream of each session.
  std::vector<TraceRequest> requests;          ///< Grouped by session, in turn order.
  std::uint32_t vocab_size = 0;                ///< Token ids are in `[0, vocab_size)`.

  /// Prompt tokens of `request`. Precondition: `request` belongs to this trace.
  [[nodiscard]] std::span<const TokenId> prompt(const TraceRequest& request) const noexcept;

  /// Output tokens of `request`. Precondition: `request` belongs to this trace.
  [[nodiscard]] std::span<const TokenId> output(const TraceRequest& request) const noexcept;

  /// Sum of prompt lengths over all requests.
  [[nodiscard]] std::uint64_t total_prompt_tokens() const noexcept;

  /// Sum of output lengths over all requests.
  [[nodiscard]] std::uint64_t total_output_tokens() const noexcept;
};

/// Generates a deterministic trace: the same workload, seed and vocabulary size give the same
/// trace. The generator uses its own PRNG instead of the implementation-defined `<random>`
/// distributions, so traces built only from fixed and uniform lengths with Zipf exponent 0 are
/// identical on every platform. Log-normal lengths and Zipf popularity go through `<cmath>`
/// (`exp`, `log`, `pow`), whose last-bit results can differ between C libraries, so such traces
/// may differ slightly from one platform to another.
/// @throws std::invalid_argument for an empty workload (zero sessions, requests, templates or
///         documents), `documents_per_query` outside `[1, documents]`, a negative or non-finite
///         Zipf exponent, or `vocab_size == 0`.
[[nodiscard]] Trace generate_trace(const Workload& workload, std::uint64_t seed,
                                   std::uint32_t vocab_size = 32000);

}  // namespace radix_kv
