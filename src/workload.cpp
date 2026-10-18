#include "radix_kv/workload.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "radix_kv/types.hpp"

namespace radix_kv {

namespace {

/// xoshiro256** (Blackman and Vigna, 2018) seeded through splitmix64: fast, fully specified, and
/// identical on every platform, unlike the engines' distributions in <random>.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) noexcept {
    for (std::uint64_t& word : state_) {
      word = splitmix64(seed);
    }
  }

  std::uint64_t next() noexcept {
    const std::uint64_t result = std::rotl(state_[1] * 5, 7) * 9;
    const std::uint64_t shifted = state_[1] << 17U;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= shifted;
    state_[3] = std::rotl(state_[3], 45);
    return result;
  }

  /// Uniform integer in `[0, bound)`, `bound > 0`, by rejection: unbiased and portable.
  std::uint64_t below(std::uint64_t bound) noexcept {
    const std::uint64_t threshold = (0 - bound) % bound;  // 2^64 mod bound
    for (;;) {
      const std::uint64_t value = next();
      if (value >= threshold) {
        return value % bound;
      }
    }
  }

  /// Uniform double in `[0, 1)` built from 53 random bits.
  double unit() noexcept { return static_cast<double>(next() >> 11U) * 0x1.0p-53; }

 private:
  static std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
  }

  std::array<std::uint64_t, 4> state_{};
};

std::uint32_t sample(const LengthDistribution& distribution, Rng& rng) {
  switch (distribution.kind()) {
    case LengthDistribution::Kind::fixed:
      return distribution.min();
    case LengthDistribution::Kind::uniform: {
      const std::uint64_t span = std::uint64_t{distribution.max()} - distribution.min() + 1;
      return distribution.min() + static_cast<std::uint32_t>(rng.below(span));
    }
    case LengthDistribution::Kind::lognormal: {
      // Box-Muller transform; 1 - unit() is in (0, 1], so the logarithm is finite.
      const double radius = std::sqrt(-2.0 * std::log(1.0 - rng.unit()));
      const double normal = radius * std::cos(2.0 * std::numbers::pi * rng.unit());
      const double value =
          std::round(distribution.median() * std::exp(distribution.sigma() * normal));
      if (!(value > distribution.min())) {  // also catches NaN
        return distribution.min();
      }
      if (value >= distribution.max()) {
        return distribution.max();
      }
      return static_cast<std::uint32_t>(value);
    }
  }
  return distribution.min();
}

void append_random(std::vector<TokenId>& out, std::size_t count, std::uint32_t vocab_size,
                   Rng& rng) {
  out.reserve(out.size() + count);
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(static_cast<TokenId>(rng.below(vocab_size)));
  }
}

std::vector<TokenId> random_tokens(std::size_t count, std::uint32_t vocab_size, Rng& rng) {
  std::vector<TokenId> tokens;
  append_random(tokens, count, vocab_size, rng);
  return tokens;
}

void check_zipf(double exponent, const char* what) {
  if (!std::isfinite(exponent) || exponent < 0.0) {
    throw std::invalid_argument(what);
  }
}

/// Unnormalised Zipf weights `1 / rank^exponent` for ranks 1..count.
std::vector<double> zipf_weights(std::uint32_t count, double exponent) {
  std::vector<double> weights(count);
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    weights[rank] =
        exponent == 0.0 ? 1.0 : 1.0 / std::pow(static_cast<double>(rank) + 1.0, exponent);
  }
  return weights;
}

double sum(std::span<const double> values) noexcept {
  double total = 0.0;
  for (const double value : values) {
    total += value;
  }
  return total;
}

/// Draws an index with probability proportional to `weights` (all positive, at least one entry).
std::uint32_t draw_weighted(std::span<const double> weights, double total, Rng& rng) {
  const double target = rng.unit() * total;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    cumulative += weights[i];
    if (target < cumulative && weights[i] > 0.0) {
      return static_cast<std::uint32_t>(i);
    }
  }
  // Rounding left `target` at or beyond the sum: take the last index that still has weight.
  for (std::size_t i = weights.size(); i-- > 0;) {
    if (weights[i] > 0.0) {
      return static_cast<std::uint32_t>(i);
    }
  }
  return 0;
}

/// Lowest index not in `taken`. Pool documents are ordered by decreasing popularity, so this is the
/// most popular document not drawn yet.
std::uint32_t first_not_taken(std::span<const std::uint32_t> taken) noexcept {
  std::uint32_t candidate = 0;
  while (std::ranges::find(taken, candidate) != taken.end()) {
    ++candidate;
  }
  return candidate;
}

std::uint32_t checked_length(std::size_t length) {
  if (length > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("generate_trace: a session exceeds 2^32 - 1 tokens");
  }
  return static_cast<std::uint32_t>(length);
}

void add_request(Trace& trace, std::vector<TokenId> prompt_then_output, std::uint32_t prompt) {
  const auto session = static_cast<std::uint32_t>(trace.sessions.size());
  const std::uint32_t total = checked_length(prompt_then_output.size());
  trace.requests.push_back(
      TraceRequest{.session = session, .prompt_tokens = prompt, .output_tokens = total - prompt});
  trace.sessions.push_back(std::move(prompt_then_output));
}

void generate(const ChatWorkload& workload, Trace& trace, Rng& rng) {
  if (workload.sessions == 0) {
    throw std::invalid_argument("ChatWorkload: sessions must be positive");
  }
  const std::vector<TokenId> system =
      random_tokens(workload.system_prompt_tokens, trace.vocab_size, rng);
  for (std::uint32_t s = 0; s < workload.sessions; ++s) {
    std::vector<TokenId> stream = system;
    const std::uint32_t turns = sample(workload.turns, rng);
    for (std::uint32_t turn = 0; turn < turns; ++turn) {
      append_random(stream, sample(workload.user_tokens, rng), trace.vocab_size, rng);
      const std::uint32_t prompt = checked_length(stream.size());
      const std::uint32_t output = sample(workload.output_tokens, rng);
      append_random(stream, output, trace.vocab_size, rng);
      trace.requests.push_back(
          TraceRequest{.session = s, .prompt_tokens = prompt, .output_tokens = output});
    }
    checked_length(stream.size());
    trace.sessions.push_back(std::move(stream));
  }
}

void generate(const FewShotWorkload& workload, Trace& trace, Rng& rng) {
  if (workload.requests == 0 || workload.templates == 0) {
    throw std::invalid_argument("FewShotWorkload: requests and templates must be positive");
  }
  check_zipf(workload.template_zipf, "FewShotWorkload: template_zipf must be finite and >= 0");
  std::vector<std::vector<TokenId>> templates;
  templates.reserve(workload.templates);
  for (std::uint32_t t = 0; t < workload.templates; ++t) {
    templates.push_back(
        random_tokens(sample(workload.template_tokens, rng), trace.vocab_size, rng));
  }
  const std::vector<double> weights = zipf_weights(workload.templates, workload.template_zipf);
  const double total = sum(weights);
  for (std::uint32_t r = 0; r < workload.requests; ++r) {
    const std::uint32_t pick = workload.template_zipf == 0.0
                                   ? static_cast<std::uint32_t>(rng.below(workload.templates))
                                   : draw_weighted(weights, total, rng);
    std::vector<TokenId> stream = templates[pick];
    append_random(stream, sample(workload.question_tokens, rng), trace.vocab_size, rng);
    const std::uint32_t prompt = checked_length(stream.size());
    append_random(stream, sample(workload.output_tokens, rng), trace.vocab_size, rng);
    add_request(trace, std::move(stream), prompt);
  }
}

void generate(const RagWorkload& workload, Trace& trace, Rng& rng) {
  if (workload.requests == 0 || workload.documents == 0) {
    throw std::invalid_argument("RagWorkload: requests and documents must be positive");
  }
  if (workload.documents_per_query == 0 || workload.documents_per_query > workload.documents) {
    throw std::invalid_argument("RagWorkload: documents_per_query must be in [1, documents]");
  }
  check_zipf(workload.document_zipf, "RagWorkload: document_zipf must be finite and >= 0");
  const std::vector<TokenId> system =
      random_tokens(workload.system_prompt_tokens, trace.vocab_size, rng);
  std::vector<std::vector<TokenId>> documents;
  documents.reserve(workload.documents);
  for (std::uint32_t d = 0; d < workload.documents; ++d) {
    documents.push_back(
        random_tokens(sample(workload.document_tokens, rng), trace.vocab_size, rng));
  }
  const std::vector<double> popularity = zipf_weights(workload.documents, workload.document_zipf);
  std::vector<double> remaining;
  std::vector<std::uint32_t> picked;
  picked.reserve(workload.documents_per_query);
  for (std::uint32_t r = 0; r < workload.requests; ++r) {
    std::vector<TokenId> stream = system;
    // Weighted sampling without replacement; the draw order is the retrieval order.
    remaining = popularity;
    picked.clear();
    for (std::uint32_t k = 0; k < workload.documents_per_query; ++k) {
      // With a very large exponent the weights of all remaining documents can underflow to zero;
      // the distribution's limit then takes the most popular document not drawn yet.
      const double total = sum(remaining);
      const std::uint32_t pick =
          total > 0.0 ? draw_weighted(remaining, total, rng) : first_not_taken(picked);
      remaining[pick] = 0.0;
      picked.push_back(pick);
      stream.insert(stream.end(), documents[pick].begin(), documents[pick].end());
    }
    append_random(stream, sample(workload.query_tokens, rng), trace.vocab_size, rng);
    const std::uint32_t prompt = checked_length(stream.size());
    append_random(stream, sample(workload.output_tokens, rng), trace.vocab_size, rng);
    add_request(trace, std::move(stream), prompt);
  }
}

void generate(const RandomWorkload& workload, Trace& trace, Rng& rng) {
  if (workload.requests == 0) {
    throw std::invalid_argument("RandomWorkload: requests must be positive");
  }
  for (std::uint32_t r = 0; r < workload.requests; ++r) {
    std::vector<TokenId> stream =
        random_tokens(sample(workload.prompt_tokens, rng), trace.vocab_size, rng);
    const std::uint32_t prompt = checked_length(stream.size());
    append_random(stream, sample(workload.output_tokens, rng), trace.vocab_size, rng);
    add_request(trace, std::move(stream), prompt);
  }
}

}  // namespace

LengthDistribution LengthDistribution::fixed(std::uint32_t value) {
  if (value == 0) {
    throw std::invalid_argument("LengthDistribution::fixed: value must be positive");
  }
  return {Kind::fixed, value, value, 0.0, 0.0};
}

LengthDistribution LengthDistribution::uniform(std::uint32_t min, std::uint32_t max) {
  if (min == 0 || min > max) {
    throw std::invalid_argument("LengthDistribution::uniform: requires 1 <= min <= max");
  }
  return {Kind::uniform, min, max, 0.0, 0.0};
}

LengthDistribution LengthDistribution::lognormal(double median, double sigma, std::uint32_t min,
                                                 std::uint32_t max) {
  if (min == 0 || min > max) {
    throw std::invalid_argument("LengthDistribution::lognormal: requires 1 <= min <= max");
  }
  if (!std::isfinite(median) || !std::isfinite(sigma) || median <= 0.0 || sigma < 0.0) {
    throw std::invalid_argument(
        "LengthDistribution::lognormal: requires finite median > 0 and sigma >= 0");
  }
  return {Kind::lognormal, min, max, median, sigma};
}

std::span<const TokenId> Trace::prompt(const TraceRequest& request) const noexcept {
  return std::span<const TokenId>(sessions[request.session]).first(request.prompt_tokens);
}

std::span<const TokenId> Trace::output(const TraceRequest& request) const noexcept {
  return std::span<const TokenId>(sessions[request.session])
      .subspan(request.prompt_tokens, request.output_tokens);
}

std::uint64_t Trace::total_prompt_tokens() const noexcept {
  std::uint64_t total = 0;
  for (const TraceRequest& request : requests) {
    total += request.prompt_tokens;
  }
  return total;
}

std::uint64_t Trace::total_output_tokens() const noexcept {
  std::uint64_t total = 0;
  for (const TraceRequest& request : requests) {
    total += request.output_tokens;
  }
  return total;
}

Trace generate_trace(const Workload& workload, std::uint64_t seed, std::uint32_t vocab_size) {
  if (vocab_size == 0) {
    throw std::invalid_argument("generate_trace: vocab_size must be positive");
  }
  Trace trace;
  trace.vocab_size = vocab_size;
  Rng rng(seed);
  std::visit([&](const auto& concrete) { generate(concrete, trace, rng); }, workload);
  return trace;
}

}  // namespace radix_kv
