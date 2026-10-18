#include "radix_kv/workload.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "radix_kv/types.hpp"

namespace radix_kv {
namespace {

// FNV-1a over every token and request field: a compact fingerprint of a whole trace.
std::uint64_t fingerprint(const Trace& trace) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  const auto mix = [&hash](std::uint64_t value) {
    hash ^= value;
    hash *= 0x100000001b3ULL;
  };
  for (const std::vector<TokenId>& session : trace.sessions) {
    mix(session.size());
    for (const TokenId token : session) {
      mix(token);
    }
  }
  for (const TraceRequest& request : trace.requests) {
    mix(request.session);
    mix(request.prompt_tokens);
    mix(request.output_tokens);
  }
  return hash;
}

// Fingerprint of the golden trace below, recorded when the generator was written.
constexpr std::uint64_t kGoldenFingerprint = 0x8fc01ebddeda90b7ULL;

bool starts_with(std::span<const TokenId> tokens, std::span<const TokenId> prefix) {
  return tokens.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

TEST(LengthDistribution, ValidatesParameters) {
  EXPECT_THROW(LengthDistribution::fixed(0), std::invalid_argument);
  EXPECT_THROW(LengthDistribution::uniform(0, 4), std::invalid_argument);
  EXPECT_THROW(LengthDistribution::uniform(5, 4), std::invalid_argument);
  EXPECT_THROW(LengthDistribution::lognormal(0.0, 1.0, 1, 4), std::invalid_argument);
  EXPECT_THROW(LengthDistribution::lognormal(2.0, -1.0, 1, 4), std::invalid_argument);
  EXPECT_THROW(LengthDistribution::lognormal(std::numeric_limits<double>::infinity(), 1.0, 1, 4),
               std::invalid_argument);
  EXPECT_THROW(LengthDistribution::lognormal(2.0, 1.0, 3, 2), std::invalid_argument);

  const LengthDistribution fixed = LengthDistribution::fixed(7);
  EXPECT_EQ(fixed.kind(), LengthDistribution::Kind::fixed);
  EXPECT_EQ(fixed.min(), 7U);
  EXPECT_EQ(fixed.max(), 7U);
  const LengthDistribution lognormal = LengthDistribution::lognormal(64, 0.5, 8, 512);
  EXPECT_EQ(lognormal.kind(), LengthDistribution::Kind::lognormal);
  EXPECT_DOUBLE_EQ(lognormal.median(), 64.0);
  EXPECT_DOUBLE_EQ(lognormal.sigma(), 0.5);
}

TEST(GenerateTrace, IsDeterministicPerSeed) {
  const Workload workload = RagWorkload{.requests = 50};
  EXPECT_EQ(fingerprint(generate_trace(workload, 7)), fingerprint(generate_trace(workload, 7)));
  EXPECT_NE(fingerprint(generate_trace(workload, 7)), fingerprint(generate_trace(workload, 8)));
}

TEST(GenerateTrace, IntegerOnlyTracesMatchAGoldenFingerprint) {
  // Uniform lengths and uniform popularity use integer sampling only, so this trace is identical
  // on every platform and standard library. A change here changes every documented result.
  const Workload workload = FewShotWorkload{.requests = 64,
                                            .templates = 3,
                                            .template_tokens = LengthDistribution::uniform(8, 64),
                                            .template_zipf = 0.0,
                                            .question_tokens = LengthDistribution::uniform(1, 9),
                                            .output_tokens = LengthDistribution::fixed(5)};
  EXPECT_EQ(fingerprint(generate_trace(workload, 42, 1000)), kGoldenFingerprint);
}

TEST(GenerateTrace, ChatTurnsExtendTheWholeConversation) {
  const ChatWorkload chat{.sessions = 20,
                          .system_prompt_tokens = 32,
                          .turns = LengthDistribution::uniform(1, 4),
                          .user_tokens = LengthDistribution::uniform(1, 16),
                          .output_tokens = LengthDistribution::uniform(1, 16)};
  const Trace trace = generate_trace(chat, 3);
  ASSERT_EQ(trace.sessions.size(), 20U);
  const std::span<const TokenId> system = std::span<const TokenId>(trace.sessions[0]).first(32);
  for (std::size_t i = 0; i < trace.requests.size(); ++i) {
    const TraceRequest& request = trace.requests[i];
    EXPECT_TRUE(starts_with(trace.prompt(request), system));
    EXPECT_GE(request.output_tokens, 1U);
    if (i + 1 < trace.requests.size() && trace.requests[i + 1].session == request.session) {
      const TraceRequest& next = trace.requests[i + 1];
      // The next turn re-sends this prompt and this output, then a new user message.
      EXPECT_GT(next.prompt_tokens, request.prompt_tokens + request.output_tokens);
      EXPECT_TRUE(starts_with(trace.prompt(next), trace.prompt(request)));
    }
  }
}

TEST(GenerateTrace, FewShotPromptsStartWithATemplate) {
  const FewShotWorkload few_shot{.requests = 100,
                                 .templates = 2,
                                 .template_tokens = LengthDistribution::fixed(40),
                                 .template_zipf = 1.0,
                                 .question_tokens = LengthDistribution::uniform(1, 8),
                                 .output_tokens = LengthDistribution::fixed(3)};
  const Trace trace = generate_trace(few_shot, 11);
  ASSERT_EQ(trace.requests.size(), 100U);
  std::vector<std::vector<TokenId>> templates;
  for (const TraceRequest& request : trace.requests) {
    const std::span<const TokenId> head = trace.prompt(request).first(40);
    if (std::ranges::none_of(templates,
                             [&](const auto& known) { return std::ranges::equal(known, head); })) {
      templates.emplace_back(head.begin(), head.end());
    }
    EXPECT_LE(request.prompt_tokens, 48U);
    EXPECT_EQ(request.output_tokens, 3U);
  }
  EXPECT_EQ(templates.size(), 2U);
}

TEST(GenerateTrace, RagPromptsShareTheSystemPromptAndPoolDocuments) {
  const RagWorkload rag{.requests = 60,
                        .system_prompt_tokens = 16,
                        .documents = 4,
                        .document_tokens = LengthDistribution::fixed(10),
                        .documents_per_query = 4,
                        .document_zipf = 2.0,
                        .query_tokens = LengthDistribution::fixed(2),
                        .output_tokens = LengthDistribution::fixed(1)};
  const Trace trace = generate_trace(rag, 5);
  const std::span<const TokenId> system = std::span<const TokenId>(trace.sessions[0]).first(16);
  for (const TraceRequest& request : trace.requests) {
    EXPECT_EQ(request.prompt_tokens, 16U + (4U * 10U) + 2U);
    EXPECT_TRUE(starts_with(trace.prompt(request), system));
    // All four documents appear exactly once: every 10-token chunk is distinct.
    std::vector<std::vector<TokenId>> chunks;
    for (std::size_t d = 0; d < 4; ++d) {
      const auto chunk = trace.prompt(request).subspan(16 + (d * 10), 10);
      chunks.emplace_back(chunk.begin(), chunk.end());
    }
    std::ranges::sort(chunks);
    EXPECT_EQ(std::ranges::adjacent_find(chunks), chunks.end());
  }
}

TEST(GenerateTrace, RagDocumentsStayDistinctWhenZipfWeightsUnderflow) {
  // 1 / rank^2000 is 0.0 in double precision for every rank above 1.
  const RagWorkload rag{.requests = 5,
                        .system_prompt_tokens = 0,
                        .documents = 5,
                        .document_tokens = LengthDistribution::fixed(4),
                        .documents_per_query = 3,
                        .document_zipf = 2000.0,
                        .query_tokens = LengthDistribution::fixed(1),
                        .output_tokens = LengthDistribution::fixed(1)};
  const Trace trace = generate_trace(rag, 7);
  const std::span<const TokenId> first = trace.prompt(trace.requests.front()).first(12);
  for (const TraceRequest& request : trace.requests) {
    std::vector<std::vector<TokenId>> chunks;
    for (std::size_t d = 0; d < 3; ++d) {
      const auto chunk = trace.prompt(request).subspan(d * 4, 4);
      chunks.emplace_back(chunk.begin(), chunk.end());
    }
    std::ranges::sort(chunks);
    EXPECT_EQ(std::ranges::adjacent_find(chunks), chunks.end());
    // In the limit the three most popular documents are always retrieved, in popularity order.
    EXPECT_TRUE(starts_with(trace.prompt(request), first));
  }
}

TEST(GenerateTrace, RandomPromptsRespectBoundsAndVocabulary) {
  const RandomWorkload random{.requests = 200,
                              .prompt_tokens = LengthDistribution::uniform(3, 9),
                              .output_tokens = LengthDistribution::lognormal(4, 2.0, 2, 6)};
  const Trace trace = generate_trace(random, 9, 50);
  EXPECT_EQ(trace.vocab_size, 50U);
  for (const TraceRequest& request : trace.requests) {
    EXPECT_GE(request.prompt_tokens, 3U);
    EXPECT_LE(request.prompt_tokens, 9U);
    EXPECT_GE(request.output_tokens, 2U);
    EXPECT_LE(request.output_tokens, 6U);
  }
  for (const std::vector<TokenId>& session : trace.sessions) {
    EXPECT_TRUE(std::ranges::all_of(session, [](TokenId token) { return token < 50; }));
  }
  EXPECT_EQ(trace.total_prompt_tokens() + trace.total_output_tokens(), [&] {
    std::uint64_t total = 0;
    for (const auto& session : trace.sessions) {
      total += session.size();
    }
    return total;
  }());
}

TEST(GenerateTrace, RejectsEmptyOrInvalidWorkloads) {
  const auto generate = [](const Workload& workload, std::uint32_t vocab_size = 32000) {
    static_cast<void>(generate_trace(workload, 1, vocab_size));
  };
  EXPECT_THROW(generate(ChatWorkload{.sessions = 0}), std::invalid_argument);
  EXPECT_THROW(generate(FewShotWorkload{.requests = 0}), std::invalid_argument);
  EXPECT_THROW(generate(FewShotWorkload{.templates = 0}), std::invalid_argument);
  EXPECT_THROW(generate(FewShotWorkload{.template_zipf = -1.0}), std::invalid_argument);
  EXPECT_THROW(generate(RagWorkload{.documents = 0}), std::invalid_argument);
  EXPECT_THROW(generate(RagWorkload{.documents = 2, .documents_per_query = 3}),
               std::invalid_argument);
  EXPECT_THROW(generate(RagWorkload{.documents_per_query = 0}), std::invalid_argument);
  EXPECT_THROW(generate(RagWorkload{.document_zipf = std::numeric_limits<double>::quiet_NaN()}),
               std::invalid_argument);
  EXPECT_THROW(generate(RandomWorkload{.requests = 0}), std::invalid_argument);
  EXPECT_THROW(generate(RandomWorkload{}, 0), std::invalid_argument);
}

}  // namespace
}  // namespace radix_kv
