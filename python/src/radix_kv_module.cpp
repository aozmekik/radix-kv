// nanobind bindings for radix-kv. Errors become Python exceptions; everything else maps 1:1 onto
// the C++ API (see python/radix_kv/__init__.py for the public Python surface).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "radix_kv/radix_kv.hpp"

namespace nb = nanobind;
using namespace nb::literals;  // NOLINT(google-build-using-namespace): the _a argument literal

namespace radix_kv::python {
namespace {

class RadixKvError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class OutOfBlocksError : public RadixKvError {
 public:
  using RadixKvError::RadixKvError;
};

class InvalidHandleError : public RadixKvError {
 public:
  using RadixKvError::RadixKvError;
};

[[noreturn]] void raise(Errc error) {
  // Plain concatenation: std::format is unavailable before macOS 13.3, which wheels may target.
  std::string message(describe(error));
  message.append(" (").append(to_string(error)).append(")");
  switch (error) {
    case Errc::out_of_blocks:
      throw OutOfBlocksError(message);
    case Errc::invalid_handle:
    case Errc::not_locked:
      throw InvalidHandleError(message);
    case Errc::ref_count_overflow:
      throw std::overflow_error(message);
    case Errc::invalid_block:
    case Errc::block_not_allocated:
    case Errc::invalid_argument:
      throw nb::value_error(message.c_str());
  }
  throw RadixKvError(message);
}

template <class T>
  requires(!std::is_void_v<T>)
T unwrap(Result<T>&& result) {
  if (!result) {
    raise(result.error());
  }
  return *std::move(result);
}

void unwrap(const Result<void>& result) {
  if (!result) {
    raise(result.error());
  }
}

std::vector<TokenId> to_vector(std::span<const TokenId> tokens) {
  return {tokens.begin(), tokens.end()};
}

/// A Sequence plus a strong reference to the Python allocator object it draws from, so the pool
/// outlives every sequence (including forks) no matter in which order Python frees them.
struct PySequence {
  nb::object owner;
  Sequence sequence;
};

PySequence make_sequence(BlockAllocator& allocator) {
  return PySequence{.owner = nb::borrow(nb::find(allocator)), .sequence = Sequence(allocator)};
}

template <class Cache>
void bind_cache(nb::module_& m, const char* name, const char* doc) {
  nb::class_<Cache>(m, name, doc)
      .def(nb::init<BlockAllocator&>(), "allocator"_a, nb::keep_alive<1, 2>(),
           "Creates an empty cache over the allocator's pool.")
      .def(
          "match_prefix",
          [](Cache& cache, const std::vector<TokenId>& tokens) {
            return cache.match_prefix(tokens);
          },
          "tokens"_a,
          "Longest cached prefix of tokens made of whole blocks; marks it recently used. Takes no "
          "references and no locks.")
      .def(
          "insert",
          [](Cache& cache, const std::vector<TokenId>& tokens, const std::vector<BlockId>& blocks) {
            return unwrap(cache.insert(tokens, blocks));
          },
          "tokens"_a, "blocks"_a,
          "Caches the full blocks of (tokens, blocks), taking a reference to each newly cached "
          "block. The blocks must belong to this cache's allocator. Raises ValueError for invalid "
          "block ids.")
      .def(
          "insert",
          [](Cache& cache, const PySequence& sequence) {
            if (&sequence.sequence.allocator() != &cache.allocator()) {
              throw nb::value_error(
                  "the sequence draws from a different BlockAllocator than this cache");
            }
            return unwrap(
                cache.insert(sequence.sequence.tokens(), sequence.sequence.block_table()));
          },
          "sequence"_a,
          "Caches the full blocks of a sequence. Raises ValueError if the sequence uses another "
          "allocator.")
      .def(
          "lock", [](Cache& cache, NodeHandle node) { unwrap(cache.lock(node)); }, "node"_a,
          "Pins the path from the root to node against eviction.")
      .def(
          "unlock", [](Cache& cache, NodeHandle node) { unwrap(cache.unlock(node)); }, "node"_a,
          "Releases one lock taken with lock(node). Raises InvalidHandleError if none is held.")
      .def("evict", &Cache::evict, "num_blocks"_a,
           "Evicts least-recently-used unlocked entries until num_blocks blocks were released or "
           "nothing is evictable; returns the number released.")
      .def_prop_ro("root", &Cache::root, "Handle of the root (the empty prefix).")
      .def("contains", &Cache::contains, "node"_a, "True when node refers to a live node.")
      .def("holds_block", &Cache::holds_block, "block"_a,
           "True when the cache holds a reference to block.")
      .def_prop_ro("stats", &Cache::stats, "Current counters.")
      .def_prop_ro("allocator", &Cache::allocator, nb::rv_policy::reference_internal,
                   "The allocator this cache was created over.")
      .def("check_invariants", &Cache::check_invariants,
           "Full structural check; returns a description of the first violation, or None.");
}

void bind_core(nb::module_& m) {
  nb::class_<NodeHandle>(m, "NodeHandle", "Opaque, generation-checked reference to a cache node.")
      .def_ro("index", &NodeHandle::index)
      .def_ro("generation", &NodeHandle::generation)
      .def(
          "__eq__", [](const NodeHandle& a, const NodeHandle& b) { return a == b; },
          nb::is_operator())
      .def("__hash__",
           [](const NodeHandle& h) {
             return (static_cast<std::uint64_t>(h.index) << 32U) | h.generation;
           })
      .def("__repr__", [](const NodeHandle& h) {
        return nb::str("NodeHandle(index={}, generation={})").format(h.index, h.generation);
      });

  nb::class_<CopyOp>(m, "CopyOp", "Copy-on-write instruction: copy block src into block dst.")
      .def_ro("src", &CopyOp::src)
      .def_ro("dst", &CopyOp::dst)
      .def(
          "__eq__", [](const CopyOp& a, const CopyOp& b) { return a == b; }, nb::is_operator())
      .def("__repr__", [](const CopyOp& op) {
        return nb::str("CopyOp(src={}, dst={})").format(op.src, op.dst);
      });

  nb::class_<AppendResult>(m, "AppendResult", "What Sequence.append/extend did.")
      .def_ro("new_blocks", &AppendResult::new_blocks)
      .def_ro("copy", &AppendResult::copy)
      .def("__repr__", [](const AppendResult& r) {
        if (const auto& copy = r.copy) {
          return nb::str("AppendResult(new_blocks={}, copy=CopyOp(src={}, dst={}))")
              .format(r.new_blocks, copy->src, copy->dst);
        }
        return nb::str("AppendResult(new_blocks={}, copy=None)").format(r.new_blocks);
      });

  nb::class_<BlockAllocator>(m, "BlockAllocator",
                             "Fixed pool of reference-counted KV-cache blocks (PagedAttention).")
      .def(nb::init<std::uint32_t, std::uint32_t>(), "num_blocks"_a, "block_size"_a,
           "Creates a pool with every block free. Raises ValueError for zero sizes.")
      .def(
          "allocate", [](BlockAllocator& a) { return unwrap(a.allocate()); },
          "Takes one free block. Raises OutOfBlocksError when the pool is exhausted.")
      .def(
          "retain", [](BlockAllocator& a, BlockId block) { unwrap(a.retain(block)); }, "block"_a,
          "Adds a reference to an allocated block.")
      .def(
          "release", [](BlockAllocator& a, BlockId block) { return unwrap(a.release(block)); },
          "block"_a, "Drops a reference; returns the remaining count (0 means freed).")
      .def("ref_count", &BlockAllocator::ref_count, "block"_a,
           "Reference count of block; 0 for free blocks and ids outside the pool.")
      .def("is_allocated", &BlockAllocator::is_allocated, "block"_a)
      .def("blocks_for", &BlockAllocator::blocks_for, "num_tokens"_a,
           "Blocks needed to hold num_tokens tokens.")
      .def("reset_peak", &BlockAllocator::reset_peak)
      .def_prop_ro("num_blocks", &BlockAllocator::num_blocks)
      .def_prop_ro("block_size", &BlockAllocator::block_size)
      .def_prop_ro("num_free", &BlockAllocator::num_free)
      .def_prop_ro("num_used", &BlockAllocator::num_used)
      .def_prop_ro("peak_used", &BlockAllocator::peak_used);

  nb::class_<PySequence>(m, "Sequence",
                         "Tokens and block table of one sequence, with copy-on-write fork().")
      .def(
          "__init__",
          [](PySequence* self, BlockAllocator& allocator) {
            new (self) PySequence(make_sequence(allocator));
          },
          "allocator"_a, "Creates an empty sequence drawing blocks from allocator.")
      .def(
          "append",
          [](PySequence& self, TokenId token) { return unwrap(self.sequence.append(token)); },
          "token"_a, "Appends one token (a decode step). Raises OutOfBlocksError.")
      .def(
          "extend",
          [](PySequence& self, const std::vector<TokenId>& tokens) {
            return unwrap(self.sequence.append(tokens));
          },
          "tokens"_a, "Appends tokens, all or nothing. Raises OutOfBlocksError.")
      .def(
          "fork",
          [](const PySequence& self) {
            return PySequence{.owner = self.owner, .sequence = unwrap(self.sequence.fork())};
          },
          "Returns a new sequence sharing every block with this one.")
      .def(
          "truncate",
          [](PySequence& self, std::size_t num_tokens) {
            unwrap(self.sequence.truncate(num_tokens));
          },
          "num_tokens"_a, "Keeps the first num_tokens tokens.")
      .def(
          "release", [](PySequence& self) { self.sequence.release(); },
          "Releases every block and clears the tokens.")
      .def(
          "assign_prefix",
          [](PySequence& self, const std::vector<TokenId>& tokens,
             const std::vector<BlockId>& blocks) {
            unwrap(self.sequence.assign_prefix(tokens, blocks));
          },
          "tokens"_a, "blocks"_a,
          "Seeds an empty sequence with full cached blocks, which must belong to this sequence's "
          "allocator (a PrefixMatch of a cache over the same pool).")
      .def_prop_ro(
          "tokens", [](const PySequence& self) { return to_vector(self.sequence.tokens()); },
          "Copy of the token ids.")
      .def_prop_ro(
          "block_table",
          [](const PySequence& self) {
            const auto table = self.sequence.block_table();
            return std::vector<BlockId>(table.begin(), table.end());
          },
          "Copy of the block table.")
      .def_prop_ro("num_blocks", [](const PySequence& self) { return self.sequence.num_blocks(); })
      .def_prop_ro("num_full_blocks",
                   [](const PySequence& self) { return self.sequence.num_full_blocks(); })
      .def("__len__", [](const PySequence& self) { return self.sequence.size(); });

  nb::class_<PrefixMatch>(m, "PrefixMatch", "Longest cached block-aligned prefix of a query.")
      .def_ro("num_tokens", &PrefixMatch::num_tokens)
      .def_ro("blocks", &PrefixMatch::blocks)
      .def_ro("node", &PrefixMatch::node);

  nb::class_<InsertResult>(m, "InsertResult")
      .def_ro("new_blocks", &InsertResult::new_blocks)
      .def_ro("node", &InsertResult::node);

  nb::class_<CacheStats>(m, "CacheStats", "Point-in-time counters of a prefix cache.")
      .def_ro("cached_blocks", &CacheStats::cached_blocks)
      .def_ro("locked_blocks", &CacheStats::locked_blocks)
      .def_ro("evictable_blocks", &CacheStats::evictable_blocks)
      .def_ro("nodes", &CacheStats::nodes)
      .def_ro("evicted_blocks", &CacheStats::evicted_blocks);

  bind_cache<HashPrefixCache>(
      m, "HashPrefixCache",
      "vLLM-style prefix cache: chained block hashes, a hash index and LRU eviction.");
  bind_cache<RadixPrefixCache>(
      m, "RadixPrefixCache",
      "SGLang-style prefix cache: a radix tree over blocks with LRU eviction of whole leaves.");
}

void bind_workloads(nb::module_& m) {
  nb::class_<LengthDistribution>(m, "LengthDistribution", "Distribution of a positive length.")
      .def_static("fixed", &LengthDistribution::fixed, "value"_a)
      .def_static("uniform", &LengthDistribution::uniform, "min"_a, "max"_a)
      .def_static("lognormal", &LengthDistribution::lognormal, "median"_a, "sigma"_a, "min"_a,
                  "max"_a)
      .def_prop_ro("kind",
                   [](const LengthDistribution& d) {
                     switch (d.kind()) {
                       case LengthDistribution::Kind::fixed:
                         return "fixed";
                       case LengthDistribution::Kind::uniform:
                         return "uniform";
                       case LengthDistribution::Kind::lognormal:
                         return "lognormal";
                     }
                     return "unknown";
                   })
      .def_prop_ro("min", &LengthDistribution::min)
      .def_prop_ro("max", &LengthDistribution::max)
      .def_prop_ro("median", &LengthDistribution::median)
      .def_prop_ro("sigma", &LengthDistribution::sigma)
      .def(
          "__eq__", [](const LengthDistribution& a, const LengthDistribution& b) { return a == b; },
          nb::is_operator())
      .def("__repr__", [](const LengthDistribution& d) {
        switch (d.kind()) {
          case LengthDistribution::Kind::fixed:
            return nb::str("LengthDistribution.fixed({})").format(d.min());
          case LengthDistribution::Kind::uniform:
            return nb::str("LengthDistribution.uniform({}, {})").format(d.min(), d.max());
          case LengthDistribution::Kind::lognormal:
            return nb::str("LengthDistribution.lognormal({}, {}, {}, {})")
                .format(d.median(), d.sigma(), d.min(), d.max());
        }
        return nb::str("LengthDistribution(?)");
      });

  const ChatWorkload chat;
  nb::class_<ChatWorkload>(m, "ChatWorkload",
                           "Multi-turn chat sharing one system prompt (synthetic).")
      .def(
          "__init__",
          [](ChatWorkload* self, std::uint32_t sessions, std::uint32_t system_prompt_tokens,
             const LengthDistribution& turns, const LengthDistribution& user_tokens,
             const LengthDistribution& output_tokens) {
            new (self) ChatWorkload{.sessions = sessions,
                                    .system_prompt_tokens = system_prompt_tokens,
                                    .turns = turns,
                                    .user_tokens = user_tokens,
                                    .output_tokens = output_tokens};
          },
          nb::kw_only(), "sessions"_a = chat.sessions,
          "system_prompt_tokens"_a = chat.system_prompt_tokens, "turns"_a = chat.turns,
          "user_tokens"_a = chat.user_tokens, "output_tokens"_a = chat.output_tokens)
      .def_rw("sessions", &ChatWorkload::sessions)
      .def_rw("system_prompt_tokens", &ChatWorkload::system_prompt_tokens)
      .def_rw("turns", &ChatWorkload::turns)
      .def_rw("user_tokens", &ChatWorkload::user_tokens)
      .def_rw("output_tokens", &ChatWorkload::output_tokens);

  const FewShotWorkload few_shot;
  nb::class_<FewShotWorkload>(m, "FewShotWorkload",
                              "Long few-shot templates followed by short questions (synthetic).")
      .def(
          "__init__",
          [](FewShotWorkload* self, std::uint32_t requests, std::uint32_t templates,
             const LengthDistribution& template_tokens, double template_zipf,
             const LengthDistribution& question_tokens, const LengthDistribution& output_tokens) {
            new (self) FewShotWorkload{.requests = requests,
                                       .templates = templates,
                                       .template_tokens = template_tokens,
                                       .template_zipf = template_zipf,
                                       .question_tokens = question_tokens,
                                       .output_tokens = output_tokens};
          },
          nb::kw_only(), "requests"_a = few_shot.requests, "templates"_a = few_shot.templates,
          "template_tokens"_a = few_shot.template_tokens,
          "template_zipf"_a = few_shot.template_zipf,
          "question_tokens"_a = few_shot.question_tokens,
          "output_tokens"_a = few_shot.output_tokens)
      .def_rw("requests", &FewShotWorkload::requests)
      .def_rw("templates", &FewShotWorkload::templates)
      .def_rw("template_tokens", &FewShotWorkload::template_tokens)
      .def_rw("template_zipf", &FewShotWorkload::template_zipf)
      .def_rw("question_tokens", &FewShotWorkload::question_tokens)
      .def_rw("output_tokens", &FewShotWorkload::output_tokens);

  const RagWorkload rag;
  nb::class_<RagWorkload>(m, "RagWorkload",
                          "System prompt, retrieved documents and a query (synthetic).")
      .def(
          "__init__",
          [](RagWorkload* self, std::uint32_t requests, std::uint32_t system_prompt_tokens,
             std::uint32_t documents, const LengthDistribution& document_tokens,
             std::uint32_t documents_per_query, double document_zipf,
             const LengthDistribution& query_tokens, const LengthDistribution& output_tokens) {
            new (self) RagWorkload{.requests = requests,
                                   .system_prompt_tokens = system_prompt_tokens,
                                   .documents = documents,
                                   .document_tokens = document_tokens,
                                   .documents_per_query = documents_per_query,
                                   .document_zipf = document_zipf,
                                   .query_tokens = query_tokens,
                                   .output_tokens = output_tokens};
          },
          nb::kw_only(), "requests"_a = rag.requests,
          "system_prompt_tokens"_a = rag.system_prompt_tokens, "documents"_a = rag.documents,
          "document_tokens"_a = rag.document_tokens,
          "documents_per_query"_a = rag.documents_per_query, "document_zipf"_a = rag.document_zipf,
          "query_tokens"_a = rag.query_tokens, "output_tokens"_a = rag.output_tokens)
      .def_rw("requests", &RagWorkload::requests)
      .def_rw("system_prompt_tokens", &RagWorkload::system_prompt_tokens)
      .def_rw("documents", &RagWorkload::documents)
      .def_rw("document_tokens", &RagWorkload::document_tokens)
      .def_rw("documents_per_query", &RagWorkload::documents_per_query)
      .def_rw("document_zipf", &RagWorkload::document_zipf)
      .def_rw("query_tokens", &RagWorkload::query_tokens)
      .def_rw("output_tokens", &RagWorkload::output_tokens);

  const RandomWorkload random;
  nb::class_<RandomWorkload>(m, "RandomWorkload",
                             "Independent random prompts with no shared prefixes (synthetic).")
      .def(
          "__init__",
          [](RandomWorkload* self, std::uint32_t requests, const LengthDistribution& prompt_tokens,
             const LengthDistribution& output_tokens) {
            new (self) RandomWorkload{.requests = requests,
                                      .prompt_tokens = prompt_tokens,
                                      .output_tokens = output_tokens};
          },
          nb::kw_only(), "requests"_a = random.requests, "prompt_tokens"_a = random.prompt_tokens,
          "output_tokens"_a = random.output_tokens)
      .def_rw("requests", &RandomWorkload::requests)
      .def_rw("prompt_tokens", &RandomWorkload::prompt_tokens)
      .def_rw("output_tokens", &RandomWorkload::output_tokens);

  nb::class_<TraceRequest>(m, "TraceRequest",
                           "A request reading prompt_tokens, then output_tokens of a session.")
      .def(
          "__init__",
          [](TraceRequest* self, std::uint32_t session, std::uint32_t prompt_tokens,
             std::uint32_t output_tokens) {
            new (self) TraceRequest{
                .session = session, .prompt_tokens = prompt_tokens, .output_tokens = output_tokens};
          },
          "session"_a, "prompt_tokens"_a, "output_tokens"_a)
      .def_ro("session", &TraceRequest::session)
      .def_ro("prompt_tokens", &TraceRequest::prompt_tokens)
      .def_ro("output_tokens", &TraceRequest::output_tokens)
      .def(
          "__eq__", [](const TraceRequest& a, const TraceRequest& b) { return a == b; },
          nb::is_operator())
      .def("__repr__", [](const TraceRequest& r) {
        return nb::str("TraceRequest(session={}, prompt_tokens={}, output_tokens={})")
            .format(r.session, r.prompt_tokens, r.output_tokens);
      });

  nb::class_<Trace>(m, "Trace", "Token streams plus the requests that read them.")
      .def(
          "__init__",
          [](Trace* self, std::vector<std::vector<TokenId>> sessions,
             std::vector<TraceRequest> requests, std::uint32_t vocab_size) {
            new (self) Trace{.sessions = std::move(sessions),
                             .requests = std::move(requests),
                             .vocab_size = vocab_size};
          },
          "sessions"_a, "requests"_a, "vocab_size"_a,
          "Builds a trace from your own token streams (validated by simulate()).")
      .def_ro("requests", &Trace::requests)
      .def_ro("vocab_size", &Trace::vocab_size)
      .def_prop_ro("num_sessions", [](const Trace& t) { return t.sessions.size(); })
      .def(
          "session",
          [](const Trace& t, std::size_t index) {
            if (index >= t.sessions.size()) {
              throw nb::index_error("session index out of range");
            }
            return t.sessions[index];
          },
          "index"_a, "Token stream of a session.")
      .def(
          "prompt",
          [](const Trace& t, const TraceRequest& request) {
            if (request.session >= t.sessions.size() ||
                std::size_t{request.prompt_tokens} > t.sessions[request.session].size()) {
              throw nb::index_error("request is outside this trace");
            }
            return to_vector(t.prompt(request));
          },
          "request"_a, "Prompt tokens of a request.")
      .def_prop_ro("total_prompt_tokens", &Trace::total_prompt_tokens)
      .def_prop_ro("total_output_tokens", &Trace::total_output_tokens)
      .def("__len__", [](const Trace& t) { return t.requests.size(); });

  m.def("generate_trace", &generate_trace, "workload"_a, "seed"_a, "vocab_size"_a = 32000,
        "Generates a deterministic synthetic trace.");
}

void bind_simulator(nb::module_& m) {
  nb::enum_<Strategy>(m, "Strategy", "Prefix-caching strategy of a simulation.")
      .value("none", Strategy::none, "No prefix caching (baseline).")
      .value("hash", Strategy::hash, "HashPrefixCache.")
      .value("radix", Strategy::radix, "RadixPrefixCache.");

  const SimulatorConfig defaults;
  nb::class_<SimulatorConfig>(m, "SimulatorConfig", "Scheduler and memory parameters.")
      .def(
          "__init__",
          [](SimulatorConfig* self, std::uint32_t num_blocks, std::uint32_t block_size,
             std::uint32_t max_running, std::uint32_t samples_per_request, bool check_invariants) {
            new (self) SimulatorConfig{.num_blocks = num_blocks,
                                       .block_size = block_size,
                                       .max_running = max_running,
                                       .samples_per_request = samples_per_request,
                                       .check_invariants = check_invariants};
          },
          nb::kw_only(), "num_blocks"_a = defaults.num_blocks, "block_size"_a = defaults.block_size,
          "max_running"_a = defaults.max_running,
          "samples_per_request"_a = defaults.samples_per_request,
          "check_invariants"_a = defaults.check_invariants)
      .def_rw("num_blocks", &SimulatorConfig::num_blocks)
      .def_rw("block_size", &SimulatorConfig::block_size)
      .def_rw("max_running", &SimulatorConfig::max_running)
      .def_rw("samples_per_request", &SimulatorConfig::samples_per_request)
      .def_rw("check_invariants", &SimulatorConfig::check_invariants);

  using Seconds = std::chrono::duration<double>;
  nb::class_<SimulationReport>(m, "SimulationReport",
                               "Counters of a simulation over a (synthetic) trace.")
      .def_ro("strategy", &SimulationReport::strategy)
      .def_ro("requests_completed", &SimulationReport::requests_completed)
      .def_ro("requests_rejected", &SimulationReport::requests_rejected)
      .def_ro("preemptions", &SimulationReport::preemptions)
      .def_ro("steps", &SimulationReport::steps)
      .def_ro("prompt_tokens", &SimulationReport::prompt_tokens)
      .def_ro("cached_tokens", &SimulationReport::cached_tokens)
      .def_ro("recompute_prompt_tokens", &SimulationReport::recompute_prompt_tokens)
      .def_ro("recompute_cached_tokens", &SimulationReport::recompute_cached_tokens)
      .def_ro("decode_tokens", &SimulationReport::decode_tokens)
      .def_ro("evicted_blocks", &SimulationReport::evicted_blocks)
      .def_ro("cow_copies", &SimulationReport::cow_copies)
      .def_ro("peak_blocks_used", &SimulationReport::peak_blocks_used)
      .def_ro("peak_blocks_active", &SimulationReport::peak_blocks_active)
      .def_ro("cache_ops", &SimulationReport::cache_ops)
      .def_ro("consistent", &SimulationReport::consistent)
      .def_ro("inconsistency", &SimulationReport::inconsistency)
      .def_prop_ro("hit_rate", &SimulationReport::hit_rate)
      .def_prop_ro("cache_ops_per_second", &SimulationReport::cache_ops_per_second)
      .def_prop_ro("cache_seconds",
                   [](const SimulationReport& r) { return Seconds(r.cache_time).count(); })
      .def_prop_ro("wall_seconds",
                   [](const SimulationReport& r) { return Seconds(r.wall_time).count(); });

  m.def(
      "simulate",
      [](const Trace& trace, Strategy strategy, const SimulatorConfig& config) {
        return simulate(trace, strategy, config);
      },
      "trace"_a, "strategy"_a, "config"_a = defaults, nb::call_guard<nb::gil_scoped_release>(),
      "Replays a trace through the scheduler simulation (releases the GIL).");
}

}  // namespace
}  // namespace radix_kv::python

NB_MODULE(_core, m) {
  namespace py = radix_kv::python;
  m.doc() = "KV-cache block manager and prefix caches for LLM serving (C++ core).";
  m.attr("__version__") = std::string(radix_kv::kVersion);

  // Registering an exception type creates the Python class and the C++ -> Python translation.
  const nb::exception<py::RadixKvError> base(m, "RadixKvError", PyExc_RuntimeError);
  const nb::exception<py::OutOfBlocksError> out_of_blocks(m, "OutOfBlocksError", base);
  const nb::exception<py::InvalidHandleError> invalid_handle(m, "InvalidHandleError", base);

  py::bind_core(m);
  py::bind_workloads(m);
  py::bind_simulator(m);
}
