"""Tests for the workload generator and the simulator bindings."""

import threading

import pytest

import radix_kv as rk

ROOMY = rk.SimulatorConfig(num_blocks=1 << 14, block_size=16, max_running=8)


def small_few_shot(seed: int = 1) -> rk.Trace:
    workload = rk.FewShotWorkload(
        requests=60,
        templates=3,
        template_tokens=rk.LengthDistribution.uniform(60, 200),
        question_tokens=rk.LengthDistribution.uniform(3, 20),
        output_tokens=rk.LengthDistribution.fixed(4),
    )
    return rk.generate_trace(workload, seed=seed)


def test_length_distributions() -> None:
    uniform = rk.LengthDistribution.uniform(2, 6)
    assert (uniform.kind, uniform.min, uniform.max) == ("uniform", 2, 6)
    assert repr(uniform) == "LengthDistribution.uniform(2, 6)"
    assert rk.LengthDistribution.fixed(3) == rk.LengthDistribution.fixed(3)
    lognormal = rk.LengthDistribution.lognormal(64.0, 0.5, 8, 512)
    assert (lognormal.kind, lognormal.median, lognormal.sigma) == ("lognormal", 64.0, 0.5)
    with pytest.raises(ValueError, match="min <= max"):
        rk.LengthDistribution.uniform(5, 4)


def test_workloads_take_keyword_arguments_with_defaults() -> None:
    chat = rk.ChatWorkload(sessions=3)
    assert chat.sessions == 3
    assert chat.system_prompt_tokens == rk.ChatWorkload().system_prompt_tokens
    rag = rk.RagWorkload(documents=10, documents_per_query=2)
    rag.document_zipf = 0.5
    assert (rag.documents, rag.documents_per_query, rag.document_zipf) == (10, 2, 0.5)
    assert rk.RandomWorkload(requests=7).requests == 7


def test_generate_trace_is_deterministic() -> None:
    first = small_few_shot(seed=3)
    second = small_few_shot(seed=3)
    assert len(first) == 60
    assert first.requests == second.requests
    assert first.session(0) == second.session(0)
    assert first.total_prompt_tokens > 0
    request = first.requests[0]
    assert first.prompt(request) == first.session(request.session)[: request.prompt_tokens]
    with pytest.raises(IndexError):
        first.session(10_000)


def test_generate_trace_validates_workloads() -> None:
    with pytest.raises(ValueError, match="documents_per_query"):
        rk.generate_trace(rk.RagWorkload(documents=2, documents_per_query=3), seed=1)
    with pytest.raises(ValueError, match="vocab_size"):
        rk.generate_trace(rk.RandomWorkload(), seed=1, vocab_size=0)


@pytest.mark.parametrize("strategy", [rk.Strategy.none, rk.Strategy.hash, rk.Strategy.radix])
def test_simulate_every_strategy(strategy: rk.Strategy) -> None:
    trace = small_few_shot()
    report = rk.simulate(trace, strategy, rk.SimulatorConfig(num_blocks=256))
    assert report.consistent, report.inconsistency
    assert report.strategy == strategy
    assert report.requests_completed == len(trace)
    assert 0.0 <= report.hit_rate <= 1.0
    assert (report.hit_rate == 0.0) == (strategy == rk.Strategy.none)
    assert report.cache_seconds >= 0.0
    assert report.wall_seconds > 0.0


def test_hash_and_radix_agree_without_memory_pressure() -> None:
    trace = small_few_shot()
    hash_report = rk.simulate(trace, rk.Strategy.hash, ROOMY)
    radix_report = rk.simulate(trace, rk.Strategy.radix, ROOMY)
    assert hash_report.evicted_blocks == radix_report.evicted_blocks == 0
    assert hash_report.cached_tokens == radix_report.cached_tokens > 0


def test_custom_traces_are_validated() -> None:
    tokens = list(range(20))
    trace = rk.Trace(
        sessions=[tokens, tokens],
        requests=[rk.TraceRequest(0, 16, 4), rk.TraceRequest(1, 16, 4)],
        vocab_size=32,
    )
    report = rk.simulate(trace, rk.Strategy.radix, rk.SimulatorConfig(block_size=4))
    assert report.cached_tokens == 12  # second request reuses 3 of its 4 prompt blocks
    broken = rk.Trace(sessions=[tokens], requests=[rk.TraceRequest(0, 18, 4)], vocab_size=32)
    with pytest.raises(ValueError, match="inside its session"):
        rk.simulate(broken, rk.Strategy.hash)


def test_turns_of_interleaved_sessions_run_in_order() -> None:
    trace = rk.Trace(
        sessions=[list(range(20)), list(range(32, 44))],
        requests=[rk.TraceRequest(0, 8, 4), rk.TraceRequest(1, 8, 4), rk.TraceRequest(0, 16, 4)],
        vocab_size=64,
    )
    report = rk.simulate(trace, rk.Strategy.hash, rk.SimulatorConfig(num_blocks=64, block_size=4))
    # Session 0's second turn waits for its first: 4 steps, then 4 more.
    assert (report.requests_completed, report.steps, report.consistent) == (3, 8, True)


def test_simulations_run_in_parallel_threads() -> None:
    trace = small_few_shot()
    config = rk.SimulatorConfig(num_blocks=128, block_size=8)
    expected = rk.simulate(trace, rk.Strategy.radix, config).cached_tokens
    results: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        tokens = rk.simulate(trace, rk.Strategy.radix, config).cached_tokens
        with lock:
            results.append(tokens)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [expected] * 4
