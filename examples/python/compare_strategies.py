"""Compare prefix-caching strategies on synthetic workloads across pool sizes.

All numbers come from synthetic traces (see radix_kv.generate_trace).
"""

import radix_kv as rk

WORKLOADS: dict[str, rk.ChatWorkload | rk.FewShotWorkload | rk.RagWorkload] = {
    "chat": rk.ChatWorkload(sessions=100),
    "few-shot": rk.FewShotWorkload(requests=300),
    "rag": rk.RagWorkload(requests=300),
}
STRATEGIES = [rk.Strategy.none, rk.Strategy.hash, rk.Strategy.radix]


def main() -> None:
    print(f"{'workload':<9} {'blocks':>6} {'strategy':<8} {'hit rate':>8} {'evicted':>8}")
    for name, workload in WORKLOADS.items():
        trace = rk.generate_trace(workload, seed=1)
        for num_blocks in (1024, 4096):
            config = rk.SimulatorConfig(num_blocks=num_blocks, block_size=16, max_running=16)
            for strategy in STRATEGIES:
                report = rk.simulate(trace, strategy, config)
                if not report.consistent:
                    raise RuntimeError(report.inconsistency)
                print(
                    f"{name:<9} {num_blocks:>6} {strategy.name:<8} "
                    f"{report.hit_rate:>8.1%} {report.evicted_blocks:>8}"
                )


if __name__ == "__main__":
    main()
