"""Compare admission policies under dependency-aware replay.

Every policy runs inside the same engine, on the same workload, with the same
slowdown model, so the only variable is which ready job gets the next free
slot. Serial is the same engine with parallel_cap forced to 1, which is the
honest baseline: it is what the repo's serial and time_aware policies both
reduce to, since their chooser only ever sees the anchor job.

Usage:
    python -m scheduler_benchmark_test.compare_dag_schedulers
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler_benchmark_test.dag_scheduler import DagJob, simulate_dag
from scheduler_benchmark_test.run_cassava_scheduler_test import (
    CUDA_CONTEXT_FLOOR_MB,
    MEMORY_BUDGET_MB,
    NOOP_SECONDS,
    PAIR_SLOWDOWN,
    PARALLEL_CAP,
    TRACE,
    load_rows,
    slowdown_at,
)

RECORDS = Path(__file__).resolve().parent.parent / "records"


def to_dag_jobs(rows: list[dict]) -> list[DagJob]:
    """Convert trace rows into dependency-aware job records."""
    jobs = []
    for row in rows:
        recorded = float(row.get("exec_duration_s") or 0.0)
        if recorded <= NOOP_SECONDS:
            continue
        jobs.append(
            DagJob(
                job_id=str(row["job_id"]),
                node_id=str(row["node_id"]),
                parent_id=row.get("parent_node_id"),
                gen_seconds=float(row.get("gen_duration_s") or 0.0),
                solo=recorded / slowdown_at(row.get("concurrency_degree") or 1),
                memory_mb=max(
                    float(row.get("delta_peak_vram_mib") or 0.0), CUDA_CONTEXT_FLOOR_MB
                ),
                root_release=float(row["release_seconds"]),
            )
        )
    return jobs


# Admission preferences. Lower sorts first.
def lpt(j: DagJob) -> tuple:
    """Longest training first, so long jobs do not form a lone tail."""
    return (-j.solo, j.job_id)


def critical_path(j: DagJob) -> tuple:
    """Longest downstream chain first.

    Under dependency-aware arrivals the makespan is driven by the longest
    chain of train-then-generate steps, so the job whose subtree carries the
    most remaining work should never wait behind a leaf.
    """
    return (-_subtree_cost.get(j.node_id, j.solo), -j.solo, j.job_id)


def density(j: DagJob) -> tuple:
    """Most training seconds per megabyte, the greedy knapsack choice."""
    return (-(j.solo / max(j.memory_mb, 1.0)), j.job_id)


def small(j: DagJob) -> tuple:
    """Smallest footprint first, maximising how many jobs fit at once."""
    return (j.memory_mb, -j.solo, j.job_id)


_subtree_cost: dict[str, float] = {}


def compute_subtree_cost(jobs: list[DagJob]) -> None:
    """Fill `_subtree_cost` with train+generate cost of each node's subtree.

    The cost of a node is its own training time plus, for every child, that
    child's generation time and subtree cost. This is the quantity the
    critical-path policy prioritises.
    """
    kids: dict[str, list[DagJob]] = {}
    known = {j.node_id for j in jobs}
    for j in jobs:
        if j.parent_id in known:
            kids.setdefault(j.parent_id, []).append(j)

    def cost(node_id: str, seen: frozenset) -> float:
        if node_id in _subtree_cost:
            return _subtree_cost[node_id]
        if node_id in seen:
            return 0.0
        me = next((x for x in jobs if x.node_id == node_id), None)
        if me is None:
            return 0.0
        total = me.solo
        for c in kids.get(node_id, []):
            total += c.gen_seconds + cost(c.node_id, seen | {node_id})
        _subtree_cost[node_id] = total
        return total

    for j in jobs:
        cost(j.node_id, frozenset())


def make_cp_guard(pair: float, protect_ratio: float):
    """Veto admissions that would slow a job carrying a long downstream chain.

    Args:
        pair          : float, slowdown contributed per co-runner
        protect_ratio : float, a running job is treated as critical when its
                        subtree cost is at least this fraction of the largest
                        subtree cost in the workload

    Returns:
        callable(running, candidate) -> bool, True when admission is allowed.

    Admitting a co-runner multiplies every running job's remaining time by
    roughly (1 + (pair - 1) / current_slowdown). Paid on an ordinary job that
    costs only its own completion, but paid on a critical-path job it delays
    every descendant too, so the whole chain shifts. The guard therefore
    permits contention freely until a critical job is running, then caps the
    running set so the chain is not stretched.
    """
    peak = max(_subtree_cost.values()) if _subtree_cost else 1.0
    threshold = peak * protect_ratio

    def guard(running: list[DagJob], candidate: DagJob) -> bool:
        critical = [j for j in running if _subtree_cost.get(j.node_id, 0.0) >= threshold]
        if not critical:
            return True
        # One co-runner alongside a critical job is tolerated because the
        # aggregate gain at N=2 outweighs the chain cost; beyond that the
        # chain loses more than the extra slot wins.
        return len(running) < 2

    return guard


def main() -> None:
    rows = load_rows(TRACE)
    jobs = to_dag_jobs(rows)
    compute_subtree_cost(jobs)

    total_work = sum(j.solo for j in jobs)
    roots = sum(1 for j in jobs if j.parent_id not in {x.node_id for x in jobs})
    print(f"trace         : {TRACE.name}")
    print(f"jobs          : {len(jobs)}   roots {roots}   total solo work {total_work/60:.1f} min")
    print(f"pair slowdown : {PAIR_SLOWDOWN:.3f}   budget {MEMORY_BUDGET_MB:.0f} MB   cap {PARALLEL_CAP}")
    print(f"arrivals      : dependency-aware, child = parent finish + gen time")
    print()

    serial = simulate_dag(
        list(jobs), pair_slowdown=PAIR_SLOWDOWN, memory_budget_mb=MEMORY_BUDGET_MB,
        parallel_cap=1, scorer=lpt, policy_name="serial (cap=1)",
    )

    results = [serial]
    for label, scorer in (
        ("occupancy-lpt", lpt),
        ("occupancy-critical-path", critical_path),
        ("occupancy-density", density),
        ("occupancy-small", small),
    ):
        results.append(simulate_dag(
            list(jobs), pair_slowdown=PAIR_SLOWDOWN,
            memory_budget_mb=MEMORY_BUDGET_MB, parallel_cap=PARALLEL_CAP,
            scorer=scorer, policy_name=label,
        ))

    # Critical-path ordering plus contention protection, swept over how much
    # of the subtree-cost range counts as critical.
    for ratio in (0.3, 0.5, 0.7, 0.9):
        results.append(simulate_dag(
            list(jobs), pair_slowdown=PAIR_SLOWDOWN,
            memory_budget_mb=MEMORY_BUDGET_MB, parallel_cap=PARALLEL_CAP,
            scorer=critical_path, guard=make_cp_guard(PAIR_SLOWDOWN, ratio),
            policy_name=f"cp+guard({ratio:g})",
        ))

    base = serial.makespan
    hdr = (f"{'policy':<26}{'makespan':>10}{'mean_flow':>11}{'p95_flow':>10}"
           f"{'max_wait':>10}{'avg_sd':>8}{'conc':>7}{'speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.policy:<26}{r.makespan/60:>9.1f}m{r.mean_flow/60:>10.1f}m"
              f"{r.p95_flow/60:>9.1f}m{r.max_wait/60:>9.1f}m"
              f"{r.average_slowdown:>8.2f}{r.mean_concurrency:>7.2f}"
              f"{base/r.makespan:>8.3f}x")

    best = min(results[1:], key=lambda r: r.makespan)
    print()
    print(f"serial baseline : {base/60:7.1f} min")
    print(f"best policy     : {best.policy}  {best.makespan/60:7.1f} min  "
          f"({base/best.makespan:.3f}x, {(base-best.makespan)/60:.1f} min saved)")

    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / "dag_scheduler_comparison.json"
    out.write_text(json.dumps([{
        "policy": r.policy,
        "makespan_min": r.makespan / 60,
        "mean_flow_min": r.mean_flow / 60,
        "p95_flow_min": r.p95_flow / 60,
        "max_wait_min": r.max_wait / 60,
        "avg_slowdown": r.average_slowdown,
        "mean_concurrency": r.mean_concurrency,
        "speedup_vs_serial": base / r.makespan,
    } for r in results], indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
