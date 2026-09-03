"""Compare this repo's scheduler against the occupancy scheduler on one trace.

Both sides consume the same deconvolved workload and the same slowdown model,
so the only difference measured is the scheduling policy.

Repo side, from localml_scheduler.scheduler.trace_simulator:

    serial                 priority-FIFO, one job at a time
    time_aware             shortest-remaining-time first, still one at a time
                           because its chooser only ever sees the anchor job
    recursive_time_aware   this repo's scheduler, incremental packing behind a
                           two-epoch colocation trial

Mine, from occupancy_scheduler, run under three admission preferences so the
choice is settled by measurement rather than assertion:

    lpt        longest processing time first
    density    most solo-seconds per megabyte first
    small      smallest footprint first

Metric definitions match trace_metrics exactly: flow is completion minus
release, wait is first dispatch minus release, makespan is the last completion
minus the first release.

Usage:
    python -m scheduler_benchmark_test.compare_schedulers
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localml_scheduler.scheduler import trace_simulator as ts
from scheduler_benchmark_test.occupancy_scheduler import (
    Job,
    score_density,
    score_lpt,
    score_small_memory,
    simulate,
)
from scheduler_benchmark_test.run_cassava_scheduler_test import (
    CUDA_CONTEXT_FLOOR_MB,
    MEMORY_BUDGET_MB,
    NOOP_SECONDS,
    PAIR_SLOWDOWN,
    TRACE,
    load_rows,
    slowdown_at,
    to_problem,
)

RECORDS = Path(__file__).resolve().parent.parent / "records"


def to_jobs(rows: list[dict]) -> list[Job]:
    """Convert trace rows into the occupancy scheduler's job records.

    Applies the same deconvolution and the same per-job memory attribution as
    the repo-side loader, so both schedulers see an identical workload.
    """
    jobs = []
    for row in rows:
        recorded = float(row.get("exec_duration_s") or 0.0)
        if recorded <= NOOP_SECONDS:
            continue
        jobs.append(
            Job(
                job_id=str(row["job_id"]),
                release=float(row["release_seconds"]),
                solo=recorded / slowdown_at(row.get("concurrency_degree") or 1),
                memory_mb=max(
                    float(row.get("delta_peak_vram_mib") or 0.0), CUDA_CONTEXT_FLOOR_MB
                ),
            )
        )
    return jobs


def main() -> None:
    rows = load_rows(TRACE)
    problem = to_problem(rows)
    jobs = to_jobs(rows)
    total_work = sum(j.solo for j in jobs)

    print(f"trace           : {TRACE.name}")
    print(f"jobs            : {len(jobs)}   total solo work {total_work/60:.1f} min")
    print(f"pair slowdown   : {PAIR_SLOWDOWN:.3f}   budget {MEMORY_BUDGET_MB:.0f} MB   cap unset")
    print(f"predictor       : profile-based (ML predictor never constructed)")
    print()

    serial = ts.simulate_policy(problem, "serial", ts._serial_choice)
    time_aware = ts.simulate_policy(problem, "time_aware", ts._time_aware_choice)
    recursive = ts.simulate_recursive_time_aware(problem)

    base = serial.makespan_seconds
    rows_out = []

    def add_repo(name, m):
        rows_out.append({
            "policy": name,
            "makespan_min": m.makespan_seconds / 60,
            "mean_flow_min": m.mean_flow_seconds / 60,
            "p95_flow_min": m.p95_flow_seconds / 60,
            "max_wait_min": m.max_wait_seconds / 60,
            "avg_slowdown": m.average_slowdown,
            "starved": m.starvation_count,
            "mean_conc": None,
            "speedup": base / m.makespan_seconds if m.makespan_seconds else 0.0,
        })

    add_repo("serial (repo)", serial)
    add_repo("time_aware (repo)", time_aware)
    add_repo("recursive_time_aware (repo)", recursive)

    for label, scorer in (
        ("occupancy-lpt (mine)", score_lpt),
        ("occupancy-density (mine)", score_density),
        ("occupancy-small (mine)", score_small_memory),
    ):
        r = simulate(
            list(jobs),
            pair_slowdown=PAIR_SLOWDOWN,
            memory_budget_mb=MEMORY_BUDGET_MB,
            parallel_cap=None,
            scorer=scorer,
            policy_name=label,
        )
        rows_out.append({
            "policy": label,
            "makespan_min": r.makespan / 60,
            "mean_flow_min": r.mean_flow / 60,
            "p95_flow_min": r.p95_flow / 60,
            "max_wait_min": r.max_wait / 60,
            "avg_slowdown": r.average_slowdown,
            "starved": r.starvation_count,
            "mean_conc": r.mean_concurrency,
            "speedup": base / r.makespan if r.makespan else 0.0,
        })

    hdr = (f"{'policy':<30}{'makespan':>10}{'mean_flow':>11}{'p95_flow':>10}"
           f"{'max_wait':>10}{'avg_sd':>8}{'starv':>7}{'conc':>7}{'speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        conc = f"{r['mean_conc']:.2f}" if r["mean_conc"] is not None else "-"
        print(f"{r['policy']:<30}{r['makespan_min']:>9.1f}m{r['mean_flow_min']:>10.1f}m"
              f"{r['p95_flow_min']:>9.1f}m{r['max_wait_min']:>9.1f}m"
              f"{r['avg_slowdown']:>8.2f}{r['starved']:>7d}{conc:>7}"
              f"{r['speedup']:>8.3f}x")

    best_repo = min(r["makespan_min"] for r in rows_out if "(repo)" in r["policy"])
    best_mine = min(r["makespan_min"] for r in rows_out if "(mine)" in r["policy"])
    print()
    print(f"best repo scheduler : {best_repo:7.1f} min")
    print(f"best mine           : {best_mine:7.1f} min")
    print(f"improvement         : {best_repo/best_mine:.3f}x  "
          f"({best_repo - best_mine:.1f} min saved)")

    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / "scheduler_comparison.json"
    out.write_text(json.dumps(rows_out, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
