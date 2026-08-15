"""Replay the recorded cassava trace through this repo's scheduler.

Uses the profile-based estimator only. The ML predictor is never constructed,
because ResourceEstimator builds it solely when
settings.prediction.mode == PREDICTION_MODE_ML_PREDICTOR; every estimate here
is sourced from recorded profile data instead.

Policies compared, all from localml_scheduler.scheduler.trace_simulator:

    serial                 _serial_choice, priority-FIFO, one job at a time
    time_aware             _time_aware_choice, shortest-remaining-time first
    recursive_time_aware   simulate_recursive_time_aware, this repo's
                           scheduler: anchor the running set, admit one
                           candidate at a time when memory allows and the
                           colocation gain clears the threshold

Deconvolving the recorded durations. Every cassava execution ran alongside
others, so no row is a solo measurement and exec_duration_s cannot be used as
solo_seconds directly. Device power sampled during the run gives the
throughput actually achieved at each concurrency level:

    N     mean power W    power above idle    aggregate vs solo
    1        127.1              70.4                1.00
    2        174.0             117.3                1.67
    3        194.6             137.9                1.96
    4        196.8             140.1                1.99
    5        251.4             194.7                2.77

Idle draw is 56.7 W. Treating work rate as proportional to power above idle,
the per-job slowdown at concurrency N is N divided by the aggregate at N, and
a recorded duration is converted back to solo time by dividing by it.

The simulator composes a member's slowdown as one plus the sum over co-runners
of (pair_slowdown - 1), so the pair value implied by the N=2 measurement is
used as default_slowdown.

Usage:
    python -m scheduler_benchmark_test.run_cassava_scheduler_test
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localml_scheduler.scheduler import trace_simulator as ts

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "traces" / "mlevolve_cassava_v100_deepseek.jsonl"
RECORDS = REPO / "records"

# Per CLAUDE.md the scheduler's memory ceiling is 31 GB.
MEMORY_BUDGET_MB = 31000.0
PARALLEL_CAP = 5

# Aggregate throughput relative to one job, measured from device power during
# the recorded run (see module docstring).
AGGREGATE_AT = {1: 1.00, 2: 1.67, 3: 1.96, 4: 1.99, 5: 2.77}
# Pair slowdown implied by the N=2 point: 2 jobs delivering 1.67 jobs' work.
PAIR_SLOWDOWN = 2.0 / 1.67

# Executions at or below this are agent no-ops, not training.
NOOP_SECONDS = 1.0

# A CUDA context plus cuDNN workspace costs roughly this much before any
# tensor is allocated. Used as a floor when the recorded delta is
# non-positive, which happens when a job started while the device baseline was
# already raised by its co-runners.
CUDA_CONTEXT_FLOOR_MB = 520.0


def slowdown_at(concurrency: int) -> float:
    """Per-job slowdown observed at a given concurrency level.

    Args:
        concurrency : int, number of executions sharing the device

    Returns:
        float, factor by which one job was slowed relative to running alone.
        Levels beyond the measured range extrapolate from the highest measured
        point, which is conservative because throughput was still flattening.
    """
    n = max(1, int(concurrency))
    if n in AGGREGATE_AT:
        return n / AGGREGATE_AT[n]
    top = max(AGGREGATE_AT)
    return n / AGGREGATE_AT[top]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_problem(rows: list[dict]) -> ts.TraceProblem:
    """Build a TraceProblem whose durations are deconvolved solo estimates.

    Args:
        rows : list[dict], recorded trace rows

    Returns:
        TraceProblem. Jobs with sub-second executions are dropped: they are
        agent no-ops that never reached training and would distort ordering
        metrics with zero-work entries.
    """
    jobs = []
    for row in rows:
        recorded = float(row.get("exec_duration_s") or 0.0)
        if recorded <= NOOP_SECONDS:
            continue
        solo = recorded / slowdown_at(row.get("concurrency_degree") or 1)
        # delta_peak_vram_mib is this job's own footprint above the device
        # baseline. device_peak_vram_mib is whole-device and reaches 32.7 GB
        # because it includes every co-runner, which would exceed the 31 GB
        # budget and make every pack infeasible.
        memory_mb = max(
            float(row.get("delta_peak_vram_mib") or 0.0), CUDA_CONTEXT_FLOOR_MB
        )
        option = ts.TraceBatchOption(
            batch_size=int(row.get("batch_size") or 0),
            memory_mb=memory_mb,
            solo_seconds=solo,
            actual_memory_mb=memory_mb,
            actual_solo_seconds=solo,
        )
        jobs.append(
            ts.TraceJob(
                job_id=str(row["job_id"]),
                release_seconds=float(row["release_seconds"]),
                priority=0,
                options=(option,),
                backend_allowlist=("cuda_process",),
                validation_metrics=(),
                planned_epochs=int(row.get("epochs") or 1),
            )
        )

    return ts.TraceProblem(
        jobs=tuple(jobs),
        memory_budget_mb=MEMORY_BUDGET_MB,
        parallel_cap=PARALLEL_CAP,
        default_slowdown=PAIR_SLOWDOWN,
        colocation_trial_epochs=2,
        colocation_min_gain=1.0,
        early_stopping_enabled=False,
        starvation_timeout_seconds=1800.0,
    )


def show(label: str, m: ts.TraceMetrics, baseline: ts.TraceMetrics | None = None) -> None:
    speedup = ""
    if baseline is not None and m.makespan_seconds > 0:
        speedup = f"  speedup={baseline.makespan_seconds / m.makespan_seconds:5.3f}x"
    print(
        f"  {label:<24s} makespan={m.makespan_seconds/60:8.1f}min  "
        f"mean_flow={m.mean_flow_seconds/60:8.1f}min  "
        f"p95_flow={m.p95_flow_seconds/60:8.1f}min  "
        f"max_wait={m.max_wait_seconds/60:7.1f}min  "
        f"avg_sd={m.average_slowdown:4.2f}  "
        f"rej={m.slowdown_rejections:3d}  "
        f"starved={m.starvation_count}{speedup}"
    )


def main() -> None:
    rows = load_rows(TRACE)
    problem = to_problem(rows)

    total_solo = sum(o.solo_seconds for j in problem.jobs for o in j.options[:1])
    print(f"trace          : {TRACE.name}")
    print(f"rows           : {len(rows)}   jobs after dropping no-ops: {len(problem.jobs)}")
    print(f"pair slowdown  : {PAIR_SLOWDOWN:.3f} (from measured N=2 power)")
    print(f"total solo work: {total_solo/60:.1f} min")
    print(f"memory budget  : {MEMORY_BUDGET_MB:.0f} MB   parallel cap: {PARALLEL_CAP}")
    print(f"predictor      : profile-based (ML predictor not constructed)")
    print()

    serial = ts.simulate_policy(problem, "serial", ts._serial_choice)
    time_aware = ts.simulate_policy(problem, "time_aware", ts._time_aware_choice)
    recursive = ts.simulate_recursive_time_aware(problem)

    print("=== policies ===")
    show("serial (priority-FIFO)", serial)
    show("time_aware (SRT-first)", time_aware, serial)
    show("recursive_time_aware", recursive, serial)
    print()
    print("recursive_time_aware is this repo's scheduler.")

    out = RECORDS / "cassava_scheduler_results.json"
    RECORDS.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "trace": TRACE.name,
        "jobs": len(problem.jobs),
        "pair_slowdown": PAIR_SLOWDOWN,
        "predictor": "profile_based",
        "policies": {
            name: {
                "makespan_s": m.makespan_seconds,
                "mean_flow_s": m.mean_flow_seconds,
                "p95_flow_s": m.p95_flow_seconds,
                "max_wait_s": m.max_wait_seconds,
                "average_slowdown": m.average_slowdown,
                "slowdown_rejections": m.slowdown_rejections,
                "starvation_count": m.starvation_count,
            }
            for name, m in (
                ("serial", serial),
                ("time_aware", time_aware),
                ("recursive_time_aware", recursive),
            )
        },
    }, indent=2))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
