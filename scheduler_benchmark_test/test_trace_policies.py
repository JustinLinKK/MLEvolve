"""Test scheduler policies against MLEBench V100 trace.

Loads trace from traces/mlebench_v100_100jobs.jsonl,
converts to TraceProblem, runs 3 policies + recursive time-aware,
prints comparison table.

Usage:
    python -m scheduler_benchmark_test.test_trace_policies
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localml_scheduler.scheduler.trace_simulator import (
    TraceBatchOption,
    TraceJob,
    TraceMetrics,
    TraceProblem,
    simulate_policy,
    simulate_recursive_time_aware,
    _serial_choice,
    _time_aware_choice,
    _fill_choice,
)


TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"
TRACE_PATH = TRACES_DIR / "mlebench_v100_100jobs.jsonl"
MEMORY_BUDGET_MB = 31000.0

# Pair slowdown measured on one V100 (no MPS). The simulator composes a member's
# slowdown as 1 + sum(pair_slowdown - 1) over co-runners, so the pair value is
# derived from the N=5 measurement as 1 + (sd_n5 - 1) / 4.
#   CNN trace      : sd_n5 ~ 5.0  -> pair 4.0
#   tabular trace  : sd_n5 = 1.067 -> pair 1.017
V100_PACKING_SLOWDOWN = 4.0
V100_TABULAR_SLOWDOWN = 1.017

# Named workloads: (trace filename, pair slowdown).
WORKLOADS = {
    "cnn": ("mlebench_v100_100jobs.jsonl", V100_PACKING_SLOWDOWN),
    "tabular": ("mlebench_tabular_v100_100jobs.jsonl", V100_TABULAR_SLOWDOWN),
}


def load_trace(path: Path) -> list[dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def trace_to_problem(
    raw_jobs: list[dict],
    *,
    early_stop: bool = False,
    pair_slowdown: float = V100_PACKING_SLOWDOWN,
) -> TraceProblem:
    """Convert raw JSONL trace to TraceProblem.

    Args:
        raw_jobs      : list[dict], loaded from JSONL trace
        early_stop    : bool, enable early stopping simulation
        pair_slowdown : float, measured per-pair colocation slowdown

    Returns:
        TraceProblem with TraceJob objects
    """
    trace_jobs = []
    for raw in raw_jobs:
        options_raw = raw["options"]
        options = tuple(
            TraceBatchOption(
                batch_size=opt["batch_size"],
                memory_mb=opt["memory_mb"],
                solo_seconds=opt["solo_seconds"],
                actual_memory_mb=opt.get("actual_memory_mb"),
                actual_solo_seconds=opt.get("actual_solo_seconds"),
            )
            for opt in options_raw
        )
        val_metrics = raw.get("validation_metrics", [])
        trace_jobs.append(
            TraceJob(
                job_id=raw["job_id"],
                release_seconds=raw["release_seconds"],
                priority=raw.get("priority", 0),
                options=options,
                backend_allowlist=tuple(raw.get("backend_allowlist", ("cuda_process",))),
                validation_metrics=tuple(val_metrics if val_metrics else ()),
                planned_epochs=raw.get("planned_epochs"),
            )
        )

    return TraceProblem(
        jobs=tuple(trace_jobs),
        memory_budget_mb=MEMORY_BUDGET_MB,
        parallel_cap=5,
        default_slowdown=pair_slowdown,
        colocation_trial_epochs=2,
        colocation_min_gain=1.0,
        early_stopping_enabled=early_stop,
        early_stopping_patience_epochs=5,
        early_stopping_min_delta=0.001,
        early_stopping_min_epochs=1,
        early_stopping_mode="max",
        starvation_timeout_seconds=1800.0,
    )


def print_metrics(label: str, m: TraceMetrics) -> None:
    print(f"  {label:<30s} "
          f"makespan={m.makespan_seconds:8.1f}s  "
          f"mean_flow={m.mean_flow_seconds:8.1f}s  "
          f"p95_flow={m.p95_flow_seconds:8.1f}s  "
          f"max_wait={m.max_wait_seconds:7.1f}s  "
          f"starved={m.starvation_count}  "
          f"violations={m.hard_constraint_violations}  "
          f"slowdown_rej={m.slowdown_rejections}  "
          f"early_stop_saved={m.early_stopped_epochs_saved}ep")


def main():
    raw_jobs = load_trace(TRACE_PATH)
    print(f"Loaded {len(raw_jobs)} jobs from {TRACE_PATH.name}")
    print(f"Arrival span: {raw_jobs[0]['release_seconds']:.1f} - {raw_jobs[-1]['release_seconds']:.1f}s")
    total_solo = sum(j["options"][0]["solo_seconds"] for j in raw_jobs)
    print(f"Total solo compute: {total_solo:.0f}s ({total_solo/60:.1f} min)")
    print()

    for early_stop in [False, True]:
        es_label = "early_stop=ON" if early_stop else "early_stop=OFF"
        print(f"=== {es_label} ===")
        problem = trace_to_problem(raw_jobs, early_stop=early_stop)

        serial = simulate_policy(problem, "serial", _serial_choice)
        print_metrics("serial (priority-FIFO)", serial)

        time_aware = simulate_policy(problem, "time_aware", _time_aware_choice)
        print_metrics("time_aware (SRT-first)", time_aware)

        recursive = simulate_recursive_time_aware(problem)
        print_metrics("recursive_time_aware", recursive)

        print()
        print(f"  Speedup vs serial:")
        for label, m in [
            ("time_aware", time_aware),
            ("recursive_time_aware", recursive),
        ]:
            speedup = serial.makespan_seconds / m.makespan_seconds if m.makespan_seconds > 0 else 0
            flow_ratio = m.mean_flow_seconds / serial.mean_flow_seconds if serial.mean_flow_seconds > 0 else 0
            print(f"    {label:<30s} makespan={speedup:.3f}x  flow_ratio={flow_ratio:.3f}")
        print()


if __name__ == "__main__":
    main()
