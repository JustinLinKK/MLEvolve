"""Run MLEBench V100 trace through scheduler policies and draw Gantt chart.

Captures dispatch timeline by intercepting trace_metrics(), which receives the
(start_time, TracePack) list each policy produced.

Usage:
    python -m scheduler_benchmark_test.run_trace_experiment
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localml_scheduler.scheduler import trace_simulator as ts

from scheduler_benchmark_test.test_trace_policies import (
    TRACES_DIR,
    WORKLOADS,
    load_trace,
    trace_to_problem,
)

RECORDS_DIR = Path(__file__).resolve().parent.parent / "records"

# Per-architecture colour so packed segments are visually distinguishable.
ARCH_COLORS = {
    "convnext_small": "#4C72B0",
    "efficientnet_b0": "#DD8452",
    "resnet101": "#55A868",
    "resnet50": "#C44E52",
    "swin_tiny_patch4_window7_224": "#8172B3",
    "tabular_mlp_w128_d1_b512": "#4C72B0",
    "tabular_mlp_w256_d2_b1024": "#DD8452",
}


def run_with_dispatches(problem, runner):
    """Run a policy, returning (metrics, dispatches).

    Args:
        problem : TraceProblem
        runner  : callable(problem) -> TraceMetrics

    Variables:
        captured : list[tuple[float, TracePack]], dispatch timeline
    """
    captured: list = []
    original = ts.trace_metrics

    def spy(prob, policy, completion, first_dispatch, *, dispatches=(), **kwargs):
        materialized = list(dispatches)
        captured.clear()
        captured.extend(materialized)
        return original(
            prob, policy, completion, first_dispatch, dispatches=materialized, **kwargs
        )

    ts.trace_metrics = spy
    try:
        metrics = runner(problem)
    finally:
        ts.trace_metrics = original
    return metrics, list(captured)


def draw_gantt(panels, arch_by_job, out_path, *, max_jobs=40):
    """Draw one Gantt subplot per policy.

    Args:
        panels      : list[(title, dispatches)] to plot, top to bottom
        arch_by_job : dict[str, str], job_id -> architecture
        out_path    : Path, PNG destination
        max_jobs    : int, only plot the first max_jobs distinct job_ids
    """
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(16, 5.0 * len(panels)), squeeze=False
    )

    for ax, (title, dispatches) in zip(axes[:, 0], panels):
        rows: dict[str, int] = {}
        for start, pack in dispatches:
            for member in pack.members:
                if member.job_id not in rows and len(rows) < max_jobs:
                    rows[member.job_id] = len(rows)

        for start, pack in dispatches:
            n_members = len(pack.members)
            for member, offset in zip(pack.members, pack.completion_offsets):
                row = rows.get(member.job_id)
                if row is None:
                    continue
                arch = arch_by_job.get(member.job_id, "")
                ax.barh(
                    row,
                    offset,
                    left=start,
                    height=0.72,
                    color=ARCH_COLORS.get(arch, "#999999"),
                    edgecolor="black" if n_members > 1 else "none",
                    linewidth=1.4 if n_members > 1 else 0,
                )

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(list(rows), fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("time (s)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    present = sorted(set(arch_by_job.values()))
    handles = [
        Patch(facecolor=ARCH_COLORS.get(a, "#999999"), label=a) for a in present
    ]
    handles.append(
        Patch(facecolor="white", edgecolor="black", linewidth=1.4, label="packed (N>1)")
    )
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    workload = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    if workload not in WORKLOADS:
        raise SystemExit(f"unknown workload {workload!r}; choose from {sorted(WORKLOADS)}")
    trace_name, pair_slowdown = WORKLOADS[workload]

    raw_jobs = load_trace(TRACES_DIR / trace_name)
    arch_by_job = {j["job_id"]: j["architecture"] for j in raw_jobs}
    problem = trace_to_problem(raw_jobs, early_stop=False, pair_slowdown=pair_slowdown)

    total_solo = sum(j["options"][0]["solo_seconds"] for j in raw_jobs)
    print(f"Workload: {workload}   trace: {trace_name}   pair slowdown: {pair_slowdown}")
    print(f"Jobs: {len(raw_jobs)}   total solo compute: {total_solo:.0f}s")
    print(f"Arrival span: 0 - {raw_jobs[-1]['release_seconds']:.1f}s\n")

    runs = [
        ("serial (priority-FIFO)", lambda p: ts.simulate_policy(p, "serial", ts._serial_choice)),
        ("time_aware (SRT-first)", lambda p: ts.simulate_policy(p, "time_aware", ts._time_aware_choice)),
        ("recursive_time_aware (packing)", ts.simulate_recursive_time_aware),
    ]

    results = []
    for title, runner in runs:
        metrics, dispatches = run_with_dispatches(problem, runner)
        results.append((title, metrics, dispatches))
        print(
            f"{title:<34s} makespan={metrics.makespan_seconds:9.1f}s  "
            f"mean_flow={metrics.mean_flow_seconds:9.1f}s  "
            f"avg_slowdown={metrics.average_slowdown:.2f}  "
            f"slowdown_rej={metrics.slowdown_rejections}  "
            f"trial_epochs={metrics.colocation_trial_epochs:.1f}  "
            f"starved={metrics.starvation_count}"
        )

    baseline = results[0][1].makespan_seconds
    print("\nMakespan vs serial:")
    for title, metrics, _ in results[1:]:
        print(f"  {title:<34s} {baseline / metrics.makespan_seconds:.3f}x")

    RECORDS_DIR.mkdir(exist_ok=True)
    out = draw_gantt(
        [(t, d) for t, _, d in results],
        arch_by_job,
        RECORDS_DIR / f"scheduling_gantt_v100_{workload}.png",
    )
    print(f"\nGantt chart: {out}")


if __name__ == "__main__":
    main()
