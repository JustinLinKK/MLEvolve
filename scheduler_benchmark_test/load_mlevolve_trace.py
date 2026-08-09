"""Convert a recorded MLEvolve trace into a scheduler TraceProblem.

Reads the JSONL emitted by run_traced.py (one row per code execution the agent
performed) and maps it onto TraceJob/TraceBatchOption.

Only measured quantities are used:
    release_seconds   when the agent submitted the job, relative to run start
    exec_duration_s   wall-clock the execution actually took on the V100
    peak_vram_mib     NVML peak across the execution window

Early stopping is left disabled, because the recorder does not capture
per-epoch validation metrics from arbitrary agent-written scripts.

Usage:
    python -m scheduler_benchmark_test.load_mlevolve_trace <trace.jsonl> [pair_slowdown]
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localml_scheduler.scheduler import trace_simulator as ts

MEMORY_BUDGET_MB = 31000.0
PARALLEL_CAP = 5
SEED = 42


def retime_poisson(rows: list[dict], lambda_per_min: float) -> list[dict]:
    """Re-sample arrival times as a Poisson process, keeping measured work.

    The recorded arrivals are paced by the agent's LLM latency, not by any
    workload model, so replaying them leaves the GPU mostly idle. Re-timing
    keeps every measured duration and VRAM figure and only changes when each
    job is released, which is what CLAUDE.md fixes at 4 jobs/min.
    """
    rng = random.Random(SEED)
    retimed = []
    arrival = 0.0
    for row in rows:
        clone = dict(row)
        clone["release_seconds"] = round(arrival, 3)
        retimed.append(clone)
        arrival += rng.expovariate(lambda_per_min / 60.0)
    return retimed


def load_rows(path: Path, *, include_buggy: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not include_buggy:
        rows = [r for r in rows if not r.get("is_buggy")]
    return rows


def rows_to_problem(rows: list[dict], *, pair_slowdown: float) -> ts.TraceProblem:
    """Build a TraceProblem from recorded MLEvolve executions.

    Args:
        rows          : list[dict], recorded trace rows
        pair_slowdown : float, measured per-pair colocation slowdown on the V100

    Returns:
        TraceProblem ready for simulate_* functions
    """
    jobs = []
    for row in rows:
        epochs = int(row.get("epochs") or 1)
        memory_mb = float(row.get("peak_vram_mib") or 0.0)
        solo_seconds = float(row.get("exec_duration_s") or 0.0)
        if solo_seconds <= 0:
            continue
        option = ts.TraceBatchOption(
            batch_size=int(row.get("batch_size") or 0),
            memory_mb=memory_mb,
            solo_seconds=solo_seconds,
            actual_memory_mb=memory_mb,
            actual_solo_seconds=solo_seconds,
        )
        jobs.append(
            ts.TraceJob(
                job_id=str(row["job_id"]),
                release_seconds=float(row["release_seconds"]),
                priority=0,
                options=(option,),
                backend_allowlist=("cuda_process",),
                validation_metrics=(),
                planned_epochs=epochs,
            )
        )

    return ts.TraceProblem(
        jobs=tuple(jobs),
        memory_budget_mb=MEMORY_BUDGET_MB,
        parallel_cap=PARALLEL_CAP,
        default_slowdown=pair_slowdown,
        colocation_trial_epochs=2,
        colocation_min_gain=1.0,
        early_stopping_enabled=False,
        starvation_timeout_seconds=1800.0,
    )


def summarize(rows: list[dict]) -> None:
    total = sum(float(r.get("exec_duration_s") or 0) for r in rows)
    vrams = [float(r["peak_vram_mib"]) for r in rows if r.get("peak_vram_mib")]
    sms = [float(r["avg_sm_util_percent"]) for r in rows if r.get("avg_sm_util_percent") is not None]
    print(f"jobs: {len(rows)}")
    print(f"total solo compute: {total:.1f}s")
    if rows:
        print(f"mean solo: {total/len(rows):.1f}s")
        print(f"arrival span: {rows[0]['release_seconds']:.1f} - {rows[-1]['release_seconds']:.1f}s")
    if vrams:
        print(f"peak VRAM: min {min(vrams):.0f} / mean {sum(vrams)/len(vrams):.0f} / max {max(vrams):.0f} MiB")
    if sms:
        print(f"avg SM during exec: mean {sum(sms)/len(sms):.1f}% / max {max(sms):.1f}%")
    fams = {}
    for r in rows:
        fams[r.get("family")] = fams.get(r.get("family"), 0) + 1
    print("families:", fams)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    pair_slowdown = float(sys.argv[2]) if len(sys.argv) > 2 else 1.017

    rows = load_rows(path)
    all_rows = load_rows(path, include_buggy=True)
    print(f"trace: {path}")
    print(f"rows total: {len(all_rows)}   non-buggy: {len(rows)}")
    summarize(rows)
    print()

    scenarios = [("as recorded", rows)]
    for lam in (4.0,):
        scenarios.append((f"Poisson lambda={lam:g}/min", retime_poisson(rows, lam)))

    for label, scenario_rows in scenarios:
        problem = rows_to_problem(scenario_rows, pair_slowdown=pair_slowdown)
        span = scenario_rows[-1]["release_seconds"] - scenario_rows[0]["release_seconds"]
        work = sum(float(r["exec_duration_s"]) for r in scenario_rows)
        print(f"=== arrivals: {label} (span {span:.0f}s, offered load {work / span if span else float('inf'):.2f}) ===")
        print(f"pair slowdown: {pair_slowdown}")

        serial = ts.simulate_policy(problem, "serial", ts._serial_choice)
        time_aware = ts.simulate_policy(problem, "time_aware", ts._time_aware_choice)
        recursive = ts.simulate_recursive_time_aware(problem)

        for name, m in (
            ("serial (priority-FIFO)", serial),
            ("time_aware (SRT-first)", time_aware),
            ("recursive_time_aware", recursive),
        ):
            print(
                f"  {name:<26s} makespan={m.makespan_seconds:9.1f}s  "
                f"mean_flow={m.mean_flow_seconds:9.1f}s  "
                f"avg_slowdown={m.average_slowdown:.2f}  "
                f"slowdown_rej={m.slowdown_rejections}  "
                f"starved={m.starvation_count}"
            )
        print("  Makespan vs serial:")
        for name, m in (("time_aware", time_aware), ("recursive_time_aware", recursive)):
            ratio = serial.makespan_seconds / m.makespan_seconds if m.makespan_seconds else 0.0
            flow = m.mean_flow_seconds / serial.mean_flow_seconds if serial.mean_flow_seconds else 0.0
            print(f"    {name:<26s} makespan={ratio:.3f}x  flow_ratio={flow:.3f}")
        print()

    records = Path(__file__).resolve().parent.parent / "records"
    records.mkdir(exist_ok=True)
    out = records / f"scheduling_gantt_{path.stem}.png"
    draw_gantt(retime_poisson(rows, 4.0), pair_slowdown, out)
    print(f"Gantt chart (lambda=4/min): {out}")


def draw_gantt(rows: list[dict], pair_slowdown: float, out_path) -> None:
    """Render serial / time_aware / recursive_time_aware for the re-timed trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from scheduler_benchmark_test.run_trace_experiment import run_with_dispatches

    colors = {"mlp": "#4C72B0", "cnn": "#DD8452"}
    arch = {r["job_id"]: (r.get("family") or "other") for r in rows}
    problem = rows_to_problem(rows, pair_slowdown=pair_slowdown)

    panels = []
    for title, runner in (
        ("serial (priority-FIFO)", lambda p: ts.simulate_policy(p, "serial", ts._serial_choice)),
        ("time_aware (SRT-first)", lambda p: ts.simulate_policy(p, "time_aware", ts._time_aware_choice)),
        ("recursive_time_aware (packing)", ts.simulate_recursive_time_aware),
    ):
        _, dispatches = run_with_dispatches(problem, runner)
        panels.append((title, dispatches))

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.6 * len(panels)), squeeze=False)
    for ax, (title, dispatches) in zip(axes[:, 0], panels):
        order: dict[str, int] = {}
        for _, pack in dispatches:
            for member in pack.members:
                order.setdefault(member.job_id, len(order))
        for start, pack in dispatches:
            packed = len(pack.members) > 1
            for member, offset in zip(pack.members, pack.completion_offsets):
                ax.barh(
                    order[member.job_id], offset, left=start, height=0.7,
                    color=colors.get(arch.get(member.job_id), "#999999"),
                    edgecolor="black" if packed else "none",
                    linewidth=1.4 if packed else 0,
                )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(list(order), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("time (s)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    handles = [Patch(facecolor=c, label=f) for f, c in colors.items()]
    handles.append(Patch(facecolor="white", edgecolor="black", linewidth=1.4, label="packed (N>1)"))
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
