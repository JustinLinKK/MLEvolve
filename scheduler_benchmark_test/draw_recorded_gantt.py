"""Draw the recorded MLEvolve trace exactly as it happened.

This is not a policy replay. It plots what the agent actually did on the GPU,
so the shape of the recorded workload is visible before any scheduler is
applied to it.

Each job gets one row with two spans:

    queue      release_seconds -> dispatch_at, the job waited
    execution  dispatch_at     -> exec_complete_at, the job held the GPU

Bars are coloured by whether the execution overlapped another job, since the
recorded durations of overlapped jobs are colocation-contaminated and cannot
be read as solo times.

Usage:
    python -m scheduler_benchmark_test.draw_recorded_gantt [trace.jsonl]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = REPO / "traces" / "mlevolve_leaf_v100_mp2.jsonl"
RECORDS = REPO / "records"

COLOR_SOLO = "#55A868"
COLOR_SHARED = "#C44E52"
COLOR_QUEUE = "#CCCCCC"
COLOR_NOOP = "#BBBBBB"
# Executions at or below this are agent no-ops, not training.
NOOP_SECONDS = 1.0


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw(rows, out_path):
    """Render the recorded timeline.

    Args:
        rows     : list[dict], trace rows in recorded order
        out_path : Path, PNG destination

    Variables:
        busy : float, summed execution seconds, the GPU-busy total
        span : float, last completion minus first release, the makespan
    """
    rows = sorted(rows, key=lambda r: float(r["release_seconds"]))
    fig, ax = plt.subplots(figsize=(16, max(8.0, 0.26 * len(rows) + 3.0)))

    for row_idx, r in enumerate(rows):
        release = float(r["release_seconds"])
        dispatch = float(r["dispatch_at"])
        complete = float(r["exec_complete_at"])
        duration = complete - dispatch
        shared = not r.get("ran_solo", False)

        if dispatch > release:
            ax.barh(row_idx, dispatch - release, left=release, height=0.5,
                    color=COLOR_QUEUE, edgecolor="none")

        if duration <= NOOP_SECONDS:
            colour = COLOR_NOOP
        else:
            colour = COLOR_SHARED if shared else COLOR_SOLO
        ax.barh(row_idx, max(duration, 4.0), left=dispatch, height=0.72,
                color=colour, edgecolor="black", linewidth=0.6)

        label = f"{duration:.0f}s" if duration > NOOP_SECONDS else "no-op"
        ax.text(complete + 25, row_idx, label, va="center", fontsize=7,
                color="#333333")

    busy = sum(float(r["exec_complete_at"]) - float(r["dispatch_at"]) for r in rows)
    span = max(float(r["exec_complete_at"]) for r in rows) - min(
        float(r["release_seconds"]) for r in rows)

    # Describe the trace from its own contents rather than hard-coding a task.
    task = next((r.get("task_name") for r in rows if r.get("task_name")), "unknown task")
    branches = len({r.get("chain_id") for r in rows if r.get("chain_id")}) or 1
    max_conc = max(int(r.get("concurrency_degree") or 1) for r in rows)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["job_id"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds from run start")
    ax.set_title(
        f"Recorded MLEvolve trace, {task} on one V100 "
        f"({branches} search branches, max concurrency {max_conc})\n"
        f"{len(rows)} executions, GPU busy {busy/60:.1f} min of "
        f"{span/60:.1f} min span, offered load {busy/span:.2f}"
    )
    ax.grid(axis="x", alpha=0.3)

    # Traces from the fixed recorder carry gen_duration_s, so their
    # release_seconds is a real generation-derived arrival. Older traces got
    # release_seconds from a global FIFO consumed in dispatch order, which
    # mismatched branches, and their wait bars must not be read as queueing.
    fixed_recorder = any(r.get("gen_duration_s") is not None for r in rows)
    wait_label = (
        "queued: generated, waiting for a worker slot"
        if fixed_recorder else
        "recorded wait (ARTIFACT: release_seconds came from the\n"
        "broken global FIFO, so these are not real queue delays)"
    )

    ax.legend(handles=[
        Patch(facecolor=COLOR_SOLO, edgecolor="black", label="execution, ran solo"),
        Patch(facecolor=COLOR_SHARED, edgecolor="black",
              label="execution, overlapped (duration contaminated)"),
        Patch(facecolor=COLOR_NOOP, edgecolor="black", label="no-op (<1s)"),
        Patch(facecolor=COLOR_QUEUE, label=wait_label),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8,
        frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return busy, span


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACE
    rows = load(path)
    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / f"recorded_gantt_{path.stem}.png"
    busy, span = draw(rows, out)
    print(f"trace      : {path}")
    print(f"executions : {len(rows)}")
    print(f"GPU busy   : {busy:.1f}s ({busy/60:.1f} min)")
    print(f"span       : {span:.1f}s ({span/60:.1f} min)")
    print(f"offered load: {busy/span:.2f}")
    print(f"chart      : {out}")


if __name__ == "__main__":
    main()
