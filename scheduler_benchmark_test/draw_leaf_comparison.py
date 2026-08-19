"""Gantt plus metric-vs-node chart for the two leaf V100 traces, one PNG each.

Both traces ran leaf-classification on a single V100 with identical settings
apart from scheduler.enabled, so the pair isolates what the scheduler and its
hardware knowledge database cost and provide.

Layout per CLAUDE.md: the Gantt of job execution and the metric-vs-node chart
go in a single image.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
TRACES = {
    "original MLEvolve": REPO / "traces/mlevolve_leaf_v100_orig_sonnet5.jsonl",
    "scheduler + hardware DB": REPO / "traces/mlevolve_leaf_v100_sched_db_sonnet5.jsonl",
}
# The recorder does not carry the metric, which is only known once MLEvolve has
# reviewed the node, so it is joined in from the run's journal by node id.
METRICS = {
    "original MLEvolve": REPO / "records/metrics_leaf_orig.json",
    "scheduler + hardware DB": REPO / "records/metrics_leaf_sched.json",
}
OUT = REPO / "records/leaf_v100_orig_vs_scheduler.png"

# exec_duration_s spans dispatch to return, so under the scheduler it contains
# queue time. reported_exec_time_s is the execution alone; the difference is
# what the scheduler path adds per node.
OK_COLOR = "#2b7bba"
BUG_COLOR = "#c44e52"
WAIT_COLOR = "#f0c05a"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def sweep_peak_concurrency(rows: list[dict]) -> int:
    """True peak overlap, from a sweep line over the execution windows.

    concurrency_degree in the row is a union over the node's whole window and
    overstates simultaneity, so it is not used here.
    """
    events: list[tuple[float, int]] = []
    for r in rows:
        events.append((float(r["dispatch_at"]), 1))
        events.append((float(r["exec_complete_at"]), -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["is_buggy"]]
    wall = [float(r["exec_duration_s"]) for r in ok]
    rep = [float(r["reported_exec_time_s"]) for r in ok if r.get("reported_exec_time_s")]
    ovh = [
        float(r["exec_duration_s"]) - float(r["reported_exec_time_s"])
        for r in ok
        if r.get("reported_exec_time_s")
    ]
    vram = [float(r["job_peak_vram_mib"]) for r in ok if r.get("job_peak_vram_mib")]
    return {
        "nodes": len(rows),
        "ok": len(ok),
        "wall": statistics.median(wall) if wall else 0.0,
        "rep": statistics.median(rep) if rep else 0.0,
        "ovh": statistics.median(ovh) if ovh else 0.0,
        "vram": statistics.median(vram) if vram else 0.0,
        "span": max(float(r["exec_complete_at"]) for r in rows) / 3600.0,
        "peak": sweep_peak_concurrency(rows),
    }


def draw(ax_gantt, ax_metric, rows: list[dict], title: str, metrics: dict[str, float]) -> dict:
    s = summarize(rows)
    ordered = sorted(rows, key=lambda r: float(r["dispatch_at"]))

    for lane, r in enumerate(ordered):
        start = float(r["dispatch_at"]) / 3600.0
        end = float(r["exec_complete_at"]) / 3600.0
        reported = r.get("reported_exec_time_s")
        colour = OK_COLOR if not r["is_buggy"] else BUG_COLOR
        if reported is not None:
            # Queue/instrumentation time first, then the execution itself, so
            # the scheduler's added cost is visible rather than folded in.
            exec_h = float(reported) / 3600.0
            wait_h = max(0.0, (end - start) - exec_h)
            if wait_h > 0:
                ax_gantt.broken_barh([(start, wait_h)], (lane - 0.4, 0.8), facecolors=WAIT_COLOR)
            ax_gantt.broken_barh([(start + wait_h, exec_h)], (lane - 0.4, 0.8), facecolors=colour)
        else:
            ax_gantt.broken_barh([(start, end - start)], (lane - 0.4, 0.8), facecolors=colour)

    ax_gantt.set_xlabel("hours since run start")
    ax_gantt.set_ylabel("node (dispatch order)")
    ax_gantt.set_title(
        f"{title}\n{s['nodes']} nodes, {s['ok']} ok "
        f"({100 * s['ok'] / s['nodes']:.0f}%), span {s['span']:.1f} h, peak concurrency {s['peak']}",
        fontsize=10,
    )
    ax_gantt.grid(axis="x", alpha=0.3)

    # Metric vs node, with the running best, so search progress is legible.
    xs, ys = [], []
    for i, r in enumerate(ordered):
        m = metrics.get(str(r.get("node_id")))
        if m is not None:
            xs.append(i)
            ys.append(float(m))
    if xs:
        ax_metric.plot(xs, ys, "o", ms=4, color=OK_COLOR, alpha=0.6, label="node metric")
        # maximize is False in both journals: this is a log loss, lower is better.
        best, run = None, []
        for y in ys:
            best = y if best is None else min(best, y)
            run.append(best)
        ax_metric.plot(xs, run, "-", color="#d1622b", lw=2, label="running best (lower is better)")
        ax_metric.set_title(
            f"best {min(ys):.4f} after {len(xs)} scored nodes", fontsize=9
        )
        ax_metric.legend(fontsize=8)
    else:
        ax_metric.text(
            0.5, 0.5, "no metric recorded in trace", ha="center", va="center",
            transform=ax_metric.transAxes, fontsize=9, color="grey",
        )
    ax_metric.set_xlabel("node (dispatch order)")
    ax_metric.set_ylabel("validation metric")
    ax_metric.grid(alpha=0.3)
    return s


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    stats = {}
    for col, (label, path) in enumerate(TRACES.items()):
        if not path.exists():
            print(f"missing trace: {path}")
            return 1
        metrics_path = METRICS[label]
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        stats[label] = draw(axes[0][col], axes[1][col], load(path), label, metrics)

    a = stats["original MLEvolve"]
    b = stats["scheduler + hardware DB"]
    summary = (
        "leaf-classification, 1x V100, identical settings except scheduler.enabled   |   "
        f"wall/node {a['wall']:.0f}s -> {b['wall']:.0f}s ({b['wall'] / a['wall']:.2f}x)   |   "
        f"pure exec {a['rep']:.0f}s -> {b['rep']:.0f}s ({b['rep'] / a['rep']:.2f}x)   |   "
        f"overhead {a['ovh']:.0f}s -> {b['ovh']:.0f}s   |   "
        f"span {a['span']:.1f}h -> {b['span']:.1f}h"
    )
    fig.suptitle(summary, fontsize=11, y=0.985)

    handles = [
        mpatches.Patch(color=OK_COLOR, label="execution (successful node)"),
        mpatches.Patch(color=BUG_COLOR, label="execution (buggy node)"),
        mpatches.Patch(color=WAIT_COLOR, label="queue / instrumentation before execution"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)

    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    for label, s in stats.items():
        print(
            f"  {label}: nodes={s['nodes']} ok={s['ok']} wall={s['wall']:.0f}s "
            f"rep={s['rep']:.0f}s ovh={s['ovh']:.0f}s vram={s['vram']:.0f}MiB "
            f"span={s['span']:.1f}h peak_conc={s['peak']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
