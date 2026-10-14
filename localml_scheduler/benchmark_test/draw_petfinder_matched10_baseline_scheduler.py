"""Render the controlled first-ten PetFinder baseline/Scheduler comparison.

The Scheduler timestamps and metrics below are transcribed from the completed
Scheduler SQLite trace while the A10 pod was reachable.  The raw trace remains
on the pod; this compact renderer intentionally uses only its first ten
frozen-source records so the plotted comparison is candidate-for-candidate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "records/2026-09-17_petfinder_0902_original_baseline_40_frozen_sources/manifest.json"
BASELINE = REPO / ".cache/petfinder_a100_a10_live/baseline_final_50_20260902/journal.json"
OUTPUT = REPO / "records/2026-09-17_petfinder_matched10_baseline_vs_scheduler.png"

SUCCESS = "#2b7bba"
BUGGY = "#c44e52"
SCHEDULER = "#3a9d5d"

# Exact terminal timestamps and metrics obtained from the completed Scheduler
# trace for the frozen source indices 1..10. None is a candidate runtime error.
SCHEDULER_ROWS = {
    1: ("2026-09-17T04:33:17.792858+00:00", "2026-09-17T04:48:42.621208+00:00", None),
    2: ("2026-09-17T04:35:17.055483+00:00", "2026-09-17T05:08:56.336214+00:00", 18.1325),
    3: ("2026-09-17T04:40:12.762859+00:00", "2026-09-17T05:10:04.255101+00:00", None),
    4: ("2026-09-17T04:38:16.073446+00:00", "2026-09-17T04:40:11.697538+00:00", None),
    5: ("2026-09-17T04:48:57.224448+00:00", "2026-09-17T05:05:03.048146+00:00", 18.550390463960046),
    6: ("2026-09-17T05:10:07.213740+00:00", "2026-09-17T05:15:09.440531+00:00", 20.616786260437955),
    7: ("2026-09-17T05:10:05.417692+00:00", "2026-09-17T05:26:20.586058+00:00", 18.7380),
    8: ("2026-09-17T05:14:21.787971+00:00", "2026-09-17T05:15:44.982219+00:00", None),
    9: ("2026-09-17T05:26:23.406117+00:00", "2026-09-17T05:29:21.691123+00:00", 18.612476773290204),
    10: ("2026-09-17T05:29:22.482464+00:00", "2026-09-17T05:48:59.663308+00:00", 18.5495),
}


def epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def baseline_rows() -> list[tuple[int, float, float, float | None]]:
    ids = [item["node_id"] for item in json.loads(MANIFEST.read_text())[:10]]
    by_id = {item["id"]: item for item in json.loads(BASELINE.read_text())["nodes"]}
    return [
        (
            index,
            epoch(str(by_id[node_id]["created_time"])),
            epoch(str(by_id[node_id]["finish_time"])),
            (by_id[node_id].get("metric") or {}).get("value"),
        )
        for index, node_id in enumerate(ids, start=1)
    ]


def scheduler_rows() -> list[tuple[int, float, float, float | None]]:
    return [(index, epoch(start), epoch(finish), metric) for index, (start, finish, metric) in SCHEDULER_ROWS.items()]


def gantt(ax: plt.Axes, rows: list[tuple[int, float, float, float | None]], title: str) -> None:
    origin = min(start for _, start, _, _ in rows)
    for index, start, finish, metric in rows:
        ax.broken_barh(
            [((start - origin) / 60, (finish - start) / 60)],
            (index - 0.36, 0.72),
            facecolors=SUCCESS if metric is not None else BUGGY,
        )
    span = (max(finish for _, _, finish, _ in rows) - origin) / 60
    ax.set_title(f"{title} — first 10 matched sources ({span:.1f} min)")
    ax.set_xlabel("minutes from first execution start")
    ax.set_ylabel("frozen source index")
    ax.set_yticks(range(1, 11))
    ax.set_ylim(10.7, 0.3)
    ax.grid(axis="x", alpha=0.25)


def metric_plot(ax: plt.Axes, baseline: list[tuple[int, float, float, float | None]], scheduler: list[tuple[int, float, float, float | None]]) -> None:
    for label, rows, color, marker in (
        ("Original MLEvolve baseline", baseline, SUCCESS, "o"),
        ("Scheduler", scheduler, SCHEDULER, "s"),
    ):
        points = [(index, float(metric)) for index, _, _, metric in rows if metric is not None]
        x, y = zip(*points)
        ax.plot(x, y, marker=marker, color=color, linewidth=1.8, markersize=5, label=label)
    ax.set_xlabel("frozen source index (scored candidates only)")
    ax.set_ylabel("validation RMSE (lower is better)")
    ax.set_title("Per-source metric comparison: aligned, but not bit-identical")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")


def main() -> None:
    baseline = baseline_rows()
    scheduler = scheduler_rows()
    baseline_span = max(item[2] for item in baseline) - min(item[1] for item in baseline)
    scheduler_span = max(item[2] for item in scheduler) - min(item[1] for item in scheduler)
    speedup = baseline_span / scheduler_span

    figure = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.55, 1.0))
    gantt(figure.add_subplot(grid[0, 0]), baseline, "Original MLEvolve baseline")
    gantt(figure.add_subplot(grid[0, 1]), scheduler, "Scheduler-only replay")
    metric_plot(figure.add_subplot(grid[1, :]), baseline, scheduler)
    figure.legend(
        handles=[
            patches.Patch(color=SUCCESS, label="successful / scored"),
            patches.Patch(color=BUGGY, label="buggy candidate runtime error"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
    )
    figure.suptitle(
        "PetFinder, identical frozen sources and seed 42: matched first-10 execution audit\n"
        f"Baseline {baseline_span / 60:.1f} min vs Scheduler {scheduler_span / 60:.1f} min ({speedup:.3f}x)",
        fontsize=15,
        y=1.03,
    )
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
