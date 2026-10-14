"""Render the one-node deterministic PetFinder control/treatment gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _baseline_observation(raw_root: Path) -> tuple[float, float]:
    journal = next(raw_root.glob("baseline/*/logs/journal.json"))
    payload = json.loads(journal.read_text(encoding="utf-8"))
    nodes = [node for node in payload["nodes"] if node.get("stage") != "root"]
    if len(nodes) != 1:
        raise ValueError(f"expected one baseline candidate, found {len(nodes)}")
    node = nodes[0]
    return float(node["metric"]["value"]), float(node["exec_time"])


def _scheduler_observation(raw_root: Path) -> tuple[float, float]:
    summary = json.loads(
        (raw_root / "scheduler" / "summary.json").read_text(encoding="utf-8")
    )
    jobs = summary["jobs"]
    if len(jobs) != 1 or jobs[0]["status"] != "COMPLETED":
        raise ValueError("expected one completed Scheduler job")
    metadata = jobs[0]["metadata"]
    return float(metadata["best_metric"]), float(metadata["runtime_live_elapsed_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_metric, baseline_seconds = _baseline_observation(args.raw_root)
    scheduler_metric, scheduler_seconds = _scheduler_observation(args.raw_root)
    if abs(baseline_metric - scheduler_metric) > 1e-12:
        raise ValueError("deterministic gate metrics diverged")

    figure = plt.figure(figsize=(10.5, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 1.15))
    gantt = figure.add_subplot(grid[0])
    metric = figure.add_subplot(grid[1])

    rows = [("Baseline", baseline_seconds), ("Scheduler-only", scheduler_seconds)]
    for y, (label, seconds) in enumerate(rows):
        gantt.barh(y, seconds, color="#2b83ba", edgecolor="#174a72", height=0.52)
        gantt.text(seconds + 3, y, f"{seconds:.2f} s", va="center", fontsize=10)
    gantt.set_yticks(range(len(rows)), [row[0] for row in rows])
    gantt.invert_yaxis()
    gantt.set_xlabel("reported execution interval (seconds; separate runs)")
    gantt.set_title("Gantt: one successful frozen candidate per run")
    gantt.grid(axis="x", alpha=0.25)
    gantt.plot([], [], color="#2b83ba", linewidth=8, label="successful job")
    gantt.legend(loc="lower right")

    metric.plot([1], [baseline_metric], "o", color="#2b83ba", label="Baseline")
    metric.plot([1], [scheduler_metric], "s", color="#1b9e77", label="Scheduler-only")
    metric.annotate(
        f"{baseline_metric:.10f}",
        (1, baseline_metric),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )
    metric.set_xlim(0.5, 1.5)
    metric.set_xticks([1], ["1"])
    metric.set_xlabel("completed scored node")
    metric.set_ylabel("best validation RMSE (lower is better)")
    metric.set_title("Metric-node alignment: exact match")
    metric.grid(alpha=0.25)
    metric.legend(loc="best")

    figure.suptitle(
        "PetFinder deterministic scheduler gate — same frozen source, seed 42, A10",
        fontsize=14,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
