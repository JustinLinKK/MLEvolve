"""Draw the complete 9/2 baseline beside the unfinished Scheduler-only trace.

This figure intentionally compares the full recorded execution timelines but
does not call the partial Scheduler trace a completed speedup experiment.
Blue bars are successful scored executions; red bars are candidate runtime
errors.  Unstarted READY/PENDING jobs have no execution bar and are annotated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
BASELINE_JOURNAL = REPO / ".cache/petfinder_a100_a10_live/baseline_final_50_20260902/journal.json"
SCHEDULER_EVENTS = REPO / "records/2026-09-15_petfinder_scheduler_only_predictor_a10_live_raw/events_terminal_stall_20260915T2031Z.jsonl"
OUTPUT = REPO / "records/2026-09-15_petfinder_sep2_baseline_vs_scheduler_partial.png"

SUCCESS = "#2b7bba"
BUGGY = "#c44e52"
BEST = "#d1622b"


def epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # The 9/2 journal serializes UTC wall-clock strings without an offset,
    # while its root timestamp is already Unix UTC seconds.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_baseline() -> tuple[list[dict[str, object]], list[tuple[float, float]]]:
    payload = json.loads(BASELINE_JOURNAL.read_text())
    root_start = next(node["ctime"] for node in payload["nodes"] if node.get("stage") == "root")
    rows: list[dict[str, object]] = []
    metrics: list[tuple[float, float]] = []
    for node in payload["nodes"]:
        if node.get("stage") == "root" or not node.get("finish_time"):
            continue
        finish = epoch(str(node["finish_time"]))
        duration = max(0.0, float(node.get("exec_time") or 0.0))
        start = max(float(node.get("ctime") or root_start), finish - duration)
        rows.append(
            {
                "start_h": (start - root_start) / 3600.0,
                "duration_h": duration / 3600.0,
                "finish": finish,
                "buggy": bool(node.get("is_buggy")),
                "metric": (node.get("metric") or {}).get("value"),
            }
        )
    for row in sorted(rows, key=lambda item: float(item["finish"])):
        metric = row["metric"]
        if metric is not None:
            metrics.append((float(row["finish"]), float(metric)))
    return sorted(rows, key=lambda item: float(item["start_h"])), metrics


def load_scheduler() -> tuple[list[dict[str, object]], list[tuple[float, float]], float]:
    events = [json.loads(line) for line in SCHEDULER_EVENTS.read_text().splitlines() if line]
    origin = min(epoch(event["created_at"]) for event in events if event["event_type"] == "job_dispatched")
    per_job: dict[str, dict[str, object]] = {}
    for event in events:
        job_id = event.get("job_id")
        if not job_id:
            continue
        record = per_job.setdefault(str(job_id), {})
        event_type = event["event_type"]
        if event_type == "job_started":
            record["start"] = epoch(event["created_at"])
        elif event_type in {"job_completed", "job_failed"}:
            record["finish"] = epoch(event["created_at"])
            record["buggy"] = event_type == "job_failed"
            record["metric"] = (event.get("payload") or {}).get("training_summary", {}).get("best_metric")
    rows: list[dict[str, object]] = []
    metrics: list[tuple[float, float]] = []
    for record in per_job.values():
        if "start" not in record or "finish" not in record:
            continue
        start = float(record["start"])
        finish = float(record["finish"])
        rows.append(
            {
                "start_h": (start - origin) / 3600.0,
                "duration_h": max(0.0, (finish - start) / 3600.0),
                "finish": finish,
                "buggy": bool(record["buggy"]),
                "metric": record.get("metric"),
            }
        )
    for row in sorted(rows, key=lambda item: float(item["finish"])):
        metric = row["metric"]
        if metric is not None and not bool(row["buggy"]):
            metrics.append((float(row["finish"]), float(metric)))
    last_event = max(epoch(event["created_at"]) for event in events)
    return sorted(rows, key=lambda item: float(item["start_h"])), metrics, (last_event - origin) / 3600.0


def draw_gantt(ax: plt.Axes, rows: list[dict[str, object]], title: str, note: str) -> None:
    for lane, row in enumerate(rows, start=1):
        color = BUGGY if bool(row["buggy"]) else SUCCESS
        ax.broken_barh([(float(row["start_h"]), float(row["duration_h"]))], (lane - 0.35, 0.7), facecolors=color)
    ax.set_ylim(0, len(rows) + 1)
    ax.set_ylabel("execution attempt\n(start order)")
    ax.set_xlabel("hours from trace start")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.01, 0.98, note, transform=ax.transAxes, va="top", fontsize=8)


def draw_metrics(ax: plt.Axes, baseline: list[tuple[float, float]], scheduler: list[tuple[float, float]]) -> None:
    def curve(values: list[tuple[float, float]]) -> tuple[list[int], list[float], list[float]]:
        scores = [metric for _, metric in values]
        best: list[float] = []
        current = float("inf")
        for score in scores:
            current = min(current, score)
            best.append(current)
        return list(range(1, len(scores) + 1)), scores, best

    baseline_x, baseline_scores, baseline_best = curve(baseline)
    scheduler_x, scheduler_scores, scheduler_best = curve(scheduler)
    ax.plot(baseline_x, baseline_scores, "o", ms=3, alpha=0.55, color=SUCCESS, label="9/2 baseline RMSE")
    ax.plot(baseline_x, baseline_best, "-", lw=2, color=SUCCESS, label="9/2 baseline running best")
    ax.plot(scheduler_x, scheduler_scores, "s", ms=4, alpha=0.7, color=BUGGY, label="Scheduler diagnostic RMSE")
    ax.plot(scheduler_x, scheduler_best, "-", lw=2, color=BEST, label="Scheduler diagnostic running best")
    ax.set_xlabel("successful completed-node rank")
    ax.set_ylabel("validation RMSE (lower is better)")
    ax.set_title("Metric by successful completion rank — diagnostic uses a different 9,912-row input")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="best")


def main() -> None:
    baseline_rows, baseline_metrics = load_baseline()
    scheduler_rows, scheduler_metrics, scheduler_span_h = load_scheduler()
    baseline_success = sum(not bool(row["buggy"]) for row in baseline_rows)
    scheduler_success = sum(not bool(row["buggy"]) for row in scheduler_rows)

    figure, axes = plt.subplots(3, 1, figsize=(15, 16), constrained_layout=True, height_ratios=[1.4, 1.0, 0.85])
    draw_gantt(
        axes[0],
        baseline_rows,
        "9/2 original MLEvolve baseline — full retained trace",
        f"{len(baseline_rows)} finished attempts: {baseline_success} successful, {len(baseline_rows) - baseline_success} buggy",
    )
    draw_gantt(
        axes[1],
        scheduler_rows,
        "15 Sep Scheduler + Predictor v3.2 diagnostic — terminal partial trace",
        f"{len(scheduler_rows)} dispatched/finished: {scheduler_success} successful, {len(scheduler_rows) - scheduler_success} buggy; "
        f"stalled after {scheduler_span_h:.2f} h with 32 READY and 2 PENDING jobs",
    )
    draw_metrics(axes[2], baseline_metrics, scheduler_metrics)
    figure.legend(
        handles=[
            mpatches.Patch(color=SUCCESS, label="successful execution"),
            mpatches.Patch(color=BUGGY, label="buggy candidate runtime error"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
