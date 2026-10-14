"""Render the audited 16 September Scheduler-only PetFinder experiment.

The top row gives execution windows, not queue/generation delay.  The bottom
row gives the score at completed search steps.  The comparison deliberately
uses the first ten *executed* attempts of the fixed 2 September baseline and
the ten Scheduler attempts that actually started in the one-hour run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO / ".cache/petfinder_a100_a10_live/baseline_final_50_20260902/journal.json"
DEFAULT_SCHEDULER = (
    REPO
    / "records/2026-09-16_petfinder_scheduler_v32_a10_one_hour_10executed_raw"
    / "mlevolve_petfinder_scheduler_v32_a10_20260916_081524"
)
DEFAULT_OUTPUT = REPO / "records/2026-09-16_petfinder_original46_vs_scheduler10_a10_combined.png"

SUCCESS = "#2b7bba"
BUG = "#c44e52"
EXTERNAL = "#8b8b8b"
METRIC = "#1f4e79"


@dataclass(frozen=True)
class Attempt:
    node_id: str
    start: float
    finish: float
    outcome: str  # success | buggy | external
    metric: float | None

    @property
    def duration(self) -> float:
        return max(0.0, self.finish - self.start)


def _timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _baseline_attempts(path: Path, target: int) -> list[Attempt]:
    """Use the original trace's historical 60-second budget contract.

    This selects the same kind of actual nodes that made up the documented
    46-node original MLEvolve baseline: all successful runs, and failures only
    after they consumed at least one minute of execution.  It is intentionally
    not a claim that short failures never ran.
    """
    rows = json.loads(path.read_text())["nodes"]
    attempts: list[Attempt] = []
    for row in rows:
        if row.get("stage") == "root":
            continue
        start_raw, finish_raw = row.get("created_time"), row.get("finish_time")
        if not start_raw or not finish_raw:
            continue
        buggy = bool(row.get("is_buggy"))
        execution = float(row.get("exec_time") or 0.0)
        if buggy and execution < 60.0:
            continue
        metric_raw = (row.get("metric") or {}).get("value")
        attempts.append(
            Attempt(
                node_id=str(row.get("id")),
                start=_timestamp(str(start_raw)),
                finish=_timestamp(str(finish_raw)),
                outcome="buggy" if buggy else "success",
                metric=float(metric_raw) if metric_raw is not None else None,
            )
        )
    # Completion order controls the first N historical budgeted nodes.
    return sorted(attempts, key=lambda item: item.finish)[:target]


def _scheduler_attempts(root: Path, target: int) -> list[Attempt]:
    db_path = root / "scheduler_runtime/db/scheduler.sqlite3"
    event_path = root / "scheduler_runtime/logs/events.jsonl"
    with sqlite3.connect(db_path) as connection:
        node_by_job = {
            job_id: str((json.loads(payload).get("metadata") or {}).get("mlevolve_node_id") or job_id)
            for job_id, payload in connection.execute("SELECT job_id, payload_json FROM jobs")
        }
    by_job: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_line in event_path.read_text().splitlines():
        event = json.loads(raw_line)
        if event["event_type"] not in {"job_started", "job_completed", "job_failed"}:
            continue
        record = by_job.setdefault(event["job_id"], {})
        record[event["event_type"]] = event

    attempts: list[Attempt] = []
    for job_id, records in by_job.items():
        started = records.get("job_started")
        terminal = records.get("job_completed") or records.get("job_failed")
        if not started or not terminal:
            continue
        terminal_payload = terminal.get("payload") or {}
        is_external = int(terminal_payload.get("returncode", terminal_payload.get("candidate_returncode", 0)) or 0) == -15
        if terminal["event_type"] == "job_completed":
            outcome = "success"
            metric_raw = (terminal_payload.get("training_summary") or {}).get("best_metric")
            metric = float(metric_raw) if metric_raw is not None else None
        elif is_external:
            outcome, metric = "external", None
        else:
            outcome, metric = "buggy", None
        attempts.append(
            Attempt(
                node_id=node_by_job.get(job_id, job_id),
                start=_timestamp(started["created_at"]),
                finish=_timestamp(terminal["created_at"]),
                outcome=outcome,
                metric=metric,
            )
        )
    return sorted(attempts, key=lambda item: item.start)[:target]


def _summary(items: Iterable[Attempt]) -> str:
    attempts = list(items)
    counts = {kind: sum(item.outcome == kind for item in attempts) for kind in ("success", "buggy", "external")}
    span = (max(item.finish for item in attempts) - min(item.start for item in attempts)) / 60.0
    return f"{len(attempts)} attempts; {counts['success']} scored; {counts['buggy']} runtime bugs; {counts['external']} external stops; {span:.1f} min span"


def _draw_gantt(ax: Any, title: str, attempts: list[Attempt]) -> None:
    origin = min(item.start for item in attempts)
    colors = {"success": SUCCESS, "buggy": BUG, "external": EXTERNAL}
    for lane, item in enumerate(attempts, start=1):
        ax.broken_barh(
            [((item.start - origin) / 60.0, max(item.duration / 60.0, 0.04))],
            (lane - 0.38, 0.76),
            facecolors=colors[item.outcome],
        )
    ax.set_title(f"{title}\n{_summary(attempts)}", fontsize=10, weight="bold", pad=9)
    ax.set_xlabel("minutes since first execution start")
    ax.set_ylabel("execution start order")
    ax.set_yticks(range(1, len(attempts) + 1))
    ax.grid(axis="x", alpha=0.25)


def _draw_metrics(ax: Any, attempts: list[Attempt], label: str) -> None:
    points = [(rank, item.metric) for rank, item in enumerate(attempts, start=1) if item.metric is not None]
    if points:
        x, y = zip(*points)
        ax.plot(x, y, "o-", color=METRIC, linewidth=1.7, markersize=4, label=label)
        for step, score in points:
            ax.annotate(f"{score:.2f}", (step, score), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no completed scores", transform=ax.transAxes, ha="center")
    ax.set_xlabel("execution attempt rank")
    ax.set_ylabel("best validation RMSE (lower is better)")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.5, len(attempts) + 0.5)


def render(baseline: Path, scheduler_root: Path, output: Path, target: int) -> None:
    baseline_attempts = _baseline_attempts(baseline, target)
    scheduler_attempts = _scheduler_attempts(scheduler_root, target)
    if len(baseline_attempts) != target or len(scheduler_attempts) != target:
        raise ValueError(f"need {target} attempts: baseline={len(baseline_attempts)}, scheduler={len(scheduler_attempts)}")

    figure, axes = plt.subplots(2, 2, figsize=(16, 11), height_ratios=(1.45, 1))
    figure.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.19, hspace=0.31, wspace=0.11)
    _draw_gantt(axes[0, 0], "Original MLEvolve baseline — first 10 of documented 46", baseline_attempts)
    _draw_gantt(axes[0, 1], "Scheduler-only (Predictor v3.2 + preflight) — 10 started", scheduler_attempts)
    _draw_metrics(axes[1, 0], baseline_attempts, "original baseline")
    _draw_metrics(axes[1, 1], scheduler_attempts, "scheduler")
    figure.legend(
        handles=[
            mpatches.Patch(color=SUCCESS, label="successful/scored execution"),
            mpatches.Patch(color=BUG, label="candidate runtime bug"),
            mpatches.Patch(color=EXTERNAL, label="external termination (not a candidate bug)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        fontsize=9,
    )
    figure.suptitle(
        "PetFinder, seed 42: equal-count execution audit (no fixed parallel-job cap)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.105,
        "This is an equal attempted-node visualization, not a speedup claim: the candidate workloads and failure mix differ.\n"
        "Scheduler run: Hardware Knowledge Database disabled; A10 execution; Qwen-3.8-27B-int8 local agent on A100.",
        ha="center",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--scheduler-root", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", type=int, default=10)
    args = parser.parse_args()
    render(args.baseline, args.scheduler_root, args.output, args.target)
    print(args.output)


if __name__ == "__main__":
    main()
