"""Draw one PetFinder comparison PNG from live MLEvolve journals.

Each run occupies one column. The top row is a Gantt view of completed node
windows and the bottom row is validation RMSE versus search step. The script
accepts any number of runs. The current comparison uses Codex GPT-5.6 Terra
Medium as the agent and runs both baseline and profile-based scheduler
workloads on the same A100.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine.node_accounting import node_counts_toward_budget

DEFAULT_OUTPUT = REPO / "records/petfinder_a100_a10_scheduler_comparison.png"
EXECUTION_COLOR = "#2b7bba"
FAILED_COLOR = "#c44e52"
BEST_COLOR = "#d1622b"


@dataclass(frozen=True, slots=True)
class RunSpec:
    label: str
    hardware: str
    journal_paths: tuple[Path, ...]
    target_nodes: int = 50
    include_all_executions: bool = False


@dataclass(frozen=True, slots=True)
class NodeWindow:
    node_id: str
    step: int
    created_at: float
    finished_at: float
    execution_seconds: float
    is_buggy: bool
    metric: float | None

    @property
    def execution_started_at(self) -> float:
        return max(self.created_at, self.finished_at - self.execution_seconds)


@dataclass(frozen=True, slots=True)
class LoadedRun:
    spec: RunSpec
    nodes: tuple[NodeWindow, ...]
    source_journal_nodes: int

    @property
    def completed_nodes(self) -> int:
        return len(self.nodes)

    @property
    def metric_points(self) -> list[tuple[int, float]]:
        return [
            (node.step, node.metric)
            for node in sorted(self.nodes, key=lambda item: item.step)
            if node.metric is not None
        ]


def _timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_run(spec: RunSpec) -> LoadedRun:
    nodes: list[NodeWindow] = []
    source_non_root_nodes = 0
    has_root = False
    for journal_path in spec.journal_paths:
        payload = json.loads(journal_path.read_text())
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ValueError(f"journal has no node list: {journal_path}")
        for raw in raw_nodes:
            if raw.get("stage") == "root":
                has_root = True
                continue
            source_non_root_nodes += 1
            if not spec.include_all_executions and not node_counts_toward_budget(raw):
                continue
            created_time = raw.get("created_time")
            finish_time = raw.get("finish_time")
            if not created_time or not finish_time:
                continue
            raw_metric = raw.get("metric") or {}
            metric_value = raw_metric.get("value")
            metric = float(metric_value) if metric_value is not None else None
            created_at = _timestamp(str(created_time))
            finished_at = _timestamp(str(finish_time))
            if finished_at < created_at:
                raise ValueError(f"node {raw.get('id')} finishes before it starts")
            nodes.append(
                NodeWindow(
                    node_id=str(raw.get("id") or "unknown"),
                    step=0,
                    created_at=created_at,
                    finished_at=finished_at,
                    execution_seconds=max(0.0, float(raw.get("exec_time") or 0.0)),
                    is_buggy=bool(raw.get("is_buggy")),
                    metric=metric,
                )
            )
    ordered_nodes = sorted(nodes, key=lambda item: item.created_at)
    if not spec.include_all_executions:
        ordered_nodes = ordered_nodes[: spec.target_nodes]
    numbered_nodes = tuple(
        NodeWindow(
            node_id=node.node_id,
            step=index,
            created_at=node.created_at,
            finished_at=node.finished_at,
            execution_seconds=node.execution_seconds,
            is_buggy=node.is_buggy,
            metric=node.metric,
        )
        for index, node in enumerate(ordered_nodes, start=1)
    )
    return LoadedRun(
        spec=spec,
        nodes=numbered_nodes,
        # Continuation journals each serialize a root, but they belong to one
        # logical search trace. Count that root once in a merged full trace.
        source_journal_nodes=source_non_root_nodes + int(has_root),
    )


def peak_execution_concurrency(nodes: Sequence[NodeWindow]) -> int:
    events: list[tuple[float, int]] = []
    for node in nodes:
        if node.execution_seconds <= 0.0:
            continue
        events.append((node.execution_started_at, 1))
        events.append((node.finished_at, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def _run_summary(run: LoadedRun) -> dict[str, float | int]:
    positive_exec = [
        node.execution_seconds
        for node in run.nodes
        if node.execution_seconds > 0.0
    ]
    if run.nodes:
        span_seconds = max(node.finished_at for node in run.nodes) - min(
            node.created_at for node in run.nodes
        )
    else:
        span_seconds = 0.0
    return {
        "completed": run.completed_nodes,
        "valid": sum(node.metric is not None for node in run.nodes),
        "span_hours": span_seconds / 3600.0,
        "median_exec_seconds": (
            statistics.median(positive_exec) if positive_exec else 0.0
        ),
        "peak_concurrency": peak_execution_concurrency(run.nodes),
    }


def _draw_gantt(ax, run: LoadedRun) -> None:
    summary = _run_summary(run)
    if not run.nodes:
        ax.text(
            0.5,
            0.5,
            "no completed nodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    origin = min(node.created_at for node in run.nodes)
    for lane, node in enumerate(run.nodes):
        start_h = (node.created_at - origin) / 3600.0
        execution_start_h = (node.execution_started_at - origin) / 3600.0
        finish_h = (node.finished_at - origin) / 3600.0
        execution_h = max(0.0, finish_h - execution_start_h)
        if execution_h > 0.0:
            ax.broken_barh(
                [(execution_start_h, execution_h)],
                (lane - 0.38, 0.76),
                facecolors=FAILED_COLOR if node.is_buggy else EXECUTION_COLOR,
            )
        elif node.is_buggy:
            ax.broken_barh(
                [(start_h, max((finish_h - start_h), 1.0 / 3600.0))],
                (lane - 0.38, 0.76),
                facecolors=FAILED_COLOR,
            )

    ax.set_xlabel("hours since first node started")
    ax.set_ylabel("node by start order")
    count_summary = (
        f"{summary['completed']} executions from {run.source_journal_nodes} journal nodes, "
        if run.spec.include_all_executions
        else f"{summary['completed']}/{run.spec.target_nodes} budget-counted, "
    )
    ax.set_title(
        f"{run.spec.label} — {run.spec.hardware}\n"
        f"{count_summary}"
        f"{summary['valid']} scored, span {summary['span_hours']:.2f} h, "
        f"peak execution concurrency {summary['peak_concurrency']}",
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.25)


def _draw_metrics(ax, run: LoadedRun) -> None:
    points = run.metric_points
    if not points:
        ax.text(
            0.5,
            0.5,
            "no scored nodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        running_best: list[float] = []
        best: float | None = None
        for value in ys:
            best = value if best is None else min(best, value)
            running_best.append(best)
        ax.plot(xs, ys, "o", ms=4, color=EXECUTION_COLOR, alpha=0.65, label="node RMSE")
        ax.plot(xs, running_best, "-", lw=2, color=BEST_COLOR, label="running best")
        ax.set_title(
            f"best RMSE {min(ys):.4f} from {len(points)} scored nodes", fontsize=9
        )
        ax.legend(fontsize=8)
    ax.set_xlabel(
        "execution order" if run.spec.include_all_executions else "budget-counted node"
    )
    ax.set_ylabel("validation RMSE (lower is better)")
    ax.set_xlim(
        0,
        max(
            run.completed_nodes if run.spec.include_all_executions else run.spec.target_nodes,
            max((node.step for node in run.nodes), default=0),
        )
        + 1,
    )
    ax.grid(alpha=0.25)


def render_comparison(runs: Sequence[LoadedRun], output: Path) -> None:
    if not runs:
        raise ValueError("at least one run is required")
    figure, axes = plt.subplots(
        2, len(runs), figsize=(6.2 * len(runs), 10), squeeze=False
    )
    for column, run in enumerate(runs):
        _draw_gantt(axes[0][column], run)
        _draw_metrics(axes[1][column], run)

    metric_values = [metric for run in runs for _, metric in run.metric_points]
    if metric_values:
        lower = min(metric_values)
        upper = max(metric_values)
        padding = max(0.25, (upper - lower) * 0.08)
        for axis in axes[1]:
            axis.set_ylim(lower - padding, upper + padding)

    figure.suptitle(
        "PetFinder Pawpularity | execution-only trace | profile-based scheduling",
        fontsize=12,
    )
    figure.legend(
        handles=[
            mpatches.Patch(color=EXECUTION_COLOR, label="successful execution"),
            mpatches.Patch(color=FAILED_COLOR, label="failed or rejected node"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.04, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _parse_run_argument(value: str, target_nodes: int) -> RunSpec:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("--run must be LABEL|HARDWARE|JOURNAL_PATH")
    paths = tuple(Path(path) for path in parts[2].split(",") if path)
    return RunSpec(parts[0], parts[1], paths, target_nodes=target_nodes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="repeat as LABEL|HARDWARE|JOURNAL_PATH",
    )
    parser.add_argument("--target-nodes", type=int, default=50)
    parser.add_argument(
        "--all-executions",
        action="store_true",
        help="draw every completed non-root execution; use for a full trace, not equal-budget timing",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    specs = [
        RunSpec(
            label=spec.label,
            hardware=spec.hardware,
            journal_paths=spec.journal_paths,
            target_nodes=spec.target_nodes,
            include_all_executions=args.all_executions,
        )
        for spec in (_parse_run_argument(value, args.target_nodes) for value in args.run)
    ]
    runs = [load_run(spec) for spec in specs]
    render_comparison(runs, args.out)
    print(f"wrote {args.out}")
    for run in runs:
        summary = _run_summary(run)
        print(
            f"{run.spec.label}: completed={summary['completed']}/{run.spec.target_nodes} "
            f"scored={summary['valid']} span_h={summary['span_hours']:.2f} "
            f"median_exec_s={summary['median_exec_seconds']:.2f} "
            f"peak_concurrency={summary['peak_concurrency']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
