"""Compare baseline and hardware-aware MLEvolve run metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json


KEYS = [
    "mode",
    "total_wall_time_seconds",
    "total_job_execution_time_seconds",
    "median_job_execution_time_seconds",
    "node_count",
    "valid_count",
    "buggy_count",
    "best_metric",
    "metric_direction",
    "time_to_best_seconds",
    "nodes_to_best",
    "jobs_per_hour",
    "packed_dispatch_count",
    "batch_probe_hit_count",
    "batch_probe_trial_count",
    "timeout_failures",
    "oom_failures",
    "peak_vram_mb",
    "average_vram_mb",
    "scheduler_backend_distribution",
]


def load_metrics(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_dir():
        direct = candidate / "comparison_metrics.json"
        nested = candidate / "logs" / "comparison_metrics.json"
        candidate = direct if direct.exists() else nested
    return json.loads(candidate.read_text(encoding="utf-8"))


def compare_metrics(baseline: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for key in KEYS:
        left = baseline.get(key)
        right = hardware.get(key)
        rows[key] = {
            "baseline": left,
            "hardware_aware": right,
            "delta": _delta(left, right),
        }
    return {
        "baseline_run_id": baseline.get("run_id"),
        "hardware_run_id": hardware.get("run_id"),
        "metrics": rows,
    }


def to_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "| Metric | Baseline | Hardware-aware | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, row in comparison["metrics"].items():
        lines.append(
            f"| `{key}` | {_fmt(row['baseline'])} | {_fmt(row['hardware_aware'])} | {_fmt(row['delta'])} |"
        )
    return "\n".join(lines)


def _delta(left: Any, right: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return right - left
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLEvolve experiment run metrics")
    parser.add_argument("baseline_run")
    parser.add_argument("hardware_run")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    comparison = compare_metrics(load_metrics(args.baseline_run), load_metrics(args.hardware_run))
    if args.format == "markdown":
        output = to_markdown(comparison)
    else:
        output = json.dumps(comparison, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
