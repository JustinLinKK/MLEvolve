"""Plot baseline vs hardware-aware MLEvolve comparison metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised by runtime environment
    raise SystemExit(
        "matplotlib is required to generate comparison plots. "
        "Install it with `pip install matplotlib` or from requirements_base.txt."
    ) from exc


DEFAULT_MODES = ("baseline", "hardware_aware")

SUMMARY_METRICS = [
    ("total_wall_time_seconds", "Total wall time (s)"),
    ("total_job_execution_time_seconds", "Job exec total (s)"),
    ("median_job_execution_time_seconds", "Median job time (s)"),
    ("time_to_best_seconds", "Time to best (s)"),
    ("jobs_per_hour", "Jobs per hour"),
    ("best_metric", "Best metric"),
    ("node_count", "Nodes"),
    ("valid_count", "Valid nodes"),
    ("buggy_count", "Buggy nodes"),
    ("timeout_failures", "Timeout failures"),
    ("oom_failures", "OOM failures"),
    ("avg_cpu_percent", "Avg CPU (%)"),
    ("max_cpu_percent", "Max CPU (%)"),
    ("avg_ram_percent", "Avg RAM (%)"),
    ("max_ram_percent", "Max RAM (%)"),
    ("avg_gpu_util_percent", "Avg GPU util (%)"),
    ("max_gpu_util_percent", "Max GPU util (%)"),
    ("avg_gpu_memory_percent", "Avg GPU memory (%)"),
    ("max_gpu_memory_percent", "Max GPU memory (%)"),
    ("peak_gpu_memory_used_mb", "Peak GPU memory (MB)"),
]

BAR_GROUPS = [
    (
        "Runtime",
        [
            ("total_wall_time_seconds", "Wall time"),
            ("total_job_execution_time_seconds", "Job total"),
            ("median_job_execution_time_seconds", "Median job"),
            ("time_to_best_seconds", "Time to best"),
        ],
        "Seconds",
    ),
    (
        "Search Outcome",
        [
            ("best_metric", "Best metric"),
            ("node_count", "Nodes"),
            ("valid_count", "Valid"),
            ("buggy_count", "Buggy"),
        ],
        "Raw value",
    ),
    (
        "Utilization",
        [
            ("avg_cpu_percent", "Avg CPU"),
            ("max_cpu_percent", "Max CPU"),
            ("avg_ram_percent", "Avg RAM"),
            ("max_ram_percent", "Max RAM"),
            ("avg_gpu_util_percent", "Avg GPU"),
            ("max_gpu_util_percent", "Max GPU"),
            ("avg_gpu_memory_percent", "Avg GPU mem"),
            ("max_gpu_memory_percent", "Max GPU mem"),
        ],
        "Percent",
    ),
    (
        "Scheduler And Failures",
        [
            ("jobs_per_hour", "Jobs/hour"),
            ("packed_dispatch_count", "Packed"),
            ("batch_probe_hit_count", "Probe hits"),
            ("batch_probe_trial_count", "Probe trials"),
            ("timeout_failures", "Timeouts"),
            ("oom_failures", "OOMs"),
        ],
        "Raw value",
    ),
]

TIME_SERIES_COLUMNS = [
    ("cpu_percent_avg", "CPU avg (%)"),
    ("ram_percent", "RAM (%)"),
    ("gpu_util_percent", "GPU util (%)"),
    ("gpu_memory_percent", "GPU memory (%)"),
]


@dataclass
class ModeArtifacts:
    mode: str
    mode_root: Path
    log_dir: Path | None
    metrics_path: Path | None
    hardware_samples_path: Path | None
    metrics: dict[str, Any]
    hardware_series: list[dict[str, float | None]]
    hardware_summary: dict[str, float | int]

    @property
    def combined_metrics(self) -> dict[str, Any]:
        return {**self.metrics, **self.hardware_summary}


def load_mode_artifacts(run_root: Path, mode: str) -> ModeArtifacts:
    mode_root = run_root / mode
    log_dir = _find_latest_log_dir(mode_root)
    metrics_path = _find_latest_file(mode_root, "comparison_metrics.json")
    hardware_samples_path = _find_latest_file(mode_root, "hardware_samples.csv")

    metrics: dict[str, Any] = {}
    if metrics_path is not None:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    raw_hardware_rows = _read_hardware_samples(hardware_samples_path) if hardware_samples_path else []
    hardware_series = _aggregate_hardware_series(raw_hardware_rows)
    hardware_summary = _summarize_hardware_series(hardware_series)

    return ModeArtifacts(
        mode=mode,
        mode_root=mode_root,
        log_dir=log_dir,
        metrics_path=metrics_path,
        hardware_samples_path=hardware_samples_path,
        metrics=metrics,
        hardware_series=hardware_series,
        hardware_summary=hardware_summary,
    )


def write_summary(artifacts: list[ModeArtifacts], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "modes": {
            artifact.mode: {
                "mode_root": str(artifact.mode_root),
                "log_dir": str(artifact.log_dir) if artifact.log_dir else None,
                "metrics_path": str(artifact.metrics_path) if artifact.metrics_path else None,
                "hardware_samples_path": (
                    str(artifact.hardware_samples_path) if artifact.hardware_samples_path else None
                ),
                "metrics": artifact.combined_metrics,
            }
            for artifact in artifacts
        },
        "delta_definition": "hardware_aware - baseline",
    }
    json_path = output_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    markdown_path = output_dir / "comparison_summary.md"
    markdown_path.write_text(_summary_markdown(artifacts) + "\n", encoding="utf-8")
    return json_path, markdown_path


def plot_metric_dashboard(artifacts: list[ModeArtifacts], output_path: Path) -> bool:
    if not any(artifact.combined_metrics for artifact in artifacts):
        return False

    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes_list = list(axes.ravel())
    colors = _mode_colors(artifacts)

    for ax, (title, metric_defs, ylabel) in zip(axes_list, BAR_GROUPS):
        available = [
            (key, label)
            for key, label in metric_defs
            if any(_numeric_value(artifact.combined_metrics.get(key)) is not None for artifact in artifacts)
        ]
        if not available:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        x_positions = list(range(len(available)))
        width = min(0.35, 0.8 / max(1, len(artifacts)))
        for index, artifact in enumerate(artifacts):
            offset = (index - (len(artifacts) - 1) / 2) * width
            values = [
                _numeric_value(artifact.combined_metrics.get(key)) or 0.0
                for key, _label in available
            ]
            bars = ax.bar(
                [x + offset for x in x_positions],
                values,
                width=width,
                label=_display_mode(artifact.mode),
                color=colors[artifact.mode],
            )
            _annotate_bars(ax, bars)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _key, label in available], rotation=25, ha="right")
        ax.margins(y=0.18)
        ax.legend(loc="best")

    fig.suptitle("Baseline vs Hardware-aware MLEvolve Metrics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_utilization_timeseries(artifacts: list[ModeArtifacts], output_path: Path) -> bool:
    available_columns = [
        (key, label)
        for key, label in TIME_SERIES_COLUMNS
        if any(_has_series_value(artifact.hardware_series, key) for artifact in artifacts)
    ]
    if not available_columns:
        return False

    _apply_style()
    fig, axes = plt.subplots(
        len(available_columns),
        1,
        figsize=(14, max(3.5, 3.0 * len(available_columns))),
        sharex=True,
    )
    if len(available_columns) == 1:
        axes = [axes]
    colors = _mode_colors(artifacts)

    for ax, (key, label) in zip(axes, available_columns):
        for artifact in artifacts:
            points = [
                (sample["elapsed_seconds"], sample[key])
                for sample in artifact.hardware_series
                if sample.get("elapsed_seconds") is not None and sample.get(key) is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, label=_display_mode(artifact.mode), color=colors[artifact.mode], linewidth=2)
        ax.set_ylabel(label)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Elapsed seconds")
    fig.suptitle("Hardware Utilization Over Time", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _find_latest_log_dir(mode_root: Path) -> Path | None:
    candidates: list[Path] = []
    for path in _candidate_log_dirs(mode_root):
        if path.is_dir():
            candidates.append(path)
    if not candidates:
        return None
    artifact_candidates = [
        path
        for path in candidates
        if (path / "comparison_metrics.json").is_file() or (path / "hardware_samples.csv").is_file()
    ]
    return max(artifact_candidates or candidates, key=_log_dir_mtime)


def _find_latest_file(mode_root: Path, filename: str) -> Path | None:
    candidates = [path / filename for path in _candidate_log_dirs(mode_root)]
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=_path_mtime)


def _candidate_log_dirs(mode_root: Path) -> list[Path]:
    candidates = [mode_root, mode_root / "logs"]
    runs_root = mode_root / "runs"
    if runs_root.is_dir():
        candidates.extend(runs_root.glob("*/logs"))
    return candidates


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _log_dir_mtime(path: Path) -> float:
    return max(
        _path_mtime(path),
        _path_mtime(path / "comparison_metrics.json"),
        _path_mtime(path / "hardware_samples.csv"),
    )


def _read_hardware_samples(path: Path | None) -> list[dict[str, float | None]]:
    if path is None:
        return []

    rows: list[dict[str, float | None]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "elapsed_seconds": _parse_float(row.get("elapsed_seconds")),
                    "cpu_percent_avg": _parse_float(row.get("cpu_percent_avg")),
                    "cpu_percent_max": _parse_float(row.get("cpu_percent_max")),
                    "ram_percent": _parse_float(row.get("ram_percent")),
                    "gpu_memory_used_mb": _parse_float(row.get("gpu_memory_used_mb")),
                    "gpu_memory_percent": _parse_float(row.get("gpu_memory_percent")),
                    "gpu_util_percent": _parse_float(row.get("gpu_util_percent")),
                    "gpu_memory_util_percent": _parse_float(row.get("gpu_memory_util_percent")),
                    "gpu_power_draw_w": _parse_float(row.get("gpu_power_draw_w")),
                }
            )
    return rows


def _aggregate_hardware_series(rows: list[dict[str, float | None]]) -> list[dict[str, float | None]]:
    grouped: dict[float, list[dict[str, float | None]]] = defaultdict(list)
    for row in rows:
        elapsed = row.get("elapsed_seconds")
        if elapsed is None:
            continue
        grouped[round(float(elapsed), 3)].append(row)

    series: list[dict[str, float | None]] = []
    for elapsed in sorted(grouped):
        samples = grouped[elapsed]
        series.append(
            {
                "elapsed_seconds": elapsed,
                "cpu_percent_avg": _mean_or_none(sample.get("cpu_percent_avg") for sample in samples),
                "cpu_percent_max": _max_or_none(sample.get("cpu_percent_max") for sample in samples),
                "ram_percent": _mean_or_none(sample.get("ram_percent") for sample in samples),
                "gpu_memory_used_mb": _max_or_none(sample.get("gpu_memory_used_mb") for sample in samples),
                "gpu_memory_percent": _max_or_none(sample.get("gpu_memory_percent") for sample in samples),
                "gpu_util_percent": _max_or_none(sample.get("gpu_util_percent") for sample in samples),
                "gpu_memory_util_percent": _max_or_none(
                    sample.get("gpu_memory_util_percent") for sample in samples
                ),
                "gpu_power_draw_w": _sum_or_none(sample.get("gpu_power_draw_w") for sample in samples),
            }
        )
    return series


def _summarize_hardware_series(series: list[dict[str, float | None]]) -> dict[str, float | int]:
    if not series:
        return {}
    summary: dict[str, float | int] = {
        "hardware_sample_count": len(series),
    }
    for source_key, summary_key, reducer in (
        ("cpu_percent_avg", "avg_cpu_percent", _mean_or_none),
        ("cpu_percent_max", "max_cpu_percent", _max_or_none),
        ("ram_percent", "avg_ram_percent", _mean_or_none),
        ("ram_percent", "max_ram_percent", _max_or_none),
        ("gpu_util_percent", "avg_gpu_util_percent", _mean_or_none),
        ("gpu_util_percent", "max_gpu_util_percent", _max_or_none),
        ("gpu_memory_percent", "avg_gpu_memory_percent", _mean_or_none),
        ("gpu_memory_percent", "max_gpu_memory_percent", _max_or_none),
        ("gpu_memory_used_mb", "peak_gpu_memory_used_mb", _max_or_none),
        ("gpu_power_draw_w", "avg_gpu_power_draw_w", _mean_or_none),
        ("gpu_power_draw_w", "max_gpu_power_draw_w", _max_or_none),
    ):
        value = reducer(sample.get(source_key) for sample in series)
        if value is not None:
            summary[summary_key] = value
    return summary


def _summary_markdown(artifacts: list[ModeArtifacts]) -> str:
    by_mode = {artifact.mode: artifact.combined_metrics for artifact in artifacts}
    baseline = by_mode.get("baseline", {})
    hardware = by_mode.get("hardware_aware", {})
    lines = [
        "# Hardware Awareness Comparison Summary",
        "",
        "Delta is `hardware_aware - baseline`.",
        "",
        "| Metric | Baseline | Hardware-aware | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in SUMMARY_METRICS:
        left = baseline.get(key)
        right = hardware.get(key)
        if left is None and right is None:
            continue
        lines.append(f"| {label} | {_format_value(left)} | {_format_value(right)} | {_format_value(_delta(left, right))} |")

    lines.extend(["", "## Artifacts", "", "| Mode | Metrics | Hardware samples |", "| --- | --- | --- |"])
    for artifact in artifacts:
        lines.append(
            "| {mode} | {metrics} | {samples} |".format(
                mode=_display_mode(artifact.mode),
                metrics=str(artifact.metrics_path) if artifact.metrics_path else "",
                samples=str(artifact.hardware_samples_path) if artifact.hardware_samples_path else "",
            )
        )
    return "\n".join(lines)


def _mode_colors(artifacts: list[ModeArtifacts]) -> dict[str, str]:
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    return {artifact.mode: palette[index % len(palette)] for index, artifact in enumerate(artifacts)}


def _display_mode(mode: str) -> str:
    return "Hardware-aware" if mode == "hardware_aware" else mode.replace("_", " ").title()


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")


def _annotate_bars(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        if not math.isfinite(height):
            continue
        ax.annotate(
            _short_number(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _has_series_value(series: list[dict[str, float | None]], key: str) -> bool:
    return any(sample.get(key) is not None for sample in series)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return _parse_float(value)


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(valid) if valid else None


def _max_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(valid) if valid else None


def _sum_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(valid) if valid else None


def _delta(left: Any, right: Any) -> float | None:
    left_number = _numeric_value(left)
    right_number = _numeric_value(right)
    if left_number is None or right_number is None:
        return None
    return right_number - left_number


def _format_value(value: Any) -> str:
    number = _numeric_value(value)
    if number is not None:
        return f"{number:.3f}"
    if value is None:
        return ""
    return str(value)


def _short_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matplotlib graphs for baseline vs hardware-aware MLEvolve runs."
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Root produced by compare_hardware_awareness.sh.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG graphs and summaries. Defaults to <run-root>/comparison_plots.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help="Mode subdirectories to compare. Defaults to baseline hardware_aware.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or (run_root / "comparison_plots")).resolve()

    artifacts = [load_mode_artifacts(run_root, mode) for mode in args.modes]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json, summary_markdown = write_summary(artifacts, output_dir)

    generated = [summary_json, summary_markdown]
    dashboard_path = output_dir / "comparison_metrics.png"
    if plot_metric_dashboard(artifacts, dashboard_path):
        generated.append(dashboard_path)

    timeseries_path = output_dir / "utilization_timeseries.png"
    if plot_utilization_timeseries(artifacts, timeseries_path):
        generated.append(timeseries_path)

    print("Generated comparison artifacts:")
    for path in generated:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
