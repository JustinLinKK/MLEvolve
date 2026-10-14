"""Create required Gantt and metric-node comparison for two- versus three-V100 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_GPU = Path("results/qwen38_v100_int8_benchmark.json")
THREE_GPU = Path("results/qwen38_v100_int8_3gpu_benchmark.json")
OUT = Path("results/qwen38_v100_int8_2v3gpu_comparison.png")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    runs = [("2 V100", load(TWO_GPU)), ("3 V100", load(THREE_GPU))]
    fig, (gantt, metrics) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=(1, 1.35))
    colors = ["#2b7bba", "#719e55"]
    cursor = 0.0
    labels: list[str] = []
    rows: list[int] = []
    row = 0
    for (name, data), color in zip(runs, colors, strict=True):
        for request, record in enumerate(data["records"], start=1):
            gantt.broken_barh([(cursor, record["total_seconds"])], (row - 0.35, 0.7), facecolors=color)
            gantt.axvline(cursor + record["ttft_seconds"], color="#d1622b", linewidth=1.2)
            labels.append(f"{name} request {request}")
            rows.append(row)
            cursor += record["total_seconds"]
            row += 1
    gantt.set_yticks(rows, labels)
    gantt.set_xlabel("Sequential serving time (seconds)")
    gantt.set_title("Gantt: Qwen3.8-27B INT8, orange = first token")
    gantt.grid(axis="x", alpha=0.25)

    names = [item[0] for item in runs]
    ttft = [item[1]["median"]["ttft_seconds"] for item in runs]
    tps = [item[1]["median"]["tokens_per_second"] for item in runs]
    xs = list(range(len(names)))
    left = metrics.bar([index - 0.18 for index in xs], ttft, width=0.36, color="#d1622b", label="TTFT (seconds)")
    right_axis = metrics.twinx()
    right = right_axis.bar([index + 0.18 for index in xs], tps, width=0.36, color="#2b7bba", label="Tokens per second")
    metrics.bar_label(left, fmt="%.3f", padding=3)
    right_axis.bar_label(right, fmt="%.3f", padding=3)
    metrics.set_xticks(xs, names)
    metrics.set_ylabel("Time to first token (seconds)", color="#d1622b")
    right_axis.set_ylabel("Tokens per second", color="#2b7bba")
    metrics.set_title("Metric-node graph: median post-warmup measurements")
    metrics.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)


if __name__ == "__main__":
    main()
