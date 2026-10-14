"""Render the Petfinder scheduler timeline and node metric evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
RESULT = RECORDS / "2026-08-28_petfinder_sonnet5090_liveprofile_result.json"
OUTPUT = RECORDS / "2026-08-28_petfinder_sonnet5090_liveprofile.png"


def main() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    summary = payload["training_summary"]
    points = summary["convergence_curve"]
    unique = {}
    for point in points:
        unique[int(point["epoch"])] = float(point["metric"])
    epochs = sorted(unique)
    metrics = [unique[epoch] for epoch in epochs]
    actual_seconds = float(payload["exec_time"])
    first_epoch_estimate = 49.0

    fig, (timeline, metric) = plt.subplots(
        2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [1, 1.35]}
    )
    timeline.broken_barh([(0, 336)], (20, 8), facecolors="#64748b", label="Sonnet generation")
    timeline.broken_barh([(336, actual_seconds)], (8, 8), facecolors="#2563eb", label="Scheduler job")
    timeline.axvline(336 + first_epoch_estimate, color="#f59e0b", linestyle="--", linewidth=1.5)
    timeline.text(336 + first_epoch_estimate + 3, 16, "epoch-1 estimate: 49.0 s", color="#92400e")
    timeline.text(338, 10, f"actual: {actual_seconds:.1f} s", color="white", va="center")
    timeline.set_ylim(4, 34)
    timeline.set_xlim(0, 336 + actual_seconds + 25)
    timeline.set_yticks([12, 24], ["RTX 5090 scheduler", "Claude Sonnet 5"])
    timeline.set_xlabel("Wall-clock seconds from run start")
    timeline.set_title("Petfinder task 1 — execution Gantt")
    timeline.legend(loc="upper right")
    timeline.grid(axis="x", alpha=0.25)

    metric.plot(epochs, metrics, marker="o", linewidth=2.4, color="#16a34a")
    for epoch, value in zip(epochs, metrics):
        metric.annotate(f"{value:.4f}", (epoch, value), xytext=(0, 9), textcoords="offset points", ha="center")
    metric.set_xticks(epochs)
    metric.set_xlabel("Epoch")
    metric.set_ylabel("Validation RMSE (lower is better)")
    metric.set_title("Scheduler node 7c4876a9 — EfficientNet-B0 tabular fusion")
    metric.grid(alpha=0.25)
    metric.text(
        0.02,
        0.06,
        "runtime profile: 49.0 s after epoch 1; observed completion: 103.3 s\n"
        "latest runner code calibrates matching profiles at completion\n"
        "parallel_job_cap = null; placement used exclusive cold-profile fallback",
        transform=metric.transAxes,
        fontsize=9,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.45"},
    )
    fig.suptitle("MLEvolve + branch-profile scheduler + HWKD — RTX 5090", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
