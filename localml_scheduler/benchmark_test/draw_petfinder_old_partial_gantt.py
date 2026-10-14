"""Draw the evidence-backed partial Gantt for the externally terminated A10 run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "records/2026-09-14_petfinder_old_trace_partial_gantt.png"


def main() -> None:
    # Times are seconds from the first scheduler dispatch.  The first three
    # windows come from persisted scheduler completion events and result files.
    # The fourth ends at its last observed epoch heartbeat: its real end was
    # lost when Nautilus terminated the pod, so the bar is intentionally a
    # lower bound rather than an invented completion time.
    rows = [
        ("node 1 — valid", 0.0, 1350.77, "#2563eb", "RMSE 20.1020"),
        ("node 2 — KeyError", 1351.0, 310.14, "#dc2626", "trained; test-label bug"),
        ("node 3 — AttributeError", 1698.0, 29.79, "#dc2626", "early candidate bug"),
        ("node 4 — external stop", 2018.0, 34.5, "#64748b", "running ≥34.5 s"),
    ]

    fig, (gantt, metric) = plt.subplots(
        2, 1, figsize=(13, 8.2), gridspec_kw={"height_ratios": [1.25, 1]}
    )
    for lane, (label, start, duration, color, annotation) in enumerate(rows):
        bar = gantt.broken_barh([(start, duration)], (lane - 0.34, 0.68), facecolors=color)
        if lane == 3:
            bar.set_hatch("//")
        gantt.text(start + duration + 18, lane, annotation, va="center", fontsize=8)
    gantt.axvspan(1350.77, 1351.0, color="#e2e8f0", alpha=0.7)
    gantt.set_yticks(range(len(rows)), [row[0] for row in rows])
    gantt.invert_yaxis()
    gantt.set_xlim(0, 2220)
    gantt.set_xlabel("seconds from first A10 scheduler dispatch")
    gantt.set_title(
        "PetFinder scheduler + Predictor v3.2 + pre-flight — old partial trace\n"
        "pod externally terminated; fourth bar is a confirmed minimum, not a completion"
    )
    gantt.grid(axis="x", alpha=0.28)
    gantt.legend(
        handles=[
            mpatches.Patch(color="#2563eb", label="successful scored node"),
            mpatches.Patch(color="#dc2626", label="buggy candidate node"),
            mpatches.Patch(color="#64748b", hatch="//", label="external pod termination"),
        ],
        loc="upper right",
        fontsize=8,
    )

    metric.plot([1], [20.10200336], "o", color="#2563eb", ms=8, label="valid RMSE")
    metric.annotate("20.1020", (1, 20.10200336), xytext=(0, 9), textcoords="offset points", ha="center")
    metric.scatter([2, 3], [20.7057, 20.7057], marker="x", color="#dc2626", s=70, label="invalid / no submission")
    metric.annotate("best validation 20.7057\n(no submission)", (2, 20.7057), xytext=(8, 8), textcoords="offset points", fontsize=8)
    metric.annotate("no metric", (3, 20.7057), xytext=(8, -15), textcoords="offset points", fontsize=8)
    metric.scatter([4], [20.7057], marker="^", color="#64748b", s=60, label="externally interrupted")
    metric.annotate("external stop\n(no final metric)", (4, 20.7057), xytext=(8, 8), textcoords="offset points", fontsize=8)
    metric.set_xticks([1, 2, 3, 4])
    metric.set_xlim(0.5, 4.5)
    metric.set_ylim(19.9, 20.9)
    metric.set_xlabel("budget-counted execution order")
    metric.set_ylabel("validation RMSE (lower is better)")
    metric.set_title("Quality evidence: one valid score; invalid nodes are not compared as results")
    metric.grid(alpha=0.28)
    metric.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
