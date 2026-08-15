"""Draw the schedules produced by each policy, one panel per policy.

Panels come from the dependency-aware replay, where a child's arrival is its
parent's finish plus generation time, so a faster schedule genuinely pulls
later work forward rather than replaying fixed arrivals.

Usage:
    python -m scheduler_benchmark_test.draw_scheduler_gantt
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler_benchmark_test.compare_dag_schedulers import (
    compute_subtree_cost,
    critical_path,
    lpt,
    make_cp_guard,
    to_dag_jobs,
)
from scheduler_benchmark_test.dag_scheduler import simulate_dag
from scheduler_benchmark_test.run_cassava_scheduler_test import (
    MEMORY_BUDGET_MB,
    PAIR_SLOWDOWN,
    PARALLEL_CAP,
    TRACE,
    load_rows,
)

RECORDS = Path(__file__).resolve().parent.parent / "records"
COLOR = "#4C72B0"


def main() -> None:
    jobs = to_dag_jobs(load_rows(TRACE))
    compute_subtree_cost(jobs)
    order = {j.job_id: i for i, j in enumerate(sorted(jobs, key=lambda x: x.job_id))}

    runs = [
        ("serial (cap=1)", dict(parallel_cap=1, scorer=lpt, guard=None)),
        ("occupancy-lpt", dict(parallel_cap=PARALLEL_CAP, scorer=lpt, guard=None)),
        ("occupancy-critical-path", dict(parallel_cap=PARALLEL_CAP, scorer=critical_path, guard=None)),
        ("cp + contention guard", dict(parallel_cap=PARALLEL_CAP, scorer=critical_path,
                                       guard=make_cp_guard(PAIR_SLOWDOWN, 0.3))),
    ]

    results = []
    for label, kwargs in runs:
        results.append((label, simulate_dag(
            list(jobs), pair_slowdown=PAIR_SLOWDOWN,
            memory_budget_mb=MEMORY_BUDGET_MB, policy_name=label, **kwargs,
        )))

    widest = max(r.makespan for _, r in results)
    fig, axes = plt.subplots(len(results), 1, figsize=(15, 3.4 * len(results)), squeeze=False)

    for ax, (label, r) in zip(axes[:, 0], results):
        for job_id, start, end in r.timeline:
            ax.barh(order[job_id], (end - start) / 60, left=start / 60, height=0.72,
                    color=COLOR, edgecolor="black", linewidth=0.4)
        ax.set_xlim(0, widest / 60 * 1.02)
        ax.set_ylim(-1, len(order))
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_ylabel("jobs")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(
            f"{label}   makespan {r.makespan/60:.1f} min   "
            f"mean concurrency {r.mean_concurrency:.2f}   "
            f"mean flow {r.mean_flow/60:.1f} min",
            fontsize=10,
        )

    axes[-1, 0].set_xlabel("minutes from first arrival")
    axes[0, 0].legend(
        handles=[Patch(facecolor=COLOR, edgecolor="black", label="training execution")],
        loc="upper right", fontsize=9,
    )
    fig.suptitle(
        "Cassava trace, dependency-aware replay: child arrives at parent finish + generation time\n"
        f"{len(jobs)} jobs, {MEMORY_BUDGET_MB:.0f} MB budget, pair slowdown {PAIR_SLOWDOWN:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    RECORDS.mkdir(exist_ok=True)
    out = RECORDS / "scheduler_gantt_cassava.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for label, r in results:
        print(f"{label:<26} makespan {r.makespan/60:7.1f} min   conc {r.mean_concurrency:.2f}")
    print(f"\nchart -> {out}")


if __name__ == "__main__":
    main()
