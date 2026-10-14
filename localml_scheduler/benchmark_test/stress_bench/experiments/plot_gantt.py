"""3-panel Gantt of job scheduling: baseline (packing) vs my solution (saturation-gated)
vs new scheduler (102a2ae). Same 16-job matched trace. One PNG."""
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path("/root/downeyflyfan/perfseer_test/exp30")
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e5e2"
C_EXCL = "#2a78d6"   # exclusive (solo)
C_PACK = "#eb6834"   # co-located (packed)

PANELS = [
    ("opt_pack",      "Baseline — memory-fit packing"),
    ("opt_gated",     "My solution — saturation-gated (scheduler 8d6dee9)"),
    ("opt_new_gated", "New scheduler — 102a2ae (corrected time packing)"),
]

def load(run):
    d = json.load(open(BASE / run / "raw.json"))
    t0 = d["t0"]; rows = []
    for j in d["per_job"]:
        si = j.get("started_at_iso"); fi = j.get("finished_at_iso")
        if not si or not fi:
            continue
        start = datetime.fromisoformat(si).timestamp() - t0
        end = datetime.fromisoformat(fi).timestamp() - t0
        rows.append({"start": start, "end": end,
                     "backend": j.get("placement_backend") or "exclusive",
                     "ts": j.get("training_seconds", end - start)})
    rows.sort(key=lambda r: r["start"])
    sm = json.load(open(BASE / run / "summary.json"))
    return rows, sm

# global x-max for shared scale
data = [(run, title, *load(run)) for run, title in PANELS]
xmax = max(r["end"] for _, _, rows, _ in data for r in rows) * 1.02

fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
fig.patch.set_facecolor(SURF)

for ax, (run, title, rows, sm) in zip(axes, data):
    ax.set_facecolor(SURF)
    for i, r in enumerate(rows):
        packed = r["backend"] != "exclusive"
        ax.barh(i, r["end"] - r["start"], left=r["start"], height=0.72,
                color=(C_PACK if packed else C_EXCL), edgecolor=SURF, linewidth=1.2,
                zorder=3)
    mk = sm["wall_seconds"]
    sigma = sum(r["ts"] for r in rows)
    npack = sum(1 for r in rows if r["backend"] != "exclusive")
    ax.axvline(mk, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(mk, len(rows) - 0.3, f" makespan {mk:.0f}s", color=INK2, fontsize=9,
            va="top", ha="left")
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=6)
    ax.text(0.0, 1.0, "", transform=ax.transAxes)
    # stats box
    ax.text(0.992, 0.06,
            f"Σ training {sigma:.0f}s   ·   {npack}/{len(rows)} co-located",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_ylabel("jobs (by start)", color=INK2, fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)

axes[-1].set_xlabel("time since first dispatch (seconds)", color=INK2, fontsize=10)
legend = [Patch(fc=C_EXCL, label="exclusive (runs solo)"),
          Patch(fc=C_PACK, label="co-located (packed, time-sliced)")]
fig.legend(handles=legend, loc="upper right", frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.995, 0.998))
fig.suptitle("Job Scheduling Timelines — 16-job matched workload (NVIDIA A10)",
             color=INK, fontsize=13.5, fontweight="bold", x=0.008, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = BASE / "scheduling_gantt.png"
fig.savefig(out, dpi=150, facecolor=SURF)
print("wrote", out)
for run, title, rows, sm in data:
    print(f"  {run:14s} makespan={sm['wall_seconds']:.0f}s jobs={len(rows)} "
          f"packed={sum(1 for r in rows if r['backend']!='exclusive')}")
