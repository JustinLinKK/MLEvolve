"""Generate plots + summary.md from the scheduler benchmark sweep.

Run after sweep_run.sh completes. Reads each config's:
  summary.json   per_job + by_status + treat_elapsed_s
  dmon.csv       nvidia-smi dmon -s pucvmet -d 1 -o T
  wall_clock.txt total elapsed sec

Outputs:
  plots/gpu_utility_overlay.png       4-panel × all configs
  plots/wallclock_bars.png            one bar per config + B1 line
  plots/speedup_heatmap.png           backend × probe × runner
  plots/arch_sensitivity.png          per-job time by {CNN, Transformer, Mixer}
  plots/agent_status_per_config.png   one subplot per config
  summary.md                          one row per config
"""
from __future__ import annotations
import csv
import datetime as dt
import json
import os
import re
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as exc:
    if exc.name == "matplotlib":
        raise SystemExit(
            "Missing plotting dependency: matplotlib\n"
            "Install benchmark dependencies with:\n"
            "  python -m pip install -r scheduler_benchmark_test/requirements.txt"
        ) from exc
    raise

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "scheduler_benchmark_test" / "main_sweep"))
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

CONFIGS_MAIN = [
    ("B1",  "serial_basic",            "exclusive", "off",          "scheduler"),
    ("B2",  "serial_batch_optimized",  "exclusive", "binary",       "scheduler"),
    ("B3",  "serial_batch_optimized",  "exclusive", "power_of_two", "scheduler"),
    ("T1",  "parallel_default",        "mps",       "off",          "scheduler"),
    ("T2",  "parallel_default",        "stream",    "off",          "scheduler"),
    ("T4",  "parallel_batch_optimized","mps",       "binary",       "scheduler"),
    ("T5",  "parallel_batch_optimized","mps",       "power_of_two", "scheduler"),
    ("T6",  "parallel_batch_optimized","stream",    "binary",       "scheduler"),
    ("T7",  "parallel_batch_optimized","stream",    "power_of_two", "scheduler"),
    ("T8",  "parallel_batch_optimized","mps",       "binary",       "torch_mp"),
    ("T9",  "parallel_batch_optimized","mps",       "power_of_two", "torch_mp"),
    ("T10", "parallel_batch_optimized","stream",    "binary",       "torch_mp"),
    ("T11", "parallel_batch_optimized","stream",    "power_of_two", "torch_mp"),
]

CONFIGS_WINDOWS = [
    ("B1",  "serial_basic",            "exclusive",    "off",          "scheduler"),
    ("B3",  "serial_batch_optimized",  "exclusive",    "power_of_two", "scheduler"),
    ("W1",  "parallel_default",        "cuda_process", "off",          "scheduler"),
    ("T2",  "parallel_default",        "stream",       "off",          "scheduler"),
    ("W4",  "parallel_batch_optimized","stream",       "power_of_two_nocache", "scheduler"),
    ("W5",  "parallel_batch_optimized","stream",       "power_of_two", "scheduler"),
    ("W6",  "parallel_batch_optimized","stream",       "power_of_two_pack3", "scheduler"),
    ("W7",  "parallel_batch_optimized","stream",       "power_of_two_pack4", "scheduler"),
    ("W8",  "parallel_batch_optimized","cuda_process", "power_of_two_cuda_pack3", "scheduler"),
    ("W9",  "parallel_batch_optimized","cuda_process", "power_of_two_cuda_pack4", "scheduler"),
    ("T11", "parallel_batch_optimized","stream",       "power_of_two", "torch_mp"),
]


def selected_configs() -> list[tuple[str, str, str, str, str]]:
    override = str(os.environ.get("CONFIG_SET", "")).strip().lower()
    if override in {"windows", "nomps", "no_mps"}:
        return CONFIGS_WINDOWS
    if override in {"main", "full", "default"}:
        return CONFIGS_MAIN
    if (RESULTS / "W1").exists() or RESULTS.name.endswith("_windows"):
        return CONFIGS_WINDOWS
    return CONFIGS_MAIN


def parse_dmon(csv_path: Path):
    """Return list of (t_sec, sm%, mem%, fb_MiB, rxpci_MB)."""
    if not csv_path.exists():
        return []
    rows = []
    base = None
    with csv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 22:
                continue
            try:
                tt = dt.datetime.strptime(parts[0], "%H:%M:%S")
                sm_v = float(parts[5])
                mem_v = float(parts[6])
                fb_v = float(parts[15])
                rx_v = float(parts[21])
            except Exception:
                continue
            if base is None:
                base = tt
            secs = (tt - base).total_seconds()
            if secs < 0:
                secs += 86400
            rows.append((secs, sm_v, mem_v, fb_v, rx_v))
    return rows


def load_summary(cfg_dir: Path) -> dict:
    s = cfg_dir / "summary.json"
    if not s.exists():
        return {}
    try:
        return json.loads(s.read_text())
    except Exception:
        return {}


def load_wallclock(cfg_dir: Path) -> float:
    w = cfg_dir / "wall_clock.txt"
    if not w.exists():
        summary = load_summary(cfg_dir)
        return float(summary.get("treat_elapsed_s") or 0.0)
    try:
        text = w.read_text().strip()
        if text:
            return float(text)
    except Exception:
        pass
    summary = load_summary(cfg_dir)
    try:
        return float(summary.get("treat_elapsed_s") or 0.0)
    except Exception:
        return 0.0


def display_probe_label(probe: str) -> str:
    if probe == "power_of_two_nocache":
        return "power_of_two (cache off)"
    if probe == "power_of_two_pack3":
        return "power_of_two (pack3)"
    if probe == "power_of_two_pack4":
        return "power_of_two (pack4)"
    if probe == "power_of_two_cuda_pack3":
        return "power_of_two (cuda pack3)"
    if probe == "power_of_two_cuda_pack4":
        return "power_of_two (cuda pack4)"
    return probe


def main():
    configs = selected_configs()
    data = {}
    for cid, mode, backend, probe, runner in configs:
        cfg_dir = RESULTS / cid
        data[cid] = {
            "mode": mode, "backend": backend, "probe": probe, "runner": runner,
            "wallclock": load_wallclock(cfg_dir),
            "summary": load_summary(cfg_dir),
            "dmon": parse_dmon(cfg_dir / "dmon.csv"),
        }

    cids = list(data.keys())
    b1_wc = data.get("B1", {}).get("wallclock", 1.0) or 1.0

    # Plot 1: GPU utility-time overlay
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=False)
    panels = [("SM_ACTIVE %", 1), ("DRAM_ACTIVE %", 2), ("FB MEM (MiB)", 3), ("PCIE RX (MB/s)", 4)]
    cmap = plt.get_cmap("tab20")
    for ax, (title, idx) in zip(axes, panels):
        for i, cid in enumerate(cids):
            d = data[cid]
            if not d["dmon"]:
                continue
            ts = [r[0] for r in d["dmon"]]
            ys = [r[idx] for r in d["dmon"]]
            ax.plot(ts, ys, color=cmap(i % 20), linewidth=0.7, label=cid, alpha=0.7)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("seconds since dmon start")
    axes[0].legend(loc="upper right", ncol=7, fontsize=7)
    axes[0].set_title(f"GPU utility timeline overlay ({len(cids)} configs)")
    plt.tight_layout()
    plt.savefig(PLOTS / "gpu_utility_overlay.png", dpi=110)
    plt.close()
    print("wrote gpu_utility_overlay.png")

    # Plot 2: Wall-clock bars
    wcs = [data[c]["wallclock"] for c in cids]
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#7f7f7f" if c.startswith("B")
              else ("#2ca02c" if data[c]["runner"] == "torch_mp" else "#9467bd")
              for c in cids]
    bars = ax.barh(cids, wcs, color=colors, edgecolor="black")
    ax.axvline(b1_wc, color="red", linestyle="--", label=f"B1 baseline ({b1_wc:.1f}s)")
    ax.set_xlabel("wall-clock seconds")
    ax.set_title(f"Total wall-clock per config (B1 baseline = {b1_wc:.0f}s)")
    for b, t in zip(bars, wcs):
        sp = b1_wc / t if t > 0 else 0
        ax.text(t + 5, b.get_y() + b.get_height() / 2, f"{t:.0f}s ({sp:.2f}x)",
                va="center", fontsize=9)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "wallclock_bars.png", dpi=110)
    plt.close()
    print("wrote wallclock_bars.png")

    # Plot 3: Speedup heatmap (backend × probe × runner)
    backends = list(dict.fromkeys(d["backend"] for d in data.values()))
    probes = list(dict.fromkeys(d["probe"] for d in data.values()))
    runners = ["scheduler", "torch_mp"]
    fig, axes = plt.subplots(1, len(runners), figsize=(12, 4), sharey=True)
    for ax_idx, runner in enumerate(runners):
        mat = []
        for probe in probes:
            row = []
            for backend in backends:
                speedup = None
                for cid, d in data.items():
                    if d["backend"] == backend and d["probe"] == probe and d["runner"] == runner:
                        if d["wallclock"] > 0 and b1_wc > 0:
                            speedup = b1_wc / d["wallclock"]
                        break
                row.append(speedup if speedup is not None else float("nan"))
            mat.append(row)
        ax = axes[ax_idx] if len(runners) > 1 else axes
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=6.0, aspect="auto")
        ax.set_xticks(range(len(backends)))
        ax.set_xticklabels(backends)
        ax.set_yticks(range(len(probes)))
        ax.set_yticklabels([display_probe_label(probe) for probe in probes])
        ax.set_title(f"runner: {runner}")
        for i, _p in enumerate(probes):
            for j, _b in enumerate(backends):
                v = mat[i][j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax)
    fig.suptitle("Speedup heatmap (vs B1 baseline)")
    plt.tight_layout()
    plt.savefig(PLOTS / "speedup_heatmap.png", dpi=110)
    plt.close()
    print("wrote speedup_heatmap.png")

    # Plot 4: arch sensitivity (per-job time aggregated by class for selected configs)
    preferred = {"B1", "T2", "T8", "W1", "W4", "W5", "W6", "W7", "W8", "W9", "T11"}
    SELECT = [c for c in cids if c in preferred][:10]
    classes = ["CNN", "Transformer", "Mixer"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_w = min(0.15, 0.8 / max(1, len(SELECT)))
    x = list(range(len(classes)))
    for i, cid in enumerate(SELECT):
        per_job = (data[cid].get("summary") or {}).get("per_job", []) or []
        # Aggregate elapsed_s per class (for torch_mp, elapsed_s in per_job; for scheduler, use difference)
        sums = {k: [] for k in classes}
        for pj in per_job:
            klass = pj.get("model_class")
            t = pj.get("elapsed_s")
            if klass in sums and t is not None:
                sums[klass].append(t)
        means = [sum(sums[k]) / len(sums[k]) if sums[k] else 0.0 for k in classes]
        ax.bar([xx + i * bar_w for xx in x], means, bar_w, label=cid)
    ax.set_xticks([xx + bar_w * (len(SELECT) - 1) / 2 for xx in x])
    ax.set_xticklabels(classes)
    ax.set_ylabel("mean per-job elapsed (s)")
    ax.set_title("Arch sensitivity: per-job time by model class")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "arch_sensitivity.png", dpi=110)
    plt.close()
    print("wrote arch_sensitivity.png")

    # Plot 5: Agent status per config
    fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=False)
    axes_flat = axes.flatten()
    for ax_idx, cid in enumerate(cids):
        ax = axes_flat[ax_idx]
        d = data[cid]
        per_job = (d.get("summary") or {}).get("per_job", []) or []
        for j_idx, j in enumerate(per_job):
            stage = j.get("agent_used") or "unknown"
            color = {"draft": "#7f7f7f", "improve": "#9467bd",
                     "debug": "#1f77b4", "evolution": "#2ca02c",
                     "fusion": "#d62728"}.get(stage, "#cccccc")
            x = j.get("step_idx", j_idx)
            ax.barh(0, 1, left=x, color=color, edgecolor="black", height=0.6)
            ax.text(x + 0.5, 0, stage[:3], ha="center", va="center", fontsize=7)
        ax.set_title(cid, fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("step idx")
    for ax_idx in range(len(cids), len(axes_flat)):
        axes_flat[ax_idx].axis("off")
    fig.suptitle("Agent status by step (per config)")
    plt.tight_layout()
    plt.savefig(PLOTS / "agent_status_per_config.png", dpi=110)
    plt.close()
    print("wrote agent_status_per_config.png")

    # summary.md
    md = []
    md.append(f"# Scheduler Benchmark Results — {len(cids)}-Config × W3 Trace\n\n")
    md.append("## Setting\n")
    md.append("- Hardware target: RTX 5090 class GPU (scheduler safe budget default 30 GiB)\n")
    md.append("- Workload: cassava-leaf-disease-classification (12 GB)\n")
    md.append("- Trace: W3 — 7 CNN + 7 Transformer + 6 Mixer (real timm models, real cassava data)\n")
    md.append("- Per-job: subset 4000, epochs 2, bs ∈ {16,24,32,48} per-arch capped\n\n")
    md.append("## Per-config wall-clock\n\n")
    md.append("| ID | Mode | Backend | Probe | Runner | Wall-clock | Speedup vs B1 |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for cid in cids:
        d = data[cid]
        sp = (b1_wc / d["wallclock"]) if d["wallclock"] > 0 else 0.0
        md.append(f"| {cid} | {d['mode']} | {d['backend']} | {display_probe_label(d['probe'])} | {d['runner']} | "
                  f"{d['wallclock']:.1f}s | {sp:.2f}x |\n")
    md.append("\n## Plots\n\n")
    for p in ["gpu_utility_overlay.png", "wallclock_bars.png", "speedup_heatmap.png",
              "arch_sensitivity.png", "agent_status_per_config.png"]:
        md.append(f"- ![{p}](plots/{p})\n")
    out = RESULTS / "summary.md"
    out.write_text("".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
