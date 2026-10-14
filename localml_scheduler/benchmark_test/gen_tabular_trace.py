"""Generate a packable MLEBench-Lite tabular trace for the V100 scheduler test.

Uses the two configurations measured to pack 4-5 jobs on one V100 while
driving the device to ~99% SM (see records/v100_tabular_packing.md):

    tps_w128_d1_b512   solo SM 11%,  N=4 SM 99.1%, slowdown 1.00x
    tps_w256_d2_b1024  solo SM 23%,  N=4 SM 99.0%, slowdown 1.00x

Both come from the tabular-playground MLEBench-Lite tasks, whose agent
solutions are MLPs rather than CNNs.

Shape symbols:
    b = batch size, f = input features, w = hidden width, c = classes

Output: traces/mlebench_tabular_v100_100jobs.jsonl
"""

import json
import random
from pathlib import Path

SEED = 42
LAMBDA_PER_MIN = 4.0
N_JOBS = 100
OUT_PATH = Path(__file__).resolve().parent.parent / "traces" / "mlebench_tabular_v100_100jobs.jsonl"

# Measured on Tesla V100-SXM2-32GB, no MPS, 50 warmup + 300 timed steps.
# steps_per_sec is the solo (N=1) rate; vram_mib is torch peak allocated.
CONFIGS = {
    "w128_d1_b512": {
        "width": 128, "depth": 1, "batch_size": 512,
        "steps_per_sec": 755.5, "vram_mib": 1.7, "sm_solo": 11.0,
    },
    "w256_d2_b1024": {
        "width": 256, "depth": 2, "batch_size": 1024,
        "steps_per_sec": 560.9, "vram_mib": 6.8, "sm_solo": 23.0,
    },
}

# MLEBench-Lite tabular tasks. train_rows are the published training-set sizes;
# n_features / n_classes define the MLP input and output dims.
TASKS = [
    {"name": "tabular-playground-series-dec-2021", "train_rows": 4_000_000, "n_features": 54, "n_classes": 7},
    {"name": "tabular-playground-series-may-2022", "train_rows": 900_000, "n_features": 31, "n_classes": 2},
]

# Per-job VRAM floor: CUDA context + cuDNN workspace dominate these tiny models.
CUDA_CONTEXT_MB = 520.0

EPOCH_CHOICES = [10, 15, 20, 30]


def plateau_metrics(epochs, rng, peak_frac=0.55, noise=0.004):
    """Validation accuracy that improves then plateaus, for early-stopping tests.

    Args:
        epochs    : int, number of epochs to emit
        rng       : random.Random
        peak_frac : float, fraction of epochs after which improvement stops
        noise     : float, std of Gaussian noise added per epoch

    Returns:
        list[float], per-epoch validation accuracy
    """
    peak = max(1, int(epochs * peak_frac))
    out = []
    for e in range(1, epochs + 1):
        base = 0.55 + 0.30 * (e / peak) if e <= peak else 0.85
        out.append(round(base + rng.gauss(0, noise), 5))
    return out


def main():
    rng = random.Random(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    arrival = 0.0
    jobs = []
    for i in range(N_JOBS):
        task = TASKS[i % len(TASKS)]
        cfg_name = rng.choice(list(CONFIGS))
        cfg = CONFIGS[cfg_name]
        epochs = rng.choice(EPOCH_CHOICES)

        batches_per_epoch = max(1, task["train_rows"] // cfg["batch_size"])
        epoch_seconds = batches_per_epoch / cfg["steps_per_sec"]
        solo_seconds = epoch_seconds * epochs * max(0.6, 1.0 + rng.gauss(0, 0.04))
        memory_mb = round(CUDA_CONTEXT_MB + cfg["vram_mib"], 1)

        jobs.append({
            "job_id": f"{task['name'][:12]}_{i:03d}",
            "release_seconds": round(arrival, 2),
            "priority": 0,
            "planned_epochs": epochs,
            "validation_metrics": plateau_metrics(epochs, rng),
            "backend_allowlist": ["cuda_process"],
            "options": [{
                "batch_size": cfg["batch_size"],
                "memory_mb": memory_mb,
                "solo_seconds": round(solo_seconds, 2),
                "actual_memory_mb": memory_mb,
                "actual_solo_seconds": round(solo_seconds, 2),
            }],
            "task_name": task["name"],
            "step_idx": i // len(TASKS),
            "architecture": f"tabular_mlp_{cfg_name}",
            "family": "tabular",
            "width": cfg["width"],
            "depth": cfg["depth"],
            "n_features": task["n_features"],
            "n_classes": task["n_classes"],
            "batches_per_epoch": batches_per_epoch,
            "sm_solo_pct": cfg["sm_solo"],
        })
        arrival += rng.expovariate(LAMBDA_PER_MIN / 60.0)

    with open(OUT_PATH, "w") as fh:
        for job in jobs:
            fh.write(json.dumps(job) + "\n")

    total = sum(j["options"][0]["solo_seconds"] for j in jobs)
    print(f"Wrote {len(jobs)} jobs -> {OUT_PATH}")
    print(f"Arrival span: 0 - {jobs[-1]['release_seconds']:.1f}s")
    print(f"Total solo compute: {total:.0f}s ({total/3600:.2f} h); mean {total/len(jobs):.1f}s/job")
    counts = {}
    for j in jobs:
        counts[j["architecture"]] = counts.get(j["architecture"], 0) + 1
    for arch, n in sorted(counts.items()):
        print(f"  {arch}: {n}")


if __name__ == "__main__":
    main()
