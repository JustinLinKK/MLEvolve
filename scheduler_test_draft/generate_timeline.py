"""
Scheduler Experiment: Generate a 100-job training timeline.

Step 1: Build a job list covering CNN / LSTM / Transformer / GBM variations.
Step 2: Assign Poisson-distributed submission timestamps.
Step 3: Save timeline as JSON so both scheduler and baseline can replay it exactly.

Output: timeline.json
Format: [[job_dict, t_seconds], [job_dict, t_seconds], ...]
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Job templates: one entry per model variant
# Each entry reflects what MLEvolve's executor would submit to the scheduler.
# VRAM figures are rough estimates at batch_size=32 on a single GPU.
# ---------------------------------------------------------------------------

JOB_TEMPLATES = [
    # ── CNN ──────────────────────────────────────────────────────────────────
    {"model_key": "resnet50",        "model_family": "cnn", "estimated_vram_mb": 3500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "resnet101",       "model_family": "cnn", "estimated_vram_mb": 6000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "resnet152",       "model_family": "cnn", "estimated_vram_mb": 8500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "efficientnet_b0", "model_family": "cnn", "estimated_vram_mb": 2000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "efficientnet_b4", "model_family": "cnn", "estimated_vram_mb": 5000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "efficientnet_b7", "model_family": "cnn", "estimated_vram_mb": 9000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "convnext_small",  "model_family": "cnn", "estimated_vram_mb": 4500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "convnext_large",  "model_family": "cnn", "estimated_vram_mb": 9500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "densenet121",     "model_family": "cnn", "estimated_vram_mb": 3000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "densenet201",     "model_family": "cnn", "estimated_vram_mb": 6500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "mobilenet_v3_s",  "model_family": "cnn", "estimated_vram_mb": 1500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "mobilenet_v3_l",  "model_family": "cnn", "estimated_vram_mb": 2500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},

    # ── LSTM ─────────────────────────────────────────────────────────────────
    {"model_key": "lstm_1layer",     "model_family": "lstm", "estimated_vram_mb": 1500, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "lstm_2layer",     "model_family": "lstm", "estimated_vram_mb": 2500, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "lstm_3layer",     "model_family": "lstm", "estimated_vram_mb": 4000, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "bilstm_1layer",   "model_family": "lstm", "estimated_vram_mb": 2000, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "bilstm_2layer",   "model_family": "lstm", "estimated_vram_mb": 3500, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "stacked_lstm",    "model_family": "lstm", "estimated_vram_mb": 5000, "requires_gpu": True,  "packing_family": "mlevolve_script"},

    # ── Transformer ──────────────────────────────────────────────────────────
    {"model_key": "vit_b_16",        "model_family": "transformer", "estimated_vram_mb": 7000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "vit_l_16",        "model_family": "transformer", "estimated_vram_mb": 14000, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "swin_tiny",       "model_family": "transformer", "estimated_vram_mb": 4000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "swin_base",       "model_family": "transformer", "estimated_vram_mb": 9000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "deit_small",      "model_family": "transformer", "estimated_vram_mb": 3500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "deit_base",       "model_family": "transformer", "estimated_vram_mb": 7000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "bert_base",       "model_family": "transformer", "estimated_vram_mb": 6000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "bert_large",      "model_family": "transformer", "estimated_vram_mb": 12000, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "roberta_base",    "model_family": "transformer", "estimated_vram_mb": 6500,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "roberta_large",   "model_family": "transformer", "estimated_vram_mb": 13000, "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "gpt2",            "model_family": "transformer", "estimated_vram_mb": 5000,  "requires_gpu": True,  "packing_family": "mlevolve_script"},
    {"model_key": "gpt2_medium",     "model_family": "transformer", "estimated_vram_mb": 10000, "requires_gpu": True,  "packing_family": "mlevolve_script"},

    # ── GBM (CPU-based, no GPU) ───────────────────────────────────────────────
    {"model_key": "lightgbm",        "model_family": "gbm", "estimated_vram_mb": 0,    "requires_gpu": False, "packing_family": "mlevolve_script"},
    {"model_key": "xgboost",         "model_family": "gbm", "estimated_vram_mb": 0,    "requires_gpu": False, "packing_family": "mlevolve_script"},
    {"model_key": "histgbm",         "model_family": "gbm", "estimated_vram_mb": 0,    "requires_gpu": False, "packing_family": "mlevolve_script"},
]

# Weight each family so the distribution resembles real MLEvolve runs
# (CNN and Transformer are most common, GBM is least)
FAMILY_WEIGHTS = {
    "cnn":         0.35,
    "transformer": 0.35,
    "lstm":        0.20,
    "gbm":         0.10,
}


def _weighted_sample(templates: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n templates according to FAMILY_WEIGHTS."""
    rng = random.Random(seed)
    by_family: dict[str, list[dict]] = {}
    for t in templates:
        by_family.setdefault(t["model_family"], []).append(t)

    families = list(FAMILY_WEIGHTS.keys())
    weights  = [FAMILY_WEIGHTS[f] for f in families]

    jobs = []
    for _ in range(n):
        family = rng.choices(families, weights=weights, k=1)[0]
        template = rng.choice(by_family[family])
        jobs.append(dict(template))
    return jobs


def generate_job_list(n: int = 100, seed: int = 42) -> list[dict]:
    """Return a list of n job dicts, each with a unique job_id."""
    templates = _weighted_sample(JOB_TEMPLATES, n, seed)
    job_list = []
    for i, tmpl in enumerate(templates, start=1):
        job = {
            "job_id":           f"job_{i:03d}_{uuid.uuid4().hex[:6]}",
            "model_key":        tmpl["model_key"],
            "model_family":     tmpl["model_family"],
            "task_type":        "mlevolve_script",
            "packing_family":   tmpl["packing_family"],
            "requires_gpu":     tmpl["requires_gpu"],
            "estimated_vram_mb": tmpl["estimated_vram_mb"],
            "priority":         0,
        }
        job_list.append(job)
    return job_list


def generate_poisson_timeline(
    job_list: list[dict],
    arrival_rate: float = 0.1,  # jobs per second on average
    seed: int = 42,
) -> list[tuple[dict, float]]:
    """
    Assign Poisson-distributed submission timestamps.

    Inter-arrival times are drawn from Exponential(1/arrival_rate),
    which is the correct continuous-time model for a Poisson process.

    arrival_rate=0.1 means one job every ~10 seconds on average.
    """
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=1.0 / arrival_rate, size=len(job_list))
    timestamps = np.cumsum(inter_arrivals)

    timeline = []
    for job, t in zip(job_list, timestamps):
        timeline.append((job, round(float(t), 3)))
    return timeline


def save_timeline(timeline: list[tuple[dict, float]], output_path: Path) -> None:
    """Save timeline as JSON: [[job_dict, t_seconds], ...]"""
    serializable = [[job, t] for job, t in timeline]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Saved {len(timeline)} jobs to {output_path}")


def print_summary(timeline: list[tuple[dict, float]]) -> None:
    from collections import Counter
    families = Counter(job["model_family"] for job, _ in timeline)
    total_duration = timeline[-1][1]
    print(f"\n=== Timeline Summary ===")
    print(f"Total jobs     : {len(timeline)}")
    print(f"Total duration : {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Family counts  : {dict(families)}")
    print(f"First job at   : t={timeline[0][1]:.2f}s")
    print(f"Last  job at   : t={timeline[-1][1]:.2f}s")


if __name__ == "__main__":
    SEED = 42
    N_JOBS = 100
    ARRIVAL_RATE = 0.1  # 1 job per ~10 seconds

    job_list = generate_job_list(n=N_JOBS, seed=SEED)
    timeline = generate_poisson_timeline(job_list, arrival_rate=ARRIVAL_RATE, seed=SEED)

    output_path = Path(__file__).parent / "timeline.json"
    save_timeline(timeline, output_path)
    print_summary(timeline)
