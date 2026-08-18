"""Paired model-quality audit for MP2 versus warm/cold stream scheduling."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from scheduler_benchmark_test.model_quality_runner import create_quality_baseline
from scheduler_benchmark_test.rtx5090_pressure_benchmark import (
    REPO,
    TERMINAL,
    _iso_epoch,
    _read_json,
    _scheduler_settings,
    _seed_scheduler_profiles,
    _write_json,
    run_mp2_baseline,
    validate_stream_placements,
)


DEFAULT_SEEDS = (104729, 130363, 155921, 181081, 206369, 231503, 256739, 282001)


def build_quality_trace(
    output_root: Path,
    *,
    replicates: int = 8,
    epochs: int = 12,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Build matched paired jobs; only the execution mode differs."""
    if replicates < 2:
        raise ValueError("the quality audit requires at least two paired replicates")
    baselines = output_root / "initial-checkpoints"
    trace: list[dict[str, Any]] = []
    train_samples = 2048 if smoke else 16384
    validation_samples = 1024 if smoke else 4096
    actual_epochs = 4 if smoke else int(epochs)
    input_dim = 128 if smoke else 256
    hidden_dim = 256 if smoke else 2048
    batch_size = 256
    for index in range(replicates):
        seed = DEFAULT_SEEDS[index % len(DEFAULT_SEEDS)] + (index // len(DEFAULT_SEEDS)) * 1_000_003
        job_id = f"quality-seed-{seed}"
        baseline_path = baselines / f"{job_id}.pt"
        if not baseline_path.exists():
            create_quality_baseline(
                baseline_path,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=4,
                seed=seed,
            )
        trace.append(
            {
                "step_idx": index,
                "job_id": job_id,
                "scenario": "quality_bf16",
                "release_s": 0.0,
                "packing_signature": "rtx5090:quality-mlp-bf16-v1",
                "baseline_model_path": str(baseline_path),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": 4,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                "batch_size": batch_size,
                "batches_per_epoch": math.ceil(train_samples / batch_size),
                "epochs": actual_epochs,
                "learning_rate": 1.5e-3,
                "weight_decay": 1e-4,
                "dataset_seed": 47017,
                "teacher_seed": 991,
                "label_noise": 0.30,
                "precision": "bf16_amp",
                # This is deliberately a light job: short CPU/data work between
                # real GPU optimizer steps makes 3-4 stream packing beneficial.
                "step_delay_ms": 20.0 if smoke else 8.0,
                "random_seed": seed,
                "target_vram_mib": 1024,
                "target_vram_fraction": 0.031,
            }
        )
    return trace


def _quality_command(_job: dict[str, Any], spec: Path, result: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scheduler_benchmark_test.model_quality_runner",
        "--spec",
        str(spec),
        "--result",
        str(result),
    ]


def calibrate_quality_trace(
    trace: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Measure one solo replicate only to seed runtime and memory admission."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "solo.spec.json"
    result_path = output_dir / "solo.result.json"
    log_path = output_dir / "solo.log"
    _write_json(spec_path, trace[0])
    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            _quality_command(trace[0], spec_path, result_path),
            cwd=REPO,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"quality calibration failed; inspect {log_path}")
    result = _read_json(result_path, {})
    epoch_seconds = [
        float(row["epoch_seconds"])
        for row in result.get("history", [])
        if row.get("epoch_seconds") is not None
    ]
    result["median_epoch_seconds"] = (
        statistics.median(epoch_seconds)
        if epoch_seconds
        else float(result["training_seconds"]) / max(1, int(trace[0]["epochs"]))
    )
    _write_json(output_dir / "calibration.json", result)
    return result


def build_quality_profile_snapshots(
    trace: list[dict[str, Any]], calibration: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    epoch_seconds = float(calibration["median_epoch_seconds"])
    total_seconds = float(calibration["training_seconds"])
    step_seconds = epoch_seconds / max(1, int(trace[0]["batches_per_epoch"]))
    solo = [
        {
            "logical_job_id": item["job_id"],
            "signature": item["packing_signature"],
            "batch_size": item["batch_size"],
            "avg_vram_mib": item["target_vram_mib"],
            "peak_vram_mib": item["target_vram_mib"],
            "step_seconds": step_seconds,
            "seconds_per_epoch": epoch_seconds,
            "total_seconds": total_seconds,
            "observations": 2,
        }
        for item in trace
    ]
    colocations: list[dict[str, Any]] = []
    for size in (2, 3, 4):
        descriptor = {
            "signature": trace[0]["packing_signature"],
            "batch_size": trace[0]["batch_size"],
            "backend_name": "stream",
        }
        members = [dict(descriptor) for _ in range(size)]
        timings = [
            {
                **descriptor,
                "seconds_per_epoch": epoch_seconds * (1.0 + 0.06 * (size - 1)),
                "observations": 2,
                "source": "quality_audit_solo_calibration",
            }
            for _ in range(size)
        ]
        colocations.append(
            {
                "name": f"quality_{size}",
                "members": members,
                "member_timings": timings,
                "observations": 2,
                "decision": "accepted",
                "slowdown": 1.0 + 0.06 * (size - 1),
            }
        )
    cold = {"kind": "cold", "solo_memory_profiles": solo, "colocation_profiles": []}
    warm = {
        "kind": "warm",
        "solo_memory_profiles": solo,
        "colocation_profiles": colocations,
    }
    _write_json(output_dir / "profiles-cold.json", cold)
    _write_json(output_dir / "profiles-warm.json", warm)
    return warm, cold


def _build_scheduler_jobs(
    trace: list[dict[str, Any]], result_dir: Path
) -> dict[str, Any]:
    from localml_scheduler.adapters.mlevolve import build_mlevolve_job
    from localml_scheduler.domain import CheckpointPolicy, ResourceRequirements, RuntimeProbeSpec

    jobs: dict[str, Any] = {}
    for item in trace:
        runner_kwargs = {
            key: item[key]
            for key in (
                "input_dim",
                "hidden_dim",
                "output_dim",
                "train_samples",
                "validation_samples",
                "batch_size",
                "epochs",
                "learning_rate",
                "weight_decay",
                "dataset_seed",
                "teacher_seed",
                "label_noise",
                "precision",
                "step_delay_ms",
            )
        }
        runner_kwargs["batches_per_epoch"] = item["batches_per_epoch"]
        runner_kwargs["result_dir"] = str(result_dir)
        job = build_mlevolve_job(
            workflow_id="rtx5090-model-quality-audit",
            baseline_model_id=item["job_id"],
            baseline_model_path=item["baseline_model_path"],
            runner_target="scheduler_benchmark_test.model_quality_runner:run_quality_job",
            runner_kwargs=runner_kwargs,
            priority=5,
            task_type="model_quality_audit",
            checkpoint_policy=CheckpointPolicy(save_every_n_steps=None, save_every_epoch=False),
            resource_requirements=ResourceRequirements(
                requires_gpu=True,
                gpu_slots=1,
                estimated_vram_mb=int(item["target_vram_mib"]),
                estimated_avg_vram_mb=int(item["target_vram_mib"]),
            ),
            packing_family="quality_bf16",
            packing_signature=item["packing_signature"],
            packing_eligible=True,
            packing_max_slowdown_ratio=3.0,
            packing_backend_allowlist=["stream"],
            runtime_probe=RuntimeProbeSpec(enabled=False, strategy="epoch_1"),
            max_epochs=int(item["epochs"]),
            metadata={
                "logical_job_id": item["job_id"],
                "step_idx": item["step_idx"],
                "scenario": item["scenario"],
                "release_s": item["release_s"],
                "quality_audit": True,
            },
        )
        jobs[item["job_id"]] = job
    return jobs


def run_quality_scheduler_mode(
    mode: str,
    trace: list[dict[str, Any]],
    snapshot: dict[str, Any],
    output_dir: Path,
    *,
    timeout_s: float,
    total_vram_mib: int,
) -> dict[str, Any]:
    from localml_scheduler.client import SchedulerClient

    runtime_root = output_dir / "runtime"
    result_dir = output_dir / "results"
    runtime_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    settings = _scheduler_settings(runtime_root, total_vram_mib, smoke=True)
    jobs = _build_scheduler_jobs(trace, result_dir)
    seeded = _seed_scheduler_profiles(settings, jobs, snapshot)
    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)
    origin = time.time()
    deadline = origin + float(timeout_s)
    submitted: dict[str, dict[str, Any]] = {}
    timed_out = False
    try:
        for item in trace:
            submitted_job = api.submit(jobs[item["job_id"]])
            submitted[submitted_job.job_id] = item
        while time.time() < deadline:
            current = api.list_jobs()
            if len(current) == len(submitted) and all(job.status.value in TERMINAL for job in current):
                break
            time.sleep(0.2)
        else:
            timed_out = True
    finally:
        service.stop()

    attempts: list[dict[str, Any]] = []
    logical: list[dict[str, Any]] = []
    for job in api.list_jobs():
        item = submitted.get(job.job_id)
        if item is None:
            continue
        result_path = result_dir / f"{job.job_id}.json"
        result = _read_json(result_path, {})
        record = {
            "logical_job_id": item["job_id"],
            "scheduler_job_id": job.job_id,
            "step_idx": item["step_idx"],
            "scenario": item["scenario"],
            "attempt": 1,
            "retry": False,
            "backend": (job.metadata or {}).get("placement_backend"),
            "stream_host_pid": (job.metadata or {}).get("stream_host_pid") or result.get("stream_host_pid"),
            "cuda_stream_id": (job.metadata or {}).get("cuda_stream_id") or result.get("cuda_stream_id"),
            "released_at": origin,
            "started_at": _iso_epoch(job.started_at),
            "finished_at": _iso_epoch(job.finished_at),
            "status": job.status.value.lower(),
            "oom": "out of memory" in str(job.status_reason or "").lower(),
            "status_reason": job.status_reason,
            "training_seconds": result.get("training_seconds"),
            "result_path": str(result_path),
        }
        attempts.append(record)
        logical.append(
            {
                "logical_job_id": item["job_id"],
                "step_idx": item["step_idx"],
                "scenario": item["scenario"],
                "release_s": 0.0,
                "status": record["status"],
                "started_at": record["started_at"],
                "finished_at": record["finished_at"],
                "attempt_count": 1,
                "oom_count": int(record["oom"]),
            }
        )
    events: list[dict[str, Any]] = []
    if settings.events_jsonl_path.exists():
        for line in settings.events_jsonl_path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {
        "mode": mode,
        "origin": origin,
        "deadline": deadline,
        "timed_out": timed_out,
        "attempts": attempts,
        "logical_jobs": logical,
        "events": events,
        "seeded_profiles": seeded,
        "profile_kind": snapshot["kind"],
        "stream_assertions": validate_stream_placements(attempts),
        "runtime_root": str(runtime_root),
    }


def _result_rows(output_root: Path, modes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        raw = _read_json(output_root / mode / "raw.json", {})
        for attempt in raw.get("attempts", []):
            if attempt.get("status") not in {"succeeded", "completed"}:
                continue
            result = _read_json(Path(str(attempt["result_path"])), {})
            if result.get("final_validation_accuracy") is None:
                continue
            rows.append(
                {
                    "mode": mode,
                    "logical_job_id": attempt["logical_job_id"],
                    "step_idx": int(attempt["step_idx"]),
                    "backend": attempt.get("backend"),
                    "training_seconds": result.get("training_seconds"),
                    "final_validation_accuracy": result["final_validation_accuracy"],
                    "final_validation_loss": result["final_validation_loss"],
                    "model_parameter_sha256": result["model_parameter_sha256"],
                    "validation_label_sha256": result["validation_label_sha256"],
                    "validation_predictions": result["validation_predictions"],
                    "history": result["history"],
                    "stream_host_pid": attempt.get("stream_host_pid"),
                    "cuda_stream_id": attempt.get("cuda_stream_id"),
                }
            )
    return rows


def _bootstrap_ci(values: list[float], *, iterations: int = 20000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    import numpy as np

    source = np.asarray(values, dtype=float)
    generator = np.random.default_rng(5090)
    means = generator.choice(source, size=(iterations, len(source)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze_quality_results(
    output_root: Path,
    *,
    modes: list[str] | None = None,
    equivalence_pp: float = 0.5,
) -> dict[str, Any]:
    modes = modes or [mode for mode in ("baseline", "warm", "cold") if (output_root / mode / "raw.json").exists()]
    rows = _result_rows(output_root, modes)
    baseline = {row["logical_job_id"]: row for row in rows if row["mode"] == "baseline"}
    comparisons: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    for mode in ("warm", "cold"):
        deltas: list[float] = []
        agreements: list[float] = []
        exact_models = 0
        for row in sorted((item for item in rows if item["mode"] == mode), key=lambda item: item["step_idx"]):
            reference = baseline.get(row["logical_job_id"])
            if reference is None:
                continue
            delta_pp = 100.0 * (
                float(row["final_validation_accuracy"])
                - float(reference["final_validation_accuracy"])
            )
            left = reference["validation_predictions"]
            right = row["validation_predictions"]
            agreement = sum(a == b for a, b in zip(left, right)) / max(1, min(len(left), len(right)))
            model_exact = row["model_parameter_sha256"] == reference["model_parameter_sha256"]
            deltas.append(delta_pp)
            agreements.append(agreement)
            exact_models += int(model_exact)
            comparisons.append(
                {
                    "mode": mode,
                    "logical_job_id": row["logical_job_id"],
                    "step_idx": row["step_idx"],
                    "baseline_accuracy": reference["final_validation_accuracy"],
                    "scheduler_accuracy": row["final_validation_accuracy"],
                    "accuracy_delta_pp": delta_pp,
                    "baseline_validation_loss": reference["final_validation_loss"],
                    "scheduler_validation_loss": row["final_validation_loss"],
                    "prediction_agreement": agreement,
                    "model_parameters_exact": model_exact,
                }
            )
        low, high = _bootstrap_ci(deltas)
        equivalent = bool(
            deltas
            and len(deltas) == len(baseline)
            and max(abs(value) for value in deltas) <= equivalence_pp
            and low >= -equivalence_pp
            and high <= equivalence_pp
        )
        mode_summaries.append(
            {
                "mode": mode,
                "paired_replicates": len(deltas),
                "mean_accuracy_delta_pp": statistics.mean(deltas) if deltas else None,
                "median_accuracy_delta_pp": statistics.median(deltas) if deltas else None,
                "max_absolute_accuracy_delta_pp": max(map(abs, deltas)) if deltas else None,
                "bootstrap_95_ci_mean_delta_pp": [low, high] if deltas else None,
                "mean_prediction_agreement": statistics.mean(agreements) if agreements else None,
                "exact_model_matches": exact_models,
                "practically_equivalent": equivalent,
            }
        )
    complete = bool(baseline) and all(
        summary["paired_replicates"] == len(baseline) for summary in mode_summaries
    )
    execution_integrity: list[dict[str, Any]] = []
    for mode in ("warm", "cold"):
        raw = _read_json(output_root / mode / "raw.json", {})
        attempts = [
            item
            for item in raw.get("attempts", [])
            if item.get("backend") == "stream"
            and item.get("started_at") is not None
            and item.get("finished_at") is not None
        ]
        points: list[tuple[float, int]] = []
        for item in attempts:
            points.append((float(item["started_at"]), 1))
            points.append((float(item["finished_at"]), -1))
        active = 0
        peak = 0
        for _timestamp, delta in sorted(points, key=lambda point: (point[0], point[1])):
            active += delta
            peak = max(peak, active)
        stream_assertions = raw.get("stream_assertions") or {}
        execution_integrity.append(
            {
                "mode": mode,
                "peak_stream_concurrency": peak,
                "shared_host_distinct_streams_valid": bool(stream_assertions.get("valid")),
                "overlap_groups_observed": len(stream_assertions.get("overlaps") or []),
                "sufficient_packed_pressure": peak >= 3,
            }
        )
    packed_evidence = bool(execution_integrity) and all(
        item["sufficient_packed_pressure"]
        and item["shared_host_distinct_streams_valid"]
        for item in execution_integrity
    )
    verdict = complete and packed_evidence and bool(mode_summaries) and all(
        summary["practically_equivalent"] for summary in mode_summaries
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_mode": "baseline",
        "modes": modes,
        "paired_replicates": len(baseline),
        "equivalence_threshold_pp": equivalence_pp,
        "complete": complete,
        "no_practical_accuracy_difference_detected": verdict,
        "packed_execution_evidence_valid": packed_evidence,
        "execution_integrity": execution_integrity,
        "mode_summaries": mode_summaries,
        "comparisons": comparisons,
        "method": {
            "task": "deterministic synthetic four-class classification",
            "precision": "BF16 autocast with FP32 loss and optimizer weights",
            "pairing": "same initial checkpoint, dataset, sample order, hyperparameters, and seed",
            "uncertainty": "paired nonparametric bootstrap over replicate accuracy deltas",
        },
    }
    _write_json(output_root / "quality-summary.json", summary)
    _write_quality_csv(output_root / "quality-results.csv", rows, comparisons)
    render_quality_chart(output_root, rows, comparisons, summary)
    render_quality_bar_chart(output_root, rows, comparisons, summary)
    _write_quality_report(output_root / "QUALITY_REPORT.md", summary)
    return summary


def _write_quality_csv(path: Path, rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    comparison_by_key = {(item["mode"], item["logical_job_id"]): item for item in comparisons}
    fields = [
        "mode",
        "logical_job_id",
        "backend",
        "training_seconds",
        "final_validation_accuracy",
        "final_validation_loss",
        "accuracy_delta_pp",
        "prediction_agreement",
        "model_parameters_exact",
        "stream_host_pid",
        "cuda_stream_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["step_idx"], item["mode"])):
            comparison = comparison_by_key.get((row["mode"], row["logical_job_id"]), {})
            writer.writerow({field: comparison.get(field, row.get(field)) for field in fields})


def render_quality_chart(
    output_root: Path,
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"baseline": "#4c566a", "warm": "#2a9d8f", "cold": "#e76f51"}
    labels = {"baseline": "MP2 baseline", "warm": "Scheduler warm", "cold": "Scheduler cold"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), gridspec_kw={"width_ratios": [1.25, 1.0, 1.2]})
    jobs = sorted({row["logical_job_id"] for row in rows}, key=lambda name: int(name.rsplit("-", 1)[-1]))
    x = np.arange(len(jobs))
    offsets = {"baseline": -0.18, "warm": 0.0, "cold": 0.18}
    by_key = {(row["mode"], row["logical_job_id"]): row for row in rows}
    for mode in ("baseline", "warm", "cold"):
        values = [100.0 * float(by_key[(mode, job)]["final_validation_accuracy"]) if (mode, job) in by_key else np.nan for job in jobs]
        axes[0].scatter(x + offsets[mode], values, s=42, color=colors[mode], label=labels[mode], zorder=3)
    axes[0].set_title("Final validation accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlabel("Paired seed")
    axes[0].set_xticks(x, [str(index + 1) for index in range(len(jobs))])
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    threshold = float(summary["equivalence_threshold_pp"])
    axes[1].axhspan(-threshold, threshold, color="#90be6d", alpha=0.18, label=f"±{threshold:g} pp equivalence band")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for mode, marker in (("warm", "o"), ("cold", "s")):
        selected = sorted((item for item in comparisons if item["mode"] == mode), key=lambda item: item["step_idx"])
        axes[1].scatter(
            [item["step_idx"] + 1 + offsets[mode] for item in selected],
            [item["accuracy_delta_pp"] for item in selected],
            marker=marker,
            s=44,
            color=colors[mode],
            label=labels[mode],
        )
    axes[1].set_title("Paired difference from MP2")
    axes[1].set_ylabel("Accuracy delta (percentage points)")
    axes[1].set_xlabel("Paired seed")
    axes[1].set_xticks(np.arange(1, len(jobs) + 1))
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for mode in ("baseline", "warm", "cold"):
        histories = [row["history"] for row in rows if row["mode"] == mode]
        if not histories:
            continue
        epochs = min(len(history) for history in histories)
        matrix = np.asarray(
            [[100.0 * float(history[index]["validation_accuracy"]) for index in range(epochs)] for history in histories]
        )
        mean = matrix.mean(axis=0)
        low = matrix.min(axis=0)
        high = matrix.max(axis=0)
        epoch_x = np.arange(1, epochs + 1)
        axes[2].plot(epoch_x, mean, color=colors[mode], label=labels[mode], linewidth=2)
        axes[2].fill_between(epoch_x, low, high, color=colors[mode], alpha=0.10)
    axes[2].set_title("Validation learning curves")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    verdict = (
        "No practical scheduler accuracy difference detected"
        if summary["no_practical_accuracy_difference_detected"]
        else "Accuracy equivalence was not established"
    )
    fig.suptitle(f"RTX 5090 scheduler model-quality audit\n{verdict}", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Paired BF16 runs: identical initialization, data, order, optimizer, and epochs. Shading on curves spans replicates.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    png = output_root / "quality_accuracy_comparison.png"
    pdf = output_root / "quality_accuracy_comparison.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def render_quality_bar_chart(
    output_root: Path,
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Render an explicit per-job grouped accuracy and degradation chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"baseline": "#4c566a", "warm": "#2a9d8f", "cold": "#e76f51"}
    labels = {
        "baseline": "MP2 baseline",
        "warm": "Scheduler warm",
        "cold": "Scheduler cold",
    }
    ordered = sorted(
        {row["logical_job_id"]: int(row["step_idx"]) for row in rows}.items(),
        key=lambda item: item[1],
    )
    jobs = [name for name, _step_idx in ordered]
    x = np.arange(len(jobs), dtype=float)
    width = 0.25
    offsets = {"baseline": -width, "warm": 0.0, "cold": width}
    by_key = {(row["mode"], row["logical_job_id"]): row for row in rows}
    comparison_by_key = {
        (item["mode"], item["logical_job_id"]): item for item in comparisons
    }

    fig, (accuracy_ax, delta_ax) = plt.subplots(
        2,
        1,
        figsize=(15.5, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    for mode in ("baseline", "warm", "cold"):
        values = [
            100.0 * float(by_key[(mode, job)]["final_validation_accuracy"])
            if (mode, job) in by_key
            else np.nan
            for job in jobs
        ]
        bars = accuracy_ax.bar(
            x + offsets[mode],
            values,
            width,
            color=colors[mode],
            edgecolor="white",
            linewidth=0.7,
            label=labels[mode],
        )
        if mode == "baseline":
            accuracy_ax.bar_label(
                bars,
                labels=[f"{value:.2f}%" if np.isfinite(value) else "" for value in values],
                padding=3,
                fontsize=8,
                rotation=90,
            )
    accuracy_ax.set_ylim(0, 100)
    accuracy_ax.set_ylabel("Final validation accuracy (%)")
    accuracy_ax.set_title("Produced model accuracy for every paired job")
    accuracy_ax.grid(axis="y", alpha=0.22)
    accuracy_ax.legend(ncol=3, frameon=False, loc="upper center")

    threshold = float(summary["equivalence_threshold_pp"])
    delta_ax.axhspan(
        -threshold,
        threshold,
        color="#90be6d",
        alpha=0.18,
        label=f"±{threshold:g} pp practical-equivalence band",
    )
    delta_ax.axhline(0, color="#222222", linewidth=0.9)
    for mode in ("warm", "cold"):
        deltas = [
            float(comparison_by_key[(mode, job)]["accuracy_delta_pp"])
            if (mode, job) in comparison_by_key
            else np.nan
            for job in jobs
        ]
        bars = delta_ax.bar(
            x + offsets[mode] / 2,
            deltas,
            width / 2,
            color=colors[mode],
            edgecolor=colors[mode],
            linewidth=1.2,
            label=labels[mode],
        )
        for bar, value in zip(bars, deltas):
            if not np.isfinite(value):
                continue
            delta_ax.annotate(
                f"{value:+.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(((-3) if mode == "warm" else 3), 5 if value >= 0 else -11),
                textcoords="offset points",
                ha="right" if mode == "warm" else "left",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
                color=colors[mode],
            )
    observed = [
        abs(float(item["accuracy_delta_pp"])) for item in comparisons
    ]
    extent = max([threshold * 1.2, *(value * 1.25 for value in observed)])
    delta_ax.set_ylim(-extent, extent)
    delta_ax.set_ylabel("Scheduler − MP2\n(percentage points)")
    delta_ax.set_xlabel("Matched training job (job number / initialization seed)")
    delta_ax.grid(axis="y", alpha=0.22)
    delta_ax.legend(ncol=3, frameon=False, loc="upper center")
    tick_labels = [
        f"Job {index + 1}\n{name.rsplit('-', 1)[-1]}"
        for index, name in enumerate(jobs)
    ]
    delta_ax.set_xticks(x, tick_labels)

    verdict = (
        "No per-job scheduler degradation detected"
        if summary["no_practical_accuracy_difference_detected"]
        else "Per-job accuracy equivalence was not established"
    )
    fig.suptitle(
        f"RTX 5090 scheduler quality comparison\n{verdict}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Each group uses the same checkpoint, data, order, BF16 precision, optimizer, and epoch count in all three modes.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.09, right=0.98)
    png = output_root / "quality_accuracy_by_job_bar.png"
    pdf = output_root / "quality_accuracy_by_job_bar.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _write_quality_report(path: Path, summary: dict[str, Any]) -> None:
    verdict = (
        "No practical accuracy difference was detected within the predeclared ±"
        f"{summary['equivalence_threshold_pp']:g} percentage-point band."
        if summary["no_practical_accuracy_difference_detected"]
        else "The audit did not establish practical accuracy equivalence."
    )
    lines = [
        "# Scheduler Model-Quality Audit",
        "",
        verdict,
        "",
        "| Mode | Paired runs | Mean Δ accuracy (pp) | 95% bootstrap CI (pp) | Max |Δ| (pp) | Prediction agreement | Exact final weights | Equivalent |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in summary["mode_summaries"]:
        ci = item["bootstrap_95_ci_mean_delta_pp"] or [float("nan"), float("nan")]
        lines.append(
            f"| {item['mode']} | {item['paired_replicates']} | {item['mean_accuracy_delta_pp']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {item['max_absolute_accuracy_delta_pp']:.4f} | "
            f"{100.0 * item['mean_prediction_agreement']:.3f}% | {item['exact_model_matches']}/{item['paired_replicates']} | "
            f"{'yes' if item['practically_equivalent'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Execution integrity",
            "",
            "| Mode | Peak concurrent streams | Shared host / distinct streams | Sustained packed pressure |",
            "|---|---:|:---:|:---:|",
            *[
                f"| {item['mode']} | {item['peak_stream_concurrency']} | "
                f"{'yes' if item['shared_host_distinct_streams_valid'] else 'no'} | "
                f"{'yes' if item['sufficient_packed_pressure'] else 'no'} |"
                for item in summary["execution_integrity"]
            ],
            "",
            "## Interpretation",
            "",
            "This is a paired execution-integrity test. The baseline and scheduler receive the same initial checkpoint, synthetic learnable dataset, deterministic sample order, BF16 precision, optimizer, and epoch count. Therefore the paired delta isolates execution-mode effects under concurrent GPU pressure.",
            "",
            "The result applies to this controlled workload and GPU/software stack. It is strong evidence against scheduler-induced numerical corruption, but it is not a substitute for validating each production model on its real dataset and domain metric.",
            "",
            "Artifacts: `quality_accuracy_by_job_bar.png`, `quality_accuracy_by_job_bar.pdf`, `quality_accuracy_comparison.png`, `quality_accuracy_comparison.pdf`, `quality-results.csv`, and `quality-summary.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_audit(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = set(modes) - {"baseline", "warm", "cold"}
    if invalid:
        raise ValueError(f"unknown modes: {sorted(invalid)}")
    trace = build_quality_trace(
        output_root,
        replicates=args.replicates,
        epochs=args.epochs,
        smoke=args.smoke,
    )
    _write_json(output_root / "quality-trace.json", trace)
    calibration = calibrate_quality_trace(trace, output_root / "calibration")
    warm, cold = build_quality_profile_snapshots(trace, calibration, output_root / "calibration")
    import torch

    total_vram_mib = int(torch.cuda.get_device_properties(args.device_index).total_memory / (1024 * 1024))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(args.device_index),
        "total_vram_mib": total_vram_mib,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "modes": modes,
        "replicates": len(trace),
        "equivalence_threshold_pp": args.equivalence_pp,
    }
    _write_json(output_root / "quality-manifest.json", manifest)
    failures: list[str] = []
    try:
        for mode in modes:
            mode_dir = output_root / mode
            if mode == "baseline":
                raw = run_mp2_baseline(
                    trace,
                    mode_dir,
                    timeout_s=args.mode_timeout,
                    command_factory=_quality_command,
                )
            else:
                raw = run_quality_scheduler_mode(
                    mode,
                    trace,
                    warm if mode == "warm" else cold,
                    mode_dir,
                    timeout_s=args.mode_timeout,
                    total_vram_mib=total_vram_mib,
                )
            _write_json(mode_dir / "raw.json", raw)
            if raw.get("timed_out"):
                failures.append(f"{mode} timed out")
    finally:
        summary = analyze_quality_results(
            output_root,
            modes=modes,
            equivalence_pp=args.equivalence_pp,
        )
    if not summary["complete"]:
        failures.append("not all paired quality jobs completed")
    if not summary["packed_execution_evidence_valid"]:
        failures.append("warm/cold scheduler modes did not both reach three-way validated stream overlap")
    if failures:
        print("; ".join(failures), file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the paired quality audit")
    run.add_argument("--output-root", required=True)
    run.add_argument("--modes", default="baseline,warm,cold")
    run.add_argument("--mode-timeout", type=float, default=900.0)
    run.add_argument("--replicates", type=int, default=8)
    run.add_argument("--epochs", type=int, default=12)
    run.add_argument("--equivalence-pp", type=float, default=0.5)
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--smoke", action="store_true")
    analyze = subparsers.add_parser("analyze", help="rerender an existing audit")
    analyze.add_argument("--output-root", required=True)
    analyze.add_argument("--equivalence-pp", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        analyze_quality_results(Path(args.output_root), equivalence_pp=args.equivalence_pp)
        return 0
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
