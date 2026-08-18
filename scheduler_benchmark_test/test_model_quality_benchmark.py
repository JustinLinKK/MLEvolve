from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scheduler_benchmark_test.model_quality_benchmark import (
    analyze_quality_results,
    build_quality_trace,
)
from scheduler_benchmark_test.model_quality_runner import (
    build_quality_dataset,
    create_quality_baseline,
    train_quality_model,
)


def test_quality_dataset_is_deterministic_and_learnable() -> None:
    left_x, left_y = build_quality_dataset(
        num_samples=64,
        input_dim=16,
        output_dim=4,
        dataset_seed=7,
        teacher_seed=11,
        label_noise=0.2,
    )
    right_x, right_y = build_quality_dataset(
        num_samples=64,
        input_dim=16,
        output_dim=4,
        dataset_seed=7,
        teacher_seed=11,
        label_noise=0.2,
    )
    assert torch.equal(left_x, right_x)
    assert torch.equal(left_y, right_y)
    assert len(torch.unique(left_y)) > 1


def test_quality_pause_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    create_quality_baseline(
        baseline_path, input_dim=16, hidden_dim=32, output_dim=4, seed=101
    )
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    common = {
        "input_dim": 16,
        "hidden_dim": 32,
        "output_dim": 4,
        "train_samples": 128,
        "validation_samples": 64,
        "batch_size": 32,
        "dataset_seed": 55,
        "teacher_seed": 91,
        "label_noise": 0.2,
        "learning_rate": 2e-3,
        "precision": "fp32_ieee",
    }
    uninterrupted = train_quality_model(
        baseline=baseline,
        params={**common, "epochs": 2},
        device=torch.device("cpu"),
    )
    captured: list[dict] = []

    def capture(_epoch, _step, _metrics, state_factory) -> None:
        captured.append(state_factory())

    train_quality_model(
        baseline=baseline,
        params={**common, "epochs": 1},
        device=torch.device("cpu"),
        epoch_callback=capture,
    )
    resumed = train_quality_model(
        baseline=baseline,
        params={**common, "epochs": 2},
        device=torch.device("cpu"),
        resume_state=captured[-1],
    )
    assert resumed["model_parameter_sha256"] == uninterrupted["model_parameter_sha256"]
    assert resumed["final_validation_accuracy"] == uninterrupted["final_validation_accuracy"]
    for resumed_epoch, full_epoch in zip(resumed["history"], uninterrupted["history"]):
        for key in ("train_loss", "validation_accuracy", "validation_loss"):
            assert resumed_epoch[key] == pytest.approx(full_epoch[key])


def test_quality_analysis_pairs_seeds_and_renders_chart(tmp_path: Path) -> None:
    trace = build_quality_trace(tmp_path, replicates=3, smoke=True)
    for mode, delta in (("baseline", 0.0), ("warm", 0.001), ("cold", -0.001)):
        mode_dir = tmp_path / mode
        results_dir = mode_dir / "results"
        results_dir.mkdir(parents=True)
        attempts = []
        for item in trace:
            result_path = results_dir / f"{item['job_id']}.json"
            result_path.write_text(
                json.dumps(
                    {
                        "final_validation_accuracy": 0.75 + delta,
                        "final_validation_loss": 0.5,
                        "model_parameter_sha256": mode,
                        "validation_label_sha256": "labels",
                        "validation_predictions": [0, 1, 2, 3],
                        "history": [
                            {"epoch": 1, "validation_accuracy": 0.70 + delta},
                            {"epoch": 2, "validation_accuracy": 0.75 + delta},
                        ],
                        "training_seconds": 1.0,
                    }
                )
            )
            attempts.append(
                {
                    "logical_job_id": item["job_id"],
                    "step_idx": item["step_idx"],
                    "status": "succeeded" if mode == "baseline" else "completed",
                    "backend": "multiprocess" if mode == "baseline" else "stream",
                    "started_at": 1.0,
                    "finished_at": 2.0,
                    "result_path": str(result_path),
                }
            )
        (mode_dir / "raw.json").write_text(
            json.dumps(
                {
                    "attempts": attempts,
                    "stream_assertions": {
                        "valid": True,
                        "overlaps": [] if mode == "baseline" else [{"jobs": ["a", "b", "c"]}],
                    },
                }
            )
        )
    summary = analyze_quality_results(tmp_path)
    assert summary["complete"]
    assert summary["no_practical_accuracy_difference_detected"]
    assert summary["mode_summaries"][0]["mean_accuracy_delta_pp"] == pytest.approx(0.1)
    assert (tmp_path / "quality_accuracy_comparison.png").stat().st_size > 0
    assert (tmp_path / "quality_accuracy_comparison.pdf").stat().st_size > 0
    assert (tmp_path / "quality_accuracy_by_job_bar.png").stat().st_size > 0
    assert (tmp_path / "quality_accuracy_by_job_bar.pdf").stat().st_size > 0
