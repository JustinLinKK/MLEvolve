"""Verify the exact PerfSeer V3.2 artifact through the scheduler adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from localml_scheduler.config import PredictionSettings
from localml_scheduler.domain import ResourceRequirements, TrainingJob
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction import MLVramPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    predictor = MLVramPredictor(
        PredictionSettings(
            mode="ml_predictor",
            require_ml_prediction=True,
            v32_source_root=str(args.source_root.resolve()),
            v32_artifact_path=str(args.artifact.resolve()),
            v32_capture_device=args.device,
            v32_async_source_capture=False,
            conversion_timeout_seconds=120,
        ),
        HardwareProfile(
            "aba-a100-80gb-pcie",
            "linux",
            "NVIDIA A100 80GB PCIe",
            81920,
            "8.0",
            "12.6",
            "aba",
        ),
    )
    if not predictor.available:
        raise RuntimeError(predictor.unavailable_reason or "predictor unavailable")

    job = TrainingJob.create(
        "localml_scheduler.examples.toy_pytorch_runner:run_toy_job",
        "predictor-verification",
        str(args.model_source.resolve()),
        job_id="predictor-v32-aba-verification",
        resource_requirements=ResourceRequirements(estimated_avg_vram_mb=777),
        metadata={
            "perfseer_model": {
                "source_path": str(args.model_source.resolve()),
                "entry": "build_model",
                "input_shapes": [["$batch", 3, 32, 32]],
                "input_dtypes": ["float32"],
                "precision": "fp32",
            }
        },
    )
    peak_vram_mib = predictor.predict_avg_vram_mb(job, 2)
    epoch_seconds = predictor.predict_seconds_per_epoch_options(job, [2])[2]
    if peak_vram_mib <= 0 or epoch_seconds <= 0:
        raise RuntimeError("predictor returned a non-positive scheduler estimate")
    print(f"source={predictor.last_sources[job.job_id]}")
    print(f"train_peak_vram_mib={peak_vram_mib:.6f}")
    print(f"train_epoch_seconds={epoch_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
