"""Generate and verify Stress Test Data v1.0."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import json
import math
import sys

import torch

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.prediction.ml_predictor import model_specification_for_job

from . import REPOSITORY_ROOT, STRESS_TEST_DATA_JOB_COUNT, STRESS_TEST_DATA_V1_0_FIXTURE


MODEL_SOURCE = STRESS_TEST_DATA_V1_0_FIXTURE / "model_source.py"
A10_ARTIFACT = (
    REPOSITORY_ROOT
    / "PerfSeer-predictor"
    / "models"
    / "nvidia_a10"
    / "student_a10_cpu.torchscript.pt"
)
BLACKWELL_ARTIFACT = (
    REPOSITORY_ROOT
    / "PerfSeer-predictor"
    / "models"
    / "nvidia_rtx_pro_6000_blackwell"
    / "student_rtx_pro_6000_blackwell_cpu.torchscript.pt"
)
DEFAULT_ARTIFACT = A10_ARTIFACT
PERFSEER_SRC = REPOSITORY_ROOT / "PerfSeer-predictor" / "src"
JOBLIST = STRESS_TEST_DATA_V1_0_FIXTURE / "joblist.json"
MANIFEST = STRESS_TEST_DATA_V1_0_FIXTURE / "manifest.json"
PRECISIONS = ("fp32_ieee", "tf32", "bf16_amp", "fp16_amp")


ARCHITECTURES: tuple[tuple[str, str, int, int], ...] = (
    ("cnn", "vgg", 8, 1),
    ("cnn", "resnet", 10, 1),
    ("cnn", "densenet", 12, 1),
    ("cnn", "convnext_compatible", 14, 2),
    ("efficient_cnn", "mobilenet_v2", 8, 1),
    ("efficient_cnn", "mobilenet_v3", 10, 2),
    ("efficient_cnn", "mbconv", 12, 2),
    ("efficient_cnn", "efficient_residual", 14, 2),
    ("mlp", "patch_mlp", 16, 1),
    ("mlp", "mixer_mlp", 20, 2),
    ("mlp", "gmlp_compatible", 24, 2),
    ("mlp", "resmlp_compatible", 28, 3),
    ("recurrent", "row_lstm", 8, 1),
    ("recurrent", "bilstm", 10, 1),
    ("recurrent", "patch_gru", 12, 2),
    ("recurrent", "conv_lstm", 14, 2),
    ("hybrid", "conv_mlp", 10, 1),
    ("hybrid", "depthwise_mlp", 12, 1),
    ("hybrid", "residual_mlp", 14, 2),
    ("hybrid", "dense_mlp", 16, 2),
)

VARIANTS: tuple[tuple[str, str, float, int, int], ...] = (
    ("relu_base", "relu", 1.00, 0, 32),
    ("gelu_wide", "gelu", 1.25, 0, 48),
    ("silu_deep", "silu", 1.00, 1, 64),
    ("relu_compact", "relu", 0.75, 0, 80),
    ("gelu_deep_wide", "gelu", 1.50, 1, 96),
)


def _scaled_width(base: int, scale: float) -> int:
    return max(4, int(round((base * scale) / 2.0)) * 2)


def build_joblist() -> list[dict[str, Any]]:
    """Return the deterministic 20-architecture by 5-variant model list."""

    source_relative = str(MODEL_SOURCE.relative_to(REPOSITORY_ROOT))
    jobs: list[dict[str, Any]] = []
    index = 1
    for family, architecture, base_width, base_depth in ARCHITECTURES:
        for variant, activation, width_scale, extra_depth, input_size in VARIANTS:
            jobs.append(
                {
                    "id": f"stress-v1-{index:03d}",
                    "family": family,
                    "architecture": architecture,
                    "variant": variant,
                    "epochs": 1,
                    "perfseer_model": {
                        "source_path": source_relative,
                        "entry": "build_model",
                        "input_shapes": [["$batch", 3, input_size, input_size]],
                        "input_dtypes": ["float32"],
                        "precision": PRECISIONS[(index - 1) % len(PRECISIONS)],
                        "constructor_kwargs": {
                            "architecture": architecture,
                            "width": _scaled_width(base_width, width_scale),
                            "depth": base_depth + extra_depth,
                            "activation": activation,
                        },
                    },
                }
            )
            index += 1
    if len(jobs) != STRESS_TEST_DATA_JOB_COUNT:
        raise AssertionError(f"expected {STRESS_TEST_DATA_JOB_COUNT} jobs, built {len(jobs)}")
    return jobs


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render_fixture() -> dict[str, bytes]:
    jobs = build_joblist()
    joblist_bytes = _json_bytes(
        {
            "schema_version": "stress-test-data-v1.0",
            "description": "Stress Test Data v1.0: 100 one-epoch model specifications restricted to the deployed 53/3/40 student vocabulary",
            "jobs": jobs,
        }
    )
    family_counts = Counter(job["family"] for job in jobs)
    architecture_counts = Counter(job["architecture"] for job in jobs)
    precision_counts = Counter(job["perfseer_model"]["precision"] for job in jobs)
    variant_counts = Counter(job["variant"] for job in jobs)
    source_hash = sha256(MODEL_SOURCE.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "stress-test-data-v1.0-manifest",
        "dataset_name": "stress_test_data",
        "dataset_version": "1.0",
        "job_count": len(jobs),
        "student_schema": {"node_dim": 53, "edge_dim": 3, "global_dim": 40},
        "source_path": str(MODEL_SOURCE.relative_to(REPOSITORY_ROOT)),
        "source_sha256": source_hash,
        "joblist_sha256": sha256(joblist_bytes).hexdigest(),
        "family_counts": dict(sorted(family_counts.items())),
        "architecture_counts": dict(sorted(architecture_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "precision_counts": dict(sorted(precision_counts.items())),
    }
    return {"joblist.json": joblist_bytes, "manifest.json": _json_bytes(manifest)}


def write_fixture(*, check: bool = False) -> list[str]:
    expected = render_fixture()
    mismatches: list[str] = []
    STRESS_TEST_DATA_V1_0_FIXTURE.mkdir(parents=True, exist_ok=True)
    for relative, contents in expected.items():
        path = STRESS_TEST_DATA_V1_0_FIXTURE / relative
        if not path.is_file() or path.read_bytes() != contents:
            mismatches.append(relative)
            if not check:
                path.write_bytes(contents)
    return mismatches


def _load_jobs() -> list[dict[str, Any]]:
    payload = json.loads(JOBLIST.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Stress Test Data v1.0 joblist has no jobs array")
    return jobs


def verify_predictions(
    *,
    batch_size: int = 2,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Encode and infer every entry using the scheduler's real metadata parser."""

    if str(PERFSEER_SRC) not in sys.path:
        sys.path.insert(0, str(PERFSEER_SRC))
    from perfseer_source_converter import SourceModelSpec, convert_source_to_networkx
    from perfseer_student import EDGE_DIM, GLOBAL_DIM, NODE_DIM, StudentRuntime, encode_source
    from perfseer_student.features import OP_INDEX

    drift = write_fixture(check=True)
    if drift:
        raise ValueError(f"Stress Test Data v1.0 drift: {', '.join(drift)}")
    selected_artifact = Path(artifact_path or DEFAULT_ARTIFACT).expanduser().resolve()
    runtime = StudentRuntime(selected_artifact)
    records: list[dict[str, Any]] = []
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for item in _load_jobs():
        metadata = {"perfseer_model": dict(item["perfseer_model"])}
        job = build_mlevolve_job(
            workflow_id="stress-test-data-v1.0",
            baseline_model_id=str(item["architecture"]),
            baseline_model_path=str(MODEL_SOURCE),
            runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
            runner_kwargs={"batch_size": batch_size, "epochs": 1, "num_samples": 8},
            packing_family=str(item["family"]),
            packing_eligible=True,
            max_epochs=1,
            metadata=metadata,
        )
        job.job_id = str(item["id"])
        specification = model_specification_for_job(job, batch_size)
        graph = convert_source_to_networkx(
            SourceModelSpec(
                source_path=specification.source_path,
                entry=specification.entry,
                input_shapes=specification.input_shapes,
                constructor_args=specification.constructor_args,
                constructor_kwargs=specification.constructor_kwargs,
                input_dtypes=specification.input_dtypes,
            )
        )
        operations = sorted(
            {
                str(data["feature"]["type"])
                for _node, data in graph.nodes(data=True)
            }
        )
        unknown = sorted(set(operations) - set(OP_INDEX))
        if unknown:
            raise AssertionError(f"{item['id']} contains unknown student operations: {unknown}")
        encoded = encode_source(
            specification.source_path,
            specification.entry,
            specification.input_shapes,
            specification.precision,
            constructor_args=specification.constructor_args,
            constructor_kwargs=specification.constructor_kwargs,
            input_dtypes=specification.input_dtypes,
        )
        if encoded.x.shape[1] != NODE_DIM:
            raise AssertionError(f"{item['id']} node_dim={encoded.x.shape[1]}, expected {NODE_DIM}")
        if encoded.edge_attr.shape[1] != EDGE_DIM:
            raise AssertionError(f"{item['id']} edge_dim={encoded.edge_attr.shape[1]}, expected {EDGE_DIM}")
        if encoded.u.shape != (1, GLOBAL_DIM):
            raise AssertionError(f"{item['id']} global shape={tuple(encoded.u.shape)}, expected (1, {GLOBAL_DIM})")
        if any(tensor.device.type != "cpu" for tensor in encoded.as_tuple()):
            raise AssertionError(f"{item['id']} produced a non-CPU predictor tensor")
        prediction = float(runtime.predict_train_mem_mb(encoded))
        if not math.isfinite(prediction) or prediction <= 0:
            raise AssertionError(f"{item['id']} produced invalid train_mem={prediction}")
        records.append(
            {
                "id": item["id"],
                "family": item["family"],
                "architecture": item["architecture"],
                "variant": item["variant"],
                "precision": specification.precision,
                "node_count": int(encoded.x.shape[0]),
                "edge_count": int(encoded.edge_index.shape[1]),
                "operations": operations,
                "train_mem_mb": prediction,
            }
        )
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if after != before:
        raise AssertionError(f"predictor CUDA allocation changed from {before} to {after}")
    return {
        "schema_version": "stress-test-data-v1.0-verification",
        "dataset_name": "stress_test_data",
        "dataset_version": "1.0",
        "artifact_path": str(selected_artifact),
        "artifact_sha256": sha256(selected_artifact.read_bytes()).hexdigest(),
        "accepted": len(records) == STRESS_TEST_DATA_JOB_COUNT,
        "job_count": len(records),
        "finite_positive_prediction_count": sum(
            math.isfinite(record["train_mem_mb"]) and record["train_mem_mb"] > 0 for record in records
        ),
        "cpu_only": True,
        "cuda_allocation_before": before,
        "cuda_allocation_after": after,
        "operation_union": sorted({operation for record in records for operation in record["operations"]}),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-predictions", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--artifact",
        help="Compatible CPU TorchScript artifact; defaults to the registered A10 artifact.",
    )
    parser.add_argument("--output-report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mismatches = write_fixture(check=args.check)
    if args.check and mismatches:
        print("Fixture drift: " + ", ".join(mismatches))
        return 1
    if args.verify_predictions:
        report = verify_predictions(
            batch_size=max(1, args.batch_size),
            artifact_path=args.artifact,
        )
        if args.output_report:
            Path(args.output_report).write_bytes(_json_bytes(report))
        print(
            json.dumps(
                {
                    "accepted": report["accepted"],
                    "job_count": report["job_count"],
                    "finite_positive_prediction_count": report["finite_positive_prediction_count"],
                    "cpu_only": report["cpu_only"],
                    "operation_union": report["operation_union"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["accepted"] else 1
    action = "Checked" if args.check else "Generated"
    print(f"{action} {STRESS_TEST_DATA_JOB_COUNT} stress-test models at {STRESS_TEST_DATA_V1_0_FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
