"""Generate the versioned 100-job standard histopathology fixture."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import json
import math
import os
import random

import numpy as np

from . import (
    A10_VRAM_CAP_MIB,
    ARRIVAL_RATE,
    DATASET_SIZE,
    DEFAULT_FIXTURE_ROOT,
    EPOCHS,
    INITIAL_BATCH_SIZE,
    INPUT_SIZE,
    JOB_COUNT,
    MAX_PROBE_BATCH_SIZE,
    SCHEMA_VERSION,
    SEED,
)


EXPECTED_LABELS_SHA256 = "b799f3ae98de631d94ed2c8f776d63facb31ad2c8cfa5997cff58c9a2b993b6c"
PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
FORBIDDEN_SOURCE_TOKENS = ("float8", "fp8", "flash_attention_3", "sm_100", "blackwell")


@dataclass(frozen=True, slots=True)
class Architecture:
    family: str
    name: str
    width: int
    depth: int


ARCHITECTURES = (
    Architecture("cnn", "vgg", 24, 2),
    Architecture("cnn", "resnet", 32, 2),
    Architecture("cnn", "densenet", 32, 3),
    Architecture("cnn", "convnext", 40, 3),
    Architecture("efficient_cnn", "mobilenet_v2", 24, 2),
    Architecture("efficient_cnn", "mobilenet_v3", 32, 2),
    Architecture("efficient_cnn", "mbconv", 32, 2),
    Architecture("efficient_cnn", "shufflenet", 40, 3),
    Architecture("mlp_mixer", "patch_mlp", 128, 2),
    Architecture("mlp_mixer", "mixer", 160, 3),
    Architecture("mlp_mixer", "gmlp", 192, 3),
    Architecture("mlp_mixer", "resmlp", 224, 4),
    Architecture("recurrent", "row_lstm", 96, 1),
    Architecture("recurrent", "bilstm", 128, 2),
    Architecture("recurrent", "patch_gru", 160, 2),
    Architecture("recurrent", "convlstm", 192, 3),
    Architecture("vision_transformer", "vit", 96, 2),
    Architecture("vision_transformer", "window_transformer", 128, 3),
    Architecture("vision_transformer", "conv_vit", 160, 3),
    Architecture("vision_transformer", "hybrid_transformer", 192, 4),
)

VARIANTS = (
    ("identity_relu_batch", "relu", "batch", 0.00),
    ("pointwise_gelu_batch", "gelu", "batch", 0.05),
    ("depthwise_silu_group", "silu", "group", 0.10),
    ("bottleneck_leaky_instance", "leaky_relu", "instance", 0.15),
    ("spatial_elu_group", "elu", "group", 0.20),
)


SOURCE_TEMPLATE = '''"""Generated standard scheduler benchmark model: {job_id}."""
from __future__ import annotations

import os
import torch
from torch import nn

from localml_scheduler.elastic import ElasticTrainingSession
from scheduler_benchmark_test.standard.training_runtime import run_generated_job

epochs = int(os.environ.get("STANDARD_BENCH_EPOCHS", "{epochs}"))
batch_size = int(os.environ.get("STANDARD_BENCH_BATCH_SIZE", "{batch_size}"))

{inline_model_library}

SPEC = {spec_json}


class Model(nn.Module):
    """Job-local architecture variant selected by the generated specification."""

    def __init__(self):
        super().__init__()
        self.network = build_architecture(
            family=SPEC["family"],
            architecture=SPEC["architecture"],
            width=SPEC["width"],
            depth=SPEC["depth"],
            activation=SPEC["activation"],
            norm=SPEC["norm"],
            dropout=SPEC["dropout"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> nn.Module:
    return Model()


def build_loader(session, dataset):
    return session.make_dataloader(
        dataset,
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )


def register_training_state(session, model, optimizer, scaler):
    session.register_training_state(model, optimizer, scaler=scaler)


def restore_training_state(session):
    return session.restore_if_present()


def optimizer_step_completed(session, samples, epoch, batch_index, global_step, metrics):
    session.optimizer_step_completed(samples, epoch, batch_index, global_step, metrics=metrics)


if __name__ == "__main__":
    session = ElasticTrainingSession.from_env()
    run_generated_job(
        spec=SPEC,
        build_model=build_model,
        build_loader=build_loader,
        register_training_state=register_training_state,
        restore_training_state=restore_training_state,
        optimizer_step_completed=optimizer_step_completed,
        session=session,
        epochs=epochs,
        batch_size=batch_size,
    )
'''


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_spec(index: int, architecture: Architecture, variant_index: int) -> dict[str, Any]:
    variant_name, activation, norm, dropout = VARIANTS[variant_index]
    architecture_index = ARCHITECTURES.index(architecture)
    precision = PRECISIONS[architecture_index % len(PRECISIONS)]
    return {
        "job_id": f"std-histo-{index:03d}",
        "family": architecture.family,
        "architecture": architecture.name,
        "variant": variant_name,
        "variant_index": variant_index,
        "precision": precision,
        "width": architecture.width,
        "depth": architecture.depth,
        "activation": activation,
        "norm": norm,
        "dropout": dropout,
        "seed": SEED * 1000 + index,
        "learning_rate": 1e-3,
        "input_shape": [3, INPUT_SIZE, INPUT_SIZE],
        "dataset_size": DATASET_SIZE,
        "epochs": EPOCHS,
        "submitted_batch_size": INITIAL_BATCH_SIZE,
        "profile_bucket": f"standard-v1:{architecture.family}:{architecture.name}:{precision}",
    }


def _render_source(spec: dict[str, Any]) -> str:
    library_source = (Path(__file__).with_name("model_library.py")).read_text(encoding="utf-8")
    inline_model_library = library_source[library_source.index("def _activation") :].rstrip()
    source = SOURCE_TEMPLATE.format(
        job_id=spec["job_id"],
        epochs=EPOCHS,
        batch_size=INITIAL_BATCH_SIZE,
        inline_model_library=inline_model_library,
        spec_json=json.dumps(spec, sort_keys=True),
    )
    lowered = source.lower()
    forbidden = [token for token in FORBIDDEN_SOURCE_TOKENS if token in lowered]
    if forbidden:
        raise ValueError(f"Generated source {spec['job_id']} contains forbidden tokens: {forbidden}")
    compile(source, f"{spec['job_id']}.py", "exec")
    return source


def _job_payload(spec: dict[str, Any], source_path: str, source_hash: str) -> dict[str, Any]:
    bucket = str(spec["profile_bucket"])
    steps = math.ceil(DATASET_SIZE / INITIAL_BATCH_SIZE)
    return {
        "job_id": spec["job_id"],
        "agent_id": "standard-benchmark",
        "workflow_id": SCHEMA_VERSION,
        "baseline_model_id": bucket,
        "baseline_model_path": source_path,
        "task_type": "mlevolve_script",
        "priority": 0,
        "status": "PENDING",
        "submitted_at": "1970-01-01T00:00:00+00:00",
        "config": {
            "runner_target": "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            "runner_kwargs": {
                "script_path": source_path,
                "working_dir": ".",
                "result_path": f"working/scheduler_results/result_{spec['job_id']}.json",
                "batch_size": INITIAL_BATCH_SIZE,
                "micro_batch_size": INITIAL_BATCH_SIZE,
                "epochs": EPOCHS,
                "max_epochs": EPOCHS,
                "probe_max_epochs": 1,
                "probe_max_batch_size": MAX_PROBE_BATCH_SIZE,
                "precision": spec["precision"],
                "input_shape": spec["input_shape"],
                "optimizer_name": "AdamW",
                "dataset_size": DATASET_SIZE,
                "steps_per_epoch": steps,
                "num_workers": 1,
                "pin_memory": True,
                "persistent_workers": False,
                "drop_last": False,
                "shuffle": True,
            },
            "loader_target": "localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            "max_steps": None,
            "max_epochs": EPOCHS,
            "seed": spec["seed"],
            "python_executable": None,
            "env": {},
        },
        "resource_requirements": {
            "requires_gpu": True,
            "estimated_vram_mb": None,
            "estimated_ram_mb": None,
            "gpu_slots": 1,
        },
        "packing": {
            "eligible": True,
            "signature": f"standard-source:{source_hash[:20]}",
            "family": bucket,
            "max_slowdown_ratio": 1.30,
            "backend_allowlist": ["cuda_process", "stream"],
        },
        "batch_probe": {
            "enabled": True,
            "probe_target": "localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
            "batch_param_name": "batch_size",
            "model_key": bucket,
            "search_mode": "power_of_two",
            "shape_hints": {
                "input_shape": spec["input_shape"],
                "precision": spec["precision"],
                "profile_bucket": bucket,
            },
            "profile_key": None,
            "profile_namespace": bucket,
            "shape_signature_override": bucket,
            "minimum_batch_size": INITIAL_BATCH_SIZE,
            "contract_version": 3,
            "reuse_only": False,
        },
        "runtime_probe": {
            "enabled": True,
            "probe_target": None,
            "model_key": bucket,
            "strategy": "epoch_1",
        },
        "checkpoint_policy": {
            "save_every_n_steps": None,
            "save_every_epoch": False,
            "keep_last_n": 0,
            "pause_mode": "step",
            "preemptible": False,
        },
        "max_steps": None,
        "max_epochs": EPOCHS,
        "resume_from_checkpoint": None,
        "preload_source": None,
        "authored_batch_size": INITIAL_BATCH_SIZE,
        "metadata": {
            **spec,
            "architecture_source": source_path,
            "architecture_source_hash": source_hash,
            "source_path": source_path,
            "source_hash": source_hash,
            "total_epochs": EPOCHS,
            "completed_epochs": 0,
            "remaining_epochs": EPOCHS,
            "profile_target": "nvidia-a10-compatible",
            "vram_budget_mib": A10_VRAM_CAP_MIB,
        },
        "queue_sequence": 0,
        "status_reason": None,
        "latest_checkpoint_path": None,
        "status_timestamps": {"PENDING": "1970-01-01T00:00:00+00:00"},
        "last_heartbeat_at": None,
        "last_dispatched_at": None,
        "started_at": None,
        "finished_at": None,
        "hold": False,
        "current_batch_size": INITIAL_BATCH_SIZE,
        "profile_state": "WAITING_FOR_DRAIN",
        "force_exclusive": False,
        "placement_generation": 0,
    }


def _settings_payload() -> dict[str, Any]:
    # Keep the model-generation and validation paths independent from the
    # scheduler package.  SchedulerSettings is only needed when rendering the
    # replay settings file, and importing it eagerly prevents validating an
    # already-versioned fixture while scheduler configuration is being edited.
    from localml_scheduler.config import SchedulerSettings

    settings = SchedulerSettings.from_dict(
        {
            "runtime_root": "scheduler_benchmark_runtime",
            "scheduler_poll_interval_seconds": 0.2,
            "baseline_cache": {"warm_queue_policy": "top_k", "warm_queue_top_k": 0, "entry_capacity": 0},
            "redis_cache": {"enabled": False},
            "log_db": {"enabled": False},
            "prediction": {
                "mode": "branch_profile",
                "timeout_ms": 1000,
                "fallback_to_exclusive": True,
                "branch": {"enabled": True, "fixed_confidence_if_uncalibrated": 0.55},
                "ml": {"enabled": False, "device": "cpu", "cache_size": 1024},
            },
            "gpu_scheduler": {
                "enabled": True,
                "mode": "adaptive",
                "backend_priority": ["cuda_process", "stream", "exclusive"],
                "max_packed_jobs_per_gpu": 8,
                "candidate_window_size": 16,
                "batch_probe_enabled": True,
                "batch_probe_min_batch_size": INITIAL_BATCH_SIZE,
                "batch_probe_max_batch_size": MAX_PROBE_BATCH_SIZE,
                "batch_probe_max_search_rounds": 14,
                "batch_probe_search_mode": "power_of_two",
                "model_family_probe_enabled": False,
                "checkpoint_preemption_enabled": False,
                "memory": {"vram_budget_fraction": A10_VRAM_CAP_MIB / 24_576.0},
                "telemetry": {"device_poll_ms": 1000, "pair_recheck_every_steps": 20},
                "early_stop": {"enabled": False, "plot_enabled": False},
                "profiling": {"warmup_steps": 2, "solo_probe_steps": 5},
                "adaptive": {
                    "exact_search_max_jobs": 8,
                    "vram_bucket_mb": 128,
                    "frontier_width": 32,
                    "finalist_limit": 64,
                    "replan_debounce_seconds": 1.0,
                },
                "submission_defaults": {
                    "requires_gpu": True,
                    "packing_eligible": True,
                    "packing_family": "standard_histopath_v1",
                    "backend_allowlist": ["cuda_process", "stream"],
                    "batch_probe_enabled": True,
                    "batch_probe_search_mode": "power_of_two",
                    "runtime_probe_enabled": True,
                    "runtime_probe_strategy": "epoch_1",
                },
                "mps": {"enabled": False},
                "cuda_process": {"enabled": True, "default_omp_num_threads": 2, "default_mkl_num_threads": 2},
                "stream": {"enabled": True, "host_poll_interval_seconds": 0.1, "host_join_timeout_seconds": 5.0},
            },
        }
    )
    payload = settings.to_dict()
    payload["runtime_root"] = "scheduler_benchmark_runtime"
    return payload


def render_fixture() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    jobs: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    index = 1
    for architecture in ARCHITECTURES:
        for variant_index in range(len(VARIANTS)):
            spec = _variant_spec(index, architecture, variant_index)
            source = _render_source(spec)
            relative_source = f"scheduler_benchmark_test/fixtures/standard_histopath_v1/sources/{spec['job_id']}.py"
            source_bytes = source.encode("utf-8")
            source_hash = _sha256_bytes(source_bytes)
            files[f"sources/{spec['job_id']}.py"] = source_bytes
            jobs.append(_job_payload(spec, relative_source, source_hash))
            specs.append(spec)
            index += 1

    if len(jobs) != JOB_COUNT:
        raise AssertionError(f"Expected {JOB_COUNT} jobs, rendered {len(jobs)}")

    arrival_jobs = list(jobs)
    random.Random(SEED).shuffle(arrival_jobs)
    rng = np.random.default_rng(SEED)
    timestamps = np.cumsum(rng.exponential(scale=1.0 / ARRIVAL_RATE, size=JOB_COUNT))
    actions = [
        {
            "action": "SUBMIT",
            "job_id": job["job_id"],
            "relative_seconds": round(float(timestamp), 3),
            "final_cleanup": False,
        }
        for job, timestamp in zip(arrival_jobs, timestamps, strict=True)
    ]
    timeline = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "arrival_process": "poisson",
        "arrival_rate_jobs_per_second": ARRIVAL_RATE,
        "actions": actions,
    }
    jobs_bytes = b"".join(
        (json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for job in jobs
    )
    files["jobs.jsonl"] = jobs_bytes
    files["timeline.json"] = _json_bytes(timeline)
    files["scheduler_settings.replay.json"] = _json_bytes(_settings_payload())
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "source_run_root": None,
        "original_input_dir": None,
        "dataset_id": "histopathologic-cancer-detection",
        "dataset_size": DATASET_SIZE,
        "epochs_per_job": EPOCHS,
        "job_count": JOB_COUNT,
        "submit_count": JOB_COUNT,
        "command_count": JOB_COUNT,
        "task_type_counts": {"mlevolve_script": JOB_COUNT},
        "reference_metrics": {},
        "profile_state_policy": "cold isolated database per arm and repetition",
    }
    files["baseline_summary.json"] = _json_bytes(baseline)

    family_counts = Counter(spec["family"] for spec in specs)
    architecture_counts = Counter(spec["architecture"] for spec in specs)
    precision_counts = Counter(spec["precision"] for spec in specs)
    variant_counts = Counter(spec["variant"] for spec in specs)
    bucket_counts = Counter(spec["profile_bucket"] for spec in specs)
    checksums = {name: _sha256_bytes(payload) for name, payload in sorted(files.items())}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "scheduler_benchmark_test.standard.generate_fixture",
        "generator_version": "1.0.0",
        "seed": SEED,
        "job_count": JOB_COUNT,
        "epochs_per_job": EPOCHS,
        "dataset": {
            "id": "histopathologic-cancer-detection",
            "labeled_image_count": DATASET_SIZE,
            "train_labels_sha256": EXPECTED_LABELS_SHA256,
            "input_shape": [3, INPUT_SIZE, INPUT_SIZE],
        },
        "arrival": {
            "process": "poisson",
            "rate_jobs_per_second": ARRIVAL_RATE,
            "first_relative_seconds": actions[0]["relative_seconds"],
            "last_relative_seconds": actions[-1]["relative_seconds"],
        },
        "training": {
            "initial_batch_size": INITIAL_BATCH_SIZE,
            "max_probe_batch_size": MAX_PROBE_BATCH_SIZE,
            "optimizer": "AdamW",
            "samples_per_job": DATASET_SIZE * EPOCHS,
            "samples_per_case": DATASET_SIZE * EPOCHS * JOB_COUNT,
        },
        "target_profile": {
            "name": "NVIDIA A10 compatible",
            "vram_cap_mib": A10_VRAM_CAP_MIB,
            "allowed_precisions": list(PRECISIONS),
            "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
        },
        "family_counts": dict(sorted(family_counts.items())),
        "architecture_counts": dict(sorted(architecture_counts.items())),
        "precision_counts": dict(sorted(precision_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "profile_bucket_counts": dict(sorted(bucket_counts.items())),
        "file_sha256": checksums,
    }
    files["manifest.json"] = _json_bytes(manifest)
    return files


def write_fixture(output_root: str | Path, *, check: bool = False) -> list[str]:
    root = Path(output_root).expanduser().resolve()
    expected = render_fixture()
    mismatches: list[str] = []
    for relative, payload in expected.items():
        path = root / relative
        if not path.exists() or path.read_bytes() != payload:
            mismatches.append(relative)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
    existing_sources = set((root / "sources").glob("*.py")) if (root / "sources").exists() else set()
    expected_sources = {root / name for name in expected if name.startswith("sources/")}
    extras = sorted(str(path.relative_to(root)) for path in existing_sources - expected_sources)
    mismatches.extend(extras)
    if not check:
        for relative in extras:
            (root / relative).unlink()
    return mismatches


def validate_dataset_identity(data_root: str | Path) -> None:
    labels = Path(data_root).expanduser().resolve() / "train_labels.csv"
    if not labels.is_file():
        raise FileNotFoundError(f"Missing dataset labels file: {labels}")
    actual = _sha256_path(labels)
    if actual != EXPECTED_LABELS_SHA256:
        raise ValueError(f"Unexpected train_labels.csv SHA-256: {actual}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_FIXTURE_ROOT))
    parser.add_argument("--check", action="store_true", help="Fail if committed fixture content differs from generation.")
    parser.add_argument("--data-root", default=os.environ.get("HISTOPATH_DATA_ROOT"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root:
        validate_dataset_identity(args.data_root)
    mismatches = write_fixture(args.output, check=args.check)
    if args.check and mismatches:
        print("Fixture drift detected:")
        for path in mismatches:
            print(f"  {path}")
        return 1
    action = "Checked" if args.check else "Generated"
    print(f"{action} {JOB_COUNT} jobs at {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
