"""Run one-epoch functional validation of every generated model in parallel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

from . import A10_VRAM_CAP_MIB, DEFAULT_FIXTURE_ROOT, JOB_COUNT, PACKAGE_ROOT
from .generate_fixture import validate_dataset_identity
from .validate import _read_jobs, _verify_fixture_checksums


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _result_errors(result: dict[str, Any], *, expected_samples: int, batch_size: int) -> list[str]:
    errors: list[str] = []
    if int(result.get("epochs", -1)) != 1:
        errors.append(f"epochs={result.get('epochs')!r}, expected 1")
    if int(result.get("samples_seen", -1)) != expected_samples:
        errors.append(f"samples_seen={result.get('samples_seen')!r}, expected {expected_samples}")
    if int(result.get("batch_size", -1)) != batch_size:
        errors.append(f"batch_size={result.get('batch_size')!r}, expected {batch_size}")
    try:
        if not math.isfinite(float(result["loss"])):
            errors.append("loss is not finite")
    except (KeyError, TypeError, ValueError):
        errors.append("loss is missing")
    if int(result.get("global_steps", -1)) != math.ceil(expected_samples / batch_size):
        errors.append("global_steps does not cover the complete validation epoch")
    return errors


def validate_fixture_models_functionally(
    *,
    fixture: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    samples_per_model: int = 256,
    batch_size: int = 4,
    parallelism: int = 4,
    resume: bool = False,
    timeout_seconds: float | None = 300.0,
) -> dict[str, Any]:
    fixture_path = Path(fixture).expanduser().resolve()
    data_path = Path(data_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    samples_per_model = max(1, int(samples_per_model))
    batch_size = max(1, min(int(batch_size), samples_per_model))
    parallelism = max(1, int(parallelism))
    output.mkdir(parents=True, exist_ok=True)

    validate_dataset_identity(data_path)
    drift = _verify_fixture_checksums(fixture_path)
    if drift:
        raise ValueError(f"Fixture checksum drift: {', '.join(drift[:5])}")
    if not torch.cuda.is_available():
        raise RuntimeError("Functional validation requires a CUDA GPU")

    jobs = _read_jobs(fixture_path)
    if len(jobs) != JOB_COUNT:
        raise ValueError(f"Expected {JOB_COUNT} jobs, found {len(jobs)}")

    def run_one(position: int, job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        job_id = str(job["job_id"])
        metadata = dict(job.get("metadata") or {})
        relative_source = str(
            metadata.get("source_path")
            or (job.get("config") or {}).get("runner_kwargs", {}).get("script_path")
        )
        source = PACKAGE_ROOT.parent.parent / relative_source
        expected_hash = str(metadata.get("source_hash") or sha256(source.read_bytes()).hexdigest())
        job_output = output / "jobs" / job_id
        metric_path = job_output / "metric.json"
        existing = _load_result(metric_path) if resume else None
        if existing and existing.get("source_hash") == expected_hash:
            errors = _result_errors(existing, expected_samples=samples_per_model, batch_size=batch_size)
            if not errors:
                return position, {**existing, "status": "PASSED", "errors": [], "resumed": True}

        job_output.mkdir(parents=True, exist_ok=True)
        stdout_path = job_output / "stdout.log"
        stderr_path = job_output / "stderr.log"
        env = dict(os.environ)
        python_path = [str(PACKAGE_ROOT.parent.parent)]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(python_path),
                "PYTHONUNBUFFERED": "1",
                "HISTOPATH_DATA_ROOT": str(data_path),
                "STANDARD_BENCH_EPOCHS": "1",
                "STANDARD_BENCH_BATCH_SIZE": str(batch_size),
                "STANDARD_BENCH_ALLOW_PARTIAL": "1",
                "STANDARD_BENCH_MAX_SAMPLES": str(samples_per_model),
                "STANDARD_BENCH_RESULT_DIR": str(job_output),
                "STANDARD_BENCH_VRAM_CAP_MIB": str(A10_VRAM_CAP_MIB),
            }
        )
        started = time.time()
        timed_out = False
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    [sys.executable, str(source)],
                    cwd=str(PACKAGE_ROOT.parent.parent),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = -1
                timed_out = True

        result = _load_result(metric_path) or {}
        errors = (
            _result_errors(result, expected_samples=samples_per_model, batch_size=batch_size)
            if result
            else ["metric.json was not produced"]
        )
        if returncode != 0:
            errors.append(f"process return code {returncode}")
        if timed_out:
            errors.append("validation timeout")
        record = {
            **result,
            "job_id": job_id,
            "source_hash": expected_hash,
            "status": "PASSED" if not errors else "FAILED",
            "errors": errors,
            "resumed": False,
            "validation_wall_seconds": time.time() - started,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        metric_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return position, record

    started = time.time()
    completed_records: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_one, position, job) for position, job in enumerate(jobs, start=1)]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            position, record = future.result()
            completed_records.append((position, record))
            print(f"[{completed_count}/{JOB_COUNT}] {record['job_id']}: {record['status']}", flush=True)

    records = [record for _position, record in sorted(completed_records)]
    failures = [record["job_id"] for record in records if record.get("status") != "PASSED"]
    report = {
        "schema_version": "standard-histopath-functional-validation-v1",
        "fixture": str(fixture_path),
        "dataset_root": str(data_path),
        "physical_device": torch.cuda.get_device_name(0),
        "job_count": len(records),
        "samples_per_model": samples_per_model,
        "batch_size": batch_size,
        "epochs_per_model": 1,
        "parallelism": parallelism,
        "passed_job_count": sum(record.get("status") == "PASSED" for record in records),
        "failed_job_ids": failures,
        "elapsed_seconds": time.time() - started,
        "functional_accepted": len(records) == JOB_COUNT and not failures,
        "full_dataset_validation": False,
        "records": records,
    }
    (output / "functional_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_ROOT))
    parser.add_argument("--data-root", default=os.environ.get("HISTOPATH_DATA_ROOT"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-model", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("--data-root or HISTOPATH_DATA_ROOT is required")
    report = validate_fixture_models_functionally(
        fixture=args.fixture,
        data_root=args.data_root,
        output_root=args.output_root,
        samples_per_model=args.samples_per_model,
        batch_size=args.batch_size,
        parallelism=args.parallelism,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("functional_accepted", "job_count", "passed_job_count", "elapsed_seconds")
            },
            indent=2,
        )
    )
    return 0 if report["functional_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
