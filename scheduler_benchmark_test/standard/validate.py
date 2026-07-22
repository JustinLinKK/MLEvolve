"""Run the mandatory one-epoch, full-data validation for the standard fixture."""

from __future__ import annotations

from collections import defaultdict
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

from . import A10_VRAM_CAP_MIB, DATASET_SIZE, DEFAULT_FIXTURE_ROOT, JOB_COUNT, PACKAGE_ROOT
from .generate_fixture import validate_dataset_identity


def _read_jobs(fixture: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (fixture / "jobs.jsonl").read_text(encoding="utf-8").splitlines() if line]


def _source_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _verify_fixture_checksums(fixture: Path) -> list[str]:
    """Verify the committed fixture without re-rendering scheduler settings."""
    manifest_path = fixture / "manifest.json"
    manifest = _load_result(manifest_path)
    if manifest is None:
        return ["manifest.json is missing or invalid"]
    checksums = manifest.get("file_sha256")
    if not isinstance(checksums, dict):
        return ["manifest.json has no file_sha256 mapping"]

    mismatches: list[str] = []
    for relative, expected in sorted(checksums.items()):
        path = fixture / str(relative)
        if not path.is_file():
            mismatches.append(f"{relative} (missing)")
        elif _source_hash(path) != str(expected):
            mismatches.append(f"{relative} (checksum)")

    expected_sources = {
        fixture / str(relative)
        for relative in checksums
        if str(relative).startswith("sources/") and str(relative).endswith(".py")
    }
    actual_sources = set((fixture / "sources").glob("*.py")) if (fixture / "sources").is_dir() else set()
    mismatches.extend(
        f"{path.relative_to(fixture)} (unexpected)" for path in sorted(actual_sources - expected_sources)
    )
    return mismatches


def _check_job_result(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(result.get("epochs", -1)) != 1:
        errors.append(f"epochs={result.get('epochs')!r}, expected 1")
    if int(result.get("samples_seen", -1)) != DATASET_SIZE:
        errors.append(f"samples_seen={result.get('samples_seen')!r}, expected {DATASET_SIZE}")
    loss_value = result.get("loss")
    try:
        if loss_value is None or not math.isfinite(float(loss_value)):
            errors.append("loss is not finite")
    except (TypeError, ValueError):
        errors.append("loss is missing")
    for field in ("peak_allocated_mib", "peak_reserved_mib"):
        value = result.get(field)
        if value is None:
            errors.append(f"{field} is missing (validation must run on CUDA)")
        elif float(value) > A10_VRAM_CAP_MIB:
            errors.append(f"{field}={float(value):.1f} exceeds {A10_VRAM_CAP_MIB} MiB")
    return errors


def validate_fixture_models(
    *,
    fixture: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    resume: bool = False,
    limit: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    fixture_path = Path(fixture).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    validate_dataset_identity(data_root)
    drift = _verify_fixture_checksums(fixture_path)
    if drift:
        raise ValueError(f"Fixture checksum drift: {', '.join(drift[:5])}")
    if not torch.cuda.is_available():
        raise RuntimeError("The standard one-epoch validation requires a CUDA GPU")

    jobs = _read_jobs(fixture_path)
    if len(jobs) != JOB_COUNT:
        raise ValueError(f"Expected {JOB_COUNT} jobs, found {len(jobs)}")
    selected_jobs = jobs[: max(0, int(limit))] if limit is not None else jobs
    records: list[dict[str, Any]] = []
    started = time.time()
    for position, job in enumerate(selected_jobs, start=1):
        job_id = str(job["job_id"])
        metadata = dict(job.get("metadata") or {})
        relative_source = str(metadata.get("source_path") or (job.get("config") or {}).get("runner_kwargs", {}).get("script_path"))
        source = PACKAGE_ROOT.parent.parent / relative_source
        if not source.is_file():
            raise FileNotFoundError(f"Missing source for {job_id}: {source}")
        job_output = output / "jobs" / job_id
        metric_path = job_output / "metric.json"
        expected_hash = str(metadata.get("source_hash") or _source_hash(source))
        existing = _load_result(metric_path) if resume else None
        if existing and existing.get("source_hash") == expected_hash and not _check_job_result(existing):
            record = dict(existing)
            record.update({"status": "PASSED", "resumed": True, "profile_bucket": metadata.get("profile_bucket")})
            records.append(record)
            continue

        job_output.mkdir(parents=True, exist_ok=True)
        stdout_path = job_output / "stdout.log"
        stderr_path = job_output / "stderr.log"
        env = dict(os.environ)
        for key in ("STANDARD_BENCH_ALLOW_PARTIAL", "STANDARD_BENCH_MAX_SAMPLES"):
            env.pop(key, None)
        python_path = [str(PACKAGE_ROOT.parent.parent)]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(python_path),
                "PYTHONUNBUFFERED": "1",
                "HISTOPATH_DATA_ROOT": str(Path(data_root).expanduser().resolve()),
                "STANDARD_BENCH_EPOCHS": "1",
                "STANDARD_BENCH_RESULT_DIR": str(job_output),
                "STANDARD_BENCH_VRAM_CAP_MIB": str(A10_VRAM_CAP_MIB),
            }
        )
        job_started = time.time()
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
                timed_out = False
            except subprocess.TimeoutExpired:
                returncode = -1
                timed_out = True
        result = _load_result(metric_path) or {}
        errors = _check_job_result(result) if result else ["metric.json was not produced"]
        if returncode != 0:
            errors.append(f"process return code {returncode}")
        if timed_out:
            errors.append("validation timeout")
        record = {
            **result,
            "job_id": job_id,
            "source_hash": expected_hash,
            "profile_bucket": metadata.get("profile_bucket"),
            "status": "PASSED" if not errors else "FAILED",
            "errors": errors,
            "resumed": False,
            "validation_wall_seconds": time.time() - job_started,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        metric_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append(record)
        print(f"[{position}/{len(selected_jobs)}] {job_id}: {record['status']}", flush=True)

    bucket_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("status") == "PASSED" and record.get("peak_allocated_mib") is not None:
            bucket_values[str(record.get("profile_bucket"))].append(float(record["peak_allocated_mib"]))
    bucket_spread: dict[str, dict[str, Any]] = {}
    for bucket, values in sorted(bucket_values.items()):
        low, high = min(values), max(values)
        spread = ((high / low) - 1.0) if low > 0 else (0.0 if high == 0 else math.inf)
        bucket_spread[bucket] = {
            "job_count": len(values),
            "minimum_peak_allocated_mib": low,
            "maximum_peak_allocated_mib": high,
            "spread_fraction": spread,
            "passed": len(values) == 5 and spread <= 0.15,
        }

    complete = len(selected_jobs) == JOB_COUNT
    failures = [record["job_id"] for record in records if record.get("status") != "PASSED"]
    bucket_failures = [bucket for bucket, item in bucket_spread.items() if not item["passed"]] if complete else []
    report = {
        "schema_version": "standard-histopath-validation-v1",
        "fixture": str(fixture_path),
        "dataset_root": str(Path(data_root).expanduser().resolve()),
        "physical_device": torch.cuda.get_device_name(0),
        "vram_cap_mib": A10_VRAM_CAP_MIB,
        "sequential": True,
        "requested_job_count": len(selected_jobs),
        "validated_job_count": len(records),
        "full_validation": complete,
        "passed_job_count": sum(record.get("status") == "PASSED" for record in records),
        "failed_job_ids": failures,
        "bucket_spread": bucket_spread,
        "failed_bucket_ids": bucket_failures,
        "elapsed_seconds": time.time() - started,
        "accepted": complete and not failures and not bucket_failures and len(bucket_spread) == 20,
        "records": records,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_ROOT))
    parser.add_argument("--data-root", default=os.environ.get("HISTOPATH_DATA_ROOT"), required=False)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("--data-root or HISTOPATH_DATA_ROOT is required")
    report = validate_fixture_models(
        fixture=args.fixture,
        data_root=args.data_root,
        output_root=args.output_root,
        resume=args.resume,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({key: report[key] for key in ("accepted", "validated_job_count", "passed_job_count", "elapsed_seconds")}, indent=2))
    return 0 if report["accepted"] or not report["full_validation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
