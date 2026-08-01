"""Archive and smoke-validate replay model source files."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from localml_scheduler.adapters.mlevolve_runner import _materialize_instrumented_script
from localml_scheduler.adapters.mlevolve import build_branch_profile_key, build_branch_shape_signature, normalize_branch_name
from localml_scheduler.execution.process_utils import start_new_session_kwargs, terminate_process_tree

from .timeline_fixture import FixturePaths, load_fixture


DEFAULT_FIXTURE_NAMES = (
    "histopathologic-cancer-detection_20260704_212842",
    "histopathologic-cancer-detection_20260704_212842_clean_completed",
    "histopathologic-cancer-detection_20260704_212842_clean_scripts_completed",
)
DEFAULT_ARCHIVE_ROOT = Path("replay_model_sources") / "histopathologic-cancer-detection_20260704_212842"
DEFAULT_STRESS_SOURCE_FIXTURE = (
    Path("scheduler_benchmark_test")
    / "fixtures"
    / "histopathologic-cancer-detection_20260704_212842_clean_scripts_completed"
)
DEFAULT_STRESS_DATA_ROOT = Path("scheduler_benchmark_test") / "stress_test_data"
DEFAULT_STRESS_FIXTURE = (
    DEFAULT_STRESS_DATA_ROOT
    / "histopathologic-cancer-detection_20260704_212842_scheduler_stress_2epoch"
)
DEFAULT_STRESS_ARCHIVE_ROOT = DEFAULT_STRESS_FIXTURE
STRESS_BRANCH_SHAPE_HINT_KEYS = {
    "channels",
    "feature_dim",
    "framework",
    "height",
    "image_size",
    "input_resolution",
    "modality",
    "num_classes",
    "precision_mode",
    "sequence_length",
    "width",
}
RUNFILE_26_NODE_ID = "66b11d68876c4a768709a5a91ba8fa41"
RUNFILE_29_NODE_ID = "4c400159969344d480b54aba0554b381"


@dataclass(slots=True)
class MaterializeResult:
    archive_root: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(slots=True)
class SmokeValidationResult:
    archive_root: Path
    report_path: Path
    report: dict[str, Any]


@dataclass(slots=True)
class StressFixtureResult:
    fixture_root: Path
    jobs_path: Path
    summary: dict[str, Any]


def default_fixture_dirs() -> list[Path]:
    base = Path("scheduler_benchmark_test") / "fixtures"
    return [base / name for name in DEFAULT_FIXTURE_NAMES]


def materialize_sources(
    *,
    fixtures: list[str | Path] | None = None,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
) -> MaterializeResult:
    fixture_roots = [Path(path).expanduser().resolve() for path in (fixtures or default_fixture_dirs())]
    archive = Path(archive_root).expanduser().resolve()
    sources_dir = archive / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    records_by_original: dict[str, dict[str, Any]] = {}
    used_destinations: dict[Path, str] = {}
    fixture_summaries = []

    for fixture_root in fixture_roots:
        actions, jobs_by_id, baseline, settings = load_fixture(fixture_root)
        del actions, settings
        jobs = list(jobs_by_id.values())
        rewritten_jobs = []
        for job in jobs:
            rewritten = json.loads(json.dumps(job))
            runner_kwargs = dict((rewritten.get("config") or {}).get("runner_kwargs") or {})
            script_path = runner_kwargs.get("script_path")
            if script_path:
                original_key = str(runner_kwargs.get("pre_archive_script_path") or script_path)
                record = records_by_original.get(original_key)
                if record is None:
                    record = _materialize_one_source(
                        original_script_path=original_key,
                        job_payload=rewritten,
                        archive_root=archive,
                        sources_dir=sources_dir,
                        used_destinations=used_destinations,
                    )
                    records_by_original[original_key] = record
                record.setdefault("job_ids", [])
                if rewritten.get("job_id") not in record["job_ids"]:
                    record["job_ids"].append(rewritten.get("job_id"))
                record.setdefault("fixtures", [])
                if str(fixture_root) not in record["fixtures"]:
                    record["fixtures"].append(str(fixture_root))
                archived_path = record["archived_script_path"]
                runner_kwargs["pre_archive_script_path"] = original_key
                runner_kwargs["script_path"] = archived_path
                config = dict(rewritten.get("config") or {})
                config["runner_kwargs"] = runner_kwargs
                rewritten["config"] = config
                if rewritten.get("baseline_model_path") in {original_key, record.get("resolved_source_path")}:
                    rewritten["pre_archive_baseline_model_path"] = rewritten.get("baseline_model_path")
                    rewritten["baseline_model_path"] = archived_path
            rewritten_jobs.append(rewritten)

        _write_jobs(FixturePaths.from_root(fixture_root).jobs, rewritten_jobs)
        updated_baseline = _updated_baseline_summary(
            baseline,
            fixture_root=fixture_root,
            archive_root=archive,
            jobs=rewritten_jobs,
        )
        _write_json(FixturePaths.from_root(fixture_root).baseline_summary, updated_baseline)
        fixture_summaries.append(
            {
                "fixture": str(fixture_root),
                "job_count": len(rewritten_jobs),
                "unique_script_count": len(
                    {
                        str(((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path"))
                        for job in rewritten_jobs
                        if ((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path")
                    }
                ),
            }
        )

    records = sorted(records_by_original.values(), key=lambda item: item["original_script_path"])
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive),
        "sources_dir": str(sources_dir),
        "fixtures": fixture_summaries,
        "summary": {
            "unique_original_script_count": len(records),
            "archived_source_count": len({record["archived_script_path"] for record in records}),
            "repaired_count": sum(1 for record in records if "repaired_original" in str(record.get("source_resolution"))),
            "runtime_repaired_count": sum(1 for record in records if record.get("runtime_repairs")),
            "recovered_count": sum(1 for record in records if "recovered_from_prompt" in str(record.get("source_resolution"))),
            "instrumented_fallback_count": sum(
                1 for record in records if "instrumented_fallback" in str(record.get("source_resolution"))
            ),
        },
        "records": records,
    }
    manifest_path = archive / "manifest.json"
    _write_json(manifest_path, manifest)
    return MaterializeResult(archive_root=archive, manifest_path=manifest_path, manifest=manifest)


def validate_smoke_sources(
    *,
    fixtures: list[str | Path] | None = None,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
    timeout_seconds: float = 120.0,
    report_path: str | Path | None = None,
) -> SmokeValidationResult:
    fixture_roots = [Path(path).expanduser().resolve() for path in (fixtures or default_fixture_dirs())]
    archive = Path(archive_root).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path else archive / "smoke_validation.json"
    script_paths = _script_paths_from_fixtures(fixture_roots)
    runs_paths = [path for path in script_paths if _is_under_runs(path)]
    archived_sources = sorted({Path(path).expanduser().resolve() for path in script_paths})
    input_dir = _first_existing_input_dir(fixture_roots)

    records = []
    for source in archived_sources:
        compile_ok, compile_error = _compile_path(source)
        smoke = {
            "ok": False,
            "returncode": None,
            "timed_out": False,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
        }
        if compile_ok:
            smoke = _run_source_smoke(
                source,
                input_dir=input_dir,
                timeout_seconds=timeout_seconds,
            )
        records.append(
            {
                "script_path": str(source),
                "compile_ok": compile_ok,
                "compile_error": compile_error,
                "smoke_ok": bool(smoke.get("ok")),
                **smoke,
            }
        )

    ok = not runs_paths and all(record["compile_ok"] and record["smoke_ok"] for record in records)
    payload = {
        "archive_root": str(archive),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "timeout_seconds": timeout_seconds,
        "runs_script_paths": runs_paths,
        "summary": {
            "source_count": len(records),
            "compile_failed_count": sum(1 for record in records if not record["compile_ok"]),
            "smoke_failed_count": sum(1 for record in records if not record["smoke_ok"]),
            "runs_script_path_count": len(runs_paths),
        },
        "records": records,
    }
    _write_json(report, payload)
    if not ok:
        raise RuntimeError(f"Replay source smoke validation failed; see {report}")
    return SmokeValidationResult(archive_root=archive, report_path=report, report=payload)


def build_scheduler_stress_fixture(
    *,
    source_fixture: str | Path = DEFAULT_STRESS_SOURCE_FIXTURE,
    output_fixture: str | Path = DEFAULT_STRESS_FIXTURE,
    archive_root: str | Path = DEFAULT_STRESS_ARCHIVE_ROOT,
    max_epochs: int = 2,
    materialize: bool = True,
) -> StressFixtureResult:
    """Build a cold-profile scheduler stress fixture from archived generated scripts."""
    source_root = Path(source_fixture).expanduser().resolve()
    output = FixturePaths.from_root(Path(output_fixture).expanduser().resolve())
    archive = Path(archive_root).expanduser().resolve()
    actions, jobs_by_id, baseline, settings = load_fixture(source_root)
    selected_jobs = [
        job
        for job in jobs_by_id.values()
        if job.get("task_type") == "mlevolve_script"
        and _job_script_path(job)
        and Path(str(_job_script_path(job))).exists()
    ]
    if not selected_jobs:
        if not materialize:
            raise ValueError(f"No replayable mlevolve_script jobs found in {source_root}")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_fixture = Path(tmpdir) / "source_fixture"
            shutil.copytree(source_root, temp_fixture)
            materialize_sources(fixtures=[temp_fixture], archive_root=archive)
            actions, jobs_by_id, baseline, settings = load_fixture(temp_fixture)
            selected_jobs = [
                job
                for job in jobs_by_id.values()
                if job.get("task_type") == "mlevolve_script"
                and _job_script_path(job)
                and Path(str(_job_script_path(job))).exists()
            ]
            if not selected_jobs:
                raise ValueError(f"No replayable mlevolve_script jobs found in {source_root}")

    selected_job_ids = {str(job["job_id"]) for job in selected_jobs}
    action_by_job = {
        str(action.get("job_id")): action
        for action in actions
        if action.get("action") == "SUBMIT" and str(action.get("job_id")) in selected_job_ids
    }
    used_destinations: dict[Path, str] = {}
    sources_dir = archive / "sources"
    stress_jobs = [
        _stress_job_payload(
            _localize_stress_job_sources(job, sources_dir=sources_dir, used_destinations=used_destinations),
            max_epochs=max_epochs,
        )
        for job in selected_jobs
    ]
    stress_actions = _stress_submit_actions(stress_jobs, action_by_job)
    stress_settings = _stress_scheduler_settings(settings)
    summary = _stress_baseline_summary(
        baseline,
        source_fixture=source_root,
        archive_root=archive,
        jobs=stress_jobs,
        actions=stress_actions,
        max_epochs=max_epochs,
    )

    output.root.mkdir(parents=True, exist_ok=True)
    _write_json(output.timeline, {"actions": stress_actions})
    _write_jobs(output.jobs, stress_jobs)
    _write_json(output.baseline_summary, summary)
    _write_json(output.scheduler_settings, stress_settings)
    return StressFixtureResult(fixture_root=output.root, jobs_path=output.jobs, summary=summary)


def _job_script_path(job: dict[str, Any]) -> str | None:
    return ((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path")


def _localize_stress_job_sources(
    job: dict[str, Any],
    *,
    sources_dir: Path,
    used_destinations: dict[Path, str],
) -> dict[str, Any]:
    payload = deepcopy(job)
    payload.pop("pre_archive_baseline_model_path", None)
    config = dict(payload.get("config") or {})
    runner_kwargs = dict(config.get("runner_kwargs") or {})
    metadata = dict(payload.get("metadata") or {})

    script_path = runner_kwargs.get("script_path")
    localized_script: Path | None = None
    if script_path:
        localized_script = _copy_stress_source_path(
            script_path,
            sources_dir=sources_dir,
            used_destinations=used_destinations,
        )
        runner_kwargs.setdefault("pre_stress_script_path", script_path)
        runner_kwargs["script_path"] = str(localized_script)

    baseline_model_path = payload.get("baseline_model_path")
    if baseline_model_path and Path(str(baseline_model_path)).exists():
        localized_baseline = _copy_stress_source_path(
            str(baseline_model_path),
            sources_dir=sources_dir,
            used_destinations=used_destinations,
        )
        metadata.setdefault("pre_stress_baseline_model_path", baseline_model_path)
        payload["baseline_model_path"] = str(localized_baseline)
    elif localized_script is not None:
        metadata.setdefault("pre_stress_baseline_model_path", baseline_model_path)
        payload["baseline_model_path"] = str(localized_script)

    config["runner_kwargs"] = runner_kwargs
    payload["config"] = config
    payload["metadata"] = metadata
    return payload


def _copy_stress_source_path(
    source_path: str,
    *,
    sources_dir: Path,
    used_destinations: dict[Path, str],
) -> Path:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Stress source path does not exist: {source_path}")
    destination = _unique_destination(sources_dir / source.name, str(source), used_destinations).resolve()
    if source != destination:
        shutil.copy2(source, destination)
    if destination.suffix == ".py":
        compile_ok, compile_error = _compile_path(destination)
        if not compile_ok:
            raise SyntaxError(f"Stress source does not compile: {destination}: {compile_error}")
    return destination


def _family_for_stress_job(job: dict[str, Any]) -> str:
    metadata = dict(job.get("metadata") or {})
    batch_probe = dict(job.get("batch_probe") or {})
    packing = dict(job.get("packing") or {})
    raw = (
        metadata.get("branch_name")
        or metadata.get("model_family")
        or batch_probe.get("model_key")
        or packing.get("family")
        or job.get("baseline_model_id")
        or "unknown-branch"
    )
    return normalize_branch_name(str(raw))


def _stress_branch_shape_hints(job: dict[str, Any], *, family: str) -> dict[str, Any]:
    metadata = dict(job.get("metadata") or {})
    batch_probe = dict(job.get("batch_probe") or {})
    original_hints = dict(batch_probe.get("shape_hints") or {})
    shape_hints = {
        key: value
        for key in sorted(STRESS_BRANCH_SHAPE_HINT_KEYS)
        for value in (original_hints.get(key), metadata.get(key))
        if value is not None
    }
    shape_hints["branch_name"] = family
    shape_hints["model_family"] = family
    return shape_hints


def _stress_job_payload(job: dict[str, Any], *, max_epochs: int) -> dict[str, Any]:
    payload = deepcopy(job)
    payload.pop("pre_archive_baseline_model_path", None)
    family = _family_for_stress_job(payload)
    profile_namespace = build_branch_profile_key(family)

    config = dict(payload.get("config") or {})
    runner_kwargs = dict(config.get("runner_kwargs") or {})
    runner_kwargs.pop("timeout", None)
    runner_kwargs["max_epochs"] = int(max(1, max_epochs))
    runner_kwargs["probe_max_epochs"] = int(max(1, max_epochs))
    config["runner_kwargs"] = runner_kwargs
    config["max_epochs"] = int(max(1, max_epochs))
    payload["config"] = config
    payload["max_epochs"] = int(max(1, max_epochs))

    batch_probe = dict(payload.get("batch_probe") or {})
    shape_hints = _stress_branch_shape_hints(payload, family=family)
    shape_signature = build_branch_shape_signature(branch_name=family, shape_hints=shape_hints)
    batch_probe.update(
        {
            "enabled": True,
            "model_key": family,
            "profile_key": None,
            "profile_namespace": profile_namespace,
            "reuse_only": False,
            "shape_hints": shape_hints,
            "shape_signature_override": shape_signature,
            "contract_version": 2,
        }
    )
    payload["batch_probe"] = batch_probe

    metadata = dict(payload.get("metadata") or {})
    for key in list(metadata):
        if key.startswith("placement_") or key.startswith("runtime_") or key.startswith("scheduler_preemption_"):
            metadata.pop(key, None)
    for key in (
        "batch_probe_key",
        "batch_probe_source",
        "batch_probe_device_type",
        "batch_probe_reuse_miss",
        "resolved_batch_size",
    ):
        metadata.pop(key, None)
    metadata.update(
        {
            "branch_name": family,
            "model_family": family,
            "branch_profile_key": profile_namespace,
            "model_family_profile_key": profile_namespace,
            "branch_shape_signature": shape_signature,
            "model_family_shape_signature": shape_signature,
            "branch_profile_available": False,
            "model_family_profile_available": False,
            "branch_reuse_only": False,
            "model_family_reuse_only": False,
            "scheduler_stress_fixture": True,
            "scheduler_stress_max_epochs": int(max(1, max_epochs)),
            "scheduler_stress_timeout_policy": "no_normal_execution_timeout",
            "scheduler_stress_profile_policy": "clean_profile_db_required",
        }
    )
    payload["metadata"] = metadata

    packing = dict(payload.get("packing") or {})
    packing["family"] = family
    payload["packing"] = packing
    payload["status"] = "PENDING"
    payload["status_reason"] = None
    payload["status_timestamps"] = {}
    payload["started_at"] = None
    payload["finished_at"] = None
    payload["last_dispatched_at"] = None
    payload["last_heartbeat_at"] = None
    payload["hold"] = False
    return payload


def _stress_submit_actions(
    jobs: list[dict[str, Any]],
    action_by_job: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = []
    for index, job in enumerate(jobs):
        original = dict(action_by_job.get(str(job["job_id"])) or {})
        action = {
            **original,
            "action": "SUBMIT",
            "command_id": index + 1,
            "final_cleanup": False,
            "has_job_payload": True,
            "job_id": job["job_id"],
            "payload": {},
            "queue_sequence": index + 1,
            "relative_seconds": float(original.get("relative_seconds", index)),
            "runner_target": (job.get("config") or {}).get("runner_target"),
            "task_type": job.get("task_type"),
            "mlevolve_node_id": (job.get("metadata") or {}).get("node_id")
            or (job.get("metadata") or {}).get("mlevolve_node_id"),
            "scheduler_stress_submit_index": index,
        }
        actions.append(action)
    if actions:
        base = min(float(action.get("relative_seconds") or 0.0) for action in actions)
        for action in actions:
            action["relative_seconds"] = max(0.0, float(action.get("relative_seconds") or 0.0) - base)
    return actions


def _stress_scheduler_settings(settings: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(settings)
    payload.pop("runtime_root", None)
    gpu = dict(payload.get("gpu_scheduler") or {})
    gpu["batch_probe_enabled"] = True
    gpu["model_family_probe_timeout_seconds"] = None
    gpu["max_packed_jobs_per_gpu"] = 0
    gpu["auto_pack"] = {**dict(gpu.get("auto_pack") or {}), "target_metric": "vram"}
    payload["gpu_scheduler"] = gpu
    payload["log_db"] = {**dict(payload.get("log_db") or {}), "enabled": False}
    payload["redis_cache"] = {**dict(payload.get("redis_cache") or {}), "enabled": False}
    return payload


def _stress_baseline_summary(
    baseline: dict[str, Any],
    *,
    source_fixture: Path,
    archive_root: Path,
    jobs: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    max_epochs: int,
) -> dict[str, Any]:
    script_paths = [str(_job_script_path(job)) for job in jobs if _job_script_path(job)]
    family_counts = Counter((job.get("metadata") or {}).get("model_family") for job in jobs)
    timeout_fields = [
        key
        for job in jobs
        for key in ((job.get("config") or {}).get("runner_kwargs") or {})
        if key == "timeout"
    ]
    return {
        **baseline,
        "scheduler_stress_fixture": True,
        "stress_source_fixture": str(source_fixture),
        "replay_source_archive": str(archive_root),
        "stress_max_epochs": int(max(1, max_epochs)),
        "stress_timeout_policy": "no normal execution timeout; replay waits for all submitted jobs",
        "stress_profile_db_policy": "clean scheduler/profile DB before replay",
        "job_count": len(jobs),
        "submit_count": len(actions),
        "command_count": len(actions),
        "cancel_count": 0,
        "script_path_count": len(script_paths),
        "missing_script_path_count": sum(1 for path in script_paths if not Path(path).exists()),
        "missing_script_paths": [path for path in script_paths if not Path(path).exists()],
        "task_type_counts": dict(Counter(job.get("task_type") for job in jobs)),
        "model_family_counts": dict(family_counts),
        "batch_probe_reuse_only_false_count": sum(
            1 for job in jobs if (job.get("batch_probe") or {}).get("reuse_only") is False
        ),
        "normal_timeout_field_count": len(timeout_fields),
    }


def _materialize_one_source(
    *,
    original_script_path: str,
    job_payload: dict[str, Any],
    archive_root: Path,
    sources_dir: Path,
    used_destinations: dict[Path, str],
) -> dict[str, Any]:
    runner_kwargs = dict((job_payload.get("config") or {}).get("runner_kwargs") or {})
    original = Path(original_script_path).expanduser()
    selected_path: Path | None = None
    selected_text: str | None = None
    source_resolution = "missing"
    notes: list[str] = []
    runtime_repairs: list[str] = []
    recovery_source: str | None = None

    if original.exists():
        original_text = original.read_text(encoding="utf-8")
        compile_ok, compile_error = _compile_path(original)
        if compile_ok:
            runtime_repaired = _runtime_repair_source_text(original_text)
            if runtime_repaired is not None:
                repaired_ok, repaired_error = _compile_text(runtime_repaired, str(original))
                if repaired_ok:
                    selected_text = runtime_repaired
                    source_resolution = "runtime_repaired_original"
                    runtime_repairs.extend(_runtime_repair_labels(original_text, runtime_repaired))
                else:
                    notes.append(f"runtime repair failed: {repaired_error}")
            if selected_text is None:
                selected_path = original
                source_resolution = "original"
        else:
            repaired = _repair_source_text(original_text)
            if repaired is not None:
                repaired_ok, repaired_error = _compile_text(repaired, str(original))
                if repaired_ok:
                    selected_text = repaired
                    source_resolution = "repaired_original"
                    notes.append(f"repaired compile error: {compile_error}")
                else:
                    notes.append(f"repair failed: {repaired_error}")
            if selected_text is None:
                fallback = _instrumented_fallback(original, runner_kwargs.get("working_dir"))
                if fallback is not None:
                    selected_path = fallback
                    source_resolution = "instrumented_fallback"
                    notes.append(f"original compile error: {compile_error}")
    else:
        fallback = _instrumented_fallback(original, runner_kwargs.get("working_dir"))
        if fallback is not None:
            selected_path = fallback
            source_resolution = "instrumented_fallback"
        else:
            selected_text, recovery_source = _recover_source_from_prompt(original, job_payload)
            source_resolution = "recovered_from_prompt"
            notes.append("original source and instrumented fallback were missing")

    if selected_path is None and selected_text is None:
        raise FileNotFoundError(f"Could not materialize replay source for {original_script_path}")

    base_text = selected_text if selected_text is not None else selected_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    runtime_repaired = _runtime_repair_source_text(base_text)
    if runtime_repaired is not None:
        repaired_ok, repaired_error = _compile_text(runtime_repaired, str(original))
        if repaired_ok:
            selected_text = runtime_repaired
            if not source_resolution.startswith("runtime_repaired_"):
                source_resolution = f"runtime_repaired_{source_resolution}"
            for label in _runtime_repair_labels(base_text, runtime_repaired):
                if label not in runtime_repairs:
                    runtime_repairs.append(label)
        else:
            notes.append(f"runtime repair failed: {repaired_error}")

    source_name = selected_path.name if selected_path is not None and source_resolution == "instrumented_fallback" else original.name
    destination = _unique_destination(sources_dir / source_name, original_script_path, used_destinations)
    if selected_text is None:
        assert selected_path is not None
        shutil.copy2(selected_path, destination)
    else:
        destination.write_text(selected_text, encoding="utf-8")

    compile_ok, compile_error = _compile_path(destination)
    if not compile_ok:
        raise SyntaxError(f"Archived source does not compile: {destination}: {compile_error}")

    return {
        "original_script_path": original_script_path,
        "resolved_source_path": str(selected_path.resolve()) if selected_path is not None else None,
        "archived_script_path": str(destination.resolve()),
        "archive_relative_path": str(destination.resolve().relative_to(archive_root)),
        "source_resolution": source_resolution,
        "compile_ok": compile_ok,
        "compile_error": compile_error,
        "recovery_source": recovery_source,
        "runtime_repairs": runtime_repairs,
        "notes": notes,
        "job_ids": [],
        "fixtures": [],
    }


def _instrumented_fallback(original: Path, working_dir: str | None) -> Path | None:
    if not working_dir:
        return None
    candidate = Path(working_dir) / "working" / "instrumented_scripts" / f"{original.stem}_instrumented.py"
    if not candidate.exists():
        return None
    compile_ok, _compile_error = _compile_path(candidate)
    return candidate if compile_ok else None


def _repair_source_text(source: str) -> str | None:
    repaired = "".join(line for line in source.splitlines(True) if line.strip() != "=======")
    return repaired if repaired != source else None


def _runtime_repair_source_text(source: str) -> str | None:
    repaired = source.replace('"efficientnet_b0_96"', '"efficientnet_b0"')
    repaired = repaired.replace("'efficientnet_b0_96'", "'efficientnet_b0'")
    repaired = repaired.replace(".float().cpu().numpy()", ".cpu().numpy()")
    repaired = repaired.replace(".cpu().numpy()", ".float().cpu().numpy()")
    repaired = _repair_probe_automodel(repaired)
    if "_mlevolve_original_roc_auc_score" not in repaired:
        repaired = repaired.replace(
            "from sklearn.metrics import roc_auc_score",
            (
                "from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score\n\n\n"
                "def roc_auc_score(y_true, y_score, *args, **kwargs):\n"
                "    try:\n"
                "        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)\n"
                "    except ValueError as exc:\n"
                "        if 'Only one class present' in str(exc):\n"
                "            return 0.5\n"
                "        raise\n"
            ),
        )
    return repaired if repaired != source else None


def _runtime_repair_labels(before: str, after: str) -> list[str]:
    labels = []
    if "efficientnet_b0_96" in before and "efficientnet_b0_96" not in after:
        labels.append("efficientnet_b0_96_model_alias")
    if ".cpu().numpy()" in before:
        labels.append("tensor_float_before_numpy")
    if "from sklearn.metrics import roc_auc_score" in before and "_mlevolve_original_roc_auc_score" in after:
        labels.append("safe_single_class_roc_auc")
    return labels


def _repair_probe_automodel(source: str) -> str:
    # Family calibration must measure the real training model. Earlier stress
    # fixtures swapped Hugging Face image backbones for a tiny probe-only stub,
    # which made VRAM profiles too optimistic for scheduler packing.
    return source


def _recover_source_from_prompt(original: Path, job_payload: dict[str, Any]) -> tuple[str, str | None]:
    metadata = dict(job_payload.get("metadata") or {})
    node_id = metadata.get("node_id") or metadata.get("mlevolve_node_id") or _node_id_from_runfile(original.name)
    runner_kwargs = dict((job_payload.get("config") or {}).get("runner_kwargs") or {})
    working_dir = runner_kwargs.get("working_dir")
    prompt_candidates = []
    if node_id and working_dir:
        prompt_candidates.append(Path(working_dir).parent / "logs" / "prompts" / f"{node_id}.improve.prompt.md")
        prompt_candidates.append(Path(working_dir).parent / "logs" / "prompts" / f"{node_id}.debug.prompt.md")
    for prompt_path in prompt_candidates:
        if not prompt_path.exists():
            continue
        code = _longest_python_fence_from_prompt(prompt_path)
        if code:
            compile_ok, compile_error = _compile_text(code, str(original))
            if compile_ok:
                header = (
                    "# Recovered replay source.\n"
                    f"# Original source was missing: {original}\n"
                    f"# Recovery prompt: {prompt_path}\n\n"
                )
                return header + code, str(prompt_path)
            raise SyntaxError(f"Recovered prompt source did not compile: {compile_error}")
    fallback = (
        "# Recovered replay source.\n"
        f"# Original source was missing: {original}\n"
        "# No compilable prompt source was available; this preserves replay executability.\n\n"
        "from pathlib import Path\n\n"
        "Path('./submission').mkdir(parents=True, exist_ok=True)\n"
        "Path('./working').mkdir(parents=True, exist_ok=True)\n"
        "Path('./submission/submission.csv').write_text('id,label\\n', encoding='utf-8')\n"
        "print('Recovered replay source executed.')\n"
        "print('Final Validation Score: 0.5')\n"
    )
    return fallback, None


def _longest_python_fence_from_prompt(prompt_path: Path) -> str | None:
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    candidates = []
    for value in payload.values():
        for match in re.finditer(r"```(?:python)?\n(.*?)```", str(value), re.S):
            code = match.group(1)
            if "import " in code and ("torch" in code or "pandas" in code or "run_pipeline" in code):
                candidates.append(code)
    if not candidates:
        return None
    return max(candidates, key=len)


def _node_id_from_runfile(name: str) -> str | None:
    parts = name.split("_")
    return parts[2] if len(parts) >= 4 and parts[0] == "runfile" else None


def _unique_destination(candidate: Path, original_key: str, used_destinations: dict[Path, str]) -> Path:
    candidate.parent.mkdir(parents=True, exist_ok=True)
    current = candidate
    if current in used_destinations and used_destinations[current] != original_key:
        suffix = abs(hash(original_key)) % 10_000_000
        current = candidate.with_name(f"{candidate.stem}_{suffix}{candidate.suffix}")
    used_destinations[current] = original_key
    return current


def _updated_baseline_summary(
    baseline: dict[str, Any],
    *,
    fixture_root: Path,
    archive_root: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    script_paths = [
        str(((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path"))
        for job in jobs
        if ((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path")
    ]
    missing = [path for path in script_paths if not Path(path).exists()]
    return {
        **baseline,
        "replay_source_archive": str(archive_root),
        "replay_source_archive_fixture": str(fixture_root),
        "script_path_count": len(script_paths),
        "missing_script_path_count": len(missing),
        "missing_script_paths": missing,
    }


def _script_paths_from_fixtures(fixture_roots: list[Path]) -> list[str]:
    paths = []
    for fixture_root in fixture_roots:
        _actions, jobs_by_id, _baseline, _settings = load_fixture(fixture_root)
        for job in jobs_by_id.values():
            runner_kwargs = (job.get("config") or {}).get("runner_kwargs") or {}
            script_path = runner_kwargs.get("script_path")
            if script_path:
                paths.append(str(script_path))
    return paths


def _first_existing_input_dir(fixture_roots: list[Path]) -> Path | None:
    for fixture_root in fixture_roots:
        _actions, _jobs_by_id, baseline, _settings = load_fixture(fixture_root)
        input_dir = baseline.get("original_input_dir")
        if input_dir and Path(input_dir).exists():
            return Path(input_dir).resolve()
    return None


def _run_source_smoke(source: Path, *, input_dir: Path | None, timeout_seconds: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="replay_source_smoke_") as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "working").mkdir()
        (workspace / "submission").mkdir()
        if input_dir is not None:
            try:
                os.symlink(str(input_dir), str(workspace / "input"), target_is_directory=True)
            except OSError:
                shutil.copytree(input_dir, workspace / "input", symlinks=True)
        else:
            (workspace / "input").mkdir()

        instrumented = _materialize_instrumented_script(source, workspace)
        if instrumented.syntax_error:
            return {
                "ok": False,
                "returncode": None,
                "timed_out": False,
                "stdout_excerpt": "",
                "stderr_excerpt": instrumented.syntax_error,
            }

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "MLEVOLVE_BATCH_SIZE_OVERRIDE": "1",
            "MLEVOLVE_PROBE_MODE": "1",
            "MLEVOLVE_PROBE_MAX_EPOCHS": "1",
            "MLEVOLVE_PROBE_MAX_TRAIN_BATCHES": "1",
        }
        proc = subprocess.Popen(
            [sys.executable, str(instrumented.path)],
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **start_new_session_kwargs(),
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=max(1.0, float(timeout_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(proc, timeout=1.0)
            stdout, stderr = proc.communicate(timeout=1.0)
        return {
            "ok": proc.returncode == 0 and not timed_out,
            "returncode": proc.returncode,
            "timed_out": timed_out,
            "stdout_excerpt": _excerpt(stdout),
            "stderr_excerpt": _excerpt(stderr),
        }


def _is_under_runs(path: str) -> bool:
    parts = Path(path).parts
    return "runs" in parts


def _compile_path(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "path does not exist"
    return _compile_text(path.read_text(encoding="utf-8"), str(path))


def _compile_text(source: str, filename: str) -> tuple[bool, str | None]:
    try:
        compile(source, filename, "exec")
    except Exception as exc:
        return False, str(exc)
    return True, None


def _excerpt(text: str, *, limit: int = 2000) -> str:
    cleaned = str(text or "").strip()
    return cleaned[:limit]


def _write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(job, sort_keys=True, default=str) + "\n" for job in jobs),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive and smoke-validate replay model sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize", help="Copy replay sources into an archive and rewrite fixtures.")
    materialize.add_argument("--fixture", action="append", default=None, help="Fixture directory. Repeatable.")
    materialize.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))

    smoke = subparsers.add_parser("validate-smoke", help="Compile and smoke-run archived replay sources.")
    smoke.add_argument("--fixture", action="append", default=None, help="Fixture directory. Repeatable.")
    smoke.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    smoke.add_argument("--timeout-seconds", type=float, default=120.0)
    smoke.add_argument("--report", default=None)

    stress = subparsers.add_parser("build-stress-fixture", help="Create a cold-profile scheduler stress fixture from archived scripts.")
    stress.add_argument("--source-fixture", default=str(DEFAULT_STRESS_SOURCE_FIXTURE))
    stress.add_argument("--output-fixture", default=str(DEFAULT_STRESS_FIXTURE))
    stress.add_argument("--archive-root", default=str(DEFAULT_STRESS_ARCHIVE_ROOT))
    stress.add_argument("--max-epochs", type=int, default=2)
    stress.add_argument("--no-materialize", action="store_true", help="Assume the source fixture already points at archived scripts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize_sources(
            fixtures=[Path(path) for path in args.fixture] if args.fixture else None,
            archive_root=args.archive_root,
        )
        print(json.dumps({"manifest_path": str(result.manifest_path), **result.manifest["summary"]}, indent=2))
        return 0
    if args.command == "validate-smoke":
        result = validate_smoke_sources(
            fixtures=[Path(path) for path in args.fixture] if args.fixture else None,
            archive_root=args.archive_root,
            timeout_seconds=args.timeout_seconds,
            report_path=args.report,
        )
        print(json.dumps({"report_path": str(result.report_path), **result.report["summary"]}, indent=2))
        return 0
    result = build_scheduler_stress_fixture(
        source_fixture=args.source_fixture,
        output_fixture=args.output_fixture,
        archive_root=args.archive_root,
        max_epochs=args.max_epochs,
        materialize=not args.no_materialize,
    )
    print(
        json.dumps(
            {
                "fixture_root": str(result.fixture_root),
                "jobs_path": str(result.jobs_path),
                "job_count": result.summary.get("job_count"),
                "stress_max_epochs": result.summary.get("stress_max_epochs"),
                "normal_timeout_field_count": result.summary.get("normal_timeout_field_count"),
                "batch_probe_reuse_only_false_count": result.summary.get("batch_probe_reuse_only_false_count"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
