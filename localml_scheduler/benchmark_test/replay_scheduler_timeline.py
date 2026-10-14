"""Replay a persisted MLEvolve scheduler command timeline."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
import argparse
import csv
import json
import os
import shutil
import sys
import time

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import JobStatus, TrainingJob, utc_now

from .timeline_fixture import load_fixture, reset_job_payload_for_replay

NOOP_RUNNER_TARGET = "scheduler_benchmark_test.noop_runner:run_noop_job"
NOOP_LOADER_TARGET = "scheduler_benchmark_test.noop_runner:load_noop_baseline"
DEFAULT_MODE = "scheduler_replay"
CANCEL_POLICIES = {"replay", "ignore"}


@dataclass(slots=True)
class ReplayResult:
    output_root: Path
    runtime_root: Path
    log_dir: Path
    summary_path: Path
    metrics_path: Path
    submitted_job_ids: list[str]
    skipped_actions: list[dict[str, Any]]


def _deep_merge_settings(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_settings(base[key], value)
        else:
            base[key] = value


def replay_fixture(
    *,
    fixture: str | Path,
    output_root: str | Path,
    runner_mode: str = "real",
    speedup: float = 1.0,
    until_seconds: float | None = None,
    include_final_cleanup_cancels: bool = False,
    dry_run: bool = False,
    post_actions_wait_seconds: float = 60.0,
    allow_missing_scripts: bool = False,
    use_instrumented_fallback: bool = True,
    strict_missing_jobs: bool = False,
    no_sleep: bool = False,
    wait_for_all: bool = False,
    cancel_policy: str = "replay",
    clean_scheduler_state: bool = False,
    settings_overlay: str | Path | None = None,
) -> ReplayResult:
    if runner_mode not in {"real", "noop"}:
        raise ValueError(f"Unsupported runner mode: {runner_mode}")
    if speedup <= 0:
        raise ValueError("--speedup must be greater than 0")
    if cancel_policy not in CANCEL_POLICIES:
        raise ValueError(f"Unsupported cancel policy: {cancel_policy}")

    output = Path(output_root).expanduser().resolve()
    runtime_root = output / "scheduler_runtime"
    workspace = output / "workspace"
    log_dir = output / "logs"
    mirror_log_dir = output / DEFAULT_MODE / "runs" / _run_id(runner_mode) / "logs"
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    actions, jobs_by_id, baseline, settings_payload = load_fixture(fixture)
    selected_actions = _select_actions(
        actions,
        until_seconds=until_seconds,
        include_final_cleanup_cancels=include_final_cleanup_cancels,
        cancel_policy=cancel_policy,
    )
    _prepare_workspace(workspace, baseline)
    if clean_scheduler_state and not dry_run:
        _clean_scheduler_state(runtime_root)

    skipped_actions: list[dict[str, Any]] = []
    submitted_job_ids: list[str] = []
    settings_payload = deepcopy(settings_payload)
    settings_payload["runtime_root"] = str(runtime_root)
    if settings_overlay:
        overlay_payload = json.loads(Path(settings_overlay).expanduser().read_text())
        _deep_merge_settings(settings_payload, overlay_payload)
    if dry_run:
        for action in selected_actions:
            if action["action"] == "SUBMIT" and action["job_id"] not in jobs_by_id:
                skipped_actions.append({**action, "skip_reason": "missing job payload"})
        metrics = _build_metrics(
            client=None,
            service=None,
            started_at=time.time(),
            finished_at=time.time(),
            output_root=output,
            baseline=baseline,
            runner_mode=runner_mode,
            selected_actions=selected_actions,
            submitted_job_ids=[],
            skipped_actions=skipped_actions,
            dry_run=True,
            wait_for_all=wait_for_all,
            cancel_policy=cancel_policy,
            clean_scheduler_state=clean_scheduler_state,
        )
        return _write_outputs(
            output,
            log_dir,
            mirror_log_dir,
            runtime_root,
            metrics,
            [],
            skipped_actions,
            submitted_job_ids=[],
        )

    # Historical fixtures remain reportable in dry-run mode. Executing one
    # requires a current settings overlay so retired backends cannot silently
    # become a different process topology.
    settings = SchedulerSettings.from_dict(settings_payload)
    client = SchedulerClient(settings)
    service = None
    started_at = time.time()
    previous_relative = 0.0
    try:
        service = client.create_service().start(background=True)
        for action in selected_actions:
            relative = float(action["relative_seconds"])
            if not no_sleep:
                sleep_seconds = max(0.0, relative - previous_relative) / speedup
                if sleep_seconds:
                    time.sleep(sleep_seconds)
            previous_relative = relative

            if action["action"] == "SUBMIT":
                payload = jobs_by_id.get(action["job_id"])
                if payload is None:
                    skipped = {**action, "skip_reason": "missing job payload"}
                    skipped_actions.append(skipped)
                    if strict_missing_jobs:
                        raise KeyError(
                            f"Missing job payload for submitted job {action['job_id']}"
                        )
                    continue
                try:
                    job = _prepare_job(
                        payload,
                        output_workspace=workspace,
                        runner_mode=runner_mode,
                        allow_missing_scripts=allow_missing_scripts,
                        use_instrumented_fallback=use_instrumented_fallback,
                        fixture_sources=Path(fixture).expanduser().resolve()
                        / "sources",
                    )
                except FileNotFoundError as exc:
                    skipped = {**action, "skip_reason": str(exc)}
                    skipped_actions.append(skipped)
                    if strict_missing_jobs:
                        raise
                    continue
                client.submit(job)
                submitted_job_ids.append(job.job_id)
            elif action["action"] == "CANCEL":
                client.cancel(action["job_id"])

        if wait_for_all:
            _wait_for_terminal(client, submitted_job_ids, timeout_seconds=None)
        elif post_actions_wait_seconds > 0:
            _wait_for_terminal(
                client, submitted_job_ids, timeout_seconds=post_actions_wait_seconds
            )
        if not wait_for_all:
            outstanding = _nonterminal_job_ids(client, submitted_job_ids)
            if outstanding:
                for job_id in outstanding:
                    client.cancel(job_id)
                _wait_for_terminal(
                    client,
                    outstanding,
                    timeout_seconds=min(10.0, post_actions_wait_seconds),
                )
    finally:
        if service is not None:
            service.stop()

    finished_at = time.time()
    metrics = _build_metrics(
        client=client,
        service=service,
        started_at=started_at,
        finished_at=finished_at,
        output_root=output,
        baseline=baseline,
        runner_mode=runner_mode,
        selected_actions=selected_actions,
        submitted_job_ids=submitted_job_ids,
        skipped_actions=skipped_actions,
        dry_run=False,
        wait_for_all=wait_for_all,
        cancel_policy=cancel_policy,
        clean_scheduler_state=clean_scheduler_state,
    )
    samples = (
        list(getattr(service, "_device_samples", []) or [])
        if service is not None
        else []
    )
    return _write_outputs(
        output,
        log_dir,
        mirror_log_dir,
        runtime_root,
        metrics,
        samples,
        skipped_actions,
        submitted_job_ids=submitted_job_ids,
    )


def _select_actions(
    actions: list[dict[str, Any]],
    *,
    until_seconds: float | None,
    include_final_cleanup_cancels: bool,
    cancel_policy: str = "replay",
) -> list[dict[str, Any]]:
    selected = []
    for action in actions:
        if cancel_policy == "ignore" and action.get("action") == "CANCEL":
            continue
        if action.get("final_cleanup") and not include_final_cleanup_cancels:
            continue
        if (
            until_seconds is not None
            and float(action["relative_seconds"]) > until_seconds
        ):
            continue
        selected.append(action)
    return selected


def _remap_to_fixture_sources(
    path_value: str | None, fixture_sources: Path | None
) -> str | None:
    if not path_value or fixture_sources is None:
        return path_value
    path = Path(path_value)
    if path.exists():
        return path_value
    candidate = fixture_sources / path.name
    if candidate.exists():
        return str(candidate)
    return path_value


def _prepare_job(
    payload: dict[str, Any],
    *,
    output_workspace: Path,
    runner_mode: str,
    allow_missing_scripts: bool,
    use_instrumented_fallback: bool,
    fixture_sources: Path | None = None,
) -> TrainingJob:
    job_payload = reset_job_payload_for_replay(payload)
    job_payload["submitted_at"] = utc_now()
    job_payload["status_timestamps"] = {}

    if job_payload.get("baseline_model_path"):
        job_payload["baseline_model_path"] = _remap_to_fixture_sources(
            str(job_payload["baseline_model_path"]), fixture_sources
        )

    config = dict(job_payload.get("config") or {})
    runner_kwargs = dict(config.get("runner_kwargs") or {})
    if runner_kwargs.get("script_path"):
        runner_kwargs["script_path"] = _remap_to_fixture_sources(
            str(runner_kwargs["script_path"]), fixture_sources
        )
    original_working_dir = runner_kwargs.get("working_dir")
    if original_working_dir:
        runner_kwargs["original_working_dir"] = original_working_dir
        runner_kwargs["working_dir"] = str(output_workspace)

    result_path = runner_kwargs.get("result_path")
    if result_path:
        runner_kwargs["original_result_path"] = result_path
        runner_kwargs["result_path"] = str(
            output_workspace / "working" / "scheduler_results" / Path(result_path).name
        )
    elif runner_mode == "noop":
        runner_kwargs["result_path"] = str(
            output_workspace
            / "working"
            / "scheduler_results"
            / f"result_{job_payload['job_id']}.json"
        )

    script_path = runner_kwargs.get("script_path")
    if script_path and runner_mode == "real":
        resolved_script = _resolve_script_path(
            script_path,
            original_working_dir=original_working_dir,
            allow_missing=allow_missing_scripts,
            use_instrumented_fallback=use_instrumented_fallback,
        )
        runner_kwargs["original_script_path"] = script_path
        runner_kwargs["script_path"] = str(resolved_script)
        if job_payload.get("baseline_model_path") == script_path:
            job_payload["baseline_model_path"] = str(resolved_script)

    if runner_mode == "noop":
        config["runner_target"] = NOOP_RUNNER_TARGET
        config["loader_target"] = NOOP_LOADER_TARGET
        config["python_executable"] = sys.executable
        runner_kwargs.setdefault("noop_sleep_seconds", 0.01)
        job_payload.setdefault("batch_probe", {})["enabled"] = False
        job_payload.setdefault("runtime_probe", {})["enabled"] = False
        job_payload.setdefault("resource_requirements", {})["requires_gpu"] = False

    metadata = dict(job_payload.get("metadata") or {})
    metadata["replay_runner_mode"] = runner_mode
    job_payload["metadata"] = metadata
    config["runner_kwargs"] = runner_kwargs
    job_payload["config"] = config
    return TrainingJob.from_dict(job_payload)


def _resolve_script_path(
    script_path: str,
    *,
    original_working_dir: str | None,
    allow_missing: bool,
    use_instrumented_fallback: bool,
) -> Path:
    path = Path(script_path)
    if path.exists():
        return path
    fallback = None
    if use_instrumented_fallback and original_working_dir:
        candidate = (
            Path(original_working_dir)
            / "working"
            / "instrumented_scripts"
            / f"{path.stem}_instrumented.py"
        )
        if candidate.exists():
            fallback = candidate
    if fallback is not None:
        return fallback
    if allow_missing:
        return path
    raise FileNotFoundError(f"Replay script path does not exist: {script_path}")


def _prepare_workspace(workspace: Path, baseline: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for dirname in ("working", "submission", "best_solution", "best_submission"):
        (workspace / dirname).mkdir(parents=True, exist_ok=True)
    original_input = baseline.get("original_input_dir")
    input_path = workspace / "input"
    if input_path.exists():
        return
    if original_input and Path(original_input).exists():
        try:
            os.symlink(
                str(Path(original_input).resolve()),
                str(input_path),
                target_is_directory=True,
            )
        except OSError:
            # Avoid copying a large dataset; real replay will fail clearly if the
            # platform disallows symlinks.
            pass


def _clean_scheduler_state(runtime_root: Path) -> None:
    db_dir = runtime_root / "db"
    if db_dir.exists():
        shutil.rmtree(db_dir)


def _wait_for_terminal(
    client: SchedulerClient, job_ids: list[str], *, timeout_seconds: float | None
) -> None:
    deadline = (
        None if timeout_seconds is None else time.time() + max(0.0, timeout_seconds)
    )
    while deadline is None or time.time() < deadline:
        if not _nonterminal_job_ids(client, job_ids):
            return
        time.sleep(0.2)


def _nonterminal_job_ids(client: SchedulerClient, job_ids: list[str]) -> list[str]:
    outstanding = []
    for job_id in job_ids:
        job = client.inspect(job_id)
        if job is not None and not job.status.is_terminal:
            outstanding.append(job_id)
    return outstanding


def _build_metrics(
    *,
    client: SchedulerClient | None,
    service: Any | None,
    started_at: float,
    finished_at: float,
    output_root: Path,
    baseline: dict[str, Any],
    runner_mode: str,
    selected_actions: list[dict[str, Any]],
    submitted_job_ids: list[str],
    skipped_actions: list[dict[str, Any]],
    dry_run: bool,
    wait_for_all: bool,
    cancel_policy: str,
    clean_scheduler_state: bool,
) -> dict[str, Any]:
    jobs = [job.to_dict() for job in client.list_jobs()] if client is not None else []
    events = client.list_events() if client is not None else []
    report = client.report() if client is not None else {}
    durations = [_job_duration_seconds(job) for job in jobs]
    durations = [value for value in durations if value is not None]
    intervals = [_job_interval(job) for job in jobs]
    intervals = [value for value in intervals if value is not None]
    makespan = _interval_makespan(intervals)
    queue_wait = _queue_wait_seconds(jobs)
    probe_time = _probe_time_seconds(events)
    hardware_summary = _hardware_summary(
        list(getattr(service, "_device_samples", []) or [])
        if service is not None
        else []
    )
    status_counts = Counter(str(job.get("status")) for job in jobs)
    task_counts = Counter(str(job.get("task_type")) for job in jobs)
    backend_distribution = _backend_distribution(jobs, events)
    candidate_total = sum(durations)
    total_wall = max(0.0, finished_at - started_at)

    metrics: dict[str, Any] = {
        "mode": DEFAULT_MODE,
        "experiment_mode": "scheduler_timeline_replay",
        "command_label": DEFAULT_MODE,
        "run_id": output_root.name,
        "exp_id": "histopathologic-cancer-detection",
        "configured_scheduler_enabled": True,
        "scheduler_client_attached": client is not None,
        "scheduler_runtime_root": str(output_root / "scheduler_runtime"),
        "replay_runner_mode": runner_mode,
        "replay_dry_run": dry_run,
        "replay_wait_for_all": wait_for_all,
        "replay_cancel_policy": cancel_policy,
        "replay_clean_scheduler_state": clean_scheduler_state,
        "replay_action_count": len(selected_actions),
        "replay_submit_action_count": sum(
            1 for action in selected_actions if action["action"] == "SUBMIT"
        ),
        "replay_cancel_action_count": sum(
            1 for action in selected_actions if action["action"] == "CANCEL"
        ),
        "replay_skipped_action_count": len(skipped_actions),
        "total_run_wall_time_seconds": total_wall,
        "total_wall_time_seconds": total_wall,
        "total_candidate_execution_time_seconds": candidate_total,
        "total_job_execution_time_seconds": candidate_total,
        "execution_time_seconds": candidate_total,
        "median_candidate_execution_time_seconds": (
            median(durations) if durations else None
        ),
        "median_job_execution_time_seconds": median(durations) if durations else None,
        "candidate_execution_makespan_seconds": makespan,
        "candidate_execution_parallelism_ratio": (
            (candidate_total / makespan) if makespan and makespan > 0 else None
        ),
        "non_candidate_overhead_wall_time_seconds": (
            max(0.0, total_wall - makespan) if makespan is not None else None
        ),
        "total_scheduler_queue_wait_seconds": queue_wait,
        "queue_wait_seconds": queue_wait,
        "total_scheduler_probe_time_seconds": probe_time,
        "probe_time_seconds": probe_time,
        "packed_dispatch_count": sum(
            1
            for event in events
            if event.get("event_type")
            in {"packed_pair_dispatched", "packed_group_dispatched"}
        ),
        "packed_fallback_count": sum(
            1
            for event in events
            if event.get("event_type")
            in {"packed_pair_fallback", "packed_group_fallback"}
        ),
        "batch_probe_hit_count": sum(
            1 for event in events if event.get("event_type") == "batch_probe_cache_hit"
        ),
        "batch_probe_trial_count": sum(
            1 for event in events if event.get("event_type") == "batch_probe_trial"
        ),
        "concurrent_gpu_active_seconds": _concurrent_active_seconds(jobs),
        "node_count": int(task_counts.get("mlevolve_script", 0)),
        "scheduler_job_count": len(jobs),
        "submitted_job_count": len(submitted_job_ids),
        "completed_job_count": int(status_counts.get("COMPLETED", 0)),
        "failed_job_count": int(status_counts.get("FAILED", 0)),
        "cancelled_job_count": int(status_counts.get("CANCELLED", 0)),
        "job_status_counts": dict(status_counts),
        "task_type_counts": dict(task_counts),
        "scheduler_backend_distribution": dict(backend_distribution),
        "scheduler_report": report,
        "baseline_source_run_root": baseline.get("source_run_root"),
        **hardware_summary,
    }
    metrics["deltas_vs_original_scheduler_on"] = _metric_deltas(
        baseline.get("reference_metrics") or {},
        metrics,
    )
    return metrics


def _write_outputs(
    output: Path,
    log_dir: Path,
    mirror_log_dir: Path,
    runtime_root: Path,
    metrics: dict[str, Any],
    samples: list[Any],
    skipped_actions: list[dict[str, Any]],
    submitted_job_ids: list[str],
) -> ReplayResult:
    metrics_path = log_dir / "comparison_metrics.json"
    summary_path = output / "replay_summary.json"
    hardware_path = log_dir / "hardware_samples.csv"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output_root": str(output),
        "runtime_root": str(runtime_root),
        "metrics_path": str(metrics_path),
        "hardware_samples_path": str(hardware_path) if samples else None,
        "skipped_actions": skipped_actions,
        "metrics": metrics,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if samples:
        _write_hardware_samples(hardware_path, samples)

    mirror_log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, mirror_log_dir / "comparison_metrics.json")
    if samples and hardware_path.exists():
        shutil.copy2(hardware_path, mirror_log_dir / "hardware_samples.csv")

    return ReplayResult(
        output_root=output,
        runtime_root=runtime_root,
        log_dir=log_dir,
        summary_path=summary_path,
        metrics_path=metrics_path,
        submitted_job_ids=submitted_job_ids,
        skipped_actions=skipped_actions,
    )


def _write_hardware_samples(path: Path, samples: list[Any]) -> None:
    first = _parse_time(getattr(samples[0], "captured_at", None)) if samples else None
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "elapsed_seconds",
                "cpu_percent_avg",
                "cpu_percent_max",
                "ram_percent",
                "gpu_memory_used_mb",
                "gpu_memory_percent",
                "gpu_util_percent",
                "gpu_memory_util_percent",
                "gpu_power_draw_w",
                "sample_count",
            ],
        )
        writer.writeheader()
        for sample in samples:
            captured = _parse_time(getattr(sample, "captured_at", None))
            elapsed = (captured - first).total_seconds() if captured and first else 0.0
            memory_total = float(getattr(sample, "memory_total_mb", 0) or 0)
            memory_used = float(getattr(sample, "memory_used_mb", 0) or 0)
            writer.writerow(
                {
                    "elapsed_seconds": elapsed,
                    "cpu_percent_avg": "",
                    "cpu_percent_max": "",
                    "ram_percent": "",
                    "gpu_memory_used_mb": memory_used,
                    "gpu_memory_percent": (
                        (memory_used / memory_total * 100.0) if memory_total > 0 else ""
                    ),
                    "gpu_util_percent": float(
                        getattr(sample, "gpu_utilization", 0.0) or 0.0
                    )
                    * 100.0,
                    "gpu_memory_util_percent": float(
                        getattr(sample, "memory_utilization", 0.0) or 0.0
                    )
                    * 100.0,
                    "gpu_power_draw_w": "",
                    "sample_count": 1,
                }
            )


def _job_duration_seconds(job: dict[str, Any]) -> float | None:
    interval = _job_interval(job)
    if interval is None:
        return None
    started, finished = interval
    return max(0.0, finished - started)


def _job_interval(job: dict[str, Any]) -> tuple[float, float] | None:
    started = _timestamp_seconds(
        job.get("started_at") or (job.get("status_timestamps") or {}).get("RUNNING")
    )
    finished = _timestamp_seconds(job.get("finished_at"))
    if started is None or finished is None or finished < started:
        return None
    return started, finished


def _interval_makespan(intervals: list[tuple[float, float]]) -> float | None:
    if not intervals:
        return None
    return max(end for _start, end in intervals) - min(
        start for start, _end in intervals
    )


def _queue_wait_seconds(jobs: list[dict[str, Any]]) -> float:
    total = 0.0
    for job in jobs:
        submitted = _parse_time(
            job.get("submitted_at")
            or (job.get("status_timestamps") or {}).get("PENDING")
        )
        started = _parse_time(
            job.get("started_at") or (job.get("status_timestamps") or {}).get("RUNNING")
        )
        if submitted and started:
            total += max(0.0, (started - submitted).total_seconds())
    return total


def _probe_time_seconds(events: list[dict[str, Any]]) -> float:
    started_by_job: dict[str, datetime] = {}
    total = 0.0
    for event in events:
        event_type = str(event.get("event_type") or "")
        job_id = str(event.get("job_id") or "")
        created_at = _parse_time(event.get("created_at"))
        if not job_id or created_at is None:
            continue
        if event_type == "batch_probe_started":
            started_by_job[job_id] = created_at
        elif (
            event_type in {"batch_probe_selected", "batch_probe_failed"}
            and job_id in started_by_job
        ):
            total += max(0.0, (created_at - started_by_job.pop(job_id)).total_seconds())
    return total


def _concurrent_active_seconds(jobs: list[dict[str, Any]]) -> float:
    points: list[tuple[float, int]] = []
    for job in jobs:
        interval = _job_interval(job)
        if interval is None:
            continue
        started, finished = interval
        points.append((started, 1))
        points.append((finished, -1))
    points.sort(key=lambda item: item[0])
    active = 0
    previous = None
    total = 0.0
    for timestamp, delta in points:
        if previous is not None and active >= 2:
            total += max(0.0, timestamp - previous)
        active += delta
        previous = timestamp
    return total


def _backend_distribution(
    jobs: list[dict[str, Any]], events: list[dict[str, Any]]
) -> Counter:
    counter: Counter = Counter()
    for job in jobs:
        metadata = job.get("metadata") or {}
        backend = metadata.get("placement_backend")
        if backend:
            counter[str(backend)] += 1
    for event in events:
        if event.get("event_type") != "job_dispatched":
            continue
        payload = event.get("payload") or {}
        backend = payload.get("placement_backend") or payload.get("backend_name")
        if backend:
            counter[str(backend)] += 1
    return counter


def _hardware_summary(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        return {}
    memory_used = [
        float(getattr(sample, "memory_used_mb", 0) or 0.0) for sample in samples
    ]
    memory_total = [
        float(getattr(sample, "memory_total_mb", 0) or 0.0) for sample in samples
    ]
    gpu_util = [
        float(getattr(sample, "gpu_utilization", 0) or 0.0) * 100.0
        for sample in samples
    ]
    memory_percent = [
        (used / total * 100.0)
        for used, total in zip(memory_used, memory_total, strict=True)
        if total > 0
    ]
    return {
        "hardware_sample_count": len(samples),
        "avg_gpu_util_percent": sum(gpu_util) / len(gpu_util) if gpu_util else None,
        "max_gpu_util_percent": max(gpu_util) if gpu_util else None,
        "avg_gpu_memory_percent": (
            sum(memory_percent) / len(memory_percent) if memory_percent else None
        ),
        "max_gpu_memory_percent": max(memory_percent) if memory_percent else None,
        "peak_gpu_memory_used_mb": max(memory_used) if memory_used else None,
    }


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    deltas = {}
    for key, right_value in right.items():
        left_value = left.get(key)
        if isinstance(left_value, (int, float)) and isinstance(
            right_value, (int, float)
        ):
            deltas[key] = float(right_value) - float(left_value)
    return deltas


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _timestamp_seconds(value: Any) -> float | None:
    parsed = _parse_time(value)
    return parsed.timestamp() if parsed else None


def _run_id(runner_mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_histopathologic-cancer-detection_{runner_mode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a persisted scheduler timeline without the MLEvolve agent."
    )
    parser.add_argument(
        "--fixture",
        required=True,
        help="Fixture directory containing timeline.json and jobs.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for replay runtime, logs, and metrics.",
    )
    parser.add_argument("--runner-mode", choices=["real", "noop"], default="real")
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Scale timeline sleeps by this factor.",
    )
    parser.add_argument(
        "--until-seconds",
        type=float,
        default=None,
        help="Only replay actions at or before this original offset.",
    )
    parser.add_argument("--include-final-cleanup-cancels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--post-actions-wait-seconds", type=float, default=60.0)
    parser.add_argument("--allow-missing-scripts", action="store_true")
    parser.add_argument("--no-instrumented-fallback", action="store_true")
    parser.add_argument("--strict-missing-jobs", action="store_true")
    parser.add_argument(
        "--no-sleep", action="store_true", help="Submit selected actions immediately."
    )
    parser.add_argument(
        "--wait-for-all",
        action="store_true",
        help="Wait until all submitted jobs finish; do not cancel after the post-action wait.",
    )
    parser.add_argument(
        "--cancel-policy", choices=sorted(CANCEL_POLICIES), default="replay"
    )
    parser.add_argument(
        "--clean-scheduler-state",
        action="store_true",
        help="Remove persisted replay scheduler state before starting.",
    )
    parser.add_argument(
        "--settings-overlay",
        default=None,
        help="JSON file deep-merged into the fixture scheduler settings (e.g. to enable ML prediction).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = replay_fixture(
        fixture=args.fixture,
        output_root=args.output_root,
        runner_mode=args.runner_mode,
        speedup=args.speedup,
        until_seconds=args.until_seconds,
        include_final_cleanup_cancels=args.include_final_cleanup_cancels,
        dry_run=args.dry_run,
        post_actions_wait_seconds=args.post_actions_wait_seconds,
        allow_missing_scripts=args.allow_missing_scripts,
        use_instrumented_fallback=not args.no_instrumented_fallback,
        strict_missing_jobs=args.strict_missing_jobs,
        no_sleep=args.no_sleep,
        wait_for_all=args.wait_for_all,
        cancel_policy=args.cancel_policy,
        clean_scheduler_state=args.clean_scheduler_state,
        settings_overlay=args.settings_overlay,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "runtime_root": str(result.runtime_root),
                "metrics_path": str(result.metrics_path),
                "summary_path": str(result.summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
