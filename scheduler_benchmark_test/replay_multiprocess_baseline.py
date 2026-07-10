"""Replay generated MLEvolve scripts with simple subprocess slots."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from threading import Event, Thread
from typing import Any
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

from localml_scheduler.adapters.mlevolve_runner import _parse_exception
from localml_scheduler.execution.process_utils import start_new_session_kwargs, terminate_process_tree
from localml_scheduler.scheduler.telemetry import NvidiaSmiTelemetrySampler

from .timeline_fixture import load_fixture, reset_job_payload_for_replay


DEFAULT_MODE = "multiprocess_baseline"
SCRIPT_TASK_TYPE = "mlevolve_script"
PROBE_TASK_TYPE = "mlevolve_model_family_probe"
CANCEL_POLICIES = {"replay", "ignore"}


@dataclass(slots=True)
class BaselineResult:
    output_root: Path
    log_dir: Path
    summary_path: Path
    metrics_path: Path
    submitted_job_ids: list[str]
    skipped_actions: list[dict[str, Any]]


@dataclass(slots=True)
class RunningProcess:
    job_id: str
    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    timeout_deadline: float | None


class HardwareSampler:
    def __init__(self, *, interval_seconds: float, device_index: int = 0):
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.sampler = NvidiaSmiTelemetrySampler(device_index)
        self.samples: list[Any] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if not self.sampler.available():
            return
        self._thread = Thread(target=self._run, name="multiprocess-baseline-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self.sampler.sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def replay_multiprocess_baseline(
    *,
    fixture: str | Path,
    output_root: str | Path,
    runner_mode: str = "real",
    parallelism: int = 2,
    speedup: float = 1.0,
    until_seconds: float | None = None,
    include_final_cleanup_cancels: bool = False,
    job_filter: str = "script",
    dry_run: bool = False,
    post_actions_wait_seconds: float = 60.0,
    allow_missing_scripts: bool = False,
    use_instrumented_fallback: bool = True,
    strict_missing_jobs: bool = False,
    no_sleep: bool = False,
    poll_interval_seconds: float = 0.2,
    job_timeout_seconds: float | None = None,
    hardware_sample_interval_seconds: float = 1.0,
    wait_for_all: bool = False,
    cancel_policy: str = "replay",
) -> BaselineResult:
    if runner_mode not in {"real", "noop"}:
        raise ValueError(f"Unsupported runner mode: {runner_mode}")
    if job_filter not in {"script", "probe", "all"}:
        raise ValueError(f"Unsupported job filter: {job_filter}")
    if parallelism <= 0:
        raise ValueError("--parallelism must be greater than 0")
    if speedup <= 0:
        raise ValueError("--speedup must be greater than 0")
    if cancel_policy not in CANCEL_POLICIES:
        raise ValueError(f"Unsupported cancel policy: {cancel_policy}")

    output = Path(output_root).expanduser().resolve()
    workspace = output / "workspace"
    log_dir = output / "logs"
    mirror_log_dir = output / DEFAULT_MODE / "runs" / _run_id(runner_mode) / "logs"
    output.mkdir(parents=True, exist_ok=True)
    actions, jobs_by_id, baseline, _settings_payload = load_fixture(fixture)
    log_dir.mkdir(parents=True, exist_ok=True)
    _prepare_workspace(workspace, baseline)
    selected_actions = _select_actions(
        actions,
        until_seconds=until_seconds,
        include_final_cleanup_cancels=include_final_cleanup_cancels,
        cancel_policy=cancel_policy,
    )

    skipped_actions: list[dict[str, Any]] = []
    submitted_job_ids: list[str] = []
    records: dict[str, dict[str, Any]] = {}

    if dry_run:
        dry_submitted: set[str] = set()
        for action in selected_actions:
            job_id = str(action["job_id"])
            if action["action"] == "SUBMIT":
                payload = jobs_by_id.get(job_id)
                if payload is None:
                    skipped_actions.append({**action, "skip_reason": "missing job payload"})
                    continue
                if not _job_matches_filter(payload, job_filter):
                    skipped_actions.append({**action, "skip_reason": f"job filter excluded {payload.get('task_type')}"})
                    continue
                dry_submitted.add(job_id)
            elif action["action"] == "CANCEL" and job_id not in dry_submitted:
                skipped_actions.append({**action, "skip_reason": "cancel target not submitted in baseline"})
        metrics = _build_metrics(
            output_root=output,
            baseline=baseline,
            selected_actions=selected_actions,
            submitted_job_ids=[],
            skipped_actions=skipped_actions,
            records=[],
            samples=[],
            runner_mode=runner_mode,
            job_filter=job_filter,
            parallelism=parallelism,
            dry_run=True,
            started_at=time.time(),
            finished_at=time.time(),
            wait_for_all=wait_for_all,
            cancel_policy=cancel_policy,
        )
        return _write_outputs(output, log_dir, mirror_log_dir, metrics, [], [], skipped_actions, submitted_job_ids=[])

    pending: deque[str] = deque()
    running: dict[str, RunningProcess] = {}
    sampler = HardwareSampler(interval_seconds=hardware_sample_interval_seconds)
    started_at = time.time()
    action_index = 0
    sampler.start()
    try:
        while action_index < len(selected_actions):
            action = selected_actions[action_index]
            target_elapsed = 0.0 if no_sleep else float(action["relative_seconds"]) / speedup
            while True:
                _reap_finished(running, records)
                _start_pending(
                    pending,
                    running,
                    records,
                    workspace=workspace,
                    log_dir=log_dir,
                    runner_mode=runner_mode,
                    parallelism=parallelism,
                    job_timeout_seconds=job_timeout_seconds,
                )
                if no_sleep or time.time() - started_at >= target_elapsed:
                    break
                remaining = target_elapsed - (time.time() - started_at)
                time.sleep(min(max(0.01, poll_interval_seconds), max(0.0, remaining)))

            _apply_action(
                action,
                jobs_by_id=jobs_by_id,
                job_filter=job_filter,
                output_workspace=workspace,
                records=records,
                pending=pending,
                running=running,
                submitted_job_ids=submitted_job_ids,
                skipped_actions=skipped_actions,
                runner_mode=runner_mode,
                allow_missing_scripts=allow_missing_scripts,
                use_instrumented_fallback=use_instrumented_fallback,
                strict_missing_jobs=strict_missing_jobs,
            )
            _reap_finished(running, records)
            _start_pending(
                pending,
                running,
                records,
                workspace=workspace,
                log_dir=log_dir,
                runner_mode=runner_mode,
                parallelism=parallelism,
                job_timeout_seconds=job_timeout_seconds,
            )
            action_index += 1

        wait_deadline = None if wait_for_all else time.time() + max(0.0, post_actions_wait_seconds)
        while pending or running:
            _reap_finished(running, records)
            _start_pending(
                pending,
                running,
                records,
                workspace=workspace,
                log_dir=log_dir,
                runner_mode=runner_mode,
                parallelism=parallelism,
                job_timeout_seconds=job_timeout_seconds,
            )
            if not pending and not running:
                break
            if wait_deadline is not None and time.time() >= wait_deadline:
                break
            time.sleep(max(0.01, poll_interval_seconds))

        if not wait_for_all:
            _cancel_outstanding(pending, running, records, reason="post-actions wait expired")
    finally:
        sampler.stop()
        _cancel_outstanding(pending, running, records, reason="baseline shutdown")

    finished_at = time.time()
    record_list = list(records.values())
    metrics = _build_metrics(
        output_root=output,
        baseline=baseline,
        selected_actions=selected_actions,
        submitted_job_ids=submitted_job_ids,
        skipped_actions=skipped_actions,
        records=record_list,
        samples=sampler.samples,
        runner_mode=runner_mode,
        job_filter=job_filter,
        parallelism=parallelism,
        dry_run=False,
        started_at=started_at,
        finished_at=finished_at,
        wait_for_all=wait_for_all,
        cancel_policy=cancel_policy,
    )
    return _write_outputs(
        output,
        log_dir,
        mirror_log_dir,
        metrics,
        sampler.samples,
        record_list,
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
        if until_seconds is not None and float(action["relative_seconds"]) > until_seconds:
            continue
        selected.append(action)
    return selected


def _job_matches_filter(payload: dict[str, Any], job_filter: str) -> bool:
    task_type = payload.get("task_type")
    if job_filter == "all":
        return True
    if job_filter == "script":
        return task_type == SCRIPT_TASK_TYPE
    return task_type == PROBE_TASK_TYPE


def _apply_action(
    action: dict[str, Any],
    *,
    jobs_by_id: dict[str, dict[str, Any]],
    job_filter: str,
    output_workspace: Path,
    records: dict[str, dict[str, Any]],
    pending: deque[str],
    running: dict[str, RunningProcess],
    submitted_job_ids: list[str],
    skipped_actions: list[dict[str, Any]],
    runner_mode: str,
    allow_missing_scripts: bool,
    use_instrumented_fallback: bool,
    strict_missing_jobs: bool,
) -> None:
    job_id = str(action["job_id"])
    if action["action"] == "SUBMIT":
        payload = jobs_by_id.get(job_id)
        if payload is None:
            skipped_actions.append({**action, "skip_reason": "missing job payload"})
            if strict_missing_jobs:
                raise KeyError(f"Missing job payload for submitted job {job_id}")
            return
        if not _job_matches_filter(payload, job_filter):
            skipped_actions.append({**action, "skip_reason": f"job filter excluded {payload.get('task_type')}"})
            return
        try:
            record = _prepare_record(
                payload,
                action=action,
                output_workspace=output_workspace,
                runner_mode=runner_mode,
                allow_missing_scripts=allow_missing_scripts,
                use_instrumented_fallback=use_instrumented_fallback,
            )
        except FileNotFoundError as exc:
            skipped_actions.append({**action, "skip_reason": str(exc)})
            if strict_missing_jobs:
                raise
            return
        records[job_id] = record
        pending.append(job_id)
        submitted_job_ids.append(job_id)
        return

    if action["action"] == "CANCEL":
        if job_id not in records:
            skipped_actions.append({**action, "skip_reason": "cancel target not submitted in baseline"})
            return
        _cancel_job(job_id, pending, running, records, reason="cancel action")


def _prepare_record(
    payload: dict[str, Any],
    *,
    action: dict[str, Any],
    output_workspace: Path,
    runner_mode: str,
    allow_missing_scripts: bool,
    use_instrumented_fallback: bool,
) -> dict[str, Any]:
    job_payload = reset_job_payload_for_replay(payload)
    config = dict(job_payload.get("config") or {})
    runner_kwargs = dict(config.get("runner_kwargs") or {})
    original_working_dir = runner_kwargs.get("working_dir")
    script_path = runner_kwargs.get("script_path")
    resolved_script: Path | None = None
    if script_path and runner_mode == "real":
        resolved_script = _resolve_script_path(
            script_path,
            original_working_dir=original_working_dir,
            allow_missing=allow_missing_scripts,
            use_instrumented_fallback=use_instrumented_fallback,
        )

    result_path = output_workspace / "working" / "multiprocess_results" / f"result_{job_payload['job_id']}.json"
    metadata = dict(job_payload.get("metadata") or {})
    return {
        "job_id": job_payload["job_id"],
        "task_type": job_payload.get("task_type"),
        "priority": job_payload.get("priority"),
        "queue_sequence": job_payload.get("queue_sequence"),
        "model_family": metadata.get("model_family"),
        "node_id": metadata.get("node_id") or metadata.get("mlevolve_node_id"),
        "runner_target": config.get("runner_target"),
        "script_path": str(resolved_script) if resolved_script is not None else script_path,
        "original_script_path": script_path,
        "working_dir": str(output_workspace),
        "original_working_dir": original_working_dir,
        "result_path": str(result_path),
        "relative_seconds": float(action["relative_seconds"]),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "submitted_wall_time": time.time(),
        "started_at": None,
        "started_wall_time": None,
        "finished_at": None,
        "finished_wall_time": None,
        "status": "PENDING",
        "status_reason": None,
        "returncode": None,
        "exec_time": None,
        "stdout_path": None,
        "stderr_path": None,
    }


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
    if use_instrumented_fallback and original_working_dir:
        candidate = Path(original_working_dir) / "working" / "instrumented_scripts" / f"{path.stem}_instrumented.py"
        if candidate.exists():
            return candidate
    if allow_missing:
        return path
    raise FileNotFoundError(f"Baseline script path does not exist: {script_path}")


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
            os.symlink(str(Path(original_input).resolve()), str(input_path), target_is_directory=True)
        except OSError:
            pass


def _start_pending(
    pending: deque[str],
    running: dict[str, RunningProcess],
    records: dict[str, dict[str, Any]],
    *,
    workspace: Path,
    log_dir: Path,
    runner_mode: str,
    parallelism: int,
    job_timeout_seconds: float | None,
) -> None:
    while pending and len(running) < parallelism:
        job_id = pending.popleft()
        record = records.get(job_id)
        if record is None or record.get("status") != "PENDING":
            continue
        running[job_id] = _start_process(
            record,
            workspace=workspace,
            log_dir=log_dir,
            runner_mode=runner_mode,
            job_timeout_seconds=job_timeout_seconds,
        )


def _start_process(
    record: dict[str, Any],
    *,
    workspace: Path,
    log_dir: Path,
    runner_mode: str,
    job_timeout_seconds: float | None,
) -> RunningProcess:
    job_id = str(record["job_id"])
    output_dir = log_dir / "job_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{job_id}.stdout.log"
    stderr_path = output_dir / f"{job_id}.stderr.log"
    result_path = Path(record["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if runner_mode == "noop":
        cmd = [
            sys.executable,
            "-c",
            (
                "import json,pathlib,sys,time;"
                "p=pathlib.Path(sys.argv[1]);s=float(sys.argv[2]);time.sleep(s);"
                "p.parent.mkdir(parents=True,exist_ok=True);"
                "p.write_text(json.dumps({'term_out':['multiprocess baseline noop\\n'],"
                "'exec_time':s,'exc_type':None,'exc_info':{},'exc_stack':[],"
                "'phase_timings':{'phase_timing_available':False}}),encoding='utf-8')"
            ),
            str(result_path),
            "0.01",
        ]
    else:
        if record.get("task_type") == PROBE_TASK_TYPE:
            cmd = [
                sys.executable,
                "-c",
                (
                    "import json,pathlib,sys;"
                    "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
                    "p.write_text(json.dumps({'kind':'model_family_probe_skipped_by_multiprocess_baseline'}),encoding='utf-8')"
                ),
                str(result_path),
            ]
        else:
            script_path = record.get("script_path")
            if not script_path:
                raise FileNotFoundError(f"Missing script path for {job_id}")
            cmd = [sys.executable, str(script_path)]

    record["status"] = "RUNNING"
    record["started_at"] = datetime.now(timezone.utc).isoformat()
    record["started_wall_time"] = time.time()
    record["stdout_path"] = str(stdout_path)
    record["stderr_path"] = str(stderr_path)
    timeout_deadline = time.time() + float(job_timeout_seconds) if job_timeout_seconds else None
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(workspace),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            **start_new_session_kwargs(),
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    return RunningProcess(job_id=job_id, process=process, stdout_path=stdout_path, stderr_path=stderr_path, timeout_deadline=timeout_deadline)


def _reap_finished(running: dict[str, RunningProcess], records: dict[str, dict[str, Any]]) -> None:
    for job_id, handle in list(running.items()):
        if handle.timeout_deadline is not None and time.time() >= handle.timeout_deadline and handle.process.poll() is None:
            _terminate_process(handle.process)
            _finish_record(records[job_id], handle, status="FAILED", reason="job timeout", timeout=True)
            running.pop(job_id, None)
            continue
        if handle.process.poll() is None:
            continue
        status = "COMPLETED" if handle.process.returncode == 0 else "FAILED"
        _finish_record(records[job_id], handle, status=status, reason=None, timeout=False)
        running.pop(job_id, None)


def _finish_record(
    record: dict[str, Any],
    handle: RunningProcess,
    *,
    status: str,
    reason: str | None,
    timeout: bool,
) -> None:
    finished = time.time()
    started = float(record.get("started_wall_time") or finished)
    stdout = handle.stdout_path.read_text(encoding="utf-8") if handle.stdout_path.exists() else ""
    stderr = handle.stderr_path.read_text(encoding="utf-8") if handle.stderr_path.exists() else ""
    exec_time = max(0.0, finished - started)
    exc_type = None
    exc_info: dict[str, Any] = {}
    exc_stack: list[Any] = []
    if timeout:
        exc_type = "TimeoutError"
        exc_info = {"message": "multiprocess baseline job timeout"}
    elif status == "FAILED":
        exc_type, exc_info, exc_stack = _parse_exception(
            stderr,
            Path(record.get("working_dir") or "."),
            Path(record.get("script_path") or ""),
        )
        reason = reason or exc_info.get("message") or "process exited non-zero"

    record.update(
        {
            "status": status,
            "status_reason": reason,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "finished_wall_time": finished,
            "returncode": handle.process.returncode,
            "exec_time": exec_time,
            "timeout": timeout,
        }
    )
    _write_result(
        Path(record["result_path"]),
        stdout=stdout,
        stderr=stderr,
        exec_time=exec_time,
        exc_type=exc_type,
        exc_info=exc_info,
        exc_stack=exc_stack,
    )


def _write_result(
    path: Path,
    *,
    stdout: str,
    stderr: str,
    exec_time: float,
    exc_type: str | None,
    exc_info: dict[str, Any],
    exc_stack: list[Any],
) -> None:
    output: list[str] = []
    if stdout:
        output.extend(stdout.splitlines(keepends=True))
    if stderr:
        output.extend(stderr.splitlines(keepends=True))
    if not output:
        output = [""]
    payload = {
        "term_out": output,
        "exec_time": exec_time,
        "exc_type": exc_type,
        "exc_info": exc_info,
        "exc_stack": exc_stack,
        "phase_timings": {"phase_timing_available": False},
        "runner_mode": DEFAULT_MODE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _cancel_job(
    job_id: str,
    pending: deque[str],
    running: dict[str, RunningProcess],
    records: dict[str, dict[str, Any]],
    *,
    reason: str,
) -> None:
    record = records.get(job_id)
    if record is None or record.get("status") in {"COMPLETED", "FAILED", "CANCELLED"}:
        return
    if record.get("status") == "PENDING":
        pending_items = [item for item in pending if item != job_id]
        pending.clear()
        pending.extend(pending_items)
        now = time.time()
        record.update(
            {
                "status": "CANCELLED",
                "status_reason": reason,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "finished_wall_time": now,
                "exec_time": 0.0,
            }
        )
        _write_result(
            Path(record["result_path"]),
            stdout="",
            stderr=f"Cancelled before start: {reason}\n",
            exec_time=0.0,
            exc_type="Cancelled",
            exc_info={"message": reason},
            exc_stack=[],
        )
        return
    handle = running.pop(job_id, None)
    if handle is not None:
        _terminate_process(handle.process)
        _finish_record(record, handle, status="CANCELLED", reason=reason, timeout=False)


def _cancel_outstanding(
    pending: deque[str],
    running: dict[str, RunningProcess],
    records: dict[str, dict[str, Any]],
    *,
    reason: str,
) -> None:
    for job_id in list(pending):
        _cancel_job(job_id, pending, running, records, reason=reason)
    for job_id in list(running):
        _cancel_job(job_id, pending, running, records, reason=reason)


def _terminate_process(process: subprocess.Popen) -> None:
    terminate_process_tree(process, timeout=2.0)


def _build_metrics(
    *,
    output_root: Path,
    baseline: dict[str, Any],
    selected_actions: list[dict[str, Any]],
    submitted_job_ids: list[str],
    skipped_actions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    samples: list[Any],
    runner_mode: str,
    job_filter: str,
    parallelism: int,
    dry_run: bool,
    started_at: float,
    finished_at: float,
    wait_for_all: bool,
    cancel_policy: str,
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status")) for record in records)
    task_counts = Counter(str(record.get("task_type")) for record in records)
    durations = [float(record["exec_time"]) for record in records if record.get("exec_time") is not None]
    intervals = [
        (float(record["started_wall_time"]), float(record["finished_wall_time"]))
        for record in records
        if record.get("started_wall_time") is not None and record.get("finished_wall_time") is not None
    ]
    makespan = _interval_makespan(intervals)
    queue_wait = sum(
        max(0.0, float(record["started_wall_time"]) - float(record["submitted_wall_time"]))
        for record in records
        if record.get("started_wall_time") is not None and record.get("submitted_wall_time") is not None
    )
    total_wall = max(0.0, finished_at - started_at)
    candidate_total = sum(durations)
    reference_by_mode = _reference_metrics_by_mode(baseline)
    metrics: dict[str, Any] = {
        "mode": DEFAULT_MODE,
        "experiment_mode": "multiprocess_timeline_replay",
        "command_label": DEFAULT_MODE,
        "run_id": output_root.name,
        "exp_id": "histopathologic-cancer-detection",
        "configured_scheduler_enabled": False,
        "scheduler_client_attached": False,
        "multiprocess_parallelism": parallelism,
        "multiprocess_job_filter": job_filter,
        "replay_runner_mode": runner_mode,
        "replay_dry_run": dry_run,
        "replay_wait_for_all": wait_for_all,
        "replay_cancel_policy": cancel_policy,
        "replay_action_count": len(selected_actions),
        "replay_submit_action_count": sum(1 for action in selected_actions if action["action"] == "SUBMIT"),
        "replay_cancel_action_count": sum(1 for action in selected_actions if action["action"] == "CANCEL"),
        "replay_skipped_action_count": len(skipped_actions),
        "submitted_job_count": len(submitted_job_ids),
        "scheduler_job_count": 0,
        "node_count": int(task_counts.get(SCRIPT_TASK_TYPE, 0)),
        "completed_job_count": int(status_counts.get("COMPLETED", 0)),
        "failed_job_count": int(status_counts.get("FAILED", 0)),
        "cancelled_job_count": int(status_counts.get("CANCELLED", 0)),
        "timeout_job_count": sum(1 for record in records if record.get("timeout")),
        "job_status_counts": dict(status_counts),
        "task_type_counts": dict(task_counts),
        "total_run_wall_time_seconds": total_wall,
        "total_wall_time_seconds": total_wall,
        "total_candidate_execution_time_seconds": candidate_total,
        "total_job_execution_time_seconds": candidate_total,
        "execution_time_seconds": candidate_total,
        "median_candidate_execution_time_seconds": median(durations) if durations else None,
        "median_job_execution_time_seconds": median(durations) if durations else None,
        "candidate_execution_makespan_seconds": makespan,
        "candidate_execution_parallelism_ratio": (candidate_total / makespan) if makespan and makespan > 0 else None,
        "non_candidate_overhead_wall_time_seconds": max(0.0, total_wall - makespan) if makespan is not None else None,
        "queue_wait_seconds": queue_wait,
        "total_scheduler_queue_wait_seconds": 0.0,
        "probe_time_seconds": 0.0,
        "total_scheduler_probe_time_seconds": 0.0,
        "packed_dispatch_count": 0,
        "packed_fallback_count": 0,
        "batch_probe_hit_count": 0,
        "batch_probe_trial_count": 0,
        "concurrent_gpu_active_seconds": _concurrent_active_seconds(intervals),
        "scheduler_backend_distribution": {},
        "baseline_source_run_root": baseline.get("source_run_root"),
        **_hardware_summary(samples),
    }
    metrics["deltas_vs_original_scheduler_off"] = _metric_deltas(reference_by_mode.get("scheduler_off") or {}, metrics)
    metrics["deltas_vs_original_scheduler_on"] = _metric_deltas(reference_by_mode.get("scheduler_on") or baseline.get("reference_metrics") or {}, metrics)
    return metrics


def _write_outputs(
    output: Path,
    log_dir: Path,
    mirror_log_dir: Path,
    metrics: dict[str, Any],
    samples: list[Any],
    records: list[dict[str, Any]],
    skipped_actions: list[dict[str, Any]],
    submitted_job_ids: list[str],
) -> BaselineResult:
    metrics_path = log_dir / "comparison_metrics.json"
    summary_path = output / "replay_summary.json"
    records_path = log_dir / "multiprocess_jobs.jsonl"
    hardware_path = log_dir / "hardware_samples.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "output_root": str(output),
        "metrics_path": str(metrics_path),
        "records_path": str(records_path),
        "hardware_samples_path": str(hardware_path) if samples else None,
        "skipped_actions": skipped_actions,
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if samples:
        _write_hardware_samples(hardware_path, samples)

    mirror_log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, mirror_log_dir / "comparison_metrics.json")
    if samples and hardware_path.exists():
        shutil.copy2(hardware_path, mirror_log_dir / "hardware_samples.csv")
    return BaselineResult(
        output_root=output,
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
                    "gpu_memory_percent": (memory_used / memory_total * 100.0) if memory_total > 0 else "",
                    "gpu_util_percent": float(getattr(sample, "gpu_utilization", 0.0) or 0.0) * 100.0,
                    "gpu_memory_util_percent": float(getattr(sample, "memory_utilization", 0.0) or 0.0) * 100.0,
                    "gpu_power_draw_w": "",
                    "sample_count": 1,
                }
            )


def _interval_makespan(intervals: list[tuple[float, float]]) -> float | None:
    if not intervals:
        return None
    return max(end for _start, end in intervals) - min(start for start, _end in intervals)


def _concurrent_active_seconds(intervals: list[tuple[float, float]]) -> float:
    points: list[tuple[float, int]] = []
    for started, finished in intervals:
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


def _hardware_summary(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        return {}
    memory_used = [float(getattr(sample, "memory_used_mb", 0) or 0.0) for sample in samples]
    memory_total = [float(getattr(sample, "memory_total_mb", 0) or 0.0) for sample in samples]
    gpu_util = [float(getattr(sample, "gpu_utilization", 0) or 0.0) * 100.0 for sample in samples]
    memory_percent = [(used / total * 100.0) for used, total in zip(memory_used, memory_total, strict=True) if total > 0]
    return {
        "hardware_sample_count": len(samples),
        "avg_gpu_util_percent": sum(gpu_util) / len(gpu_util) if gpu_util else None,
        "max_gpu_util_percent": max(gpu_util) if gpu_util else None,
        "avg_gpu_memory_percent": sum(memory_percent) / len(memory_percent) if memory_percent else None,
        "max_gpu_memory_percent": max(memory_percent) if memory_percent else None,
        "peak_gpu_memory_used_mb": max(memory_used) if memory_used else None,
    }


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    deltas = {}
    for key, right_value in right.items():
        left_value = left.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            deltas[key] = float(right_value) - float(left_value)
    return deltas


def _reference_metrics_by_mode(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source = baseline.get("source_run_root")
    if not source:
        return {}
    summary_path = Path(source) / "comparison_plots" / "comparison_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    modes = payload.get("modes") or {}
    return {str(mode): dict((item or {}).get("metrics") or {}) for mode, item in modes.items()}


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


def _run_id(runner_mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_histopathologic-cancer-detection_{runner_mode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay fixture scripts through simple subprocess slots.")
    parser.add_argument("--fixture", required=True, help="Fixture directory containing timeline.json and jobs.jsonl.")
    parser.add_argument("--output-root", required=True, help="Output directory for baseline logs and metrics.")
    parser.add_argument("--runner-mode", choices=["real", "noop"], default="real")
    parser.add_argument("--parallelism", type=int, default=2, help="Number of simple subprocess slots.")
    parser.add_argument("--speedup", type=float, default=1.0, help="Scale timeline sleeps by this factor.")
    parser.add_argument("--until-seconds", type=float, default=None, help="Only replay actions at or before this original offset.")
    parser.add_argument("--include-final-cleanup-cancels", action="store_true")
    parser.add_argument("--job-filter", choices=["script", "probe", "all"], default="script")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--post-actions-wait-seconds", type=float, default=60.0)
    parser.add_argument("--allow-missing-scripts", action="store_true")
    parser.add_argument("--no-instrumented-fallback", action="store_true")
    parser.add_argument("--strict-missing-jobs", action="store_true")
    parser.add_argument("--no-sleep", action="store_true", help="Submit selected actions immediately.")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.2)
    parser.add_argument("--job-timeout-seconds", type=float, default=None)
    parser.add_argument("--hardware-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--wait-for-all", action="store_true", help="Wait until all submitted jobs finish; do not cancel after the post-action wait.")
    parser.add_argument("--cancel-policy", choices=sorted(CANCEL_POLICIES), default="replay")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = replay_multiprocess_baseline(
        fixture=args.fixture,
        output_root=args.output_root,
        runner_mode=args.runner_mode,
        parallelism=args.parallelism,
        speedup=args.speedup,
        until_seconds=args.until_seconds,
        include_final_cleanup_cancels=args.include_final_cleanup_cancels,
        job_filter=args.job_filter,
        dry_run=args.dry_run,
        post_actions_wait_seconds=args.post_actions_wait_seconds,
        allow_missing_scripts=args.allow_missing_scripts,
        use_instrumented_fallback=not args.no_instrumented_fallback,
        strict_missing_jobs=args.strict_missing_jobs,
        no_sleep=args.no_sleep,
        poll_interval_seconds=args.poll_interval_seconds,
        job_timeout_seconds=args.job_timeout_seconds,
        hardware_sample_interval_seconds=args.hardware_sample_interval_seconds,
        wait_for_all=args.wait_for_all,
        cancel_policy=args.cancel_policy,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
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
