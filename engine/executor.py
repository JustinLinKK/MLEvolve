"""
Python interpreter for executing code snippets via subprocess.
- Executes code in a separate Python process (avoids CUDA/fork issues).
- Captures stdout/stderr, exceptions and stack traces, execution time limit.
- Supports multiple parallel slots (max_parallel_run) with CPU pinning.
"""

import logging
import os
import signal
import sys
import threading
import time
import traceback
import subprocess
import json
import uuid
from dataclasses import dataclass
from hashlib import sha1
from multiprocessing import Lock
from pathlib import Path
from typing import Any

import humanize
from dataclasses_json import DataClassJsonMixin
from engine.script_introspection import (
    code_supports_batch_probe as _code_supports_batch_probe,
    detect_initial_batch_size as _detect_initial_batch_size,
    introspect_training_script as _introspect_training_script,
    normalized_mlevolve_script_signature as _normalized_mlevolve_script_signature,
)

logger = logging.getLogger("MLEvolve")

_BATCH_PROBE_EVENT_TYPES = {
    "batch_probe_cache_hit",
    "batch_probe_cache_miss",
    "batch_probe_started",
    "batch_probe_trial",
    "batch_probe_selected",
    "batch_probe_warning",
    "batch_probe_failed",
}


def _build_scheduler_preload_source(scheduler_cfg: Any) -> dict[str, str] | None:
    if scheduler_cfg is None:
        return None
    raw_model_path = getattr(scheduler_cfg, "preload_source_model_path", None)
    if not raw_model_path:
        return None
    resolved_path = str(Path(raw_model_path).resolve())
    explicit_model_id = getattr(scheduler_cfg, "preload_source_model_id", None)
    model_id = str(explicit_model_id or f"mlevolve-preload:{sha1(resolved_path.encode('utf-8')).hexdigest()[:16]}")
    loader_target = getattr(scheduler_cfg, "preload_source_loader_target", None) or "localml_scheduler.adapters.mlevolve_runner:load_raw_file"
    return {
        "model_id": model_id,
        "model_path": resolved_path,
        "loader_target": str(loader_target),
    }

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


@dataclass
class _PreparedSchedulerJob:
    node_id: str
    process_id: int
    runfile_path: Path
    result_path: Path
    job: Any
    runner_kwargs: dict[str, Any]
    job_metadata: dict[str, Any]
    scheduler_mode: str | None
    detected_batch_size: int | None
    proposed_epochs: int | None
    model_key: str | None
    framework: str | None
    uses_amp: bool | None
    requires_gpu: bool | None
    script_signature: str
    start_time: float
    job_id: str | None = None
    last_probe_event_id: int = 0



class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 3,
        cfg=None,
        pipeline_logger=None,
        **kwargs,
    ):
        """
        Executes Python code in subprocess(es). No fork/multiprocessing.Process to avoid CUDA issues.

        Args:
            working_dir: working directory of the agent
            timeout: timeout per code execution (seconds)
            agent_file_name: base name for runfile (actual names are runfile_0.py, ...)
            max_parallel_run: max concurrent execution slots
            cfg: config (start_cpu_id, cpu_number, agent.search.parallel_search_num)
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.cfg = cfg
        self.pipeline_logger = pipeline_logger
        self.timeout = timeout
        self.max_parallel_run = (
            cfg.agent.search.parallel_search_num if (cfg and getattr(cfg.agent.search, "parallel_search_num", None)) else max_parallel_run
        )
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.current_parallel_run = 0
        self.status_map = [0] * self.max_parallel_run
        self.start_cpu_id = int(cfg.start_cpu_id) if cfg else 0
        self.cpu_number = int(cfg.cpu_number) if cfg else 1
        if self.cpu_number < self.max_parallel_run:
            raise ValueError(
                "The maximum level of parallelism exceeds the number of allocated CPU cores; "
                "ensure that each process has at least one CPU core."
            )
        self.lock = Lock()
        self._procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        self.scheduler_client = None
        self.scheduler_cfg = None
        self._scheduler_service = None
        self._scheduler_job_ids: set[str] = set()
        self._scheduler_jobs_lock = threading.Lock()

    def _format_scheduler_probe_event(self, event: dict[str, Any]) -> str | None:
        event_type = str(event.get("event_type", ""))
        if event_type not in _BATCH_PROBE_EVENT_TYPES:
            return None
        payload = event.get("payload") or {}
        job_id = event.get("job_id") or "unknown"
        prefix = f"[scheduler][batch_probe][job={job_id}]"

        if event_type == "batch_probe_cache_hit":
            return (
                f"{prefix} cache hit: batch_size={payload.get('resolved_batch_size')} "
                f"device={payload.get('device_type')} key={payload.get('probe_key')}"
            )
        if event_type == "batch_probe_cache_miss":
            return f"{prefix} cache miss: device={payload.get('device_type')} key={payload.get('probe_key')}"
        if event_type == "batch_probe_started":
            return (
                f"{prefix} started: model={payload.get('model_key')} device={payload.get('device_type')} "
                f"start_batch_size={payload.get('start_batch_size')} key={payload.get('probe_key')}"
            )
        if event_type == "batch_probe_trial":
            detail = str(payload.get("message") or "").strip().replace("\n", " ")
            if len(detail) > 180:
                detail = f"{detail[:177]}..."
            suffix = f" detail={detail}" if detail else ""
            return (
                f"{prefix} trial: batch_size={payload.get('batch_size')} fits={payload.get('fits')} "
                f"within_budget={payload.get('within_budget')} peak_vram_mb={payload.get('peak_vram_mb')} "
                f"target_budget_mb={payload.get('target_budget_mb')}{suffix}"
            )
        if event_type == "batch_probe_selected":
            return (
                f"{prefix} selected: batch_size={payload.get('resolved_batch_size')} "
                f"target_budget_mb={payload.get('target_budget_mb')} key={payload.get('probe_key')}"
            )
        if event_type == "batch_probe_warning":
            detail = str(payload.get("warning_message") or "").strip().replace("\n", " ")
            if len(detail) > 220:
                detail = f"{detail[:217]}..."
            return (
                f"{prefix} warning: source={payload.get('source')} "
                f"reason={payload.get('warning_reason')} detail={detail}"
            )
        if event_type == "batch_probe_failed":
            detail = str(payload.get("reason") or "").strip().replace("\n", " ")
            if len(detail) > 220:
                detail = f"{detail[:217]}..."
            return f"{prefix} failed: {detail}"
        return None

    def _log_scheduler_probe_updates(self, job_id: str, last_event_id: int) -> int:
        if self.scheduler_client is None:
            return last_event_id
        try:
            events = self.scheduler_client.store.list_events(job_id=job_id)
        except Exception as exc:
            logger.debug(f"Skipping scheduler event polling for {job_id}: {exc}")
            return last_event_id
        next_event_id = last_event_id
        for event in events:
            event_id = int(event.get("event_id") or 0)
            if event_id <= last_event_id:
                continue
            next_event_id = max(next_event_id, event_id)
            self._pipeline_emit(
                "scheduler_event",
                job_id=job_id,
                payload=event,
            )
            formatted = self._format_scheduler_probe_event(event)
            if formatted:
                logger.info(formatted)
        return next_event_id

    def attach_scheduler(self, client: Any, scheduler_cfg: Any) -> None:
        """Route future executions through localml_scheduler."""
        self.scheduler_client = client
        self.scheduler_cfg = scheduler_cfg

    def _pipeline_emit(self, event_type: str, **kwargs: Any) -> None:
        if self.pipeline_logger is None:
            return
        try:
            self.pipeline_logger.emit(event_type, **kwargs)
        except Exception:
            return

    def _pipeline_upsert_job(self, job_id: str, **fields: Any) -> None:
        if self.pipeline_logger is None:
            return
        try:
            self.pipeline_logger.upsert_job_packet(job_id, **fields)
        except Exception:
            return

    def _available_cpus(self) -> list[int]:
        try:
            return sorted(os.sched_getaffinity(0))
        except AttributeError:
            return list(range(max(1, os.cpu_count() or 1)))

    def _scheduler_bridge_start_service_enabled(self) -> bool:
        if self.scheduler_cfg is None:
            return True
        return bool(getattr(self.scheduler_cfg, "start_service", True))

    def _ensure_scheduler_service_available(self) -> None:
        if self.scheduler_client is None:
            return
        if self._scheduler_service is not None:
            return
        if self._scheduler_bridge_start_service_enabled():
            return
        try:
            if self.scheduler_client.scheduler_service_active():
                return
        except Exception as exc:
            logger.warning("Could not verify external scheduler service heartbeat: %s", exc)
        logger.warning(
            "No active external scheduler service detected at %s; starting an in-process scheduler service fallback.",
            self.scheduler_client.settings.runtime_root,
        )
        self._scheduler_service = self.scheduler_client.create_service().start(background=True)

    def _normalized_raw_packing_defaults(self, submission_defaults: Any) -> tuple[bool, list[str]]:
        packing_eligible = bool(getattr(submission_defaults, "packing_eligible", False))
        configured_allowlist = getattr(submission_defaults, "backend_allowlist", None)
        if not configured_allowlist:
            configured_allowlist = ["mps", "cuda_process"]
        supported_backends = {"mps", "cuda_process"}
        normalized_allowlist = [str(name) for name in configured_allowlist if str(name) in supported_backends]
        if packing_eligible and not normalized_allowlist:
            logger.warning(
                "Raw MLEvolve script jobs do not support the configured packed backends %r; disabling packing for this submission.",
                list(configured_allowlist),
            )
            return False, []
        return packing_eligible, normalized_allowlist

    def _stop_managed_scheduler_service(self) -> None:
        if self._scheduler_service is None:
            return
        try:
            self._scheduler_service.stop()
        except Exception as exc:
            logger.warning("Error stopping fallback scheduler service: %s", exc)
        finally:
            self._scheduler_service = None

    def terminate_all_subprocesses(self) -> None:
        """Terminate all active subprocesses (for graceful Ctrl+C exit)."""
        with self._procs_lock:
            procs = list(self._active_procs.items())
            self._active_procs.clear()
        for slot_id, proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Error terminating subprocess slot {slot_id}: {e}")
        if self.scheduler_client is not None:
            with self._scheduler_jobs_lock:
                job_ids = list(self._scheduler_job_ids)
            for job_id in job_ids:
                try:
                    self.scheduler_client.cancel(job_id)
                except Exception as e:
                    logger.warning(f"Error cancelling scheduler job {job_id}: {e}")
        self._stop_managed_scheduler_service()

    def check_current_status(self):
        """Check current parallel run number."""
        return self.current_parallel_run < self.max_parallel_run

    def isolate_submission_path(self, code: str, _id) -> str:
        """Per-process submission filename to avoid write conflicts."""
        target = f"submission_{_id}.csv"

        code = code.replace("submission/submission.csv", f"submission/{target}")
        code = code.replace("/submission.csv", f"/{target}")

        for quote in ("'", '"'):
            code = code.replace(
                f"to_csv({quote}submission.csv",
                f"to_csv({quote}submission/{target}",
            )

        for quote in ("'", '"'):
            code = code.replace(f"{quote}submission.csv{quote}", f"{quote}{target}{quote}")

        return code
    
    def isolate_model_path(self, code, _id):
        """Replace generic model filenames in code to avoid multi-process conflicts."""
        if '.pth' not in code and '.bin' not in code and '.pt' not in code:
            return code

        modified_code = code

        generic_model_names = [
            "best_model.pth",
            "best_model.bin",
            "best_model.pt",
            "model_best.pth",
            "model_best.bin",
            "model_best.pt",
            "model.pth",
            "model.pt",
            "model.bin",
            "checkpoint.pth",
            "checkpoint.pt",
            "checkpoint.bin",
        ]

        generic_model_names.sort(key=len, reverse=True)

        for filename in generic_model_names:
            if filename not in modified_code:
                continue

            name, ext = filename.rsplit('.', 1)
            new_filename = f"{name}_{_id}.{ext}"

            modified_code = modified_code.replace(f"/{filename}", f"/{new_filename}")
            modified_code = modified_code.replace(f'"{filename}"', f'"{new_filename}"')
            modified_code = modified_code.replace(f"'{filename}'", f"'{new_filename}'")

        return modified_code
    
    def cleanup_session(self, process_id: int = -1) -> None:
        """Clean up resources for the given process slot."""
        if process_id == -1:
            self._stop_managed_scheduler_service()

    def run(self, code: str, id, reset_session=True, working_dir: str | None = None):
        """
        Execute the provided Python command in a subprocess and return its output.

        Parameters:
            code: Python code to execute.
            reset_session: Reserved for future use.
            working_dir: Optional per-run working directory.

        Returns:
            ExecutionResult: output, exec_time, exc_type, exc_info, exc_stack.
        """
        if self.scheduler_client is not None:
            return self._run_scheduler_job(code=code, id=id, working_dir=working_dir)
        return self._run_subprocess(code=code, id=id, working_dir=working_dir)

    def run_many(
        self,
        items: list[tuple[str, Any]] | list[dict[str, Any]],
        *,
        working_dir: str | None = None,
    ) -> dict[str, ExecutionResult]:
        """Execute a round of node codes, submitting scheduler jobs as one visible packet."""
        if not items:
            return {}
        normalized_items: list[tuple[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized_items.append((str(item.get("code") or ""), item.get("id") or item.get("node_id")))
            else:
                code, node_id = item
                normalized_items.append((code, node_id))

        if self.scheduler_client is None:
            return {
                str(node_id): self.run(code=code, id=node_id, working_dir=working_dir)
                for code, node_id in normalized_items
            }

        if len(normalized_items) > self.max_parallel_run:
            raise ValueError(
                f"Scheduler round has {len(normalized_items)} jobs but interpreter capacity is {self.max_parallel_run}"
            )

        logger.info("REPL is submitting %s code candidates to localml_scheduler as one round", len(normalized_items))
        prepared: list[_PreparedSchedulerJob] = []
        results: dict[str, ExecutionResult] = {}
        try:
            self._ensure_scheduler_service_available()
            for code, node_id in normalized_items:
                prepared.append(self._prepare_scheduler_round_job(code=code, id=node_id, working_dir=working_dir))

            candidates = [dict(prepared_job.job_metadata) for prepared_job in prepared]
            packet_context: dict[str, Any] = {}
            if hasattr(self.scheduler_client, "plan_job_packet"):
                try:
                    packet_context = self.scheduler_client.plan_job_packet(candidates=candidates)
                except Exception as exc:
                    logger.debug("Skipping scheduler packet planning before submission: %s", exc)
                    packet_context = {}

            jobs = [prepared_job.job for prepared_job in prepared]
            if hasattr(self.scheduler_client, "submit_many"):
                submitted_jobs = self.scheduler_client.submit_many(jobs)
            else:
                submitted_jobs = [self.scheduler_client.submit(job) for job in jobs]

            for prepared_job, submitted in zip(prepared, submitted_jobs):
                prepared_job.job_id = submitted.job_id
                with self._scheduler_jobs_lock:
                    self._scheduler_job_ids.add(submitted.job_id)
                self._pipeline_upsert_job(
                    submitted.job_id,
                    node_id=prepared_job.node_id,
                    scheduler_mode=prepared_job.scheduler_mode,
                    placement_mode="scheduler",
                    placement_backend=None,
                    status=getattr(submitted.status, "value", str(submitted.status)),
                    submitted_at=getattr(submitted, "submitted_at", None),
                    detected_batch_size=prepared_job.detected_batch_size,
                    resolved_batch_size=None,
                    proposed_epochs=prepared_job.proposed_epochs,
                    model_key=prepared_job.model_key,
                    framework=prepared_job.framework,
                    uses_amp=prepared_job.uses_amp,
                    requires_gpu=prepared_job.requires_gpu,
                    script_signature=prepared_job.script_signature,
                    payload={
                        "job": submitted.to_dict() if hasattr(submitted, "to_dict") else {},
                        "round_packet": packet_context,
                    },
                )
                self._pipeline_emit(
                    "scheduler_submission_created",
                    node_id=prepared_job.node_id,
                    job_id=submitted.job_id,
                    stage="execution",
                    payload=prepared_job.job_metadata,
                )

            self._pipeline_emit(
                "scheduler_round_submitted",
                stage="execution",
                payload={
                    "packet_id": packet_context.get("packet_id"),
                    "job_ids": [job.job_id for job in submitted_jobs],
                    "node_ids": [prepared_job.node_id for prepared_job in prepared],
                    "job_count": len(submitted_jobs),
                    "packet_context": packet_context,
                },
            )
            logger.info(
                "Submitted scheduler round with %s job(s): %s",
                len(submitted_jobs),
                ", ".join(job.job_id for job in submitted_jobs),
            )

            wait_timeout = getattr(self.scheduler_cfg, "wait_timeout_seconds", None)
            if wait_timeout is None:
                wait_timeout = self.timeout * max(1, len(prepared)) + 60
            poll_interval = max(0.1, float(getattr(self.scheduler_cfg, "wait_poll_interval_seconds", 1.0)))
            deadline = time.time() + float(wait_timeout)
            pending = {prepared_job.job_id: prepared_job for prepared_job in prepared if prepared_job.job_id}
            final_jobs: dict[str, Any] = {}
            while pending and time.time() < deadline:
                for job_id, prepared_job in list(pending.items()):
                    final_job = self.scheduler_client.inspect(job_id)
                    prepared_job.last_probe_event_id = self._log_scheduler_probe_updates(job_id, prepared_job.last_probe_event_id)
                    if final_job is not None and final_job.status.is_terminal:
                        final_jobs[job_id] = final_job
                        pending.pop(job_id, None)
                if pending:
                    time.sleep(poll_interval)

            for job_id, prepared_job in list(pending.items()):
                try:
                    self.scheduler_client.cancel(job_id)
                except Exception:
                    pass
                exec_time = time.time() - prepared_job.start_time
                self._pipeline_upsert_job(
                    job_id,
                    node_id=prepared_job.node_id,
                    status="timeout",
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    duration_seconds=exec_time,
                    detected_batch_size=prepared_job.detected_batch_size,
                    proposed_epochs=prepared_job.proposed_epochs,
                    model_key=prepared_job.model_key,
                    framework=prepared_job.framework,
                    uses_amp=prepared_job.uses_amp,
                    requires_gpu=prepared_job.requires_gpu,
                    script_signature=prepared_job.script_signature,
                    payload={"reason": "scheduler wait timeout", "wait_timeout_seconds": wait_timeout},
                )
                results[prepared_job.node_id] = ExecutionResult(
                    term_out=[
                        f"Execution time: TimeoutError: Scheduler job {job_id} exceeded wait limit of {humanize.naturaldelta(wait_timeout)}"
                    ],
                    exec_time=exec_time,
                    exc_type="TimeoutError",
                    exc_info={"message": "scheduler wait timeout", "job_id": job_id},
                    exc_stack=[],
                )

            for prepared_job in prepared:
                if prepared_job.node_id in results:
                    continue
                job_id = prepared_job.job_id
                final_job = final_jobs.get(job_id) if job_id else None
                if job_id is not None:
                    prepared_job.last_probe_event_id = self._log_scheduler_probe_updates(job_id, prepared_job.last_probe_event_id)
                results[prepared_job.node_id] = self._scheduler_execution_result_from_final(prepared_job, final_job)

            return results
        except Exception as e:
            logger.error("Error in scheduler round execution: %s", e)
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            for prepared_job in prepared:
                results.setdefault(
                    prepared_job.node_id,
                    ExecutionResult(
                        term_out=[f"Scheduler round execution error: {str(e)}", error_trace],
                        exec_time=time.time() - prepared_job.start_time,
                        exc_type="RuntimeError",
                        exc_info={"error": str(e)},
                        exc_stack=[],
                    ),
                )
            return results
        finally:
            for prepared_job in prepared:
                if prepared_job.job_id is not None:
                    with self._scheduler_jobs_lock:
                        self._scheduler_job_ids.discard(prepared_job.job_id)
                try:
                    if prepared_job.runfile_path.exists():
                        os.remove(prepared_job.runfile_path)
                except Exception as exc:
                    logger.warning("Failed to remove scheduler round runfile after execution: %s", exc)
                with self.lock:
                    self.status_map[prepared_job.process_id] = 0
                    self.current_parallel_run = max(0, self.current_parallel_run - 1)

    def _prepare_scheduler_round_job(self, code: str, id, working_dir: str | None = None) -> _PreparedSchedulerJob:
        from localml_scheduler.adapters.mlevolve import build_mlevolve_job
        from localml_scheduler.domain import BatchProbeSpec, PreloadSource, ResourceRequirements, RuntimeProbeSpec

        process_id = None
        start_time = time.time()
        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    logger.info("Assigned scheduler round submission slot: %s", process_id)
                    break
            if process_id is None:
                self.current_parallel_run -= 1
                raise ValueError("reach max process parallel number")

        cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
        avail_cpus = self._available_cpus()
        start = process_id * cpu_number_per_session
        cpu_set = set(avail_cpus[start:start + cpu_number_per_session]) or set(avail_cpus)
        pre_code = ""
        if hasattr(os, "sched_setaffinity"):
            pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)

        node_id = str(id)
        code = self.isolate_submission_path(code=code, _id=id)
        code = self.isolate_model_path(code=code, _id=id)
        code = pre_code + code

        run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
        run_wd.mkdir(parents=True, exist_ok=True)
        runfile_path = run_wd / self.agent_file_name[process_id]
        runfile_path.write_text(code, encoding="utf-8")
        script_metadata = _introspect_training_script(code)
        script_signature = script_metadata.get("script_signature") or _normalized_mlevolve_script_signature(code)
        detected_batch_size = script_metadata.get("proposed_batch_size") or _detect_initial_batch_size(code)
        proposed_epochs = script_metadata.get("proposed_epochs")
        model_key = script_metadata.get("model_key")
        framework = script_metadata.get("framework")
        uses_amp = script_metadata.get("uses_amp")
        requires_gpu = script_metadata.get("requires_gpu")
        self._pipeline_emit(
            "job_script_created",
            node_id=node_id,
            stage="execution",
            payload={
                "runfile_path": str(runfile_path),
                "submission_slot": process_id,
                "script_signature": script_signature,
                "detected_batch_size": detected_batch_size,
                "proposed_epochs": proposed_epochs,
                "model_key": model_key,
                "framework": framework,
                "uses_amp": uses_amp,
                "requires_gpu": requires_gpu,
            },
        )

        result_dir = run_wd / "working" / "scheduler_results"
        result_path = result_dir / f"result_{id}_{process_id}_{uuid.uuid4().hex}.json"
        scheduler_settings = self.scheduler_client.settings
        submission_defaults = scheduler_settings.gpu_scheduler.submission_defaults
        packing_eligible, packing_backend_allowlist = self._normalized_raw_packing_defaults(submission_defaults)
        runner_kwargs = {
            "script_path": str(runfile_path),
            "working_dir": str(run_wd),
            "result_path": str(result_path),
            "timeout": self.timeout,
            "probe_timeout_seconds": int(submission_defaults.batch_probe_probe_timeout_seconds),
            "probe_poll_interval_seconds": float(submission_defaults.batch_probe_poll_interval_seconds),
        }
        batch_probe_enabled = bool(submission_defaults.batch_probe_enabled) and _code_supports_batch_probe(code)
        if detected_batch_size is not None:
            probe_max_multiplier = max(1, int(submission_defaults.batch_probe_max_multiplier))
            runner_kwargs["batch_size"] = detected_batch_size
            runner_kwargs["probe_max_batch_size"] = max(detected_batch_size, detected_batch_size * probe_max_multiplier)
        task_id = str(getattr(self.cfg, "exp_id", "mlevolve"))
        batch_probe = BatchProbeSpec(
            enabled=batch_probe_enabled,
            probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job" if batch_probe_enabled else None,
            batch_param_name="batch_size",
            model_key=submission_defaults.batch_probe_model_key or f"mlevolve-task:{task_id}",
            search_mode=submission_defaults.batch_probe_search_mode,
            shape_hints={
                "task_id": task_id,
                "script_signature": _normalized_mlevolve_script_signature(code),
            },
        )
        runtime_probe = RuntimeProbeSpec(
            enabled=bool(getattr(submission_defaults, "runtime_probe_enabled", False)),
            probe_target=getattr(submission_defaults, "runtime_probe_target", None),
            model_key=getattr(submission_defaults, "runtime_probe_model_key", None) or f"mlevolve-task:{task_id}",
            strategy=getattr(submission_defaults, "runtime_probe_strategy", "epoch_1"),
        )
        resource_requirements = ResourceRequirements(
            requires_gpu=bool(submission_defaults.requires_gpu),
            estimated_vram_mb=submission_defaults.estimated_vram_mb,
            estimated_ram_mb=submission_defaults.estimated_ram_mb,
        )
        preload_source_payload = _build_scheduler_preload_source(self.scheduler_cfg)
        experiment_mode = str(getattr(getattr(self.cfg, "experiment", None), "mode", "hardware_aware"))
        job_metadata = {
            "mlevolve_node_id": node_id,
            "node_id": node_id,
            "submission_slot": process_id,
            "experiment_mode": experiment_mode,
            "detected_batch_size": detected_batch_size,
            "proposed_batch_size": detected_batch_size,
            "proposed_epochs": proposed_epochs,
            "model_key": model_key,
            "framework": framework,
            "uses_amp": uses_amp,
            "requires_gpu": requires_gpu,
            "script_signature": script_signature,
            "scheduler_mode": getattr(scheduler_settings.gpu_scheduler, "mode", None),
            "batch_probe_enabled": batch_probe_enabled,
            "runtime_probe_enabled": bool(getattr(submission_defaults, "runtime_probe_enabled", False)),
            "packing_eligible": packing_eligible,
            "packing_backend_allowlist": packing_backend_allowlist,
        }
        job = build_mlevolve_job(
            workflow_id=str(getattr(self.cfg, "exp_name", "mlevolve")),
            baseline_model_id=f"mlevolve-script-{id}",
            baseline_model_path=str(runfile_path),
            runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            runner_kwargs=runner_kwargs,
            task_type="mlevolve_script",
            loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            batch_probe=batch_probe,
            runtime_probe=runtime_probe,
            resource_requirements=resource_requirements,
            packing_family=submission_defaults.packing_family,
            packing_eligible=packing_eligible,
            packing_max_slowdown_ratio=submission_defaults.packing_max_slowdown_ratio,
            packing_backend_allowlist=packing_backend_allowlist,
            preload_source=PreloadSource.from_dict(preload_source_payload),
            metadata=job_metadata,
        )
        return _PreparedSchedulerJob(
            node_id=node_id,
            process_id=process_id,
            runfile_path=runfile_path,
            result_path=result_path,
            job=job,
            runner_kwargs=runner_kwargs,
            job_metadata=job_metadata,
            scheduler_mode=getattr(scheduler_settings.gpu_scheduler, "mode", None),
            detected_batch_size=detected_batch_size,
            proposed_epochs=proposed_epochs,
            model_key=model_key,
            framework=framework,
            uses_amp=uses_amp,
            requires_gpu=requires_gpu,
            script_signature=script_signature,
            start_time=start_time,
        )

    def _scheduler_execution_result_from_final(
        self,
        prepared_job: _PreparedSchedulerJob,
        final_job: Any | None,
    ) -> ExecutionResult:
        job_id = prepared_job.job_id
        final_payload = final_job.to_dict() if final_job is not None and hasattr(final_job, "to_dict") else {}
        final_metadata = final_payload.get("metadata") or {}
        resolved_batch_size = (
            final_metadata.get("resolved_batch_size")
            or final_metadata.get("batch_size")
            or prepared_job.runner_kwargs.get("batch_size")
        )
        placement_backend = final_metadata.get("placement_backend") or final_metadata.get("backend_name")
        self._pipeline_upsert_job(
            job_id or f"unknown-{prepared_job.node_id}",
            node_id=prepared_job.node_id,
            scheduler_mode=prepared_job.scheduler_mode,
            placement_mode="scheduler",
            placement_backend=placement_backend,
            status=final_payload.get("status") or (getattr(final_job.status, "value", str(final_job.status)) if final_job is not None else None),
            submitted_at=final_payload.get("submitted_at"),
            started_at=final_payload.get("started_at"),
            finished_at=final_payload.get("finished_at"),
            duration_seconds=time.time() - prepared_job.start_time,
            detected_batch_size=prepared_job.detected_batch_size,
            resolved_batch_size=resolved_batch_size,
            proposed_epochs=prepared_job.proposed_epochs,
            model_key=prepared_job.model_key,
            framework=prepared_job.framework,
            uses_amp=prepared_job.uses_amp,
            requires_gpu=prepared_job.requires_gpu,
            script_signature=prepared_job.script_signature,
            payload=final_payload,
        )
        self._pipeline_emit(
            "job_finished",
            node_id=prepared_job.node_id,
            job_id=job_id,
            stage="execution",
            payload={
                "status": final_payload.get("status"),
                "duration_seconds": time.time() - prepared_job.start_time,
                "placement_backend": placement_backend,
                "resolved_batch_size": resolved_batch_size,
            },
        )
        if prepared_job.result_path.exists():
            payload = json.loads(prepared_job.result_path.read_text(encoding="utf-8"))
            exec_time = float(payload.get("exec_time", time.time() - prepared_job.start_time))
            if job_id is not None:
                self._pipeline_upsert_job(
                    job_id,
                    node_id=prepared_job.node_id,
                    status="result_available",
                    duration_seconds=exec_time,
                    payload={"execution_result": payload},
                )
                self._record_scheduler_tuning_outcome(
                    job_id=job_id,
                    resolved_batch_size=resolved_batch_size,
                    proposed_epochs=prepared_job.proposed_epochs,
                    execution_payload=payload,
                    final_payload=final_payload,
                )
            return ExecutionResult(
                term_out=payload.get("term_out", [""]),
                exec_time=exec_time,
                exc_type=payload.get("exc_type"),
                exc_info=payload.get("exc_info") or {},
                exc_stack=payload.get("exc_stack") or [],
            )

        reason = final_job.status_reason if final_job is not None else "scheduler job finished without result"
        return ExecutionResult(
            term_out=[f"Scheduler job {job_id} finished without an execution result: {reason}\n"],
            exec_time=time.time() - prepared_job.start_time,
            exc_type="RuntimeError",
            exc_info={"message": reason, "job_id": job_id},
            exc_stack=[],
        )

    def _record_scheduler_tuning_outcome(
        self,
        *,
        job_id: str,
        resolved_batch_size: Any,
        proposed_epochs: Any,
        execution_payload: dict[str, Any],
        final_payload: dict[str, Any],
    ) -> None:
        if self.scheduler_client is None or not hasattr(self.scheduler_client, "record_tuning_outcome"):
            return
        try:
            chosen_batch_size = int(resolved_batch_size) if resolved_batch_size is not None else None
        except (TypeError, ValueError):
            chosen_batch_size = None
        try:
            chosen_epochs = int(proposed_epochs) if proposed_epochs is not None else None
        except (TypeError, ValueError):
            chosen_epochs = None
        try:
            self.scheduler_client.record_tuning_outcome(
                job_id=job_id,
                chosen_batch_size=chosen_batch_size,
                chosen_epochs=chosen_epochs,
                recommendation_source="scheduler_round",
                outcome_metrics={
                    "exec_time": execution_payload.get("exec_time"),
                    "exc_type": execution_payload.get("exc_type"),
                    "status": final_payload.get("status"),
                    "resolved_batch_size": chosen_batch_size,
                },
                notes="recorded from MLEvolve scheduler round execution",
            )
        except Exception as exc:
            logger.debug("Skipping scheduler tuning outcome record for %s: %s", job_id, exc)

    def _run_scheduler_job(self, code: str, id, working_dir: str | None = None) -> ExecutionResult:
        """Submit generated code as a localml_scheduler job and wait for its result."""
        from localml_scheduler.adapters.mlevolve import build_mlevolve_job
        from localml_scheduler.domain import BatchProbeSpec, PreloadSource, ResourceRequirements, RuntimeProbeSpec

        if self.scheduler_client is None:
            return self._run_subprocess(code=code, id=id, working_dir=working_dir)

        logger.info("REPL is submitting code to localml_scheduler")
        process_id = None
        job_id = None
        runfile_path = None
        start_time = time.time()

        try:
            self._ensure_scheduler_service_available()
            with self.lock:
                self.current_parallel_run += 1
                for idx in range(self.max_parallel_run):
                    if self.status_map[idx] == 0:
                        self.status_map[idx] = 1
                        process_id = idx
                        logger.info(f"Assigned scheduler submission slot: {process_id}")
                        break
                    elif idx == self.max_parallel_run - 1:
                        logger.info("reach max process parallel number")
                        raise ValueError("reach max process parallel number")

            cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
            avail_cpus = self._available_cpus()
            start = process_id * cpu_number_per_session
            cpu_set = set(avail_cpus[start:start + cpu_number_per_session])
            if not cpu_set:
                cpu_set = set(avail_cpus)
            pre_code = ""
            if hasattr(os, "sched_setaffinity"):
                pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)

            code = self.isolate_submission_path(code=code, _id=id)
            code = self.isolate_model_path(code=code, _id=id)
            code = pre_code + code

            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            run_wd.mkdir(parents=True, exist_ok=True)
            runfile_path = run_wd / self.agent_file_name[process_id]
            runfile_path.write_text(code, encoding="utf-8")
            script_metadata = _introspect_training_script(code)
            script_signature = script_metadata.get("script_signature") or _normalized_mlevolve_script_signature(code)
            detected_batch_size = script_metadata.get("proposed_batch_size")
            proposed_epochs = script_metadata.get("proposed_epochs")
            model_key = script_metadata.get("model_key")
            framework = script_metadata.get("framework")
            uses_amp = script_metadata.get("uses_amp")
            requires_gpu = script_metadata.get("requires_gpu")
            self._pipeline_emit(
                "job_script_created",
                node_id=str(id),
                stage="execution",
                payload={
                    "runfile_path": str(runfile_path),
                    "submission_slot": process_id,
                    "script_signature": script_signature,
                    "detected_batch_size": detected_batch_size,
                    "proposed_epochs": proposed_epochs,
                    "model_key": model_key,
                    "framework": framework,
                    "uses_amp": uses_amp,
                    "requires_gpu": requires_gpu,
                },
            )

            result_dir = run_wd / "working" / "scheduler_results"
            result_path = result_dir / f"result_{id}_{process_id}_{uuid.uuid4().hex}.json"
            scheduler_cfg = self.scheduler_cfg
            scheduler_settings = self.scheduler_client.settings
            submission_defaults = scheduler_settings.gpu_scheduler.submission_defaults
            packing_eligible, packing_backend_allowlist = self._normalized_raw_packing_defaults(submission_defaults)
            runner_kwargs = {
                "script_path": str(runfile_path),
                "working_dir": str(run_wd),
                "result_path": str(result_path),
                "timeout": self.timeout,
                "probe_timeout_seconds": int(submission_defaults.batch_probe_probe_timeout_seconds),
                "probe_poll_interval_seconds": float(submission_defaults.batch_probe_poll_interval_seconds),
            }
            batch_probe_enabled = bool(submission_defaults.batch_probe_enabled) and _code_supports_batch_probe(code)
            detected_batch_size = detected_batch_size or _detect_initial_batch_size(code)
            if detected_batch_size is not None:
                probe_max_multiplier = max(1, int(submission_defaults.batch_probe_max_multiplier))
                runner_kwargs["batch_size"] = detected_batch_size
                runner_kwargs["probe_max_batch_size"] = max(
                    detected_batch_size,
                    detected_batch_size * probe_max_multiplier,
                )
            task_id = str(getattr(self.cfg, "exp_id", "mlevolve"))
            batch_probe = BatchProbeSpec(
                enabled=batch_probe_enabled,
                probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job" if batch_probe_enabled else None,
                batch_param_name="batch_size",
                model_key=submission_defaults.batch_probe_model_key or f"mlevolve-task:{task_id}",
                search_mode=submission_defaults.batch_probe_search_mode,
                shape_hints={
                    "task_id": task_id,
                    "script_signature": _normalized_mlevolve_script_signature(code),
                },
            )
            runtime_probe = RuntimeProbeSpec(
                enabled=bool(getattr(submission_defaults, "runtime_probe_enabled", False)),
                probe_target=getattr(submission_defaults, "runtime_probe_target", None),
                model_key=getattr(submission_defaults, "runtime_probe_model_key", None) or f"mlevolve-task:{task_id}",
                strategy=getattr(submission_defaults, "runtime_probe_strategy", "epoch_1"),
            )
            resource_requirements = ResourceRequirements(
                requires_gpu=bool(submission_defaults.requires_gpu),
                estimated_vram_mb=submission_defaults.estimated_vram_mb,
                estimated_ram_mb=submission_defaults.estimated_ram_mb,
            )
            preload_source_payload = _build_scheduler_preload_source(scheduler_cfg)
            experiment_mode = str(getattr(getattr(self.cfg, "experiment", None), "mode", "hardware_aware"))
            job_metadata = {
                "mlevolve_node_id": str(id),
                "submission_slot": process_id,
                "experiment_mode": experiment_mode,
                "detected_batch_size": detected_batch_size,
                "proposed_epochs": proposed_epochs,
                "model_key": model_key,
                "framework": framework,
                "uses_amp": uses_amp,
                "requires_gpu": requires_gpu,
                "script_signature": script_signature,
                "scheduler_mode": getattr(scheduler_settings.gpu_scheduler, "mode", None),
                "batch_probe_enabled": batch_probe_enabled,
                "runtime_probe_enabled": bool(getattr(submission_defaults, "runtime_probe_enabled", False)),
                "packing_eligible": packing_eligible,
                "packing_backend_allowlist": packing_backend_allowlist,
            }
            job = build_mlevolve_job(
                workflow_id=str(getattr(self.cfg, "exp_name", "mlevolve")),
                baseline_model_id=f"mlevolve-script-{id}",
                baseline_model_path=str(runfile_path),
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs=runner_kwargs,
                task_type="mlevolve_script",
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                batch_probe=batch_probe,
                runtime_probe=runtime_probe,
                resource_requirements=resource_requirements,
                packing_family=submission_defaults.packing_family,
                packing_eligible=packing_eligible,
                packing_max_slowdown_ratio=submission_defaults.packing_max_slowdown_ratio,
                packing_backend_allowlist=packing_backend_allowlist,
                preload_source=PreloadSource.from_dict(preload_source_payload),
                metadata=job_metadata,
            )
            submitted = self.scheduler_client.submit(job)
            job_id = submitted.job_id
            self._pipeline_upsert_job(
                job_id,
                node_id=str(id),
                scheduler_mode=getattr(scheduler_settings.gpu_scheduler, "mode", None),
                placement_mode="scheduler",
                placement_backend=None,
                status=getattr(submitted.status, "value", str(submitted.status)),
                submitted_at=getattr(submitted, "submitted_at", None),
                detected_batch_size=detected_batch_size,
                resolved_batch_size=None,
                proposed_epochs=proposed_epochs,
                model_key=model_key,
                framework=framework,
                uses_amp=uses_amp,
                requires_gpu=requires_gpu,
                script_signature=script_signature,
                payload=submitted.to_dict() if hasattr(submitted, "to_dict") else {},
            )
            self._pipeline_emit(
                "scheduler_submission_created",
                node_id=str(id),
                job_id=job_id,
                stage="execution",
                payload=job_metadata,
            )
            with self._scheduler_jobs_lock:
                self._scheduler_job_ids.add(job_id)
            logger.info(f"Submitted scheduler job {job_id} for node {id}")

            wait_timeout = getattr(scheduler_cfg, "wait_timeout_seconds", None)
            if wait_timeout is None:
                wait_timeout = self.timeout * max(1, self.max_parallel_run) + 60
            poll_interval = max(0.1, float(getattr(scheduler_cfg, "wait_poll_interval_seconds", 1.0)))
            deadline = time.time() + float(wait_timeout)
            final_job = None
            last_probe_event_id = 0
            while time.time() < deadline:
                final_job = self.scheduler_client.inspect(job_id)
                last_probe_event_id = self._log_scheduler_probe_updates(job_id, last_probe_event_id)
                if final_job is not None and final_job.status.is_terminal:
                    break
                time.sleep(poll_interval)
            else:
                self.scheduler_client.cancel(job_id)
                exec_time = time.time() - start_time
                self._pipeline_upsert_job(
                    job_id,
                    node_id=str(id),
                    status="timeout",
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    duration_seconds=exec_time,
                    detected_batch_size=detected_batch_size,
                    proposed_epochs=proposed_epochs,
                    model_key=model_key,
                    framework=framework,
                    uses_amp=uses_amp,
                    requires_gpu=requires_gpu,
                    script_signature=script_signature,
                    payload={"reason": "scheduler wait timeout", "wait_timeout_seconds": wait_timeout},
                )
                return ExecutionResult(
                    term_out=[
                        f"Execution time: TimeoutError: Scheduler job {job_id} exceeded wait limit of {humanize.naturaldelta(wait_timeout)}"
                    ],
                    exec_time=exec_time,
                    exc_type="TimeoutError",
                    exc_info={"message": "scheduler wait timeout", "job_id": job_id},
                    exc_stack=[],
                )

            last_probe_event_id = self._log_scheduler_probe_updates(job_id, last_probe_event_id)
            final_payload = final_job.to_dict() if final_job is not None and hasattr(final_job, "to_dict") else {}
            final_metadata = final_payload.get("metadata") or {}
            resolved_batch_size = (
                final_metadata.get("resolved_batch_size")
                or final_metadata.get("batch_size")
                or runner_kwargs.get("batch_size")
            )
            placement_backend = final_metadata.get("placement_backend") or final_metadata.get("backend_name")
            self._pipeline_upsert_job(
                job_id,
                node_id=str(id),
                scheduler_mode=getattr(scheduler_settings.gpu_scheduler, "mode", None),
                placement_mode="scheduler",
                placement_backend=placement_backend,
                status=final_payload.get("status") or (getattr(final_job.status, "value", str(final_job.status)) if final_job is not None else None),
                submitted_at=final_payload.get("submitted_at"),
                started_at=final_payload.get("started_at"),
                finished_at=final_payload.get("finished_at"),
                duration_seconds=time.time() - start_time,
                detected_batch_size=detected_batch_size,
                resolved_batch_size=resolved_batch_size,
                proposed_epochs=proposed_epochs,
                model_key=model_key,
                framework=framework,
                uses_amp=uses_amp,
                requires_gpu=requires_gpu,
                script_signature=script_signature,
                payload=final_payload,
            )
            self._pipeline_emit(
                "job_finished",
                node_id=str(id),
                job_id=job_id,
                stage="execution",
                payload={
                    "status": final_payload.get("status"),
                    "duration_seconds": time.time() - start_time,
                    "placement_backend": placement_backend,
                    "resolved_batch_size": resolved_batch_size,
                },
            )

            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                self._pipeline_upsert_job(
                    job_id,
                    node_id=str(id),
                    status="result_available",
                    duration_seconds=float(payload.get("exec_time", time.time() - start_time)),
                    payload={"execution_result": payload},
                )
                return ExecutionResult(
                    term_out=payload.get("term_out", [""]),
                    exec_time=float(payload.get("exec_time", time.time() - start_time)),
                    exc_type=payload.get("exc_type"),
                    exc_info=payload.get("exc_info") or {},
                    exc_stack=payload.get("exc_stack") or [],
                )

            reason = final_job.status_reason if final_job is not None else "scheduler job finished without result"
            return ExecutionResult(
                term_out=[f"Scheduler job {job_id} finished without an execution result: {reason}\n"],
                exec_time=time.time() - start_time,
                exc_type="RuntimeError",
                exc_info={"message": reason, "job_id": job_id},
                exc_stack=[],
            )
        except Exception as e:
            logger.error(f"Error in _run_scheduler_job: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            if job_id is not None:
                self._pipeline_upsert_job(
                    job_id,
                    node_id=str(id),
                    status="executor_error",
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    duration_seconds=time.time() - start_time,
                    payload={"error": str(e), "traceback": error_trace},
                )
            return ExecutionResult(
                term_out=[f"Scheduler execution error: {str(e)}", error_trace],
                exec_time=time.time() - start_time,
                exc_type="RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
        finally:
            if job_id is not None:
                with self._scheduler_jobs_lock:
                    self._scheduler_job_ids.discard(job_id)
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove scheduler runfile after execution: {e}")
            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1

    def _run_subprocess(self, code: str, id, working_dir: str | None = None):
        """
        Execute code via subprocess (avoids CUDA fork issues).
        Aligned with multiprocessing mode for consistency.
        """
        logger.info("REPL is executing code via subprocess")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None

        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    logger.info(f"Assigned process_id: {process_id}")
                    break
                elif idx == self.max_parallel_run - 1:
                    logger.info("reach max process parallel number")
                    raise ValueError("reach max process parallel number")

        start_time = time.time()
        runfile_path = None
        proc = None
        
        try:
            cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
            avail_cpus = self._available_cpus()
            start = process_id * cpu_number_per_session
            cpu_set = set(avail_cpus[start:start + cpu_number_per_session])
            if not cpu_set:
                cpu_set = set(avail_cpus)
            logger.info(f"has set process_id:{process_id} to use cpu: {cpu_set}")
            pre_code = ""
            if hasattr(os, "sched_setaffinity"):
                pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)

            code = self.isolate_submission_path(code=code, _id=id)
            code = self.isolate_model_path(code=code, _id=id)
            code = pre_code + code

            # decide runfile location and cwd
            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            runfile_path = run_wd / self.agent_file_name[process_id]
            run_wd.mkdir(parents=True, exist_ok=True)
            with open(runfile_path, "w") as f:
                f.write(code)

            cmd = [sys.executable, str(runfile_path)]
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_wd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            with self._procs_lock:
                self._active_procs[process_id] = proc

            child_in_overtime = False
            exc_type = None
            exc_info = {}
            exc_stack = []
            
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                exec_time = time.time() - start_time
                
                if proc.returncode != 0:
                    exc_type = "RuntimeError"
                    exc_info = {}
                    exc_stack = []
                    
                    if stderr:
                        stderr_text = stderr
                        exc_patterns = [
                            ("KeyboardInterrupt", "KeyboardInterrupt"),
                            ("TimeoutError", "TimeoutError"),
                            ("CUDA", "RuntimeError"),
                            ("cuda", "RuntimeError"),
                            ("ValueError", "ValueError"),
                            ("TypeError", "TypeError"),
                            ("AttributeError", "AttributeError"),
                            ("KeyError", "KeyError"),
                            ("IndexError", "IndexError"),
                            ("FileNotFoundError", "FileNotFoundError"),
                            ("ImportError", "ImportError"),
                            ("AssertionError", "AssertionError"),
                            ("NameError", "NameError"),
                            ("RuntimeError", "RuntimeError"),
                        ]
                        
                        for pattern, exc_name in exc_patterns:
                            if pattern in stderr_text:
                                exc_type = exc_name
                                break
                        
                        stderr_lines = stderr_text.splitlines()
                        for line in stderr_lines:
                            if 'File "' in line and 'line' in line:
                                try:
                                    file_start = line.find('File "') + 6
                                    file_end = line.find('"', file_start)
                                    if file_end > file_start:
                                        filename = line[file_start:file_end]
                                        line_start = line.find('line ') + 5
                                        line_end = line.find(',', line_start)
                                        if line_end == -1:
                                            line_end = line.find('\n', line_start)
                                        if line_end == -1:
                                            line_end = len(line)
                                        line_num_str = line[line_start:line_end].strip()
                                        if line_num_str.isdigit():
                                            line_num = int(line_num_str)
                                            func_name = ""
                                            if 'in ' in line:
                                                func_start = line.find('in ') + 3
                                                func_end = line.find('\n', func_start)
                                                if func_end == -1:
                                                    func_end = len(line)
                                                func_name = line[func_start:func_end].strip()
                                            
                                            filename_short = filename.replace(str(self.working_dir / self.agent_file_name[process_id]), self.agent_file_name[process_id])
                                            filename_short = os.path.basename(filename_short)
                                            exc_stack.append((filename_short, line_num, func_name, ""))
                                except Exception:
                                    pass
                        
                        for line in reversed(stderr_lines):
                            line = line.strip()
                            if line and not line.startswith("File") and not line.startswith("Traceback"):
                                if ":" in line:
                                    parts = line.split(":", 1)
                                    if len(parts) == 2:
                                        exc_info["message"] = parts[1].strip()
                                    break
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess timeout, sending SIGINT...")
                try:
                    proc.send_signal(signal.SIGINT)
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("Subprocess failed to terminate after SIGINT, killing...")
                    proc.kill()
                    stdout, stderr = proc.communicate()
                
                exec_time = time.time() - start_time
                exc_type = "TimeoutError"
                exc_info = {}
                exc_stack = []
            
            output: list[str] = []
            if stdout:
                output.extend(stdout.splitlines(keepends=True))
            if stderr:
                output.extend(stderr.splitlines(keepends=True))
            if not output:
                output = [""]
            if output and output[-1] and not output[-1].endswith("\n"):
                output.append("\n")

            if exc_type == "TimeoutError":
                output.append(
                    f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
                )
            
            return ExecutionResult(output, exec_time, exc_type, exc_info, exc_stack)
            
        except Exception as e:
            logger.error(f"Error in _run_subprocess: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            exec_time = time.time() - start_time if start_time else 0
            return ExecutionResult(
                term_out=[f"Subprocess execution error: {str(e)}", error_trace],
                exec_time=exec_time,
                exc_type="RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
        finally:
            if process_id is not None:
                with self._procs_lock:
                    self._active_procs.pop(process_id, None)
            if proc is not None:
                try:
                    if proc.poll() is None:
                        logger.warning(f"Subprocess {process_id} still running, terminating...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Subprocess {process_id} failed to terminate, killing...")
                            proc.kill()
                            proc.wait()
                except Exception as e:
                    logger.warning(f"Error cleaning up subprocess {process_id}: {e}")
            
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove runfile after subprocess execution: {e}")
            
            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1
