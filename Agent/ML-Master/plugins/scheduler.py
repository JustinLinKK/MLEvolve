"""Submit ML-Master scripts through the shared scheduler's raw-script runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any

from .paths import MLEVOLVE_ROOT, MLMASTER_ROOT, expose_components


class SchedulerPlugin:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.workspace = Path(cfg.workspace_dir).resolve()
        self.client: Any = None
        self.service: Any = None

    def _ensure_client(self) -> Any:
        if self.client is not None:
            return self.client
        root = expose_components()
        # The scheduler launches workers in new processes. Preserve access to
        # its package after the ML-Master entry process leaves this scope.
        inherited = os.environ.get("PYTHONPATH", "")
        parts = [str(MLEVOLVE_ROOT), str(MLMASTER_ROOT), str(root)]
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [*parts, *[part for part in inherited.split(os.pathsep) if part and part not in parts]]
        )
        from localml_scheduler.client import SchedulerClient
        from localml_scheduler.config import SchedulerConfig

        settings = SchedulerConfig(
            runtime_root=self.workspace / "working" / "scheduler_runtime"
        )
        settings.gpu_scheduler.device_index = int(
            os.environ.get("LIGHTNING_SCHEDULER_GPU", "0")
        )
        # Incremental admission determines concurrency; no job-count cap.
        settings.gpu_scheduler.parallel_job_cap = None
        self.client = SchedulerClient(settings)
        if not self.client.scheduler_service_active():
            self.service = self.client.create_service().start(background=True)
            deadline = time.monotonic() + 10.0
            while not self.client.scheduler_service_active():
                if time.monotonic() >= deadline:
                    self.service.stop()
                    self.service = None
                    raise RuntimeError("Scheduler service did not become ready")
                time.sleep(0.1)
        return self.client

    def execute(
        self,
        code: str,
        node_id: str,
        *,
        prediction: dict[str, float] | None = None,
    ):
        from interpreter.interpreter_parallel import ExecutionResult, Interpreter

        client = self._ensure_client()
        from localml_scheduler.adapters.mlevolve import build_mlevolve_job

        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(node_id))
        source_dir = self.workspace / "working" / "scheduler_sources"
        result_dir = self.workspace / "working" / "scheduler_results"
        source_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        source = source_dir / f"runfile_{safe_id}.py"
        result_path = result_dir / f"result_{safe_id}.json"
        source.write_text(
            Interpreter.replace_submission_name(code, node_id), encoding="utf-8"
        )
        job = build_mlevolve_job(
            workflow_id=str(self.cfg.exp_name),
            baseline_model_id=f"ml-master:{safe_id}",
            baseline_model_path=str(source),
            runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            runner_kwargs={
                "script_path": str(source),
                "working_dir": str(self.workspace),
                "result_path": str(result_path),
                "timeout": int(self.cfg.exec.timeout),
            },
            task_type="ml_master_candidate",
            packing_eligible=False,
            metadata={
                "node_id": str(node_id),
                "supports_checkpoint_resume": False,
                **({"perfseer_a10_advisory": prediction} if prediction else {}),
            },
        )
        submitted = client.submit(job)
        while True:
            state = client.inspect(submitted.job_id)
            if state is None:
                raise RuntimeError(f"Scheduler lost ML-Master job {submitted.job_id}")
            if state.status.value in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            if not client.scheduler_service_active():
                raise RuntimeError("Scheduler service stopped while ML-Master job was queued")
            time.sleep(0.5)
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            return ExecutionResult(
                term_out=list(result.get("term_out") or []),
                exec_time=float(result.get("exec_time") or 0.0),
                exc_type=result.get("exc_type"),
                exc_info=result.get("exc_info"),
                exc_stack=result.get("exc_stack"),
            )
        raise RuntimeError(
            f"Scheduler job {submitted.job_id} ended {state.status.value}: {state.status_reason}"
        )

    def close(self) -> None:
        if self.service is not None:
            self.service.stop()
            self.service = None
