"""Stable ML-Master hook surface for optional shared components."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


class PluginManager:
    def __init__(self, cfg: Any, *, enabled: set[str] | None = None) -> None:
        self.cfg = cfg
        self.enabled = enabled if enabled is not None else {
            item.strip().lower()
            for item in os.environ.get("LIGHTNING_PLUGINS", "").split(",")
            if item.strip()
        }
        unknown = self.enabled - {"preflight", "scheduler", "predictor", "hwkd"}
        if unknown:
            raise ValueError(f"Unknown Lightning plugins: {', '.join(sorted(unknown))}")
        self._scheduler: Any = None
        self._predictor: Any = None

    def enrich_task(self, description: Any) -> Any:
        if "hwkd" not in self.enabled:
            return description
        from .hwkd import HardwareKnowledgePlugin

        return HardwareKnowledgePlugin().enrich_task(description)

    def hardware_guidance(
        self,
        stage: str,
        task_description: str,
        candidate_code: str = "",
        **context: Any,
    ) -> str | None:
        if "hwkd" not in self.enabled or not os.environ.get("MLEVOLVE_HWKG_GRAPH_PATH"):
            return None
        from .hwkd import HardwareKnowledgePlugin

        return HardwareKnowledgePlugin().guidance(
            stage, task_description, candidate_code, **context
        )

    def hardware_audit(self, record: dict[str, Any]) -> None:
        if "hwkd" not in self.enabled or not os.environ.get("MLEVOLVE_HWKG_GRAPH_PATH"):
            return
        from HWKD.hwkg.routing import _audit

        _audit(record)

    def execute(
        self,
        code: str,
        node_id: str,
        default_execute: Callable[[], Any],
    ) -> Any:
        if "preflight" in self.enabled:
            from .preflight import PreflightPlugin

            outcome = PreflightPlugin(Path(self.cfg.workspace_dir)).check(code, node_id)
            if not outcome["admitted"]:
                from interpreter.interpreter_parallel import ExecutionResult

                messages = [
                    f"Preflight rejected candidate; report: {outcome['report_path']}",
                    *[
                        str(item.get("message") or item.get("code"))
                        for item in outcome["diagnostics"]
                    ],
                ]
                return ExecutionResult(
                    term_out=messages,
                    exec_time=0.0,
                    exc_type="PreflightRejected",
                    exc_info={"report_path": outcome["report_path"]},
                    exc_stack=[],
                )
        prediction = None
        if "predictor" in self.enabled:
            if self._predictor is None:
                from .predictor import PredictorPlugin

                self._predictor = PredictorPlugin(Path(self.cfg.workspace_dir))
            prediction = self._predictor.inspect_candidate(code, node_id)
        if "scheduler" in self.enabled:
            if self._scheduler is None:
                from .scheduler import SchedulerPlugin

                self._scheduler = SchedulerPlugin(self.cfg)
            return self._scheduler.execute(code, node_id, prediction=prediction)
        return default_execute()

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
