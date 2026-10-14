"""Fast runner used by scheduler timeline replay smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from localml_scheduler.execution.runner_protocol import RunnerContext


def load_noop_baseline(path: str) -> bytes:
    """Return a lightweight baseline object for scheduler cache preload checks."""
    return b"scheduler replay noop baseline"


def run_noop_job(context: RunnerContext) -> dict[str, Any]:
    """Complete a scheduler job without running the original training script."""
    sleep_seconds = float(context.job.config.runner_kwargs.get("noop_sleep_seconds") or 0.01)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    result_path = context.job.config.runner_kwargs.get("result_path")
    if result_path:
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "term_out": ["scheduler replay noop\n"],
                    "exec_time": sleep_seconds,
                    "exc_type": None,
                    "exc_info": {},
                    "exc_stack": [],
                    "phase_timings": {
                        "phase_durations_seconds": {
                            "training": sleep_seconds,
                            "inference": 0.0,
                            "validation": 0.0,
                            "other_candidate": 0.0,
                        },
                        "phase_timing_available": True,
                        "phase_instrumented": False,
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return {
        "reason": "scheduler replay noop",
        "execution_result_path": str(result_path) if result_path else None,
        "candidate_returncode": 0,
        "replay_runner_mode": "noop",
    }
