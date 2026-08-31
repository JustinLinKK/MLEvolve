from __future__ import annotations

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import run as run_module


class _StopAfterContext(RuntimeError):
    pass


class _PipelineLogger:
    def emit(self, *args, **kwargs):
        return None

    def record_run_metrics(self, *args, **kwargs):
        return None

    def close(self):
        return None


class _HardwareMonitor:
    def start(self):
        return None

    def stop(self):
        return None


class _Agent:
    metric_maximize = False

    def close_cuda_docs(self):
        return None


def test_run_prepares_context_cache_once_before_interpreter(monkeypatch, tmp_path: Path) -> None:
    """A run must not duplicate context-cache preparation side effects."""

    cfg = OmegaConf.create(
        {
            "torch_hub_dir": "",
            "exp_name": "context-once",
            "exp_id": "petfinder-pawpularity-score",
            "experiment": {"mode": "hardware_aware"},
            "scheduler": {"enabled": False},
            "agent": {"seed": 42},
            "coldstart": {"use_coldstart": False},
            "resume_journal": None,
            "workspace_dir": tmp_path / "workspace",
            "log_dir": tmp_path / "logs",
            "exec": {},
        },
        flags={"allow_objects": True},
    )
    calls: list[str] = []

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(run_module, "set_global_seed", lambda seed: None)
    monkeypatch.setattr(run_module, "setup_logging", lambda cfg: logging.getLogger("test-run"))
    monkeypatch.setattr(run_module, "HardwareMonitor", lambda cfg, logger: _HardwareMonitor())
    monkeypatch.setattr(run_module, "PipelineActionLogger", lambda *args, **kwargs: _PipelineLogger())
    monkeypatch.setattr(run_module, "load_task_desc", lambda cfg: "task")
    monkeypatch.setattr(run_module, "prep_agent_workspace", lambda cfg: Path(cfg.workspace_dir).mkdir(parents=True))
    monkeypatch.setattr(run_module, "Agent", lambda **kwargs: _Agent())
    monkeypatch.setattr(run_module, "build_comparison_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_module, "write_comparison_metrics", lambda *args, **kwargs: None)

    def prepare_once(cfg):
        calls.append(cfg.exp_name)
        return []

    monkeypatch.setattr("context_cache.coordinator.prepare_run_context_cache", prepare_once)
    monkeypatch.setattr(
        run_module,
        "Interpreter",
        lambda *args, **kwargs: (_ for _ in ()).throw(_StopAfterContext()),
    )

    with pytest.raises(_StopAfterContext):
        run_module.run()

    assert calls == ["context-once"]
