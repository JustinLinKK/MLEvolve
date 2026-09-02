from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

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


class _RoundInterpreter:
    max_parallel_run = 2

    def __init__(self) -> None:
        self.packet_sizes: list[int] = []

    def run(self, *args, **kwargs):
        raise AssertionError("deferred scheduler generation must not execute directly")

    def run_many(self, items):
        self.packet_sizes.append(len(items))
        return {str(item["id"]): object() for item in items}


class _RoundAgent:
    def __init__(self, journal, candidate_count: int, skipped_steps: int = 0) -> None:
        self.journal = journal
        self.remaining = candidate_count
        self.skipped_steps = skipped_steps

    def has_selectable_work(self) -> bool:
        return self.remaining > 0

    def step(self, *, exec_callback, node, execute_immediately):
        assert execute_immediately is False
        assert node is None
        if self.skipped_steps:
            self.skipped_steps -= 1
            return None
        self.remaining -= 1
        return SimpleNamespace(
            id=f"candidate-{self.remaining}",
            stage="draft",
            is_buggy=False,
            exec_time=120.0,
        )

    def execute_deferred_nodes(self, nodes, exec_many_callback):
        exec_many_callback([{"id": node.id, "code": "pass"} for node in nodes])
        self.journal.nodes.extend(nodes)
        return nodes


def test_scheduler_submits_each_candidate_as_soon_as_generation_finishes() -> None:
    runner = getattr(run_module, "_run_scheduler_rounds", None)
    assert runner is not None

    journal = SimpleNamespace(
        nodes=[SimpleNamespace(stage="root", is_buggy=False, exec_time=None)]
    )
    agent = _RoundAgent(journal, candidate_count=5)
    interpreter = _RoundInterpreter()
    saves: list[int] = []

    completed = runner(
        agent=agent,
        interpreter=interpreter,
        cfg=SimpleNamespace(agent=SimpleNamespace(steps=5)),
        journal=journal,
        logger=logging.getLogger("test-scheduler-round"),
        save_callback=lambda cfg, current_journal: saves.append(len(current_journal.nodes)),
    )

    assert completed == 5
    assert interpreter.packet_sizes == [1, 1, 1, 1, 1]
    assert saves == [2, 3, 4, 5, 6]


def test_scheduler_round_retries_a_transient_generation_skip() -> None:
    journal = SimpleNamespace(
        nodes=[SimpleNamespace(stage="root", is_buggy=False, exec_time=None)]
    )
    agent = _RoundAgent(journal, candidate_count=1, skipped_steps=1)
    interpreter = _RoundInterpreter()

    completed = run_module._run_scheduler_rounds(
        agent=agent,
        interpreter=interpreter,
        cfg=SimpleNamespace(agent=SimpleNamespace(steps=1)),
        journal=journal,
        logger=logging.getLogger("test-scheduler-round-retry"),
        save_callback=lambda cfg, current_journal: None,
    )

    assert completed == 1
    assert interpreter.packet_sizes == [1]


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
