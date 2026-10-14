from __future__ import annotations

import logging
import threading
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


def test_pipeline_sqlite_can_use_runtime_storage_and_is_persisted(tmp_path, monkeypatch) -> None:
    durable_root = tmp_path / "durable"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("MLEVOLVE_PIPELINE_SQLITE_DIR", str(runtime_root))

    active, durable = run_module._pipeline_database_path(
        SimpleNamespace(log_dir=durable_root)
    )

    assert active == runtime_root / "pipeline.sqlite3"
    assert durable == durable_root / "pipeline.sqlite3"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"sqlite-state")
    run_module._persist_pipeline_database(active, durable)
    assert durable.read_bytes() == b"sqlite-state"


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


def test_scheduler_expands_draft_budget_when_search_tree_is_exhausted() -> None:
    class ExhaustedAgent:
        def __init__(self) -> None:
            self.scfg = SimpleNamespace(num_drafts=5)

        def has_selectable_work(self) -> bool:
            return self.scfg.num_drafts >= 50

    agent = ExhaustedAgent()
    cfg = SimpleNamespace(agent=SimpleNamespace(search=SimpleNamespace(num_drafts=5)))

    expanded = run_module._ensure_scheduler_generation_capacity(
        agent=agent,
        cfg=cfg,
        total_steps=50,
        logger=logging.getLogger("test-scheduler-capacity"),
    )

    assert expanded is True
    assert cfg.agent.search.num_drafts == 50


def test_scheduler_does_not_expand_root_drafts_while_a_candidate_is_running() -> None:
    class WaitingForProfileAgent:
        def __init__(self) -> None:
            self.scfg = SimpleNamespace(num_drafts=5)

        def has_selectable_work(self) -> bool:
            return False

    agent = WaitingForProfileAgent()
    cfg = SimpleNamespace(agent=SimpleNamespace(search=SimpleNamespace(num_drafts=5)))

    expanded = run_module._ensure_scheduler_generation_capacity(
        agent=agent,
        cfg=cfg,
        total_steps=50,
        inflight_count=1,
        logger=logging.getLogger("test-scheduler-capacity"),
    )

    assert expanded is False
    assert cfg.agent.search.num_drafts == 5


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
    assert len(saves) == 5
    assert saves == sorted(saves)
    assert saves[-1] == 6


def test_scheduler_submits_pre_generated_drafts_as_one_scheduler_packet() -> None:
    journal = SimpleNamespace(
        nodes=[SimpleNamespace(stage="root", is_buggy=False, exec_time=None)]
    )
    agent = _RoundAgent(journal, candidate_count=0)
    interpreter = _RoundInterpreter()
    drafts = [
        SimpleNamespace(id="draft-1", stage="model", is_buggy=False, exec_time=120.0),
        SimpleNamespace(id="draft-2", stage="model", is_buggy=False, exec_time=120.0),
    ]

    completed = run_module._run_scheduler_rounds(
        agent=agent,
        interpreter=interpreter,
        cfg=SimpleNamespace(agent=SimpleNamespace(steps=2)),
        journal=journal,
        logger=logging.getLogger("test-scheduler-pre-generated"),
        save_callback=lambda cfg, current_journal: None,
        initial_candidates=drafts,
    )

    assert completed == 2
    # Initial drafts are generated before execution specifically so the
    # Scheduler can compare them in one admission decision.  Sending one
    # draft at a time forces the first workload into a solo placement before
    # its contemporaries are visible.
    assert interpreter.packet_sizes == [2]


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


def test_scheduler_generation_overlaps_an_inflight_execution() -> None:
    journal = SimpleNamespace(
        nodes=[SimpleNamespace(stage="root", is_buggy=False, exec_time=None)]
    )
    agent = _RoundAgent(journal, candidate_count=2)
    interpreter = _RoundInterpreter()
    second_candidate_generated = threading.Event()
    original_step = agent.step
    original_execute = agent.execute_deferred_nodes
    generated = 0

    def step(**kwargs):
        nonlocal generated
        node = original_step(**kwargs)
        if node is not None:
            generated += 1
            if generated == 2:
                second_candidate_generated.set()
        return node

    first_execution_observed_overlap: list[bool] = []

    def execute_deferred_nodes(nodes, exec_many_callback):
        if nodes[0].id == "candidate-1":
            first_execution_observed_overlap.append(
                second_candidate_generated.wait(timeout=1.0)
            )
        return original_execute(nodes, exec_many_callback)

    agent.step = step
    agent.execute_deferred_nodes = execute_deferred_nodes

    completed = run_module._run_scheduler_rounds(
        agent=agent,
        interpreter=interpreter,
        cfg=SimpleNamespace(agent=SimpleNamespace(steps=2)),
        journal=journal,
        logger=logging.getLogger("test-scheduler-overlap"),
        save_callback=lambda cfg, current_journal: None,
    )

    assert completed == 2
    assert first_execution_observed_overlap == [True]
    assert interpreter.packet_sizes == [1, 1]


def test_scheduler_checkpoints_a_completed_node_while_next_generation_is_running() -> None:
    journal = SimpleNamespace(
        nodes=[SimpleNamespace(stage="root", is_buggy=False, exec_time=None)]
    )
    agent = _RoundAgent(journal, candidate_count=2)
    interpreter = _RoundInterpreter()
    second_generation_started = threading.Event()
    checkpointed = threading.Event()
    original_step = agent.step
    generated = 0

    def step(**kwargs):
        nonlocal generated
        if generated == 1:
            second_generation_started.set()
            assert checkpointed.wait(timeout=1.0)
        node = original_step(**kwargs)
        generated += node is not None
        return node

    original_execute = agent.execute_deferred_nodes

    def execute_deferred_nodes(nodes, exec_many_callback):
        if nodes[0].id == "candidate-1":
            assert second_generation_started.wait(timeout=1.0)
        return original_execute(nodes, exec_many_callback)

    agent.step = step
    agent.execute_deferred_nodes = execute_deferred_nodes

    completed = run_module._run_scheduler_rounds(
        agent=agent,
        interpreter=interpreter,
        cfg=SimpleNamespace(agent=SimpleNamespace(steps=2)),
        journal=journal,
        logger=logging.getLogger("test-scheduler-immediate-checkpoint"),
        save_callback=lambda cfg, current_journal: checkpointed.set(),
    )

    assert completed == 2
    assert checkpointed.is_set()


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
