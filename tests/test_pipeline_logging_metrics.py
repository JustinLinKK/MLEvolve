from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import time

from engine.executor import Interpreter
from engine.search_node import Journal, SearchNode
from utils.compare_experiment_runs import compare_metrics, to_markdown
from utils.experiment_metrics import build_comparison_metrics, write_comparison_metrics
from utils.metric import MetricValue
from utils.pipeline_logging import PipelineActionLogger


def test_pipeline_action_logger_writes_sqlite_tables(tmp_path: Path) -> None:
    logger = PipelineActionLogger(tmp_path / "pipeline.sqlite3", run_id="run-a", mode="baseline")

    logger.emit("run_started", payload={"hello": "world"})
    logger.record_node_action(
        node_id="node-1",
        action_type="node_created",
        stage="draft",
        branch_id=1,
        metric=0.5,
        is_buggy=False,
        is_valid=True,
        exec_time=12.0,
    )
    logger.upsert_job_packet(
        "job-1",
        node_id="node-1",
        status="running",
        detected_batch_size=16,
        model_key="vit",
        framework="pytorch",
        uses_amp=True,
        requires_gpu=True,
        script_signature="abc",
    )
    logger.update_job_packet_for_node("node-1", metric=0.75, status="parsed_valid")
    logger.record_run_metrics({"mode": "baseline", "node_count": 1})
    snapshot = logger.record_prompt_snapshot(
        node_id="node-1",
        parent_node_id="root",
        branch_id=1,
        stage="draft",
        prompt_text="# Prompt\nUse stage-filtered hardware knowledge.",
    )
    debug_snapshot = logger.record_debug_report(
        node_id="node-2",
        parent_node_id="node-1",
        branch_id=1,
        bug_report="Root cause: missing submission.csv.",
        fix_report="Write submission.csv with the sample submission columns.",
        payload={"parent_exc_type": "FileNotFoundError"},
    )

    with sqlite3.connect(tmp_path / "pipeline.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM node_actions").fetchone()[0] == 1
        row = conn.execute("SELECT status, metric, detected_batch_size FROM job_packets").fetchone()
        assert row == ("parsed_valid", 0.75, 16)
        assert conn.execute("SELECT COUNT(*) FROM run_metrics").fetchone()[0] == 1
        prompt_row = conn.execute(
            "SELECT node_id, prompt_chars, prompt_path, prompt_text FROM prompt_snapshots"
        ).fetchone()
        assert prompt_row[0] == "node-1"
        assert prompt_row[1] == snapshot["prompt_chars"]
        assert prompt_row[3] == "# Prompt\nUse stage-filtered hardware knowledge."
        debug_row = conn.execute(
            "SELECT node_id, bug_report, fix_report, report_path FROM debug_reports"
        ).fetchone()
        assert debug_row[0] == "node-2"
        assert debug_row[1] == "Root cause: missing submission.csv."
        assert debug_row[2] == "Write submission.csv with the sample submission columns."

    prompt_path = Path(snapshot["prompt_path"])
    assert prompt_path.exists()
    assert prompt_path.read_text(encoding="utf-8") == "# Prompt\nUse stage-filtered hardware knowledge."
    debug_path = Path(debug_snapshot["debug_report_path"])
    assert debug_path.exists()
    assert "## Bug Report" in debug_path.read_text(encoding="utf-8")


def test_comparison_metrics_generation_and_compare(tmp_path: Path) -> None:
    started = time.time() - 20
    journal = Journal()
    journal.nodes = []
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="BATCH_SIZE = 16", plan="draft", parent=root, stage="draft")
    node.metric = MetricValue(0.8, maximize=True)
    node.is_buggy = False
    node.is_valid = True
    node.exec_time = 7.0
    node.ctime = started + 5
    journal.append(root)
    journal.append(node)
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="hardware_aware"),
        scheduler=SimpleNamespace(enabled=False),
        hardware_knowledge=SimpleNamespace(enabled=True, include_profile_evidence=True),
        agent=SimpleNamespace(hardware_context_enabled=True),
        exp_name="run-hw",
        exp_id="task-a",
        log_dir=tmp_path,
    )
    hardware_client = SimpleNamespace(
        include_profile_evidence=True,
        profile_evidence_used=True,
        probe_status={"ok": True, "source": "hardware_probe_subprocess"},
    )

    metrics = build_comparison_metrics(
        cfg,
        journal,
        started_at=started,
        finished_at=started + 20,
        hardware_knowledge_client=hardware_client,
    )
    path = write_comparison_metrics(metrics, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    comparison = compare_metrics(
        {"run_id": "run-base", "mode": "baseline", "node_count": 1, "best_metric": 0.5},
        loaded,
    )
    markdown = to_markdown(comparison)

    assert metrics["mode"] == "hardware_aware"
    assert metrics["node_count"] == 1
    assert metrics["best_metric"] == 0.8
    assert metrics["time_to_best_seconds"] == 12
    assert metrics["configured_hardware_knowledge_enabled"] is True
    assert metrics["hardware_knowledge_client_attached"] is True
    assert metrics["hardware_probe_source"] == "hardware_probe_subprocess"
    assert metrics["hardware_probe_success"] is True
    assert metrics["hardware_knowledge_include_profile_evidence"] is True
    assert metrics["hardware_profile_evidence_used"] is True
    assert comparison["metrics"]["best_metric"]["delta"] == 0.30000000000000004
    assert "| `best_metric` |" in markdown


def test_comparison_metrics_runtime_breakdown_from_pipeline_and_phase_data(tmp_path: Path) -> None:
    started = time.time() - 40
    logger = PipelineActionLogger(tmp_path / "pipeline.sqlite3", run_id="run-metrics", mode="hardware_aware")
    journal = Journal()
    root = SearchNode(code="", plan="root", stage="root")
    node_a = SearchNode(code="def train_model(): pass", plan="draft", parent=root, stage="draft")
    node_a.metric = MetricValue(0.8, maximize=True)
    node_a.is_buggy = False
    node_a.is_valid = True
    node_a.exec_time = 10.0
    node_a.phase_timings = {
        "phase_durations_seconds": {
            "training": 7.0,
            "inference": 2.0,
            "validation": 1.0,
            "other_candidate": 0.0,
        },
        "phase_timing_available": True,
        "phase_timing_coverage_seconds": 10.0,
    }
    node_a.instrumentation = {"phase_instrumented": True}
    node_b = SearchNode(code="print('bad')", plan="draft", parent=root, stage="draft")
    node_b.metric = MetricValue(-1, maximize=True)
    node_b.is_buggy = True
    node_b.exec_time = 5.0
    node_b.phase_timings = {
        "phase_durations_seconds": {
            "training": 0.0,
            "inference": 0.0,
            "validation": 0.0,
            "other_candidate": 5.0,
        },
        "phase_timing_available": False,
        "phase_timing_coverage_seconds": 0.0,
    }
    node_b.instrumentation = {"phase_instrumented": True}
    journal.append(root)
    journal.append(node_a)
    journal.append(node_b)

    logger.emit(
        "llm_call_completed",
        stage="llm",
        payload={"provider": "openai", "model": "m", "interface": "query", "wall_time_seconds": 3.0},
    )
    logger.emit(
        "llm_call_failed",
        stage="llm",
        payload={"provider": "openai", "model": "m", "interface": "generate", "wall_time_seconds": 2.0},
    )
    logger.record_node_action(node_id=node_a.id, action_type="execution_started")
    time.sleep(0.01)
    logger.record_node_action(node_id=node_a.id, action_type="execution_finished")

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="hardware_aware"),
        exp_name="run-metrics",
        exp_id="task-a",
        log_dir=tmp_path,
    )
    metrics = build_comparison_metrics(
        cfg,
        journal,
        started_at=started,
        finished_at=started + 40,
        pipeline_logger=logger,
    )

    assert metrics["total_run_wall_time_seconds"] == 40
    assert metrics["total_wall_time_seconds"] == 40
    assert metrics["total_candidate_execution_time_seconds"] == 15.0
    assert metrics["total_job_execution_time_seconds"] == 15.0
    assert metrics["candidate_execution_makespan_seconds"] is not None
    assert metrics["candidate_execution_parallelism_ratio"] is not None
    assert metrics["total_llm_call_wall_time_seconds"] == 5.0
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_error_count"] == 1
    assert metrics["total_training_wall_time_seconds"] == 7.0
    assert metrics["total_inference_wall_time_seconds"] == 2.0
    assert metrics["phase_instrumented_node_count"] == 2
    assert metrics["phase_instrumentation_miss_count"] == 0


def test_local_executor_records_phase_timing(tmp_path: Path) -> None:
    interpreter = Interpreter(working_dir=tmp_path, timeout=30, max_parallel_run=1)
    code = """
import time

def train_model():
    for epoch in range(2):
        time.sleep(0.01)

def generate_submission():
    time.sleep(0.01)

train_model()
generate_submission()
"""

    result = interpreter.run(code, id="node-phase", working_dir=str(tmp_path))

    assert result.exc_type is None
    assert result.instrumentation and result.instrumentation["phase_instrumented"] is True
    assert result.phase_timings is not None
    durations = result.phase_timings["phase_durations_seconds"]
    assert durations["training"] > 0
    assert durations["inference"] > 0


def test_phase_metrics_report_null_when_no_candidate_is_instrumented(tmp_path: Path) -> None:
    started = time.time() - 5
    journal = Journal()
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="print('hello')", plan="draft", parent=root, stage="draft")
    node.metric = MetricValue(-1, maximize=True)
    node.is_buggy = True
    node.exec_time = 1.0
    node.instrumentation = {"phase_instrumented": False, "phase_instrumentation_reason": "no_recognized_phase_regions"}
    journal.append(root)
    journal.append(node)
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="hardware_aware"),
        exp_name="run-null",
        exp_id="task-a",
        log_dir=tmp_path,
    )

    metrics = build_comparison_metrics(cfg, journal, started_at=started, finished_at=started + 5)

    assert metrics["total_training_wall_time_seconds"] is None
    assert metrics["total_inference_wall_time_seconds"] is None
    assert metrics["phase_instrumented_node_count"] == 0
    assert metrics["phase_instrumentation_miss_count"] == 1
