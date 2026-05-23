from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import time

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

    with sqlite3.connect(tmp_path / "pipeline.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM node_actions").fetchone()[0] == 1
        row = conn.execute("SELECT status, metric, detected_batch_size FROM job_packets").fetchone()
        assert row == ("parsed_valid", 0.75, 16)
        assert conn.execute("SELECT COUNT(*) FROM run_metrics").fetchone()[0] == 1


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
        exp_name="run-hw",
        exp_id="task-a",
        log_dir=tmp_path,
    )

    metrics = build_comparison_metrics(cfg, journal, started_at=started, finished_at=started + 20)
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
    assert metrics["time_to_best_seconds"] == 5
    assert comparison["metrics"]["best_metric"]["delta"] == 0.30000000000000004
    assert "| `best_metric` |" in markdown
