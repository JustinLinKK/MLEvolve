from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sqlite3

from agents.debug_agent import _build_debug_reports, _extract_debug_reports
from agents.triggers import register_node
from engine.search_node import Journal, SearchNode
from utils.pipeline_logging import PipelineActionLogger
from utils.serialize import dumps_json, loads_json


def test_extract_debug_reports_from_debug_response() -> None:
    response = """Bug Report:
- Root cause: submission.csv was never written.
- Evidence: evaluator reported missing output file.

Fix Report:
- Add submission generation using sample_submission.csv columns.
- Keep the existing model and validation metric unchanged.

<<<<<<< SEARCH
pass
=======
write_submission()
>>>>>>> REPLACE
"""

    bug_report, fix_report = _extract_debug_reports(response)

    assert "submission.csv was never written" in bug_report
    assert "missing output file" in bug_report
    assert "sample_submission.csv columns" in fix_report
    assert "SEARCH" not in fix_report


def test_build_debug_reports_falls_back_to_parent_error() -> None:
    parent = SearchNode(code="raise RuntimeError()", plan="draft", stage="draft")
    parent.analysis = "The script crashed before creating predictions."
    parent.exc_type = "RuntimeError"
    parent._term_out = ["Traceback: missing predictions\n"]

    bug_report, fix_report = _build_debug_reports(
        report_source_text="Plan: add prediction export\n<<<<<<< SEARCH\n",
        parent_node=parent,
        plan="add prediction export",
    )

    assert "RuntimeError" in bug_report
    assert "crashed before creating predictions" in bug_report
    assert "add prediction export" in fix_report


def test_debug_reports_survive_journal_round_trip() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    parent = SearchNode(code="x = 1", plan="draft", parent=root, stage="draft")
    child = SearchNode(
        code="x = 2",
        plan="debug",
        parent=parent,
        stage="debug",
        bug_report="Root cause: bad shape.",
        fix_report="Reshape tensors before metric calculation.",
    )

    journal = Journal()
    journal.append(root)
    journal.append(parent)
    journal.append(child)
    loaded = loads_json(dumps_json(journal), Journal)

    assert loaded.nodes[2].bug_report == "Root cause: bad shape."
    assert loaded.nodes[2].fix_report == "Reshape tensors before metric calculation."


def test_register_node_writes_debug_report_log(tmp_path: Path) -> None:
    logger = PipelineActionLogger(tmp_path / "pipeline.sqlite3", run_id="run-a", mode="baseline")
    root = SearchNode(code="", plan="root", stage="root")
    parent = SearchNode(code="x = 1", plan="draft", parent=root, stage="draft")
    parent.branch_id = 3
    parent.exc_type = "ValueError"
    parent.analysis = "The output shape did not match sample_submission.csv."
    node = SearchNode(
        code="x = 2",
        plan="debug",
        parent=parent,
        stage="debug",
        bug_report="Root cause: output rows were misaligned.",
        fix_report="Align predictions to sample_submission.csv before writing.",
    )

    agent = SimpleNamespace(
        pipeline_logger=logger,
        next_branch_id=1,
        branch_all_nodes={3: []},
        branch_successful_nodes={3: []},
        _serialize_prompt=lambda prompt: str(prompt),
    )

    register_node(agent, node, "# Prompt", parent_node=parent)

    with sqlite3.connect(tmp_path / "pipeline.sqlite3") as conn:
        report_row = conn.execute(
            "SELECT node_id, parent_node_id, bug_report, fix_report, report_path FROM debug_reports"
        ).fetchone()
        action_payload = conn.execute(
            "SELECT payload_json FROM node_actions WHERE action_type='node_created'"
        ).fetchone()[0]

    assert report_row[0] == node.id
    assert report_row[1] == parent.id
    assert report_row[2] == "Root cause: output rows were misaligned."
    assert report_row[3] == "Align predictions to sample_submission.csv before writing."
    assert Path(report_row[4]).exists()
    assert "debug_report_path" in action_payload
