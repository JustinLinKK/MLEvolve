from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt

from localml_scheduler.benchmark_test.draw_petfinder_comparison import (
    RunSpec,
    _draw_gantt,
    _parse_run_argument,
    load_run,
    peak_execution_concurrency,
    render_comparison,
)


def test_run_argument_accepts_a_per_run_node_budget() -> None:
    spec = _parse_run_argument("Scheduler|A100 80GB|20|scheduler.json", 50)

    assert spec.label == "Scheduler"
    assert spec.hardware == "A100 80GB"
    assert spec.target_nodes == 20
    assert spec.journal_paths == (Path("scheduler.json"),)


def _write_journal(path) -> None:
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "root",
                        "stage": "root",
                        "step": 0,
                        "created_time": None,
                        "finish_time": None,
                        "exec_time": None,
                        "metric": {"value": None, "maximize": None},
                    },
                    {
                        "id": "first",
                        "stage": "draft",
                        "step": 1,
                        "created_time": "2026-08-30T00:00:00",
                        "finish_time": "2026-08-30T00:05:00",
                        "exec_time": 60.0,
                        "metric": {"value": 18.5, "maximize": False},
                        "is_buggy": False,
                    },
                    {
                        "id": "second",
                        "stage": "improve",
                        "step": 2,
                        "created_time": "2026-08-30T00:03:00",
                        "finish_time": "2026-08-30T00:06:00",
                        "exec_time": 120.0,
                        "metric": {"value": 18.2, "maximize": False},
                        "is_buggy": False,
                    },
                    {
                        "id": "rejected",
                        "stage": "draft",
                        "step": 3,
                        "created_time": "2026-08-30T00:07:00",
                        "finish_time": "2026-08-30T00:08:00",
                        "exec_time": 0.0,
                        "metric": {"value": None, "maximize": None},
                        "is_buggy": True,
                    },
                    {
                        "id": "long_failure",
                        "stage": "improve",
                        "step": 4,
                        "created_time": "2026-08-30T00:09:00",
                        "finish_time": "2026-08-30T00:11:00",
                        "exec_time": 75.0,
                        "metric": {"value": None, "maximize": None},
                        "is_buggy": True,
                        "exc_type": "RuntimeError",
                        "exc_info": {"message": "tensor sizes do not match"},
                    },
                ]
            }
        )
    )


def test_load_run_excludes_quick_failures_from_completed_nodes(tmp_path) -> None:
    journal = tmp_path / "journal.json"
    _write_journal(journal)

    run = load_run(RunSpec("A100 baseline", "A100", (journal,), target_nodes=50))

    assert [node.step for node in run.nodes] == [1, 2, 3]
    assert run.metric_points == [(1, 18.5), (2, 18.2)]
    assert peak_execution_concurrency(run.nodes) == 2
    assert run.completed_nodes == 3


def test_load_run_truncates_each_run_to_the_matched_node_budget(tmp_path) -> None:
    journal = tmp_path / "journal.json"
    _write_journal(journal)

    run = load_run(RunSpec("A100 baseline", "A100", (journal,), target_nodes=2))

    assert [node.node_id for node in run.nodes] == ["first", "second"]
    assert run.completed_nodes == 2


def test_render_three_runs_to_one_png(tmp_path) -> None:
    journal = tmp_path / "journal.json"
    output = tmp_path / "comparison.png"
    _write_journal(journal)
    run = load_run(RunSpec("run", "GPU", (journal,), target_nodes=50))

    render_comparison([run, run, run], output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_load_run_can_draw_a_full_trace_with_short_failures(tmp_path) -> None:
    journal = tmp_path / "journal.json"
    _write_journal(journal)

    run = load_run(
        RunSpec("original", "A100", (journal,), target_nodes=50, include_all_executions=True)
    )

    assert [node.node_id for node in run.nodes] == [
        "first", "second", "rejected", "long_failure"
    ]
    assert run.source_journal_nodes == 5


def test_gantt_excludes_preflight_rejects_but_shows_runtime_failures_in_red(tmp_path) -> None:
    """A pre-execution rejection must not be presented as a buggy runtime node."""
    journal = tmp_path / "journal.json"
    _write_journal(journal)
    run = load_run(
        RunSpec("original", "A100", (journal,), target_nodes=50, include_all_executions=True)
    )
    figure, axis = plt.subplots()
    try:
        _draw_gantt(axis, run)
        # first + second + long_failure: the zero-second rejected candidate is omitted.
        assert len(axis.collections) == 3
        red = [
            collection
            for collection in axis.collections
            if tuple(collection.get_facecolor()[0][:3]) == (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
        ]
        assert len(red) == 1
    finally:
        plt.close(figure)


def test_gantt_draws_external_termination_gray_not_as_a_runtime_bug(tmp_path) -> None:
    """A scheduler/watchdog SIGTERM is not evidence of a buggy candidate."""
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "root", "stage": "root"},
                    {
                        "id": "success",
                        "stage": "draft",
                        "created_time": "2026-08-30T00:00:00",
                        "finish_time": "2026-08-30T00:01:00",
                        "exec_time": 60.0,
                        "metric": {"value": 18.5, "maximize": False},
                        "is_buggy": False,
                    },
                    {
                        "id": "runtime-bug",
                        "stage": "draft",
                        "created_time": "2026-08-30T00:02:00",
                        "finish_time": "2026-08-30T00:03:00",
                        "exec_time": 60.0,
                        "metric": {"value": None, "maximize": None},
                        "is_buggy": True,
                        "exc_type": "RuntimeError",
                        "exc_info": {"message": "tensor sizes do not match"},
                    },
                    {
                        "id": "external-stop",
                        "stage": "draft",
                        "created_time": "2026-08-30T00:04:00",
                        "finish_time": "2026-08-30T00:05:00",
                        "exec_time": 60.0,
                        "metric": {"value": None, "maximize": None},
                        "is_buggy": True,
                        "exc_type": "RuntimeError",
                        "exc_info": {"message": "worker exited with code -15"},
                    },
                ]
            }
        )
    )
    run = load_run(RunSpec("run", "A10", (journal,), include_all_executions=True))
    figure, axis = plt.subplots()
    try:
        _draw_gantt(axis, run)
        colors = [tuple(collection.get_facecolor()[0][:3]) for collection in axis.collections]
        assert colors.count((0.7686274509803922, 0.3058823529411765, 0.3215686274509804)) == 1
        assert colors.count((0.5450980392156862, 0.5450980392156862, 0.5450980392156862)) == 1
    finally:
        plt.close(figure)


def test_gantt_does_not_treat_time_limit_context_as_external_stop(tmp_path) -> None:
    """An ordinary exception stays red even when its report states the configured limit."""
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "root", "stage": "root"},
                    {
                        "id": "runtime-bug",
                        "stage": "draft",
                        "created_time": "2026-08-30T00:00:00",
                        "finish_time": "2026-08-30T00:01:00",
                        "exec_time": 60.0,
                        "metric": {"value": None, "maximize": None},
                        "is_buggy": True,
                        "exc_type": "TypeError",
                        "exc_info": {"message": "tensor sizes do not match"},
                        "_term_out": [
                            "Execution time: a minute seconds (time limit is 2 hours)."
                        ],
                    },
                ]
            }
        )
    )

    run = load_run(RunSpec("run", "A100", (journal,), include_all_executions=True))

    assert run.nodes[0].is_runtime_bug


def test_load_run_can_limit_all_execution_traces_to_equal_node_count(tmp_path) -> None:
    journal = tmp_path / "journal.json"
    _write_journal(journal)

    run = load_run(
        RunSpec(
            "original",
            "A100",
            (journal,),
            target_nodes=3,
            include_all_executions=True,
            limit_executions=True,
        )
    )

    assert [node.node_id for node in run.nodes] == ["first", "second", "rejected"]
    assert run.completed_nodes == 3


def test_load_run_deduplicates_continuation_roots_in_full_trace_count(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_journal(first)
    _write_journal(second)

    run = load_run(
        RunSpec("original", "A100", (first, second), include_all_executions=True)
    )

    assert run.source_journal_nodes == 9


def test_plotter_runs_as_a_direct_script_from_the_repository_root(tmp_path) -> None:
    journal = tmp_path / "journal.json"
    output = tmp_path / "comparison.png"
    _write_journal(journal)
    repository = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "localml_scheduler/benchmark_test/draw_petfinder_comparison.py",
            "--run",
            f"run|A100|{journal}",
            "--target-nodes",
            "2",
            "--out",
            str(output),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
