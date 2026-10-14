from pathlib import Path


def test_control_pair_watcher_requires_live_metric_alignment_before_ten_node_run() -> None:
    watcher = Path("localml_scheduler/deployments/watch_petfinder_control_pair_a10.sh").read_text(
        encoding="utf-8"
    )

    assert "final_score.txt" in watcher
    assert "summary.json" in watcher
    assert "METRIC_TOLERANCE" in watcher
    assert 'r"Final Validation Score:\\s*([0-9.]+)"' in watcher
    assert 'PETFINDER_NODE_BUDGET=10' in watcher
    assert "parallel_job_cap" not in watcher
