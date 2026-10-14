from pathlib import Path


def test_seeded_baseline_launcher_is_a_scheduler_free_seeded_control() -> None:
    launcher = Path("localml_scheduler/deployments/run_petfinder_seeded_baseline_a10.sh").read_text(
        encoding="utf-8"
    )

    assert "scheduler.enabled=false" in launcher
    assert "hardware_knowledge.enabled=false" in launcher
    assert "preflight.enabled=false" in launcher
    assert "agent.seed=42" in launcher
    assert "PYTHONHASHSEED=42" in launcher
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher
    assert "agent.code.provider=openai-compatible" in launcher
    assert "agent.feedback.provider=openai-compatible" in launcher
    assert "use_grading_server=false" in launcher
    assert "parallel_job_cap" not in launcher
