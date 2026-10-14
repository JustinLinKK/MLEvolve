from pathlib import Path


def test_fixed_scheduler_replay_uses_tmpfs_and_has_no_fixed_parallel_cap() -> None:
    launcher = Path("localml_scheduler/deployments/run_petfinder_fixed_scheduler_replay_a10.sh").read_text(
        encoding="utf-8"
    )

    assert 'volatile_root="/dev/shm/' in launcher
    assert "require_ml_prediction" not in launcher
    assert "parallel_job_cap" not in launcher
    assert "CUDA_VISIBLE_DEVICES=0" in launcher
    assert "PYTHONHASHSEED=42" in launcher
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher
