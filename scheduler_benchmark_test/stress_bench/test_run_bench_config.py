from pathlib import Path

from scheduler_benchmark_test.stress_bench.run_bench import build_scheduler_settings


def test_scheduler_stress_settings_leave_parallel_cap_unset(tmp_path: Path) -> None:
    settings = build_scheduler_settings(
        gpu_vram_gib=31.0,
        prediction_mode="branch_profile",
        runtime_root=tmp_path / "runtime",
    )

    assert settings.gpu_scheduler.mode == "parallel_time_aware"
    assert settings.gpu_scheduler.parallel_job_cap is None
