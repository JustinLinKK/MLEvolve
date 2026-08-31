from pathlib import Path

from scheduler_benchmark_test.replay_scheduler import build_settings


def test_replay_scheduler_uses_incremental_admission(tmp_path: Path) -> None:
    settings = build_settings(
        mode="parallel_time_aware",
        backend="cuda_process",
        gpu_vram_gib=16.0,
        runtime_root=tmp_path / "runtime",
        cache_warm_top_k=0,
        cache_warm_policy="budget_only",
        cache_entry_capacity=None,
        cache_max_ram_percent=None,
        cache_memory_budget_gib=1.0,
    )

    assert settings.gpu_scheduler.parallel_job_cap is None
