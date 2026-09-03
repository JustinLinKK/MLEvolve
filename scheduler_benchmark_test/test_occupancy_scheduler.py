from scheduler_benchmark_test.occupancy_scheduler import Job, score_density, simulate


def test_memory_admission_has_no_fixed_parallel_job_limit() -> None:
    jobs = [
        Job(job_id=f"job-{index}", release=0.0, solo=10.0, memory_mb=100.0)
        for index in range(3)
    ]

    result = simulate(
        jobs,
        pair_slowdown=1.0,
        memory_budget_mb=500.0,
        parallel_cap=None,
        scorer=score_density,
    )

    assert result.peak_concurrency == 3
