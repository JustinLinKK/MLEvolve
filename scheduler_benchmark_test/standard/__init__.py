"""Deterministic scheduler stress-test fixtures."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
STRESS_TEST_DATA_V1_0_FIXTURE = (
    REPOSITORY_ROOT / "scheduler_benchmark_test" / "fixtures" / "stress_test_data_v1.0"
)
STRESS_TEST_DATA_JOB_COUNT = 100
