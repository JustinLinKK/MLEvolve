from __future__ import annotations

from pathlib import Path

from benchmarks.context_cache_bench import (
    comparison_metrics,
    summarize_observations,
    write_outputs,
)
from benchmarks.scenarios import (
    cold_trial_marker,
    inject_cold_marker,
    required_scenarios,
)


def _rows():
    shared = {
        "provider": "openrouter",
        "upstream_provider": "openai",
        "model": "openai/gpt-5.6",
        "agent_role": "reviewer",
        "context_length": "4k",
        "reuse": "same-agent",
        "concurrency": 1,
        "idle_gap_minutes": 0,
        "routing": "pinned",
        "cache_family_id": "family",
        "prompt_tokens": 4000,
        "output_tokens": 10,
        "error_type": None,
    }
    return [
        {
            **shared,
            "benchmark_mode": "baseline",
            "ttft_ms": 100,
            "end_to_end_ms": 200,
            "db_retrieval_ms": 20,
            "pack_build_ms": 10,
        },
        {
            **shared,
            "benchmark_mode": "local-only",
            "ttft_ms": 100,
            "end_to_end_ms": 170,
            "db_retrieval_ms": 0,
            "pack_build_ms": 1,
        },
        {
            **shared,
            "benchmark_mode": "both",
            "ttft_ms": 50,
            "end_to_end_ms": 100,
            "db_retrieval_ms": 0,
            "pack_build_ms": 1,
            "cache_read_tokens": 3000,
        },
    ]


def test_required_matrix_and_cold_markers() -> None:
    scenarios = required_scenarios()
    assert {scenario.mode for scenario in scenarios} == {
        "baseline",
        "local-only",
        "provider-only",
        "both",
    }
    assert {scenario.concurrency for scenario in scenarios} == {1, 4}
    assert cold_trial_marker("one") != cold_trial_marker("two")
    marked = inject_cold_marker([{"role": "system", "content": "stable"}], "trial")
    assert marked[0]["content"].startswith("[MLEVOLVE_CONTEXT_CACHE_COLD_TRIAL:")


def test_summary_outputs_p50_p95_and_speedups(tmp_path: Path) -> None:
    summaries = summarize_observations(_rows())
    comparisons = comparison_metrics(summaries)

    assert len(summaries) == 3
    assert comparisons[0]["ttft_speedup"] == 2.0
    assert comparisons[0]["end_to_end_speedup"] == 2.0
    assert comparisons[0]["local_retrieval_saved_ms"] == 29.0

    paths = write_outputs(_rows(), tmp_path, environment={"route": "pinned"})
    assert set(paths) == {"jsonl", "csv", "markdown"}
    assert all(path.exists() for path in paths.values())
    assert "Latency-only changes remain inferences" in paths["markdown"].read_text(
        encoding="utf-8"
    )
