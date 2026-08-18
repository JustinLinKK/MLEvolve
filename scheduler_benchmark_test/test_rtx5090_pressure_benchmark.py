from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap
import time

import pytest

from scheduler_benchmark_test.rtx5090_pressure_benchmark import (
    TRACE_TEMPLATES,
    _attempt_peak_concurrency,
    _phase_marker,
    build_profile_snapshots,
    build_trace,
    render_gantt,
    run_mp2_baseline,
    trace_target_minutes,
    validate_stream_placements,
    validate_trace,
)


def test_trace_composition_releases_and_duration_total() -> None:
    trace = build_trace(32607)
    validate_trace(trace)
    assert len(trace) == 16
    assert trace_target_minutes() == pytest.approx(113.0)
    assert [item.release_s for item in TRACE_TEMPLATES] == [
        0,
        0,
        60,
        120,
        120,
        120,
        120,
        360,
        720,
        720,
        1080,
        1080,
        1440,
        1440,
        1800,
        2160,
    ]
    assert sum(item["target_seconds"] for item in trace) == pytest.approx(113 * 60)
    assert all(item["backend_allowlist"] == ["stream"] for item in trace)
    assert sum(
        item["target_vram_fraction"] for item in trace if item["scenario"] == "boundary"
    ) == pytest.approx(0.82)


def test_warm_and_cold_snapshots_share_solo_but_not_colocation(tmp_path: Path) -> None:
    trace = build_trace(32607, smoke=True)
    archetypes = {item["archetype"] for item in trace}
    for item in trace:
        item.update(
            {
                "calibrated_step_seconds": 0.1,
                "calibrated_solo_seconds": item["target_seconds"],
                "epochs": 3,
                "batches_per_epoch": 2,
            }
        )
    solo = {
        archetype: [
            {"training_seconds": 0.2, "global_steps": 2, "peak_reserved_mib": 1000}
        ]
        for archetype in archetypes
    }
    qualification = {
        "attempts": [
            {
                "solo": solo,
                "groups": {
                    "compute_pair": [{}],
                    "light_four": [{}],
                    "boundary_pair": [{}],
                },
                "compute_pair_slowdown": 1.5,
                "light_speedup_vs_mp2": 1.7,
            }
        ]
    }
    warm, cold = build_profile_snapshots(trace, qualification, tmp_path)
    assert warm["solo_memory_profiles"] == cold["solo_memory_profiles"]
    assert warm["colocation_profiles"]
    assert cold["colocation_profiles"] == []


def _fake_worker_script(path: Path) -> None:
    path.write_text(textwrap.dedent("""
            import json, pathlib, sys, time
            spec_path, result_path = map(pathlib.Path, sys.argv[1:])
            spec = json.loads(spec_path.read_text())
            if spec['job_id'] == 'oom-job' and 'attempt1' in spec_path.name:
                print('torch.cuda.OutOfMemoryError: CUDA out of memory', file=sys.stderr)
                raise SystemExit(1)
            time.sleep(float(spec.get('fake_sleep', 0.05)))
            result_path.write_text(json.dumps({'training_seconds': float(spec.get('fake_sleep', 0.05))}))
            """))


def _baseline_trace() -> list[dict]:
    return [
        {
            "step_idx": 0,
            "job_id": "oom-job",
            "scenario": "near_exclusive",
            "release_s": 0.0,
            "fake_sleep": 0.02,
        },
        {
            "step_idx": 1,
            "job_id": "survivor",
            "scenario": "light_pack",
            "release_s": 0.0,
            "fake_sleep": 0.20,
        },
        {
            "step_idx": 2,
            "job_id": "after",
            "scenario": "short_flow",
            "release_s": 0.0,
            "fake_sleep": 0.02,
        },
    ]


def test_unconditional_mp2_oom_drains_retries_alone_then_resumes(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "fake_worker.py"
    _fake_worker_script(helper)

    def command(_job: dict, spec: Path, result: Path) -> list[str]:
        return [sys.executable, str(helper), str(spec), str(result)]

    raw = run_mp2_baseline(
        _baseline_trace(), tmp_path / "run", timeout_s=5, command_factory=command
    )
    oom_attempt = next(
        item
        for item in raw["attempts"]
        if item["logical_job_id"] == "oom-job" and item["attempt"] == 1
    )
    retry = next(
        item
        for item in raw["attempts"]
        if item["logical_job_id"] == "oom-job" and item["attempt"] == 2
    )
    survivor = next(
        item for item in raw["attempts"] if item["logical_job_id"] == "survivor"
    )
    after = next(item for item in raw["attempts"] if item["logical_job_id"] == "after")
    assert oom_attempt["status"] == "oom"
    assert retry["status"] == "succeeded" and retry["retry"]
    assert retry["started_at"] >= survivor["finished_at"]
    assert after["started_at"] >= retry["finished_at"]


def test_baseline_timeout_retains_partial_attempt(tmp_path: Path) -> None:
    helper = tmp_path / "fake_worker.py"
    _fake_worker_script(helper)
    trace = [
        {
            "step_idx": 0,
            "job_id": "slow",
            "scenario": "compute_heavy",
            "release_s": 0.0,
            "fake_sleep": 5.0,
        }
    ]
    raw = run_mp2_baseline(
        trace,
        tmp_path / "timeout",
        timeout_s=0.2,
        command_factory=lambda _job, spec, result: [
            sys.executable,
            str(helper),
            str(spec),
            str(result),
        ],
    )
    assert raw["timed_out"]
    assert raw["attempts"][0]["status"] == "timeout"


def test_partial_gantt_and_resume_marker(tmp_path: Path) -> None:
    trace = build_trace(32607, smoke=True)
    origin = time.time()
    raw = {
        "origin": origin,
        "deadline": origin + 30,
        "attempts": [
            {
                "logical_job_id": trace[0]["job_id"],
                "started_at": origin + 0.1,
                "finished_at": origin + 0.8,
                "oom": True,
                "retry": False,
                "backend": "multiprocess",
                "scenario": trace[0]["scenario"],
            }
        ],
        "events": [],
    }
    png, pdf = render_gantt(tmp_path, trace, {"baseline": raw})
    assert png.stat().st_size > 0 and pdf.stat().st_size > 0
    assert not _phase_marker(tmp_path, "baseline").exists()
    _phase_marker(tmp_path, "baseline").parent.mkdir(parents=True)
    _phase_marker(tmp_path, "baseline").write_text(json.dumps({"complete": True}))
    assert _phase_marker(tmp_path, "baseline").exists()


def test_shared_host_and_distinct_stream_assertion() -> None:
    attempts = [
        {
            "logical_job_id": "a",
            "backend": "stream",
            "started_at": 1.0,
            "finished_at": 3.0,
            "stream_host_pid": 44,
            "cuda_stream_id": 100,
        },
        {
            "logical_job_id": "b",
            "backend": "stream",
            "started_at": 2.0,
            "finished_at": 4.0,
            "stream_host_pid": 44,
            "cuda_stream_id": 200,
        },
    ]
    assert validate_stream_placements(attempts)["valid"]
    attempts[1]["cuda_stream_id"] = 100
    assert not validate_stream_placements(attempts)["valid"]


def test_gantt_concurrency_annotation_uses_peak_not_all_intersections() -> None:
    target = {"started_at": 0.0, "finished_at": 10.0}
    attempts = [
        target,
        {"started_at": 1.0, "finished_at": 3.0},
        {"started_at": 4.0, "finished_at": 6.0},
    ]
    assert _attempt_peak_concurrency(target, attempts) == 2
