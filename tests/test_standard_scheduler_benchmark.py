from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import os
import random

import numpy as np
import pytest
import torch

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.config import SCHEDULER_MODE_ADAPTIVE, SchedulerSettings
from localml_scheduler.domain import ResourceRequirements, TrainingJob
from localml_scheduler.hardware import HardwareProfile, build_hardware_key
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.storage import StateStore
from scheduler_benchmark_test.replay_multiprocess_baseline import replay_multiprocess_baseline
from scheduler_benchmark_test.replay_scheduler_timeline import replay_fixture
from scheduler_benchmark_test.standard import (
    A10_VRAM_CAP_MIB,
    ARRIVAL_RATE,
    DATASET_SIZE,
    DEFAULT_FIXTURE_ROOT,
    EPOCHS,
    JOB_COUNT,
    SEED,
)
from scheduler_benchmark_test.standard.generate_fixture import (
    FORBIDDEN_SOURCE_TOKENS,
    render_fixture,
    write_fixture,
)
from scheduler_benchmark_test.standard.reporting import REPORT_METRICS, aggregate_reports, summarize_case
from scheduler_benchmark_test.standard.run_benchmark import (
    DEFAULT_ARMS,
    _assert_primary_healthy,
    resolve_vram_budget_fraction,
    rotated_arm_order,
    scheduler_settings_overlay,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = DEFAULT_FIXTURE_ROOT


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jobs() -> list[dict]:
    return [json.loads(line) for line in (FIXTURE / "jobs.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_generation_is_deterministic_and_committed_fixture_has_no_drift() -> None:
    first = render_fixture()
    second = render_fixture()
    assert first == second
    assert write_fixture(FIXTURE, check=True) == []
    assert len(first) == 105  # 100 sources plus four payload files and manifest.


def test_manifest_schemas_checksums_and_exact_distributions() -> None:
    manifest = _json(FIXTURE / "manifest.json")
    jobs = _jobs()
    assert manifest["schema_version"] == "standard-histopath-v1"
    assert manifest["generator_version"] == "1.0.0"
    assert manifest["job_count"] == JOB_COUNT == len(jobs)
    assert manifest["dataset"]["labeled_image_count"] == DATASET_SIZE
    assert manifest["training"]["samples_per_job"] == DATASET_SIZE * EPOCHS
    assert manifest["training"]["samples_per_case"] == DATASET_SIZE * EPOCHS * JOB_COUNT
    assert manifest["family_counts"] == {
        "cnn": 20,
        "efficient_cnn": 20,
        "mlp_mixer": 20,
        "recurrent": 20,
        "vision_transformer": 20,
    }
    assert manifest["precision_counts"] == {"bf16": 25, "fp16": 25, "fp32": 25, "tf32": 25}
    assert len(manifest["profile_bucket_counts"]) == 20
    assert set(manifest["profile_bucket_counts"].values()) == {5}
    assert len(manifest["architecture_counts"]) == 20
    assert set(manifest["architecture_counts"].values()) == {5}
    assert len(manifest["variant_counts"]) == 5
    assert set(manifest["variant_counts"].values()) == {20}
    for relative, expected in manifest["file_sha256"].items():
        assert sha256((FIXTURE / relative).read_bytes()).hexdigest() == expected


def test_training_job_payloads_include_complete_prediction_inputs() -> None:
    jobs = _jobs()
    ids = [job["job_id"] for job in jobs]
    assert ids == [f"std-histo-{index:03d}" for index in range(1, 101)]
    assert not any("uuid" in json.dumps(job).lower() for job in jobs)
    for payload in jobs:
        job = TrainingJob.from_dict(payload)
        kwargs = job.config.runner_kwargs
        metadata = job.metadata
        assert metadata["architecture_source"]
        assert len(metadata["architecture_source_hash"]) == 64
        assert metadata["precision"] in {"fp32", "tf32", "fp16", "bf16"}
        assert metadata["input_shape"] == [3, 96, 96]
        assert kwargs["optimizer_name"] == "AdamW"
        assert kwargs["dataset_size"] == DATASET_SIZE
        assert kwargs["epochs"] == EPOCHS
        assert job.batch_probe.profile_namespace == metadata["profile_bucket"]
        assert job.batch_probe.shape_signature_override == metadata["profile_bucket"]
        assert job.batch_probe.minimum_batch_size == 32
        assert kwargs["probe_max_batch_size"] == 256
        assert job.checkpoint_policy.preemptible is False


def test_poisson_timeline_is_exact_and_strictly_ordered() -> None:
    timeline = _json(FIXTURE / "timeline.json")
    actions = timeline["actions"]
    expected_ids = [job["job_id"] for job in _jobs()]
    random.Random(SEED).shuffle(expected_ids)
    expected_times = np.cumsum(np.random.default_rng(SEED).exponential(scale=1.0 / ARRIVAL_RATE, size=JOB_COUNT))
    assert [action["job_id"] for action in actions] == expected_ids
    assert [action["relative_seconds"] for action in actions] == [round(float(value), 3) for value in expected_times]
    assert all(left["relative_seconds"] < right["relative_seconds"] for left, right in zip(actions, actions[1:]))
    assert all(action["action"] == "SUBMIT" for action in actions)


def test_all_sources_compile_are_inline_and_avoid_a10_forbidden_features() -> None:
    for path in sorted((FIXTURE / "sources").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        lowered = source.lower()
        assert "from scheduler_benchmark_test.standard.model_library import" not in source
        assert "class transformerclassifier" in lowered
        assert "run_generated_job" in source
        assert not [token for token in FORBIDDEN_SOURCE_TOKENS if token in lowered]


def test_scheduler_fixture_uses_adaptive_bounds_and_a10_budget(tmp_path: Path) -> None:
    payload = _json(FIXTURE / "scheduler_settings.replay.json")
    settings = SchedulerSettings.from_dict({**payload, "runtime_root": str(tmp_path / "runtime")})
    gpu = settings.gpu_scheduler
    assert gpu.mode == SCHEDULER_MODE_ADAPTIVE
    assert gpu.max_packed_jobs_per_gpu == 8
    assert gpu.candidate_window_size == 16
    assert gpu.batch_probe_max_batch_size == 256
    assert gpu.adaptive.vram_bucket_mb == 128
    assert gpu.checkpoint_preemption_enabled is False
    assert gpu.early_stop.enabled is False
    assert gpu.memory.budget_mb(24_576) == pytest.approx(A10_VRAM_CAP_MIB)
    assert resolve_vram_budget_fraction(24_576) * 24_576 == pytest.approx(A10_VRAM_CAP_MIB)


def test_arm_rotation_and_overlays_preserve_adaptive_bounds() -> None:
    assert rotated_arm_order(1) == DEFAULT_ARMS
    assert rotated_arm_order(2) == DEFAULT_ARMS[2:] + DEFAULT_ARMS[:2]
    assert rotated_arm_order(3) == DEFAULT_ARMS[4:] + DEFAULT_ARMS[:4]
    for prediction, backend in (
        ("branch_profile", "cuda_process"),
        ("branch_profile", "stream"),
        ("ml_predictor", "cuda_process"),
        ("ml_predictor", "stream"),
    ):
        overlay = scheduler_settings_overlay(prediction_mode=prediction, backend=backend, total_vram_mib=24_576)
        assert overlay["gpu_scheduler"]["max_packed_jobs_per_gpu"] == 8
        assert overlay["gpu_scheduler"]["candidate_window_size"] == 16
        assert overlay["gpu_scheduler"]["backend_priority"] == [backend, "exclusive"]
        assert overlay["prediction"]["mode"] == prediction
    primary = scheduler_settings_overlay(
        prediction_mode="ml_predictor", backend="cuda_process", total_vram_mib=24_576
    )
    with pytest.raises(RuntimeError, match="ml_predictor preflight failed: missing_checkpoint"):
        _assert_primary_healthy(primary)


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        hardware_key=build_hardware_key(
            os_name="linux",
            gpu_name="a10-budget-test",
            total_vram_mb=24_576,
            compute_capability="8.6",
            cuda_runtime="12.0",
            torch_version="2.5",
        ),
        os_name="linux",
        gpu_name="a10-budget-test",
        total_vram_mb=24_576,
        compute_capability="8.6",
        cuda_runtime="12.0",
        torch_version="2.5",
    )


def test_adaptive_planner_can_select_three_lightweight_jobs(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        gpu_scheduler={
            "mode": "adaptive",
            "backend_priority": ["cuda_process", "exclusive"],
            "max_packed_jobs_per_gpu": 8,
            "candidate_window_size": 16,
            "memory": {"vram_budget_fraction": A10_VRAM_CAP_MIB / 24_576},
        },
        prediction={"mode": "ml_predictor", "fallback_to_exclusive": True},
    )
    store = StateStore(settings)
    store._hardware_profile = _hardware()
    planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
    jobs = []
    for index in range(3):
        job = build_mlevolve_job(
            workflow_id="standard-test",
            baseline_model_id=f"light-{index}",
            baseline_model_path=f"/tmp/light-{index}.py",
            runner_target="pkg.runner:train",
            runner_kwargs={"batch_size": 32, "probe_max_batch_size": 256},
            resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
            packing_family="light",
            packing_eligible=True,
            packing_backend_allowlist=["cuda_process"],
            task_type="classification",
        )
        job.queue_sequence = index + 1
        jobs.append(job)
    plan = planner.choose_plan(jobs, backend_available={"cuda_process": True, "exclusive": True})
    assert plan is not None
    assert plan.mode == "packed_group"
    assert len(plan.job_ids) == 3


def test_large_window_is_bounded_by_candidate_and_pack_limits(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        gpu_scheduler={"mode": "adaptive", "max_packed_jobs_per_gpu": 8, "candidate_window_size": 16},
        prediction={"mode": "ml_predictor", "fallback_to_exclusive": True},
    )
    store = StateStore(settings)
    planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))
    jobs = [
        build_mlevolve_job(
            workflow_id="window-test",
            baseline_model_id=f"window-{index}",
            baseline_model_path=f"/tmp/window-{index}.py",
            runner_target="pkg.runner:train",
            runner_kwargs={"batch_size": 32},
            resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=64),
            packing_family="window",
            packing_eligible=True,
            task_type="classification",
        )
        for index in range(100)
    ]
    plan = planner.choose_plan(jobs, backend_available={"stream": True, "cuda_process": True, "exclusive": True})
    assert plan is not None
    assert len(plan.job_ids) <= 8
    assert set(plan.job_ids).issubset({job.job_id for job in jobs[:16]})


def test_fifo_noop_replay_is_submission_ordered_and_strictly_sequential(tmp_path: Path) -> None:
    result = replay_multiprocess_baseline(
        fixture=FIXTURE,
        output_root=tmp_path / "fifo",
        runner_mode="noop",
        parallelism=1,
        no_sleep=True,
        wait_for_all=True,
        cancel_policy="ignore",
    )
    records = [json.loads(line) for line in (result.log_dir / "multiprocess_jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = [action["job_id"] for action in _json(FIXTURE / "timeline.json")["actions"]]
    assert [record["job_id"] for record in records] == expected
    summary = summarize_case(
        result.output_root,
        arm="fifo",
        repetition=1,
        runner_mode="noop",
        backend=None,
        prediction_mode=None,
    )
    assert summary["completed_job_count"] == JOB_COUNT
    assert summary["maximum_observed_concurrency"] == 1
    assert {record["status"] for record in records} == {"COMPLETED"}


def _single_job_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "one-job-fixture"
    root.mkdir()
    jobs = _jobs()
    wanted_id = _json(FIXTURE / "timeline.json")["actions"][0]["job_id"]
    job = next(payload for payload in jobs if payload["job_id"] == wanted_id)
    (root / "jobs.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
    (root / "timeline.json").write_text(json.dumps({"actions": [{"action": "SUBMIT", "job_id": wanted_id, "relative_seconds": 0.0}]}) + "\n", encoding="utf-8")
    (root / "baseline_summary.json").write_text(json.dumps({"job_count": 1, "reference_metrics": {}}) + "\n", encoding="utf-8")
    (root / "scheduler_settings.replay.json").write_bytes((FIXTURE / "scheduler_settings.replay.json").read_bytes())
    return root


@pytest.mark.parametrize(
    ("prediction_mode", "backend"),
    (("branch_profile", "cuda_process"), ("branch_profile", "stream"), ("ml_predictor", "cuda_process"), ("ml_predictor", "stream")),
)
def test_noop_replay_covers_each_scheduler_arm(tmp_path: Path, prediction_mode: str, backend: str) -> None:
    fixture = _single_job_fixture(tmp_path)
    output = tmp_path / f"{prediction_mode}-{backend}"
    result = replay_fixture(
        fixture=fixture,
        output_root=output,
        runner_mode="noop",
        no_sleep=True,
        wait_for_all=True,
        cancel_policy="ignore",
        clean_profile_db=True,
        settings_overrides=scheduler_settings_overlay(
            prediction_mode=prediction_mode,
            backend=backend,
            total_vram_mib=24_576,
        ),
    )
    records = [json.loads(line) for line in (result.log_dir / "scheduler_jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "COMPLETED"
    assert (result.log_dir / "scheduler_events.jsonl").is_file()


def test_aggregate_report_uses_sample_stddev_student_t_and_matched_comparisons(tmp_path: Path) -> None:
    cases = []
    for repetition, fifo_wall, branch_wall in ((1, 12.0, 8.0), (2, 13.0, 9.0), (3, 14.0, 10.0)):
        for arm, wall in (("fifo", fifo_wall), ("branch_cuda", branch_wall)):
            item = {metric: wall for metric in REPORT_METRICS}
            item.update(
                {
                    "arm": arm,
                    "repetition": repetition,
                    "accepted": True,
                    "jobs_per_hour": 360_000.0 / wall,
                    "images_per_second": 1000.0 / wall,
                }
            )
            cases.append(item)
    report = aggregate_reports(cases, tmp_path)
    wall = report["arms"]["fifo"]["total_wall_time_seconds"]
    assert wall["n"] == 3
    assert wall["mean"] == pytest.approx(13.0)
    assert wall["sample_stddev"] == pytest.approx(1.0)
    assert wall["ci95_low"] == pytest.approx(13.0 - 4.303 / np.sqrt(3))
    comparison = report["matched_comparisons"]["branch_cuda_vs_fifo"]["total_wall_time_seconds"]
    assert comparison["delta"]["mean"] == pytest.approx(-4.0)
    assert comparison["ratio"]["n"] == 3
    assert (tmp_path / "aggregate_report.md").is_file()


@pytest.mark.skipif(os.environ.get("RUN_STANDARD_GPU_INTEGRATION") != "1", reason="opt-in real-data CUDA integration")
def test_small_real_data_gpu_scheduler_calibrates_reuses_packs_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = os.environ["HISTOPATH_DATA_ROOT"]
    fixture = tmp_path / "real-mini-fixture"
    fixture.mkdir()
    selected = _jobs()[:6]
    for payload in selected:
        payload["config"]["max_epochs"] = 1
        payload["config"]["runner_kwargs"]["epochs"] = 1
        payload["config"]["runner_kwargs"]["max_epochs"] = 1
        payload["max_epochs"] = 1
        payload["metadata"]["epochs"] = 1
        payload["metadata"]["total_epochs"] = 1
        payload["metadata"]["remaining_epochs"] = 1
    fallback = selected[-1]
    fallback["packing"]["eligible"] = False
    fallback["batch_probe"]["enabled"] = False
    (fixture / "jobs.jsonl").write_text(
        "".join(json.dumps(payload) + "\n" for payload in selected), encoding="utf-8"
    )
    (fixture / "timeline.json").write_text(
        json.dumps(
            {
                "actions": [
                    {"action": "SUBMIT", "job_id": payload["job_id"], "relative_seconds": 0.0}
                    for payload in selected
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (fixture / "baseline_summary.json").write_text(
        json.dumps({"job_count": len(selected), "reference_metrics": {}}) + "\n", encoding="utf-8"
    )
    (fixture / "scheduler_settings.replay.json").write_bytes(
        (FIXTURE / "scheduler_settings.replay.json").read_bytes()
    )
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("HISTOPATH_DATA_ROOT", data_root)
    monkeypatch.setenv("STANDARD_BENCH_EPOCHS", "1")
    monkeypatch.setenv("STANDARD_BENCH_ALLOW_PARTIAL", "1")
    monkeypatch.setenv("STANDARD_BENCH_MAX_SAMPLES", "64")

    result = replay_fixture(
        fixture=fixture,
        output_root=tmp_path / "real-gpu",
        runner_mode="real",
        no_sleep=True,
        wait_for_all=True,
        cancel_policy="ignore",
        clean_profile_db=True,
        settings_overrides=scheduler_settings_overlay(
            prediction_mode="branch_profile",
            backend="cuda_process",
            total_vram_mib=torch.cuda.get_device_properties(0).total_memory / (1024**2),
        ),
    )
    records = _read_jsonl_for_test(result.log_dir / "scheduler_jobs.jsonl")
    events = _read_jsonl_for_test(result.log_dir / "scheduler_events.jsonl")
    assert Counter(record["status"] for record in records) == {"COMPLETED": 6}
    assert any(event["event_type"] == "batch_probe_selected" for event in events)
    assert any(event["event_type"] == "batch_probe_cache_hit" for event in events)
    assert any(
        event["event_type"] == "packed_group_dispatched"
        and len((event.get("payload") or {}).get("job_ids") or []) >= 3
        for event in events
    )
    fallback_record = next(record for record in records if record["job_id"] == fallback["job_id"])
    assert fallback_record["metadata"]["placement_backend"] == "exclusive"
    assert any(int(record["metadata"].get("placement_batch_size") or 0) > 32 for record in records[:-1])
    metrics = [
        _json(path)
        for path in (tmp_path / "real-gpu").rglob("metric.json")
    ]
    assert metrics and all(metric["samples_seen"] == 64 and np.isfinite(metric["loss"]) for metric in metrics)


def _read_jsonl_for_test(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
