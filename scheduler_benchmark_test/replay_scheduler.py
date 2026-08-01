"""Replay driver A: feed trace through localml_scheduler.

This path uses a structured scheduler runner so the benchmark exercises:
- current scheduler settings layout
- MPS / stream / exclusive backends
- baseline RAM cache directly via ``context.load_baseline_object()``
- configurable batch-size threshold windows for packed optimization
- configurable packed-group width (2/3/4 jobs per GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, REPO)

try:
    from .benchmark_support import (
        DEFAULT_BINARY_RANGE_DOWN,
        DEFAULT_BINARY_RANGE_UP,
        DEFAULT_CACHE_ENTRY_CAPACITY,
        DEFAULT_CACHE_MAX_RAM_PERCENT,
        DEFAULT_CACHE_MEMORY_BUDGET_GIB,
        DEFAULT_CACHE_WARM_TOP_K,
        DEFAULT_POWER_OF_TWO_RANGE_DOWN,
        DEFAULT_POWER_OF_TWO_RANGE_UP,
        DEFAULT_VRAM_BUDGET_GIB,
    )
except ImportError:  # Direct ``python scheduler_benchmark_test/replay_scheduler.py`` execution.
    from benchmark_support import (
        DEFAULT_BINARY_RANGE_DOWN,
        DEFAULT_BINARY_RANGE_UP,
        DEFAULT_CACHE_ENTRY_CAPACITY,
        DEFAULT_CACHE_MAX_RAM_PERCENT,
        DEFAULT_CACHE_MEMORY_BUDGET_GIB,
        DEFAULT_CACHE_WARM_TOP_K,
        DEFAULT_POWER_OF_TWO_RANGE_DOWN,
        DEFAULT_POWER_OF_TWO_RANGE_UP,
        DEFAULT_VRAM_BUDGET_GIB,
    )
from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.client import SchedulerClient
from localml_scheduler.domain import (
    BatchProbeSpec,
    BatchSizeObservation,
    CheckpointPolicy,
    ResourceRequirements,
    SchedulingClass,
    SoloProfile,
    TrainingJob,
    build_batch_probe_shape_signature,
    parse_timestamp,
)
from localml_scheduler.config import (
    GpuMemorySettings,
    GpuProfilingSettings,
    GpuSchedulerSettings,
    GpuThresholdSettings,
    MPSSettings,
    ParallelOptimizerSettings,
    SchedulerSettings,
    StreamSettings,
)
from localml_scheduler.observability.outcomes import classify_job_outcome


def _mps_directory_env(var_name: str, default: str) -> str:
    return str(os.environ.get(var_name) or os.environ.get(f"CUDA_{var_name[6:]}") or default)


def build_settings(
    *,
    mode: str,
    backend: str,
    batch_search: str | None,
    max_packed_jobs_per_gpu: int,
    vram_budget_gib: float,
    runtime_root: Path,
    cache_warm_top_k: int,
    cache_warm_policy: str,
    cache_entry_capacity: int | None,
    cache_max_ram_percent: float | None,
    cache_memory_budget_gib: float,
    binary_range_up: int,
    binary_range_down: int,
    power_of_two_range_up: int,
    power_of_two_range_down: int,
    target_vram_fraction: float,
    predicted_budget_fraction: float = 0.85,
) -> SchedulerSettings:
    gpu = GpuSchedulerSettings()
    gpu.mode = mode
    mps_pipe_directory = _mps_directory_env("BENCH_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    mps_log_directory = _mps_directory_env("BENCH_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log")
    if mode == "parallel_time_aware":
        gpu.memory = GpuMemorySettings(
            safe_vram_budget_gib=vram_budget_gib,
            predicted_budget_fraction=predicted_budget_fraction,
            live_admission_stop_fraction=0.90,
            live_admission_resume_fraction=0.85,
        )
    else:
        gpu.memory = GpuMemorySettings(
            safe_vram_budget_gib=vram_budget_gib,
            predicted_budget_fraction=predicted_budget_fraction,
            live_admission_stop_fraction=0.92,
            live_admission_resume_fraction=0.87,
        )
    gpu.profiling = GpuProfilingSettings(
        warmup_steps=3,
        solo_probe_steps=6,
        pair_probe_steps=4,
        reuse_profile_if_confidence_ge=0.8,
    )
    gpu.max_packed_jobs_per_gpu = max(1, int(max_packed_jobs_per_gpu))
    gpu.parallel_job_cap = max(1, int(max_packed_jobs_per_gpu))
    gpu.allow_three_way_packing = gpu.max_packed_jobs_per_gpu >= 3

    if backend == "exclusive":
        gpu.backend_priority = ["exclusive"]
    elif backend == "mps":
        gpu.backend_priority = ["mps", "exclusive"]
        gpu.mps = MPSSettings(
            enabled=True,
            pipe_directory=mps_pipe_directory,
            log_directory=mps_log_directory,
        )
    elif backend == "stream":
        gpu.backend_priority = ["stream", "exclusive"]
        gpu.stream = StreamSettings(enabled=True)
    elif backend == "cuda_process":
        gpu.backend_priority = ["cuda_process", "exclusive"]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    gpu.thresholds = GpuThresholdSettings(
        pack_reject_max_slowdown=(1.30 if mode == "parallel_time_aware" else 1.50),
        latency_sensitive_max_slowdown=1.30,
    )
    gpu.parallel_optimizer = ParallelOptimizerSettings(
        batch_search_mode=batch_search or "binary",
        target_vram_fraction=target_vram_fraction,
        max_probe_jobs=max(3, int(max_packed_jobs_per_gpu)),
        binary_range_up=binary_range_up,
        binary_range_down=binary_range_down,
        power_of_two_range_up=power_of_two_range_up,
        power_of_two_range_down=power_of_two_range_down,
    )

    if batch_search in ("binary", "power_of_two"):
        gpu.batch_probe_enabled = True
        gpu.batch_probe_search_mode = batch_search
    else:
        gpu.batch_probe_enabled = False

    return SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        baseline_cache={
            "warm_queue_policy": cache_warm_policy,
            "warm_queue_top_k": max(0, int(cache_warm_top_k)),
            "entry_capacity": cache_entry_capacity,
            "max_ram_percent": cache_max_ram_percent,
            "memory_budget_bytes": max(0, int(float(cache_memory_budget_gib) * (1024**3))),
        },
        gpu_scheduler=gpu,
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
    )


def seed_solo_profile_for_job(api: SchedulerClient, job, vram_mb: int) -> None:
    """Inject placeholder solo profile so packed planners can make progress quickly."""
    sig = getattr(job.packing, "signature", None)
    if not sig:
        return
    api.upsert_solo_profile(
        SoloProfile(
            signature=sig,
            family=getattr(job.packing, "family", None),
            peak_vram_mb=vram_mb,
            avg_gpu_utilization=0.4,
            avg_memory_utilization=0.4,
            sample_count=1,
            last_job_id=job.job_id,
            metadata={"seeded": True, "source": "replay_scheduler"},
        )
    )


def _elapsed_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    started = parse_timestamp(started_at)
    finished = parse_timestamp(finished_at)
    if started is None or finished is None:
        return None
    return round((finished - started).total_seconds(), 3)


def _current_jobs(api: SchedulerClient, submitted_ids: list[str]) -> list[TrainingJob]:
    submitted = set(submitted_ids)
    return [job for job in api.list_jobs() if job.job_id in submitted]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _trace_metrics(jobs: list[TrainingJob], *, starvation_timeout_seconds: float) -> dict[str, float | int | bool]:
    completed = [job for job in jobs if parse_timestamp(job.submitted_at) and parse_timestamp(job.finished_at)]
    flows = [
        (parse_timestamp(job.finished_at) - parse_timestamp(job.submitted_at)).total_seconds()  # type: ignore[operator]
        for job in completed
    ]
    waits = [
        (parse_timestamp(job.started_at) - parse_timestamp(job.submitted_at)).total_seconds()  # type: ignore[operator]
        for job in jobs
        if parse_timestamp(job.started_at) and parse_timestamp(job.submitted_at)
    ]
    releases = [parse_timestamp(job.submitted_at) for job in completed]
    finishes = [parse_timestamp(job.finished_at) for job in completed]
    makespan = (
        (max(finishes) - min(releases)).total_seconds()  # type: ignore[type-var,operator]
        if releases and finishes
        else 0.0
    )
    minimum_priority = min((job.priority for job in completed), default=0)
    weights = [1.0 + 0.10 * (job.priority - minimum_priority) for job in completed]
    weighted_flow = (
        sum(weight * flow for weight, flow in zip(weights, flows, strict=True)) / sum(weights)
        if weights
        else 0.0
    )
    return {
        "complete": len(completed) == len(jobs),
        "makespan_seconds": makespan,
        "total_flow_seconds": sum(flows),
        "mean_flow_seconds": statistics.fmean(flows) if flows else 0.0,
        "weighted_mean_flow_seconds": weighted_flow,
        "median_flow_seconds": statistics.median(flows) if flows else 0.0,
        "p95_flow_seconds": _percentile(flows, 0.95),
        "max_wait_seconds": max(waits, default=0.0),
        "starvation_count": sum(wait >= starvation_timeout_seconds for wait in waits),
        "jobs_per_hour": 3600.0 * len(completed) / makespan if makespan > 0 else 0.0,
    }


def _load_time_aware_profiles(api: SchedulerClient, profile_path: Path) -> int:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported time-aware profile schema in {profile_path}")
    hardware = api.store.hardware_profile()
    manifest_hardware = payload.get("hardware") or {}
    if manifest_hardware.get("hardware_key") != hardware.hardware_key:
        raise ValueError(
            "Time-aware profile hardware mismatch: "
            f"manifest={manifest_hardware.get('gpu_name')} ({manifest_hardware.get('hardware_key')}), "
            f"current={hardware.gpu_name} ({hardware.hardware_key})"
        )
    observations = payload.get("batch_size_observations") or []
    for raw in observations:
        observation = BatchSizeObservation(**dict(raw))
        if observation.hardware_key != hardware.hardware_key:
            raise ValueError("Time-aware profile contains an observation for different hardware")
        api.upsert_batch_size_observation(observation)
    return len(observations)


def _write_time_aware_profiles(
    api: SchedulerClient,
    profile_path: Path,
    jobs: list[TrainingJob],
) -> int:
    hardware = api.store.hardware_profile()
    observations: dict[str, BatchSizeObservation] = {}
    for job in jobs:
        for observation in api.list_batch_size_observations(
            model_key=str(job.batch_probe.model_key or job.baseline_model_id),
            shape_signature=build_batch_probe_shape_signature(job),
            hardware_key=hardware.hardware_key,
            backend_name="exclusive",
        ):
            observations[observation.observation_key] = observation
    payload = {
        "schema_version": 1,
        "hardware": hardware.to_dict(),
        "batch_size_observations": [
            observations[key].to_dict() for key in sorted(observations)
        ],
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(observations)


def _placement_memory_metrics(
    api: SchedulerClient,
    jobs: list[TrainingJob],
    *,
    mode: str,
    backend: str,
    predicted_budget_fraction: float,
) -> dict[str, float | int]:
    submitted = {job.job_id for job in jobs}
    jobs_by_id = {job.job_id: job for job in jobs}
    predicted: list[float] = []
    actual: list[float] = []
    over_budget = 0
    for profile in api.list_combination_profiles(
        hardware_key=api.store.hardware_key(),
        backend_name=backend,
        scheduler_mode=mode,
    ):
        metadata = profile.metadata or {}
        if not submitted.intersection(str(job_id) for job_id in metadata.get("job_ids", [])):
            continue
        breakdown = metadata.get("objective_breakdown") or {}
        candidate_memory = breakdown.get("candidate_memory_mb")
        if isinstance(candidate_memory, (int, float)):
            predicted.append(float(candidate_memory))
        else:
            member_estimates = [
                jobs_by_id[str(job_id)].resource_requirements.estimated_avg_vram_mb
                for job_id in metadata.get("job_ids", [])
                if str(job_id) in jobs_by_id
            ]
            if member_estimates and all(value is not None for value in member_estimates):
                predicted.append(sum(float(value) for value in member_estimates if value is not None))
        if profile.avg_vram_mb is not None:
            actual.append(float(profile.avg_vram_mb))
        if profile.avg_vram_mb is not None and profile.memory_total_mb:
            budget = profile.memory_total_mb * predicted_budget_fraction
            over_budget += int(profile.avg_vram_mb > budget)
    if not actual:
        for job in jobs:
            if not job.packing.signature:
                continue
            profile = api.get_solo_profile(job.packing.signature)
            if profile is None or profile.last_job_id != job.job_id or profile.avg_vram_mb is None:
                continue
            actual.append(float(profile.avg_vram_mb))
            estimate = job.resource_requirements.estimated_avg_vram_mb
            if estimate is not None:
                predicted.append(float(estimate))
            if profile.avg_vram_mb is not None and api.store.hardware_profile().total_vram_mb:
                budget = api.store.hardware_profile().total_vram_mb * predicted_budget_fraction
                over_budget += int(profile.avg_vram_mb > budget)
    return {
        "predicted_avg_vram_mb": statistics.fmean(predicted) if predicted else 0.0,
        "actual_avg_vram_mb": statistics.fmean(actual) if actual else 0.0,
        "actual_memory_over_budget_count": over_budget,
        "measured_placement_count": len(actual),
    }


def _execution_metrics(api: SchedulerClient, jobs: list[TrainingJob], *, backend: str) -> dict[str, float | int]:
    job_ids = {job.job_id for job in jobs}
    slowdowns: list[float] = []
    for profile in api.store.list_pair_profiles(
        hardware_key=api.store.hardware_key(),
        backend_name=backend,
    ):
        per_member = (profile.metadata or {}).get("per_member_slowdown") or {}
        matched = [
            float(value)
            for job_id, value in per_member.items()
            if str(job_id) in job_ids and isinstance(value, (int, float))
        ]
        if matched:
            slowdowns.extend(matched)
    return {
        "average_slowdown": statistics.fmean(slowdowns) if slowdowns else 1.0,
        "measured_slowdown_members": len(slowdowns),
        "early_stopped_epochs_saved": sum(
            int((job.metadata.get("early_stopping_result") or {}).get("epochs_saved", 0))
            for job in jobs
        ),
        "early_stopped_wall_time_saved_seconds": sum(
            float(
                (job.metadata.get("early_stopping_result") or {}).get(
                    "estimated_wall_time_saved_seconds",
                    0.0,
                )
            )
            for job in jobs
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "serial_basic",
            "serial_batch_optimized",
            "parallel_default",
            "parallel_batch_optimized",
            "parallel_time_aware",
        ],
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["exclusive", "mps", "stream", "cuda_process"],
    )
    parser.add_argument("--batch-search", default="off", choices=["off", "binary", "power_of_two"])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--data-root", help="Override each trace row's dataset root for this replay.")
    parser.add_argument("--vram-budget-gib", type=float, default=DEFAULT_VRAM_BUDGET_GIB)
    parser.add_argument("--max-packed-jobs-per-gpu", type=int, default=2)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--duration-s", type=float, default=2700.0)
    parser.add_argument("--cache-warm-policy", choices=["top_k", "budget_only"], default="top_k")
    parser.add_argument("--cache-warm-top-k", type=int, default=DEFAULT_CACHE_WARM_TOP_K)
    parser.add_argument("--cache-entry-capacity", type=int, default=DEFAULT_CACHE_ENTRY_CAPACITY)
    parser.add_argument("--cache-max-ram-percent", type=float, default=DEFAULT_CACHE_MAX_RAM_PERCENT)
    parser.add_argument("--cache-memory-budget-gib", type=float, default=DEFAULT_CACHE_MEMORY_BUDGET_GIB)
    parser.add_argument("--binary-range-up", type=int, default=DEFAULT_BINARY_RANGE_UP)
    parser.add_argument("--binary-range-down", type=int, default=DEFAULT_BINARY_RANGE_DOWN)
    parser.add_argument("--power-of-two-range-up", type=int, default=DEFAULT_POWER_OF_TWO_RANGE_UP)
    parser.add_argument("--power-of-two-range-down", type=int, default=DEFAULT_POWER_OF_TWO_RANGE_DOWN)
    parser.add_argument("--target-vram-fraction", type=float, default=0.97)
    parser.add_argument("--predicted-budget-fraction", type=float, default=0.85)
    parser.add_argument(
        "--time-aware-profile-input",
        help="Hardware-matched five-option profile manifest produced by a calibration replay.",
    )
    parser.add_argument(
        "--time-aware-profile-output",
        help="Write hardware-tagged five-option measurements collected by this replay.",
    )
    parser.add_argument(
        "--calibrate-time-aware",
        action="store_true",
        help="Run every trace job as an exclusive five-option probe and skip its full training body.",
    )
    args = parser.parse_args()

    if args.calibrate_time_aware and args.mode != "parallel_time_aware":
        parser.error("--calibrate-time-aware requires --mode parallel_time_aware")
    if args.calibrate_time_aware and args.batch_search == "off":
        parser.error("--calibrate-time-aware requires --batch-search power_of_two")
    if args.mode == "parallel_time_aware" and not args.calibrate_time_aware and not args.time_aware_profile_input:
        parser.error(
            "measured parallel_time_aware replay requires --time-aware-profile-input; "
            "create it with --calibrate-time-aware --time-aware-profile-output"
        )

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.code_cache_dir).mkdir(parents=True, exist_ok=True)

    runtime_root = Path(args.runtime_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    workdir_root = Path(os.environ.get("REPLAY_WORKDIR_ROOT", f"/tmp/replay_workdirs/{args.config_id}")).resolve()
    workdir_root.mkdir(parents=True, exist_ok=True)

    trace: list[dict[str, Any]] = []
    with Path(args.trace).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                trace.append(json.loads(line))

    code_paths: dict[int, str] = {}
    for step in trace:
        step_idx = int(step["step_idx"])
        code = str(step.get("code") or "")
        if not code:
            continue
        code_path = Path(args.code_cache_dir) / f"step_{step_idx:03d}.py"
        code_path.write_text(code, encoding="utf-8")
        code_paths[step_idx] = str(code_path.resolve())

    batch_search = None if args.batch_search == "off" else args.batch_search
    settings = build_settings(
        mode=args.mode,
        backend=args.backend,
        batch_search=batch_search,
        max_packed_jobs_per_gpu=args.max_packed_jobs_per_gpu,
        vram_budget_gib=args.vram_budget_gib,
        runtime_root=runtime_root,
        cache_warm_policy=args.cache_warm_policy,
        cache_warm_top_k=args.cache_warm_top_k,
        cache_entry_capacity=args.cache_entry_capacity,
        cache_max_ram_percent=args.cache_max_ram_percent,
        cache_memory_budget_gib=args.cache_memory_budget_gib,
        binary_range_up=args.binary_range_up,
        binary_range_down=args.binary_range_down,
        power_of_two_range_up=args.power_of_two_range_up,
        power_of_two_range_down=args.power_of_two_range_down,
        target_vram_fraction=args.target_vram_fraction,
        predicted_budget_fraction=args.predicted_budget_fraction,
    )

    api = SchedulerClient(settings)
    imported_profile_count = 0
    if args.time_aware_profile_input:
        imported_profile_count = _load_time_aware_profiles(api, Path(args.time_aware_profile_input))
    service = api.create_service().start(background=True)

    seed_solo = args.mode in {"parallel_batch_optimized", "parallel_time_aware"}
    submitted_ids: list[str] = []
    submit_t0 = time.time()
    deadline = submit_t0 + args.duration_s
    try:
        for step in sorted(trace, key=lambda item: (float(item.get("exec_submit_at") or 0.0), int(item["step_idx"]))):
            release_offset = (
                0.0
                if args.calibrate_time_aware
                else max(0.0, float(step.get("exec_submit_at") or 0.0))
            )
            remaining_delay = submit_t0 + release_offset - time.time()
            if remaining_delay > 0:
                time.sleep(remaining_delay)
            step_idx = int(step["step_idx"])
            startpoint_path = str(step["startpoint_path"])
            model_name = str(step["model_name"])
            batch_size = int(step["bs"])
            max_batch_size = int(step.get("max_bs") or batch_size)
            dataset_seed = int(step.get("dataset_seed") or 42)
            working_dir = (workdir_root / f"step_{step_idx:03d}").resolve()
            working_dir.mkdir(parents=True, exist_ok=True)
            runner_kwargs = {
                "data_root": str(args.data_root or step["data_root"]),
                "subset_size": int(step.get("subset") or 4000),
                "batch_size": batch_size,
                "epochs": int(step.get("epochs") or 1),
                "learning_rate": float(step.get("learning_rate") or 1e-3),
                "working_dir": str(working_dir),
                "dataset_seed": dataset_seed,
                "model_name": model_name,
                "num_classes": 5,
                "probe_max_batch_size": max_batch_size,
            }
            batch_probe = BatchProbeSpec(
                enabled=batch_search in {"binary", "power_of_two"},
                probe_target=(
                    "localml_scheduler.examples.benchmark_timm_runner:probe_timm_benchmark_batch_size" if batch_search in {"binary", "power_of_two"} else None
                ),
                batch_param_name="batch_size",
                model_key=str(step.get("startpoint_id") or model_name),
                search_mode=batch_search,
                shape_hints={
                    "model_name": model_name,
                    "subset_size": int(step.get("subset") or 4000),
                    "image_size": 224,
                },
            )
            vram_est = int(step.get("estimated_vram_mb") or 4000)
            job = build_mlevolve_job(
                workflow_id=f"replay-{args.config_id}",
                baseline_model_id=str(step.get("startpoint_id") or model_name),
                baseline_model_path=startpoint_path,
                runner_target="localml_scheduler.examples.benchmark_timm_runner:run_timm_benchmark_job",
                runner_kwargs=runner_kwargs,
                priority=5,
                task_type=str(step.get("agent_used") or "unknown"),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=None, save_every_epoch=False),
                batch_probe=batch_probe,
                resource_requirements=ResourceRequirements(
                    requires_gpu=True,
                    gpu_slots=1,
                    estimated_vram_mb=vram_est,
                ),
                packing_family=str(step.get("model_class") or "unknown"),
                packing_signature=f"{step.get('model_class') or 'unknown'}:{model_name}:bs{batch_size}",
                packing_eligible=True,
                max_steps=None,
                max_epochs=int(step.get("epochs") or 1),
                metadata={
                    "step_idx": step_idx,
                    "agent_used": step.get("agent_used"),
                    "branch_id": step.get("branch_id"),
                    "model_class": step.get("model_class"),
                    "model_name": model_name,
                    "bs": batch_size,
                    "max_bs": max_batch_size,
                    "code_path": code_paths.get(step_idx) or step.get("code_path"),
                    "trace_release_offset_seconds": release_offset,
                    "benchmark_probe_only": bool(args.calibrate_time_aware),
                },
            )
            if args.calibrate_time_aware and batch_size > 0 and (batch_size & (batch_size - 1)) == 0:
                job.scheduling_class = SchedulingClass.EXCLUSIVE_PROBE
            job = api.submit(job)
            if seed_solo:
                seed_solo_profile_for_job(api, job, vram_mb=vram_est)
            submitted_ids.append(job.job_id)

        while time.time() < deadline:
            jobs = _current_jobs(api, submitted_ids)
            if all(job.status.is_terminal for job in jobs):
                break
            time.sleep(2.0)

        # Workers persist terminal status before the service consumes their
        # final snapshots. Give the scheduler loop a bounded chance to close
        # run groups and write measured VRAM/slowdown profiles before reporting.
        reconciliation_deadline = min(deadline, time.time() + 10.0)
        while service._active_runs and time.time() < reconciliation_deadline:
            time.sleep(max(0.05, settings.scheduler_poll_interval_seconds))

        current_jobs = _current_jobs(api, submitted_ids)
        deadline_reached = time.time() >= deadline and any(not job.status.is_terminal for job in current_jobs)

        treat_elapsed = time.time() - submit_t0

        pack_events = 0
        exclusive_events = 0
        pack_pair_keys: dict[str, int] = {}
        packed_group_size_counts: dict[str, int] = {}
        placement_mode_counts: dict[str, int] = {}
        cache_event_counts: dict[str, int] = {}
        batch_probe_event_counts: dict[str, int] = {}
        events_path = settings.logs_dir / "events.jsonl"
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    event_type = str(event.get("event_type") or "")
                    if event_type.startswith("cache_"):
                        cache_event_counts[event_type] = cache_event_counts.get(event_type, 0) + 1
                    if event_type.startswith("batch_probe_"):
                        batch_probe_event_counts[event_type] = batch_probe_event_counts.get(event_type, 0) + 1
                    if event_type == "job_dispatched":
                        payload = event.get("payload") or {}
                        placement_mode = str(payload.get("placement_mode") or "")
                        placement_mode_counts[placement_mode] = placement_mode_counts.get(placement_mode, 0) + 1
                        if placement_mode == "exclusive":
                            exclusive_events += 1
                    elif event_type == "packed_pair_dispatched":
                        pack_events += 1
                        payload = event.get("payload") or {}
                        ids = payload.get("job_ids") or []
                        if len(ids) == 2:
                            key = "+".join(sorted(str(item) for item in ids))
                            pack_pair_keys[key] = pack_pair_keys.get(key, 0) + 1
                            packed_group_size_counts["2"] = packed_group_size_counts.get("2", 0) + 1
                    elif event_type == "packed_group_dispatched":
                        pack_events += 1
                        payload = event.get("payload") or {}
                        ids = payload.get("job_ids") or []
                        if ids:
                            size_key = str(len(ids))
                            packed_group_size_counts[size_key] = packed_group_size_counts.get(size_key, 0) + 1

        per_job = []
        by_status: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        for job in current_jobs:
            by_status[job.status.value] = by_status.get(job.status.value, 0) + 1
            outcome = classify_job_outcome(job, externally_timed_out=deadline_reached and not job.status.is_terminal)
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
            per_job.append(
                {
                    "job_id": job.job_id,
                    "step_idx": (job.metadata or {}).get("step_idx"),
                    "agent_used": (job.metadata or {}).get("agent_used"),
                    "model_class": (job.metadata or {}).get("model_class"),
                    "model_name": (job.metadata or {}).get("model_name"),
                    "bs": (job.metadata or {}).get("bs"),
                    "resolved_batch_size": (job.metadata or {}).get("resolved_batch_size"),
                    "status": job.status.value,
                    "outcome": outcome,
                    "submitted_at": job.submitted_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "elapsed_s": _elapsed_seconds(job.started_at, job.finished_at),
                }
            )

        cache_snapshot = api.cache_stats().get("stats", {})
        trace_metrics = _trace_metrics(
            current_jobs,
            starvation_timeout_seconds=settings.gpu_scheduler.starvation_timeout_seconds,
        )
        memory_metrics = _placement_memory_metrics(
            api,
            current_jobs,
            mode=args.mode,
            backend=args.backend,
            predicted_budget_fraction=args.predicted_budget_fraction,
        )
        execution_metrics = _execution_metrics(api, current_jobs, backend=args.backend)
        exported_profile_count = 0
        if args.time_aware_profile_output:
            exported_profile_count = _write_time_aware_profiles(
                api,
                Path(args.time_aware_profile_output),
                current_jobs,
            )
        summary = {
            "config_id": args.config_id,
            "mode": args.mode,
            "backend": args.backend,
            "batch_search": args.batch_search,
            "vram_budget_gib": args.vram_budget_gib,
            "target_vram_fraction": args.target_vram_fraction,
            "trace_path": args.trace,
            "n_jobs": len(submitted_ids),
            "by_status": by_status,
            "by_outcome": by_outcome,
            "external_deadline_reached": deadline_reached,
            "treat_elapsed_s": round(treat_elapsed, 3),
            "trace_metrics": trace_metrics,
            "placement_memory_metrics": memory_metrics,
            "execution_metrics": execution_metrics,
            "hardware": api.store.hardware_profile().to_dict(),
            "time_aware_profiles": {
                "input": args.time_aware_profile_input,
                "imported_observations": imported_profile_count,
                "output": args.time_aware_profile_output,
                "exported_observations": exported_profile_count,
                "calibration_only": bool(args.calibrate_time_aware),
            },
            "n_pack_dispatches": pack_events,
            "n_exclusive_dispatches": exclusive_events,
            "pack_rate": pack_events / max(1, pack_events + exclusive_events),
            "pack_pair_keys": pack_pair_keys,
            "packed_group_size_counts": packed_group_size_counts,
            "placement_mode_counts": placement_mode_counts,
            "packing_policy": {
                "max_packed_jobs_per_gpu": args.max_packed_jobs_per_gpu,
            },
            "cache_policy": {
                "warm_queue_policy": args.cache_warm_policy,
                "warm_queue_top_k": args.cache_warm_top_k,
                "entry_capacity": args.cache_entry_capacity,
                "max_ram_percent": args.cache_max_ram_percent,
                "memory_budget_gib": args.cache_memory_budget_gib,
            },
            "cache_stats": cache_snapshot,
            "cache_event_counts": cache_event_counts,
            "optimizer_thresholds": {
                "binary_range_up": args.binary_range_up,
                "binary_range_down": args.binary_range_down,
                "power_of_two_range_up": args.power_of_two_range_up,
                "power_of_two_range_down": args.power_of_two_range_down,
            },
            "batch_probe_event_counts": batch_probe_event_counts,
            "per_job": per_job,
        }
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "per_job"},
                indent=2,
            )
        )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
