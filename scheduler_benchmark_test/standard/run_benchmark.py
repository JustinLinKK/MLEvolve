"""Execute or resume the five-arm standard scheduler benchmark matrix."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import argparse
import json
import os
import time

import torch

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.prediction import PredictionRouter
from scheduler_benchmark_test.replay_multiprocess_baseline import replay_multiprocess_baseline
from scheduler_benchmark_test.replay_scheduler_timeline import replay_fixture

from . import A10_VRAM_CAP_MIB, DEFAULT_FIXTURE_ROOT, JOB_COUNT, PACKAGE_ROOT
from .generate_fixture import validate_dataset_identity, write_fixture
from .reporting import aggregate_reports, summarize_case


ARM_CONFIGS: dict[str, dict[str, str | None]] = {
    "fifo": {"prediction_mode": None, "backend": None},
    "branch_cuda": {"prediction_mode": "branch_profile", "backend": "cuda_process"},
    "branch_stream": {"prediction_mode": "branch_profile", "backend": "stream"},
    "ml_cuda": {"prediction_mode": "ml_predictor", "backend": "cuda_process"},
    "ml_stream": {"prediction_mode": "ml_predictor", "backend": "stream"},
}
DEFAULT_ARMS = ("fifo", "branch_cuda", "branch_stream", "ml_cuda", "ml_stream")


def rotated_arm_order(repetition: int, arms: tuple[str, ...] = DEFAULT_ARMS) -> tuple[str, ...]:
    if not arms:
        return ()
    offset = ((int(repetition) - 1) * 2) % len(arms)
    return arms[offset:] + arms[:offset]


def resolve_vram_budget_fraction(total_vram_mib: int | float | None) -> float:
    total = float(total_vram_mib or 0)
    if total <= 0:
        return A10_VRAM_CAP_MIB / 24_576.0
    return min(0.95, A10_VRAM_CAP_MIB / total)


def scheduler_settings_overlay(*, prediction_mode: str, backend: str, total_vram_mib: float | None) -> dict[str, Any]:
    return {
        "prediction": {
            "mode": prediction_mode,
            "fallback_to_exclusive": True,
            "branch": {"enabled": True},
            "ml": {"enabled": prediction_mode == "ml_predictor"},
        },
        "gpu_scheduler": {
            "enabled": True,
            "mode": "adaptive",
            "backend_priority": [backend, "exclusive"],
            "max_packed_jobs_per_gpu": 8,
            "candidate_window_size": 16,
            "checkpoint_preemption_enabled": False,
            "batch_probe_enabled": True,
            "batch_probe_min_batch_size": 32,
            "batch_probe_max_batch_size": 256,
            "batch_probe_search_mode": "power_of_two",
            "memory": {"vram_budget_fraction": resolve_vram_budget_fraction(total_vram_mib)},
            "early_stop": {"enabled": False, "plot_enabled": False},
            "mps": {"enabled": False},
            "cuda_process": {"enabled": backend == "cuda_process"},
            "stream": {"enabled": backend == "stream"},
            "submission_defaults": {"backend_allowlist": [backend]},
        },
    }


def _assert_primary_healthy(overlay: dict[str, Any]) -> None:
    settings = SchedulerSettings.from_dict(overlay)
    router = PredictionRouter.from_settings(settings)
    provider = router.ml_provider
    health = provider.health() if provider is not None and hasattr(provider, "health") else None
    if health is None or not health.healthy:
        reason = getattr(health, "reason", "provider unavailable")
        raise RuntimeError(f"ml_predictor preflight failed: {reason}")


def _device_info() -> tuple[str | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    properties = torch.cuda.get_device_properties(0)
    return properties.name, properties.total_memory / (1024**2)


@contextmanager
def _benchmark_environment(data_root: str | Path | None) -> Iterator[None]:
    keys = (
        "HISTOPATH_DATA_ROOT",
        "STANDARD_BENCH_VRAM_CAP_MIB",
        "STANDARD_BENCH_EPOCHS",
        "STANDARD_BENCH_ALLOW_PARTIAL",
        "STANDARD_BENCH_MAX_SAMPLES",
        "STANDARD_BENCH_RESULT_DIR",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
    )
    previous = {key: os.environ.get(key) for key in keys}
    repo_root = str(PACKAGE_ROOT.parent.parent)
    path_parts = [repo_root]
    if previous["PYTHONPATH"]:
        path_parts.append(str(previous["PYTHONPATH"]))
    try:
        for key in (
            "STANDARD_BENCH_EPOCHS",
            "STANDARD_BENCH_ALLOW_PARTIAL",
            "STANDARD_BENCH_MAX_SAMPLES",
            "STANDARD_BENCH_RESULT_DIR",
        ):
            os.environ.pop(key, None)
        if data_root is not None:
            os.environ["HISTOPATH_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
        os.environ["STANDARD_BENCH_VRAM_CAP_MIB"] = str(A10_VRAM_CAP_MIB)
        os.environ["PYTHONPATH"] = os.pathsep.join(path_parts)
        os.environ["PYTHONUNBUFFERED"] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _attempt_root(arm_root: Path, *, resume: bool) -> tuple[Path | None, dict[str, Any] | None]:
    attempts = sorted(path for path in arm_root.glob("attempt-*") if path.is_dir()) if arm_root.exists() else []
    if attempts:
        latest_status = json.loads((attempts[-1] / "case_status.json").read_text(encoding="utf-8")) if (attempts[-1] / "case_status.json").exists() else None
        if resume and latest_status and latest_status.get("state") == "complete":
            return None, latest_status
        if not resume:
            raise FileExistsError(f"Evidence already exists for {arm_root}; pass --resume to preserve it and continue")
    attempt = arm_root / f"attempt-{len(attempts) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt, None


def run_matrix(
    *,
    fixture: str | Path,
    output_root: str | Path,
    data_root: str | Path | None,
    repetitions: int = 3,
    arms: tuple[str, ...] = DEFAULT_ARMS,
    runner_mode: str = "real",
    resume: bool = False,
    no_sleep: bool = False,
    speedup: float = 1.0,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    fixture_path = Path(fixture).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    drift = write_fixture(fixture_path, check=True)
    if drift:
        raise ValueError(f"Fixture checksum drift: {', '.join(drift[:5])}")
    if runner_mode == "real":
        if data_root is None:
            raise ValueError("Real benchmark execution requires --data-root or HISTOPATH_DATA_ROOT")
        validate_dataset_identity(data_root)
        if not torch.cuda.is_available():
            raise RuntimeError("Real standard benchmark execution requires a CUDA GPU")
    unknown = sorted(set(arms) - set(ARM_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown benchmark arms: {', '.join(unknown)}")

    gpu_name, total_vram_mib = _device_info()
    summaries: list[dict[str, Any]] = []
    for repetition in range(1, int(repetitions) + 1):
        order = rotated_arm_order(repetition, arms)
        for order_index, arm in enumerate(order):
            config = ARM_CONFIGS[arm]
            arm_root = output / f"rep-{repetition:02d}" / arm
            attempt, previous_status = _attempt_root(arm_root, resume=resume)
            if attempt is None:
                summary_path = Path(str(previous_status.get("summary_path"))) if previous_status else Path()
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
                if summary:
                    summaries.append(summary)
                print(f"rep {repetition} {arm}: already complete", flush=True)
                continue
            status_path = attempt / "case_status.json"
            metadata = {
                "schema_version": "standard-histopath-case-state-v1",
                "state": "running",
                "arm": arm,
                "repetition": repetition,
                "rotated_order": list(order),
                "order_index": order_index,
                "runner_mode": runner_mode,
                "speedup": speedup,
                "no_sleep": no_sleep,
                "physical_gpu": gpu_name,
                "physical_vram_mib": total_vram_mib,
                "vram_safety_cap_mib": A10_VRAM_CAP_MIB,
                "started_unix_seconds": time.time(),
            }
            status_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"rep {repetition} {arm}: starting {attempt.name}", flush=True)
            try:
                with _benchmark_environment(data_root):
                    if arm == "fifo":
                        replay_multiprocess_baseline(
                            fixture=fixture_path,
                            output_root=attempt,
                            runner_mode=runner_mode,
                            parallelism=1,
                            speedup=speedup,
                            no_sleep=no_sleep,
                            wait_for_all=True,
                            cancel_policy="ignore",
                            job_filter="script",
                        )
                    else:
                        overlay = scheduler_settings_overlay(
                            prediction_mode=str(config["prediction_mode"]),
                            backend=str(config["backend"]),
                            total_vram_mib=total_vram_mib,
                        )
                        if config["prediction_mode"] == "ml_predictor":
                            _assert_primary_healthy(overlay)
                        replay_fixture(
                            fixture=fixture_path,
                            output_root=attempt,
                            runner_mode=runner_mode,
                            speedup=speedup,
                            no_sleep=no_sleep,
                            wait_for_all=True,
                            cancel_policy="ignore",
                            clean_profile_db=True,
                            settings_overrides=overlay,
                        )
                summary = summarize_case(
                    attempt,
                    arm=arm,
                    repetition=repetition,
                    runner_mode=runner_mode,
                    backend=config["backend"],
                    prediction_mode=config["prediction_mode"],
                )
                if arm == "fifo" and summary["maximum_observed_concurrency"] != 1:
                    raise AssertionError(
                        f"FIFO maximum concurrency was {summary['maximum_observed_concurrency']}, expected exactly one"
                    )
                if runner_mode == "real" and (no_sleep or speedup != 1.0):
                    summary["accepted"] = False
                    summary["acceptance_note"] = "arrival timing was altered"
                    (attempt / "case_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                summaries.append(summary)
                metadata.update(
                    {
                        "state": "complete",
                        "accepted": summary["accepted"],
                        "summary_path": str(attempt / "case_summary.json"),
                        "finished_unix_seconds": time.time(),
                    }
                )
            except Exception as exc:
                metadata.update({"state": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_unix_seconds": time.time()})
                status_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if not continue_on_error:
                    raise
                print(f"rep {repetition} {arm}: FAILED: {exc}", flush=True)
                continue
            status_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"rep {repetition} {arm}: complete", flush=True)
    aggregate_reports(summaries, output)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", default=os.environ.get("HISTOPATH_DATA_ROOT"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARM_CONFIGS), default=list(DEFAULT_ARMS))
    parser.add_argument("--runner-mode", choices=("real", "noop"), default="real")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = run_matrix(
        fixture=args.fixture,
        output_root=args.output_root,
        data_root=args.data_root,
        repetitions=args.repetitions,
        arms=tuple(args.arms),
        runner_mode=args.runner_mode,
        resume=args.resume,
        no_sleep=args.no_sleep,
        speedup=args.speedup,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps({"case_count": len(summaries), "output_root": str(Path(args.output_root).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
