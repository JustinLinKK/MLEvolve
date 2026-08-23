"""Run and aggregate the repeated real-GPU time-aware scheduler benchmark.

The driver first records hardware-specific five-option probe measurements, then
runs a one-job-cap time-aware control and normal ``parallel_time_aware`` from
isolated runtime directories. Historical fill-policy comparisons belong to the
deterministic simulator, not the production replay driver. It refuses to label a report as A10 data
unless the detected GPU matches the requested hardware expression.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from localml_scheduler.hardware import HardwareProfile, detect_hardware_profile


METRIC_KEYS = (
    "makespan_seconds",
    "total_flow_seconds",
    "mean_flow_seconds",
    "weighted_mean_flow_seconds",
    "median_flow_seconds",
    "p95_flow_seconds",
    "max_wait_seconds",
    "starvation_count",
    "jobs_per_hour",
)


@dataclass(frozen=True, slots=True)
class Policy:
    report_name: str
    mode: str
    backend: str
    needs_profiles: bool = False


def hardware_matches(profile: HardwareProfile, required_gpu_name: str) -> bool:
    return re.search(required_gpu_name, profile.gpu_name, flags=re.IGNORECASE) is not None


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"n": 0, "mean": 0.0, "sample_variance": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(materialized),
        "mean": statistics.fmean(materialized),
        "sample_variance": statistics.variance(materialized) if len(materialized) > 1 else 0.0,
        "stdev": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
        "min": min(materialized),
        "max": max(materialized),
    }


def aggregate_summaries(
    summaries: dict[str, list[dict[str, Any]]],
    *,
    hardware: HardwareProfile,
    required_gpu_name: str,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy_name, runs in summaries.items():
        metrics: dict[str, Any] = {}
        for metric_name in METRIC_KEYS:
            metrics[metric_name] = _stats(
                float((run.get("trace_metrics") or {}).get(metric_name, 0.0))
                for run in runs
            )
        for metric_name in (
            "predicted_avg_vram_mb",
            "actual_avg_vram_mb",
            "actual_memory_over_budget_count",
            "measured_placement_count",
        ):
            metrics[metric_name] = _stats(
                float((run.get("placement_memory_metrics") or {}).get(metric_name, 0.0))
                for run in runs
            )
        for metric_name in (
            "average_slowdown",
            "early_stopped_epochs_saved",
            "early_stopped_wall_time_saved_seconds",
        ):
            metrics[metric_name] = _stats(
                float((run.get("execution_metrics") or {}).get(metric_name, 0.0))
                for run in runs
            )
        policies[policy_name] = {
            "runs": len(runs),
            "all_complete": all(bool((run.get("trace_metrics") or {}).get("complete")) for run in runs),
            "external_deadline_count": sum(bool(run.get("external_deadline_reached")) for run in runs),
            "metrics": metrics,
        }

    serial_makespan = float(
        (((policies.get("serial_fifo") or {}).get("metrics") or {}).get("makespan_seconds") or {}).get("mean", 0.0)
    )
    for policy in policies.values():
        mean_makespan = float(policy["metrics"]["makespan_seconds"]["mean"])
        policy["speedup_vs_serial"] = serial_makespan / mean_makespan if mean_makespan > 0 else 0.0

    return {
        "schema_version": 1,
        "hardware": hardware.to_dict(),
        "required_gpu_name": required_gpu_name,
        "hardware_requirement_met": hardware_matches(hardware, required_gpu_name),
        "policies": policies,
    }


def markdown_report(report: dict[str, Any]) -> str:
    hardware = report["hardware"]
    lines = [
        "# Repeated time-aware scheduler benchmark",
        "",
        f"- GPU: `{hardware['gpu_name']}`",
        f"- Compute capability: `{hardware.get('compute_capability')}`",
        f"- VRAM: `{hardware.get('total_vram_mb')} MiB`",
        f"- Required GPU expression: `{report['required_gpu_name']}`",
        f"- Hardware requirement met: `{'yes' if report['hardware_requirement_met'] else 'no'}`",
        "",
        "Values are mean ± sample standard deviation across isolated measured runs; the JSON report also records sample variance, minimum, maximum, and every raw run.",
        "",
        "| Policy | Runs | Complete | Makespan (s) | Total flow (s) | Mean flow (s) | Weighted flow (s) | Median flow (s) | p95 flow (s) | Max wait (s) | Jobs/hour | Slowdown | Pred/actual VRAM (MiB) | Actual over-budget packs | Early epochs/time saved | Speedup vs serial |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("serial_fifo", "legacy_vram_fill", "parallel_time_aware"):
        policy = report["policies"].get(name)
        if policy is None:
            continue
        metrics = policy["metrics"]

        def formatted(metric_name: str) -> str:
            value = metrics[metric_name]
            return f"{value['mean']:.2f} ± {value['stdev']:.2f}"

        predicted = metrics["predicted_avg_vram_mb"]["mean"]
        actual = metrics["actual_avg_vram_mb"]["mean"]
        memory_text = (
            f"{predicted:.1f}/{actual:.1f}"
            if metrics["measured_placement_count"]["mean"] > 0
            else "-"
        )
        lines.append(
            f"| {name} | {policy['runs']} | {'yes' if policy['all_complete'] else 'no'} | "
            f"{formatted('makespan_seconds')} | {formatted('total_flow_seconds')} | "
            f"{formatted('mean_flow_seconds')} | {formatted('weighted_mean_flow_seconds')} | "
            f"{formatted('median_flow_seconds')} | {formatted('p95_flow_seconds')} | "
            f"{formatted('max_wait_seconds')} | {formatted('jobs_per_hour')} | "
            f"{formatted('average_slowdown')} | "
            f"{memory_text} | "
            f"{metrics['actual_memory_over_budget_count']['mean']:.2f} | "
            f"{metrics['early_stopped_epochs_saved']['mean']:.1f}/"
            f"{metrics['early_stopped_wall_time_saved_seconds']['mean']:.1f}s | "
            f"{policy['speedup_vs_serial']:.3f}x |"
        )
    return "\n".join(lines) + "\n"


def _run_logged(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark command failed with exit code {completed.returncode}; see {log_path}")


def _replay_command(
    args: argparse.Namespace,
    *,
    config_id: str,
    policy: Policy,
    case_dir: Path,
    profile_input: Path | None = None,
    profile_output: Path | None = None,
    calibration: bool = False,
) -> list[str]:
    command = [
        args.python,
        str(Path(__file__).with_name("replay_scheduler.py")),
        "--config-id",
        config_id,
        "--mode",
        policy.mode,
        "--backend",
        policy.backend,
        "--trace",
        str(args.trace),
        "--runtime-root",
        str(case_dir / "runtime"),
        "--results-dir",
        str(case_dir / "results"),
        "--summary",
        str(case_dir / "summary.json"),
        "--code-cache-dir",
        str(case_dir / "code_cache"),
        "--duration-s",
        str(args.duration_s),
        "--gpu-vram-gib",
        str(args.gpu_vram_gib),
        "--predicted-budget-fraction",
        str(args.predicted_budget_fraction),
    ]
    if profile_input is not None:
        command.extend(["--time-aware-profile-input", str(profile_input)])
    if profile_output is not None:
        command.extend(["--time-aware-profile-output", str(profile_output)])
    if calibration:
        command.append("--calibrate-time-aware")
    if args.data_root:
        command.extend(["--data-root", str(args.data_root)])
    return command


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("trace_metrics"), dict):
        raise ValueError(f"Replay summary lacks trace_metrics: {path}")
    return payload


def _selected_batch(job: dict[str, Any]) -> int:
    return int(job.get("resolved_batch_size") or job.get("bs") or 1)


def attach_matched_batch_slowdown(
    packed_summary: dict[str, Any],
    solo_summary: dict[str, Any],
) -> None:
    solo_by_step = {
        int(job["step_idx"]): job
        for job in solo_summary.get("per_job", [])
        if job.get("step_idx") is not None
    }
    ratios: list[float] = []
    for packed_job in packed_summary.get("per_job", []):
        step_idx = packed_job.get("step_idx")
        if step_idx is None or int(step_idx) not in solo_by_step:
            continue
        solo_job = solo_by_step[int(step_idx)]
        packed_elapsed = packed_job.get("elapsed_s")
        solo_elapsed = solo_job.get("elapsed_s")
        if not isinstance(packed_elapsed, (int, float)) or not isinstance(solo_elapsed, (int, float)):
            continue
        if float(solo_elapsed) <= 0 or _selected_batch(packed_job) != _selected_batch(solo_job):
            continue
        ratios.append(max(1.0, float(packed_elapsed) / float(solo_elapsed)))
    execution = packed_summary.setdefault("execution_metrics", {})
    execution["average_slowdown"] = statistics.fmean(ratios) if ratios else 0.0
    execution["measured_slowdown_members"] = len(ratios)
    execution["slowdown_source"] = "matched_batch_solo" if ratios else "unavailable"


def write_matched_batch_trace(
    source_trace: Path,
    packed_summary: dict[str, Any],
    output_path: Path,
) -> None:
    selected_by_step = {
        int(job["step_idx"]): _selected_batch(job)
        for job in packed_summary.get("per_job", [])
        if job.get("step_idx") is not None
    }
    rows: list[dict[str, Any]] = []
    for line in source_trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step_idx = int(row["step_idx"])
        if step_idx in selected_by_step:
            selected = selected_by_step[step_idx]
            row["bs"] = selected
            row["max_bs"] = max(selected, int(row.get("max_bs") or selected))
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path(__file__).with_name("workload_trace_W3.jsonl"))
    parser.add_argument("--data-root", type=Path, help="Override the dataset root stored in the trace.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--required-gpu-name", default=r"NVIDIA A10(?:\s|$)")
    parser.add_argument("--allow-hardware-mismatch", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--duration-s", type=float, default=2700.0)
    parser.add_argument("--gpu-vram-gib", type=float, default=22.0)
    parser.add_argument("--predicted-budget-fraction", type=float, default=0.85)
    parser.add_argument("--time-aware-backend", choices=["mps", "stream", "cuda_process"], default="stream")
    parser.add_argument("--profile-input", type=Path)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 so variance is defined")
    if not args.trace.is_file():
        parser.error(f"trace does not exist: {args.trace}")
    hardware = detect_hardware_profile()
    if not hardware_matches(hardware, args.required_gpu_name) and not args.allow_hardware_mismatch:
        parser.error(
            f"detected GPU {hardware.gpu_name!r} does not match required expression "
            f"{args.required_gpu_name!r}; use --allow-hardware-mismatch only for a clearly labelled local validation"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.profile_input or args.results_dir / "calibration" / "time_aware_profiles.json"
    if not args.skip_calibration:
        calibration_policy = Policy(
            "calibration",
            "parallel_time_aware",
            "exclusive",
        )
        calibration_dir = args.results_dir / "calibration"
        command = _replay_command(
            args,
            config_id="time-aware-calibration",
            policy=calibration_policy,
            case_dir=calibration_dir,
            profile_output=profile_path,
            calibration=True,
        )
        _run_logged(command, log_path=calibration_dir / "replay.log")
        manifest = json.loads(profile_path.read_text(encoding="utf-8"))
        if not manifest.get("batch_size_observations"):
            raise RuntimeError("calibration produced no five-option observations")
    elif not profile_path.is_file():
        parser.error("--skip-calibration requires an existing --profile-input")

    policies = (
        Policy("serial_fifo", "parallel_time_aware", "exclusive", True),
        Policy(
            "parallel_time_aware",
            "parallel_time_aware",
            args.time_aware_backend,
            True,
        ),
    )
    raw: dict[str, list[dict[str, Any]]] = {policy.report_name: [] for policy in policies}
    matched_solo_runs: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        repetition_summaries: dict[str, dict[str, Any]] = {}
        for policy in policies:
            case_dir = args.results_dir / f"run_{repetition:02d}" / policy.report_name
            command = _replay_command(
                args,
                config_id=f"{policy.report_name}-r{repetition:02d}",
                policy=policy,
                case_dir=case_dir,
                profile_input=profile_path if policy.needs_profiles else None,
            )
            _run_logged(command, log_path=case_dir / "replay.log")
            summary = _load_summary(case_dir / "summary.json")
            raw[policy.report_name].append(summary)
            repetition_summaries[policy.report_name] = summary

        attach_matched_batch_slowdown(
            repetition_summaries["serial_fifo"],
            repetition_summaries["serial_fifo"],
        )
        matched_dir = args.results_dir / f"run_{repetition:02d}" / "parallel_time_aware_matched_solo"
        matched_trace = matched_dir / "trace.jsonl"
        write_matched_batch_trace(
            args.trace,
            repetition_summaries["parallel_time_aware"],
            matched_trace,
        )
        matched_policy = Policy(
            "parallel_time_aware_matched_solo",
            "parallel_time_aware",
            "exclusive",
            True,
        )
        command = _replay_command(
            args,
            config_id=f"parallel_time_aware_matched_solo-r{repetition:02d}",
            policy=matched_policy,
            case_dir=matched_dir,
            profile_input=profile_path,
        )
        trace_argument_index = command.index("--trace") + 1
        command[trace_argument_index] = str(matched_trace)
        _run_logged(command, log_path=matched_dir / "replay.log")
        matched_summary = _load_summary(matched_dir / "summary.json")
        matched_solo_runs.append(matched_summary)
        attach_matched_batch_slowdown(
            repetition_summaries["parallel_time_aware"],
            matched_summary,
        )

    report = aggregate_summaries(
        raw,
        hardware=hardware,
        required_gpu_name=args.required_gpu_name,
    )
    report["repetitions"] = args.repetitions
    report["trace"] = str(args.trace.resolve())
    report["calibration_profile"] = str(profile_path.resolve())
    report["raw_runs"] = raw
    report["matched_batch_solo_runs"] = matched_solo_runs
    (args.results_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown = markdown_report(report)
    (args.results_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)

    if not all(policy["all_complete"] for policy in report["policies"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
