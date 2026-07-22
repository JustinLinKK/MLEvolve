"""Case-level and three-repetition reporting for the standard benchmark."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable
import argparse
import csv
import json
import math

from . import DATASET_SIZE, EPOCHS, JOB_COUNT


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def _timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _maximum_concurrency(intervals: Iterable[tuple[float, float]]) -> int:
    points: list[tuple[float, int]] = []
    for started, finished in intervals:
        points.extend(((started, 1), (finished, -1)))
    # End events precede start events at the same instant.
    points.sort(key=lambda item: (item[0], item[1]))
    active = maximum = 0
    for _timestamp_value, delta in points:
        active += delta
        maximum = max(maximum, active)
    return maximum


def _load_job_metrics(case_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in case_root.rglob("metric.json"):
        payload = _read_json(path, {})
        job_id = str(payload.get("job_id") or "")
        if job_id:
            results[job_id] = payload
    return results


def _hardware_samples(case_root: Path) -> list[dict[str, float]]:
    paths = list(case_root.glob("logs/hardware_samples.csv"))
    if not paths:
        paths = list(case_root.rglob("hardware_samples.csv"))
    if not paths:
        return []
    rows: list[dict[str, float]] = []
    with paths[0].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value) if value not in (None, "") else math.nan
                except ValueError:
                    parsed[key] = math.nan
            rows.append(parsed)
    return rows


def _scheduler_job_row(job: dict[str, Any], metric: dict[str, Any] | None) -> dict[str, Any]:
    metadata = dict(job.get("metadata") or {})
    submitted = _timestamp(job.get("submitted_at"))
    started = _timestamp(job.get("started_at") or (job.get("status_timestamps") or {}).get("RUNNING"))
    finished = _timestamp(job.get("finished_at"))
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "queue_wait_seconds": max(0.0, started - submitted) if started is not None and submitted is not None else None,
        "duration_seconds": max(0.0, finished - started) if finished is not None and started is not None else None,
        "submitted_batch_size": metadata.get("submitted_batch_size", 32),
        "optimized_batch_size": metadata.get("placement_batch_size") or metadata.get("resolved_batch_size") or 32,
        "throughput_images_per_second": (metric or {}).get("throughput_images_per_second"),
        "training_loss": (metric or {}).get("loss"),
        "processed_samples": (metric or {}).get("samples_seen"),
        "profile_source": metadata.get("batch_probe_source") or metadata.get("runtime_profile_source"),
        "probe_source": metadata.get("batch_probe_source"),
        "placement_backend": metadata.get("placement_backend"),
        "packed_group_id": metadata.get("placement_group_id"),
        "placement_mode": metadata.get("placement_mode"),
        "failure_status": job.get("status_reason"),
    }


def _fifo_job_row(job: dict[str, Any], metric: dict[str, Any] | None) -> dict[str, Any]:
    submitted = _timestamp(job.get("submitted_wall_time"))
    started = _timestamp(job.get("started_wall_time"))
    finished = _timestamp(job.get("finished_wall_time"))
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "queue_wait_seconds": max(0.0, started - submitted) if started is not None and submitted is not None else None,
        "duration_seconds": max(0.0, finished - started) if finished is not None and started is not None else job.get("exec_time"),
        "submitted_batch_size": 32,
        "optimized_batch_size": 32,
        "throughput_images_per_second": (metric or {}).get("throughput_images_per_second"),
        "training_loss": (metric or {}).get("loss"),
        "processed_samples": (metric or {}).get("samples_seen"),
        "profile_source": None,
        "probe_source": None,
        "placement_backend": "fifo_sequential",
        "packed_group_id": None,
        "placement_mode": "exclusive",
        "failure_status": job.get("status_reason"),
    }


def _write_job_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["job_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_case(
    case_root: str | Path,
    *,
    arm: str,
    repetition: int,
    runner_mode: str,
    backend: str | None,
    prediction_mode: str | None,
) -> dict[str, Any]:
    root = Path(case_root).expanduser().resolve()
    replay_summary = _read_json(root / "replay_summary.json", {})
    replay_metrics = dict(replay_summary.get("metrics") or {})
    case_state = _read_json(root / "case_status.json", {})
    job_metrics = _load_job_metrics(root)
    is_fifo = arm == "fifo"
    record_name = "multiprocess_jobs.jsonl" if is_fifo else "scheduler_jobs.jsonl"
    records = _read_jsonl(root / "logs" / record_name)
    events = [] if is_fifo else _read_jsonl(root / "logs" / "scheduler_events.jsonl")
    rows = [
        (_fifo_job_row(job, job_metrics.get(str(job.get("job_id")))) if is_fifo else _scheduler_job_row(job, job_metrics.get(str(job.get("job_id")))))
        for job in records
    ]
    _write_job_csv(root / "per_job.csv", rows)

    durations = [float(row["duration_seconds"]) for row in rows if row.get("duration_seconds") is not None]
    queue_waits = [float(row["queue_wait_seconds"]) for row in rows if row.get("queue_wait_seconds") is not None]
    intervals = []
    for record in records:
        started = _timestamp(record.get("started_wall_time") if is_fifo else record.get("started_at"))
        finished = _timestamp(record.get("finished_wall_time") if is_fifo else record.get("finished_at"))
        if started is not None and finished is not None:
            intervals.append((started, finished))
    statuses = Counter(str(row.get("status")) for row in rows)
    total_wall = float(replay_metrics.get("total_wall_time_seconds") or 0.0)
    total_samples = sum(int(row.get("processed_samples") or 0) for row in rows)
    group_sizes: Counter[str] = Counter()
    for event in events:
        if event.get("event_type") in {"packed_pair_dispatched", "packed_group_dispatched"}:
            payload = dict(event.get("payload") or {})
            group_sizes[str(len(payload.get("job_ids") or []))] += 1
    hardware = _hardware_samples(root)
    gpu_util = [row["gpu_util_percent"] for row in hardware if math.isfinite(row.get("gpu_util_percent", math.nan))]
    vram_util = [row["gpu_memory_percent"] for row in hardware if math.isfinite(row.get("gpu_memory_percent", math.nan))]
    fallback_events = [event for event in events if "fallback" in str(event.get("event_type") or "")]
    oom_evidence = [
        item
        for item in [*events, *records]
        if "oom" in json.dumps(item).lower() or "out of memory" in json.dumps(item).lower()
    ]
    predictor = dict(replay_metrics.get("predictor_health") or {})
    primary_predictor_accepted = bool(
        prediction_mode != "ml_predictor"
        or (
            float(predictor.get("selection_coverage") or 0.0) >= 0.95
            and int(predictor.get("adapter_failure_count") or 0) == 0
        )
    )
    accepted = (
        runner_mode == "real"
        and len(rows) == JOB_COUNT
        and statuses.get("COMPLETED", 0) == JOB_COUNT
        and total_samples == DATASET_SIZE * EPOCHS * JOB_COUNT
        and not statuses.get("FAILED", 0)
        and not statuses.get("CANCELLED", 0)
        and not oom_evidence
        and int(replay_metrics.get("replay_skipped_action_count") or 0) == 0
        and not sum(bool(record.get("timeout")) for record in records)
        and primary_predictor_accepted
    )
    summary = {
        "schema_version": "standard-histopath-case-report-v1",
        "arm": arm,
        "repetition": repetition,
        "runner_mode": runner_mode,
        "prediction_mode": prediction_mode,
        "packed_backend": backend,
        "fifo_sequential": is_fifo,
        "packing_count_limit": None if is_fifo else 8,
        "physical_gpu": case_state.get("physical_gpu") or replay_metrics.get("gpu_name") or replay_metrics.get("physical_gpu"),
        "total_wall_time_seconds": total_wall,
        "mean_job_duration_seconds": mean(durations) if durations else None,
        "median_job_duration_seconds": median(durations) if durations else None,
        "p95_job_duration_seconds": _percentile(durations, 0.95),
        "mean_queue_wait_seconds": mean(queue_waits) if queue_waits else None,
        "p95_queue_wait_seconds": _percentile(queue_waits, 0.95),
        "jobs_per_hour": (statuses.get("COMPLETED", 0) / total_wall * 3600.0) if total_wall > 0 else None,
        "images_per_second": total_samples / total_wall if total_wall > 0 else None,
        "processed_samples": total_samples,
        "probe_overhead_seconds": replay_metrics.get("probe_time_seconds", 0.0),
        "batch_probe_cache_hits": replay_metrics.get("batch_probe_hit_count", 0),
        "packed_group_size_distribution": dict(sorted(group_sizes.items())),
        "maximum_observed_concurrency": _maximum_concurrency(intervals),
        "fallback_count": len(fallback_events),
        "oom_count": len(oom_evidence),
        "skipped_action_count": int(replay_metrics.get("replay_skipped_action_count") or 0),
        "mean_sm_util_percent": mean(gpu_util) if gpu_util else replay_metrics.get("avg_gpu_util_percent"),
        "p95_sm_util_percent": _percentile(gpu_util, 0.95),
        "mean_vram_util_percent": mean(vram_util) if vram_util else replay_metrics.get("avg_gpu_memory_percent"),
        "p95_vram_util_percent": _percentile(vram_util, 0.95),
        "peak_vram_used_mib": replay_metrics.get("peak_gpu_memory_used_mb"),
        "status_counts": dict(statuses),
        "completed_job_count": statuses.get("COMPLETED", 0),
        "failed_job_count": statuses.get("FAILED", 0),
        "cancelled_job_count": statuses.get("CANCELLED", 0),
        "timed_out_job_count": sum(bool(record.get("timeout")) for record in records),
        "predictor_health": predictor,
        "prediction_error": {
            "sample_count": 0,
            "mean_absolute_percentage_error": None,
            "note": "Unavailable until a predictor emits retained per-job estimates paired with observed metrics.",
        },
        "ml_performance_claim_eligible": prediction_mode == "ml_predictor" and primary_predictor_accepted,
        "accepted": accepted,
        "acceptance_note": None if accepted else ("no-op replay is not a performance result" if runner_mode == "noop" else "full 100-job/50-epoch completion criteria not met"),
    }
    (root / "case_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


REPORT_METRICS = (
    "total_wall_time_seconds",
    "mean_job_duration_seconds",
    "median_job_duration_seconds",
    "p95_job_duration_seconds",
    "jobs_per_hour",
    "images_per_second",
    "probe_overhead_seconds",
    "maximum_observed_concurrency",
    "mean_sm_util_percent",
    "mean_vram_util_percent",
)


def _statistics(values: list[float]) -> dict[str, Any]:
    count = len(values)
    average = mean(values) if values else None
    sample_stddev = stdev(values) if count >= 2 else None
    critical = {2: 12.706, 3: 4.303}.get(count, 1.96)
    half_width = critical * sample_stddev / math.sqrt(count) if sample_stddev is not None else None
    return {
        "n": count,
        "mean": average,
        "sample_stddev": sample_stddev,
        "ci95_low": average - half_width if average is not None and half_width is not None else None,
        "ci95_high": average + half_width if average is not None and half_width is not None else None,
    }


def aggregate_reports(case_summaries: list[dict[str, Any]], output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for summary in case_summaries:
        by_arm.setdefault(str(summary["arm"]), []).append(summary)
    aggregate: dict[str, Any] = {}
    for arm, items in sorted(by_arm.items()):
        aggregate[arm] = {
            metric: _statistics([float(item[metric]) for item in items if isinstance(item.get(metric), (int, float))])
            for metric in REPORT_METRICS
        }
        aggregate[arm]["accepted_repetition_count"] = sum(bool(item.get("accepted")) for item in items)

    by_key = {(str(item["arm"]), int(item["repetition"])): item for item in case_summaries}
    comparisons: dict[str, Any] = {}
    for arm in sorted(by_arm):
        if arm == "fifo":
            continue
        references = ["fifo"]
        if arm == "ml_cuda":
            references.append("branch_cuda")
        elif arm == "ml_stream":
            references.append("branch_stream")
        for reference in references:
            key = f"{arm}_vs_{reference}"
            comparisons[key] = {}
            for metric in ("total_wall_time_seconds", "jobs_per_hour", "images_per_second"):
                deltas: list[float] = []
                ratios: list[float] = []
                for item in by_arm[arm]:
                    left = item.get(metric)
                    right_item = by_key.get((reference, int(item["repetition"])))
                    right = right_item.get(metric) if right_item else None
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        deltas.append(float(left) - float(right))
                        if float(right) != 0:
                            ratios.append(float(left) / float(right))
                comparisons[key][metric] = {"delta": _statistics(deltas), "ratio": _statistics(ratios)}

    report = {
        "schema_version": "standard-histopath-aggregate-report-v1",
        "case_count": len(case_summaries),
        "arms": aggregate,
        "matched_comparisons": comparisons,
        "student_t_interval": "two-sided 95%; df=n-1; t=4.303 for n=3",
    }
    (output / "aggregate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Standard Histopathology Scheduler Benchmark", "", "| Arm | n | Wall time mean ± SD (s) | Jobs/hour | Images/s | Accepted |", "|---|---:|---:|---:|---:|---:|"]
    for arm, values in aggregate.items():
        wall = values["total_wall_time_seconds"]
        jobs_hour = values["jobs_per_hour"]
        images_second = values["images_per_second"]
        lines.append(
            f"| {arm} | {wall['n']} | {_fmt(wall['mean'])} ± {_fmt(wall['sample_stddev'])} | "
            f"{_fmt(jobs_hour['mean'])} | {_fmt(images_second['mean'])} | {values['accepted_repetition_count']} |"
        )
    lines.extend(("", "Intervals and matched comparisons are in `aggregate_report.json`."))
    (output / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.results_root).expanduser().resolve()
    summaries = [_read_json(path, {}) for path in sorted(root.rglob("case_summary.json"))]
    summaries = [item for item in summaries if item]
    report = aggregate_reports(summaries, args.output_root or root)
    print(json.dumps({"case_count": report["case_count"], "output_root": str(Path(args.output_root or root).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
