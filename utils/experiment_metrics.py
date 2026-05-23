"""Comparison metric generation for MLEvolve experiment runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any
import json
import time


def build_comparison_metrics(
    cfg: Any,
    journal: Any,
    *,
    started_at: float,
    finished_at: float | None = None,
    scheduler_client: Any | None = None,
    metric_maximize: bool | None = None,
) -> dict[str, Any]:
    finished_at = finished_at or time.time()
    mode = str(getattr(getattr(cfg, "experiment", None), "mode", "hardware_aware"))
    nodes = [node for node in list(getattr(journal, "nodes", []) or []) if getattr(node, "stage", None) != "root"]
    valid_nodes = [
        node for node in nodes
        if getattr(node, "is_buggy", None) is False and getattr(getattr(node, "metric", None), "value", None) is not None
    ]
    buggy_nodes = [node for node in nodes if getattr(node, "is_buggy", None) is True]
    metric_maximize = _infer_metric_direction(valid_nodes, metric_maximize)
    best_node = _best_node(valid_nodes, metric_maximize)

    node_exec_times = [
        float(getattr(node, "exec_time"))
        for node in nodes
        if getattr(node, "exec_time", None) is not None
    ]
    scheduler_jobs = _scheduler_jobs(scheduler_client)
    scheduler_events = _scheduler_events(scheduler_client)
    scheduler_job_times = [_job_duration_seconds(job) for job in scheduler_jobs]
    scheduler_job_times = [value for value in scheduler_job_times if value is not None]
    job_times = scheduler_job_times or node_exec_times

    batch_probe_hit_count = sum(1 for event in scheduler_events if event.get("event_type") == "batch_probe_cache_hit")
    batch_probe_trial_count = sum(1 for event in scheduler_events if event.get("event_type") == "batch_probe_trial")
    timeout_failures = _count_failures(nodes, scheduler_jobs, "timeout")
    oom_failures = _count_failures(nodes, scheduler_jobs, "out of memory", "oom", "cuda memory")
    vram_values = _vram_values(nodes, scheduler_events)
    backend_distribution = _backend_distribution(scheduler_jobs, scheduler_events)
    total_wall_time = max(0.0, float(finished_at) - float(started_at))

    return {
        "mode": mode,
        "run_id": str(getattr(cfg, "exp_name", "")),
        "exp_id": str(getattr(cfg, "exp_id", "")),
        "total_wall_time_seconds": total_wall_time,
        "total_job_execution_time_seconds": sum(job_times),
        "median_job_execution_time_seconds": median(job_times) if job_times else None,
        "node_count": len(nodes),
        "valid_count": len(valid_nodes),
        "buggy_count": len(buggy_nodes),
        "best_metric": getattr(getattr(best_node, "metric", None), "value", None) if best_node else None,
        "best_node_id": getattr(best_node, "id", None) if best_node else None,
        "metric_direction": "maximize" if metric_maximize else "minimize",
        "time_to_best_seconds": _time_to_best(best_node, started_at) if best_node else None,
        "nodes_to_best": getattr(best_node, "step", None) if best_node else None,
        "jobs_per_hour": (len(nodes) / (total_wall_time / 3600.0)) if total_wall_time > 0 else None,
        "packed_dispatch_count": _packed_dispatch_count(scheduler_jobs, scheduler_events),
        "batch_probe_hit_count": batch_probe_hit_count,
        "batch_probe_trial_count": batch_probe_trial_count,
        "timeout_failures": timeout_failures,
        "oom_failures": oom_failures,
        "peak_vram_mb": max(vram_values) if vram_values else None,
        "average_vram_mb": (sum(vram_values) / len(vram_values)) if vram_values else None,
        "scheduler_backend_distribution": dict(backend_distribution),
    }


def write_comparison_metrics(metrics: dict[str, Any], log_dir: str | Path) -> Path:
    path = Path(log_dir) / "comparison_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _infer_metric_direction(nodes: list[Any], explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    for node in nodes:
        metric = getattr(node, "metric", None)
        maximize = getattr(metric, "maximize", None)
        if maximize is not None:
            return bool(maximize)
    return True


def _best_node(nodes: list[Any], maximize: bool) -> Any | None:
    if not nodes:
        return None
    return (max if maximize else min)(nodes, key=lambda node: getattr(node.metric, "value"))


def _scheduler_jobs(scheduler_client: Any | None) -> list[dict[str, Any]]:
    if scheduler_client is None:
        return []
    try:
        return [job.to_dict() if hasattr(job, "to_dict") else dict(job) for job in scheduler_client.list_jobs()]
    except Exception:
        return []


def _scheduler_events(scheduler_client: Any | None) -> list[dict[str, Any]]:
    if scheduler_client is None:
        return []
    try:
        return [dict(event) for event in scheduler_client.list_events()]
    except Exception:
        try:
            return [dict(event) for event in scheduler_client.store.list_events()]
        except Exception:
            return []


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _job_duration_seconds(job: dict[str, Any]) -> float | None:
    started = _parse_time(job.get("started_at") or (job.get("status_timestamps") or {}).get("running"))
    finished = _parse_time(job.get("finished_at"))
    if not started or not finished:
        return None
    return max(0.0, (finished - started).total_seconds())


def _count_failures(nodes: list[Any], jobs: list[dict[str, Any]], *needles: str) -> int:
    count = 0
    for node in nodes:
        text = " ".join(
            str(value or "")
            for value in (
                getattr(node, "exc_type", None),
                getattr(node, "exc_info", None),
                getattr(node, "analysis", None),
                _safe_node_term_out(node),
            )
        ).lower()
        if any(needle in text for needle in needles):
            count += 1
    for job in jobs:
        status = str(job.get("status") or "").lower()
        reason = str(job.get("status_reason") or "").lower()
        if status in {"failed", "cancelled", "timeout"} and any(needle in reason for needle in needles):
            count += 1
    return count


def _safe_node_term_out(node: Any) -> str:
    try:
        raw = getattr(node, "_term_out", None)
        if raw is None:
            return ""
        return getattr(node, "term_out", "") or ""
    except Exception:
        return ""


def _vram_values(nodes: list[Any], events: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for node in nodes:
        value = getattr(node, "peak_vram_mb", None)
        if value is not None:
            values.append(float(value))
    for event in events:
        payload = event.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        for key in ("peak_vram_mb", "peak_vram_mib", "max_vram_mb"):
            if payload.get(key) is not None:
                try:
                    values.append(float(payload[key]))
                except (TypeError, ValueError):
                    pass
    return values


def _backend_distribution(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> Counter:
    distribution: Counter = Counter()
    for job in jobs:
        metadata = job.get("metadata") or {}
        backend = metadata.get("placement_backend") or metadata.get("backend_name")
        if backend:
            distribution[str(backend)] += 1
    for event in events:
        if event.get("event_type") not in {"job_dispatched", "job_started", "scheduler_placement"}:
            continue
        payload = event.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        backend = payload.get("backend_name") or payload.get("placement_backend")
        if backend:
            distribution[str(backend)] += 1
    return distribution


def _packed_dispatch_count(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> int:
    packed = 0
    for job in jobs:
        packing = job.get("packing") or {}
        metadata = job.get("metadata") or {}
        if packing.get("eligible") and (metadata.get("packed") or metadata.get("placement_backend") in {"mps", "stream", "cuda_process"}):
            packed += 1
    packed += sum(1 for event in events if event.get("event_type") in {"packed_dispatch", "job_packed", "scheduler_packed_dispatch"})
    return packed


def _time_to_best(best_node: Any, started_at: float) -> float | None:
    ctime = getattr(best_node, "ctime", None)
    try:
        return max(0.0, float(ctime) - float(started_at))
    except (TypeError, ValueError):
        return None
