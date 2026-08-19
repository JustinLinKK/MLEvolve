"""Extract and load scheduler timeline replay fixtures."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import sqlite3

from localml_scheduler.config import SchedulerSettings

REPLAY_PENDING_TIMESTAMP = "1970-01-01T00:00:00+00:00"
DEFAULT_SOURCE_RUN = Path(
    "runs/profile_scheduler_compare_histopathologic-cancer-detection_20260704_212842"
)

TRANSIENT_METADATA_KEYS = {
    "batch_probe_device_type",
    "batch_probe_key",
    "batch_probe_reuse_miss",
    "batch_probe_source",
    "backend_name",
    "checkpoint_resume_from",
    "checkpoint_resume_strategy",
    "resolved_batch_size",
}
TRANSIENT_METADATA_PREFIXES = (
    "placement_",
    "runtime_",
    "scheduler_preemption_",
)
REPLAY_BOOKKEEPING_JOB_KEYS = {
    "pre_archive_baseline_model_path",
}


@dataclass(frozen=True)
class FixturePaths:
    root: Path
    timeline: Path
    jobs: Path
    baseline_summary: Path
    scheduler_settings: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "FixturePaths":
        path = Path(root)
        return cls(
            root=path,
            timeline=path / "timeline.json",
            jobs=path / "jobs.jsonl",
            baseline_summary=path / "baseline_summary.json",
            scheduler_settings=path / "scheduler_settings.replay.json",
        )


def resolve_scheduler_runtime_root(source: str | Path) -> Path:
    """Resolve either the comparison run root or a scheduler runtime root."""
    path = Path(source).expanduser().resolve()
    candidates = [
        path,
        path / "scheduler_runtime",
        path / "scheduler_on" / "scheduler_runtime",
    ]
    for candidate in candidates:
        if (candidate / "db" / "scheduler.sqlite3").exists():
            return candidate
    raise FileNotFoundError(f"Could not find scheduler.sqlite3 under {path}")


def extract_fixture(source: str | Path, output_dir: str | Path) -> FixturePaths:
    """Extract replay fixture files from a persisted scheduler runtime DB."""
    runtime_root = resolve_scheduler_runtime_root(source)
    source_root = _infer_source_root(runtime_root)
    output = FixturePaths.from_root(output_dir)
    output.root.mkdir(parents=True, exist_ok=True)

    db_path = runtime_root / "db" / "scheduler.sqlite3"
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        jobs = _load_jobs(connection)
        commands = _load_commands(connection)
        events = _load_events(connection)

    timeline = _build_timeline(commands, jobs)
    job_payloads = [reset_job_payload_for_replay(job["payload"]) for job in jobs]
    settings_payload = _load_replay_settings(runtime_root)
    baseline_summary = _build_baseline_summary(
        source_root=source_root,
        runtime_root=runtime_root,
        db_path=db_path,
        jobs=jobs,
        commands=commands,
        timeline=timeline,
        events=events,
    )

    _write_json(output.timeline, {"actions": timeline})
    output.jobs.write_text(
        "".join(
            json.dumps(payload, sort_keys=True, default=str) + "\n"
            for payload in job_payloads
        ),
        encoding="utf-8",
    )
    _write_json(output.baseline_summary, baseline_summary)
    _write_json(output.scheduler_settings, settings_payload)
    return output


def load_fixture(
    root: str | Path,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    """Load a fixture and normalize archived payloads to the current schema."""
    paths = FixturePaths.from_root(root)
    timeline_payload = json.loads(paths.timeline.read_text(encoding="utf-8"))
    jobs_by_id: dict[str, dict[str, Any]] = {}
    with paths.jobs.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = normalize_replay_job_payload(json.loads(line))
            jobs_by_id[str(payload["job_id"])] = payload
    baseline = json.loads(paths.baseline_summary.read_text(encoding="utf-8"))
    settings = normalize_replay_settings(
        json.loads(paths.scheduler_settings.read_text(encoding="utf-8"))
    )
    return list(timeline_payload.get("actions") or []), jobs_by_id, baseline, settings


def normalize_replay_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop fields used by pre-refactor job schemas from an archived job."""
    job = deepcopy(payload)
    batch_probe = dict(job.get("batch_probe") or {})
    for key in (
        "contract_version",
        "profile_key",
        "profile_namespace",
        "reuse_only",
        "search_mode",
        "shape_signature_override",
    ):
        batch_probe.pop(key, None)
    job["batch_probe"] = batch_probe

    checkpoint_policy = dict(job.get("checkpoint_policy") or {})
    checkpoint_policy.pop("preemptible", None)
    job["checkpoint_policy"] = checkpoint_policy
    return job


def normalize_replay_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Retain only settings understood by the current scheduler dataclasses."""
    raw = dict(payload or {})
    raw.pop("runtime_root", None)
    raw.pop("redis_cache", None)
    cleaned = _known_dataclass_fields(raw, SchedulerSettings())
    gpu = dict(cleaned.get("gpu_scheduler") or {})
    # Archived fixtures predate the single production placement policy. Their
    # timelines remain valid inputs, but placement is replayed by the current
    # time-aware scheduler rather than resurrecting removed modes.
    gpu["mode"] = "parallel_time_aware"
    cleaned["gpu_scheduler"] = gpu
    return cleaned


def _known_dataclass_fields(payload: dict[str, Any], template: Any) -> dict[str, Any]:
    """Recursively filter a mapping using an initialized dataclass as schema."""
    cleaned: dict[str, Any] = {}
    for field_info in fields(template):
        if not field_info.init or field_info.name not in payload:
            continue
        value = payload[field_info.name]
        current = getattr(template, field_info.name)
        if field_info.name == "prediction" and isinstance(value, dict):
            try:
                cleaned[field_info.name] = current.from_dict(value).to_dict()
            except (TypeError, ValueError):
                cleaned[field_info.name] = current.to_dict()
        elif is_dataclass(current) and isinstance(value, dict):
            cleaned[field_info.name] = _known_dataclass_fields(value, current)
        else:
            cleaned[field_info.name] = value
    return cleaned


def reset_job_payload_for_replay(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a replay-safe job payload with prior runtime decisions removed."""
    job = deepcopy(payload)
    metadata = dict(job.get("metadata") or {})
    replay_original = {
        "status": job.get("status"),
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "status_reason": job.get("status_reason"),
        "placement_backend": metadata.get("placement_backend"),
        "placement_mode": metadata.get("placement_mode"),
        "resolved_batch_size": metadata.get("resolved_batch_size"),
        "batch_probe_source": metadata.get("batch_probe_source"),
    }

    for key in list(metadata):
        if key in TRANSIENT_METADATA_KEYS or key.startswith(
            TRANSIENT_METADATA_PREFIXES
        ):
            metadata.pop(key, None)
    metadata["replay_original"] = {
        key: value for key, value in replay_original.items() if value is not None
    }
    metadata["replay_fixture_source"] = "scheduler_timeline"

    job["metadata"] = metadata
    job["status"] = "PENDING"
    job["submitted_at"] = REPLAY_PENDING_TIMESTAMP
    job["status_reason"] = None
    job["latest_checkpoint_path"] = None
    job["status_timestamps"] = {}
    job["last_heartbeat_at"] = None
    job["last_dispatched_at"] = None
    job["started_at"] = None
    job["finished_at"] = None
    job["hold"] = False
    for key in REPLAY_BOOKKEEPING_JOB_KEYS:
        job.pop(key, None)
    return job


def _infer_source_root(runtime_root: Path) -> Path:
    parts = runtime_root.parts
    if len(parts) >= 2 and parts[-2:] == ("scheduler_on", "scheduler_runtime"):
        return runtime_root.parents[1]
    if runtime_root.name == "scheduler_runtime":
        return runtime_root.parent
    return runtime_root


def _load_jobs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("""
        SELECT job_id, status, priority, baseline_model_id, submitted_at, queue_sequence, payload_json, updated_at
        FROM jobs
        ORDER BY queue_sequence ASC
        """).fetchall()
    jobs = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        jobs.append({**dict(row), "payload": payload})
    return jobs


def _load_commands(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("""
        SELECT command_id, job_id, command_type, payload_json, created_at, processed_at
        FROM commands
        ORDER BY command_id ASC
        """).fetchall()
    commands = []
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        commands.append({**dict(row), "payload": payload})
    return commands


def _load_events(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("""
        SELECT event_id, job_id, event_type, payload_json, created_at
        FROM events
        ORDER BY event_id ASC
        """).fetchall()
    events = []
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        events.append({**dict(row), "payload": payload})
    return events


def _build_timeline(
    commands: list[dict[str, Any]], jobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not commands:
        return []
    base = _parse_time(commands[0]["created_at"])
    last_submit_index = max(
        (
            index
            for index, command in enumerate(commands)
            if command["command_type"] == "SUBMIT"
        ),
        default=-1,
    )
    jobs_by_id = {job["job_id"]: job["payload"] for job in jobs}
    timeline = []
    for index, command in enumerate(commands):
        created_at = _parse_time(command["created_at"])
        job_payload = jobs_by_id.get(command["job_id"])
        metadata = (job_payload or {}).get("metadata") or {}
        final_cleanup = (
            command["command_type"] == "CANCEL" and index > last_submit_index
        )
        timeline.append(
            {
                "command_id": command["command_id"],
                "action": command["command_type"],
                "job_id": command["job_id"],
                "relative_seconds": max(0.0, (created_at - base).total_seconds()),
                "created_at": command["created_at"],
                "processed_at": command["processed_at"],
                "payload": command["payload"],
                "final_cleanup": final_cleanup,
                "has_job_payload": job_payload is not None,
                "queue_sequence": (
                    job_payload.get("queue_sequence") if job_payload else None
                ),
                "task_type": job_payload.get("task_type") if job_payload else None,
                "runner_target": (
                    (job_payload.get("config") or {}).get("runner_target")
                    if job_payload
                    else None
                ),
                "mlevolve_node_id": metadata.get("node_id")
                or metadata.get("mlevolve_node_id"),
            }
        )
    return timeline


def _load_replay_settings(runtime_root: Path) -> dict[str, Any]:
    settings_path = runtime_root / "scheduler_settings.json"
    if settings_path.exists():
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload = normalize_replay_settings(payload)
    payload.setdefault("gpu_scheduler", {})
    payload.setdefault("baseline_cache", {})
    for legacy_key in (
        "graph" + "_db",
        "hardware_feature" + "_db",
        "hardware_knowledge" + "_graph",
    ):
        payload.pop(legacy_key, None)
    payload["log_db"] = {**dict(payload.get("log_db") or {}), "enabled": False}
    payload.pop("redis_cache", None)
    payload["baseline_cache"] = {
        **dict(payload.get("baseline_cache") or {}),
        "warm_queue_policy": "budget_only",
        "warm_queue_top_k": 0,
        "entry_capacity": 0,
        "max_ram_percent": 0,
        "memory_budget_bytes": 0,
    }
    return payload


def _build_baseline_summary(
    *,
    source_root: Path,
    runtime_root: Path,
    db_path: Path,
    jobs: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    command_counts = Counter(command["command_type"] for command in commands)
    event_counts = Counter(event["event_type"] for event in events)
    status_counts = Counter(job["payload"].get("status") for job in jobs)
    task_type_counts = Counter(job["payload"].get("task_type") for job in jobs)
    runner_counts = Counter(
        (job["payload"].get("config") or {}).get("runner_target") for job in jobs
    )
    script_paths = _script_paths(job["payload"] for job in jobs)
    missing_script_paths = [path for path in script_paths if not Path(path).exists()]
    original_workspace = _common_working_dir(job["payload"] for job in jobs)
    reference_metrics = _load_reference_metrics(source_root)

    final_cleanup_actions = [
        action for action in timeline if action.get("final_cleanup")
    ]
    mid_run_cancel_count = sum(
        1
        for action in timeline
        if action["action"] == "CANCEL" and not action.get("final_cleanup")
    )

    return {
        "source_run_root": str(source_root),
        "scheduler_runtime_root": str(runtime_root),
        "scheduler_db_path": str(db_path),
        "original_workspace": original_workspace,
        "original_input_dir": (
            str(Path(original_workspace) / "input") if original_workspace else None
        ),
        "first_command_at": commands[0]["created_at"] if commands else None,
        "last_command_at": commands[-1]["created_at"] if commands else None,
        "command_count": len(commands),
        "command_counts": dict(command_counts),
        "submit_count": int(command_counts.get("SUBMIT", 0)),
        "cancel_count": int(command_counts.get("CANCEL", 0)),
        "mid_run_cancel_count": mid_run_cancel_count,
        "final_cleanup_cancel_count": len(final_cleanup_actions),
        "final_cleanup_start_relative_seconds": (
            min(action["relative_seconds"] for action in final_cleanup_actions)
            if final_cleanup_actions
            else None
        ),
        "job_count": len(jobs),
        "job_status_counts": dict(status_counts),
        "task_type_counts": dict(task_type_counts),
        "runner_target_counts": dict(runner_counts),
        "event_counts": dict(event_counts),
        "batch_probe_trial_count": int(event_counts.get("batch_probe_trial", 0)),
        "batch_probe_cache_hit_count": int(
            event_counts.get("batch_probe_cache_hit", 0)
        ),
        "packed_pair_dispatch_count": int(
            event_counts.get("packed_pair_dispatched", 0)
        ),
        "packed_group_dispatch_count": int(
            event_counts.get("packed_group_dispatched", 0)
        ),
        "packed_dispatch_count": int(
            event_counts.get("packed_pair_dispatched", 0)
            + event_counts.get("packed_group_dispatched", 0)
        ),
        "packed_fallback_count": int(
            event_counts.get("packed_pair_fallback", 0)
            + event_counts.get("packed_group_fallback", 0)
        ),
        "script_path_count": len(script_paths),
        "missing_script_path_count": len(missing_script_paths),
        "missing_script_paths": missing_script_paths,
        "reference_metrics": reference_metrics,
    }


def _script_paths(payloads: Iterable[dict[str, Any]]) -> list[str]:
    paths = []
    for payload in payloads:
        runner_kwargs = (payload.get("config") or {}).get("runner_kwargs") or {}
        script_path = runner_kwargs.get("script_path")
        if script_path:
            paths.append(str(script_path))
    return paths


def _common_working_dir(payloads: Iterable[dict[str, Any]]) -> str | None:
    counter: Counter[str] = Counter()
    for payload in payloads:
        runner_kwargs = (payload.get("config") or {}).get("runner_kwargs") or {}
        working_dir = runner_kwargs.get("working_dir")
        if working_dir:
            counter[str(working_dir)] += 1
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _load_reference_metrics(source_root: Path) -> dict[str, Any]:
    summary_path = source_root / "comparison_plots" / "comparison_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return dict(
            ((payload.get("modes") or {}).get("scheduler_on") or {}).get("metrics")
            or {}
        )
    except Exception:
        return {}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
