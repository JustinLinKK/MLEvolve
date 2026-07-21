"""Collect stress-style evidence for profile scheduler comparison runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import yaml


SENSITIVE_KEY_PARTS = ("api_key", "password", "secret", "token", "dsn")


def _run_command(command: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {timeout:g}s",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _sanitize(value: Any, key: str = "") -> Any:
    if any(part in str(key).lower() for part in SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, dict):
        return {item_key: _sanitize(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_loads(text: str | None) -> Any:
    try:
        return json.loads(text or "{}")
    except Exception:
        return {}


def _sqlite_rows(path: Path, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
    except Exception:
        return []
    return [dict(row) for row in rows]


def _capture_environment(output_dir: Path, config_path: Path | None) -> None:
    environment = {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "git_commit": _run_command(["git", "rev-parse", "HEAD"]),
        "git_status": _run_command(["git", "status", "--short"]),
        "codex_path": _run_command(["which", "codex"]),
        "codex_version": _run_command(["codex", "--version"]),
        "nvidia_smi_l": _run_command(["nvidia-smi", "-L"]),
        "nvidia_smi_query": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ]
        ),
        "docker_ps": _run_command(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"]),
    }
    _write_json(output_dir / "environment.json", environment)
    lines = [
        "# Profile Scheduler Evidence Environment",
        f"- Git commit: {environment['git_commit'].get('stdout') or 'unknown'}",
        f"- Codex CLI: {environment['codex_version'].get('stdout') or 'unknown'} at {environment['codex_path'].get('stdout') or 'unknown'}",
        f"- GPU: {environment['nvidia_smi_query'].get('stdout') or environment['nvidia_smi_l'].get('stdout') or 'unavailable'}",
        f"- Config: {config_path or 'unknown'}",
    ]
    _write_text(output_dir / "environment.md", "\n".join(lines) + "\n")

    if config_path and config_path.exists():
        sanitized = _sanitize(_read_yaml(config_path))
        _write_text(output_dir / "config.sanitized.yaml", yaml.safe_dump(sanitized, sort_keys=False))


def _latest_journal(mode_root: Path) -> Path | None:
    journals = sorted(mode_root.glob("runs/*/logs/journal.json"), key=lambda path: path.stat().st_mtime)
    return journals[-1] if journals else None


def _metric_value(metric: Any) -> Any:
    if isinstance(metric, dict):
        return metric.get("value")
    return None


def _term_excerpt(node: dict[str, Any], limit: int = 500) -> str:
    term = node.get("_term_out") or node.get("term_out") or []
    if isinstance(term, list):
        text = "\n".join(str(item) for item in term)
    else:
        text = str(term or "")
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _node_rows_for_mode(mode: str, journal_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [{"mode": mode, "journal_path": str(journal_path), "parse_error": str(exc)}]
    nodes = payload.get("nodes") if isinstance(payload, dict) else payload
    if not isinstance(nodes, list):
        return [{"mode": mode, "journal_path": str(journal_path), "parse_error": "journal nodes are not a list"}]

    rows: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if node.get("stage") == "root":
            continue
        metric = _metric_value(node.get("metric"))
        is_buggy = node.get("is_buggy")
        outcome = node.get("outcome")
        exc_type = node.get("exc_type")
        has_metric = metric is not None
        if is_buggy is True or exc_type or outcome not in (None, "", "valid") or not has_metric:
            rows.append(
                {
                    "mode": mode,
                    "node_index": index,
                    "node_id": node.get("id"),
                    "parent_id": (payload.get("node2parent") or {}).get(node.get("id")) if isinstance(payload, dict) else None,
                    "stage": node.get("stage"),
                    "is_buggy": is_buggy,
                    "is_valid": node.get("is_valid"),
                    "outcome": outcome,
                    "metric": metric,
                    "exc_type": exc_type,
                    "failure_fingerprint": node.get("failure_fingerprint"),
                    "quarantine_reason": node.get("quarantine_reason"),
                    "scheduler_job_id": node.get("scheduler_job_id"),
                    "backend_name": node.get("backend_name"),
                    "resolved_batch_size": node.get("resolved_batch_size"),
                    "exec_time": node.get("exec_time"),
                    "analysis": node.get("analysis"),
                    "term_out_excerpt": _term_excerpt(node),
                    "journal_path": str(journal_path),
                }
            )
    return rows


def _collect_node_index(run_root: Path, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for mode in ("scheduler_off", "scheduler_on"):
        journal = _latest_journal(run_root / mode)
        if journal:
            rows.extend(_node_rows_for_mode(mode, journal))
    log_dir = output_dir / "log_excerpts"
    _write_json(log_dir / "node_evidence_index.json", rows)
    txt_lines = ["# Node Evidence Index", ""]
    for row in rows:
        txt_lines.append(
            "- {mode} node={node_id} stage={stage} buggy={is_buggy} outcome={outcome} exc={exc_type} metric={metric}".format(
                **{key: row.get(key) for key in ("mode", "node_id", "stage", "is_buggy", "outcome", "exc_type", "metric")}
            )
        )
    _write_text(log_dir / "node_evidence_index.txt", "\n".join(txt_lines) + "\n")
    if rows:
        csv_path = log_dir / "node_evidence_index.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _collect_scheduler_db_summary(run_root: Path, output_dir: Path) -> None:
    summary: dict[str, Any] = {}
    job_index_rows: list[dict[str, Any]] = []
    for mode in ("scheduler_off", "scheduler_on"):
        runtime_dir = run_root / mode / "scheduler_runtime"
        scheduler_db = runtime_dir / "db" / "scheduler.sqlite3"
        branch_db = runtime_dir / "db" / "branch_profile.sqlite3"
        event_counts = _sqlite_rows(
            scheduler_db,
            "SELECT event_type, COUNT(*) AS count FROM events GROUP BY event_type ORDER BY count DESC, event_type",
        )
        jobs = _sqlite_rows(
            scheduler_db,
            "SELECT job_id, status, baseline_model_id, submitted_at, updated_at, payload_json FROM jobs ORDER BY queue_sequence",
        )
        branch_profiles = _sqlite_rows(
            branch_db,
            """
            SELECT probe_key, model_key, profile_namespace, hardware_key, shape_signature,
                   search_mode, contract_version, resolved_batch_size, peak_vram_mb,
                   memory_total_mb, target_budget_mb, observations, last_job_id, updated_at
            FROM batch_probe_profiles
            ORDER BY updated_at DESC
            """,
        )
        summary[mode] = {
            "scheduler_db": str(scheduler_db) if scheduler_db.exists() else None,
            "branch_profile_db": str(branch_db) if branch_db.exists() else None,
            "event_counts": event_counts,
            "branch_profile_count": len(branch_profiles),
            "branch_profiles": branch_profiles,
        }
        for row in jobs:
            payload = _json_loads(row.get("payload_json"))
            metadata = payload.get("metadata") if isinstance(payload, dict) else {}
            batch_probe = payload.get("batch_probe") if isinstance(payload, dict) else {}
            runner_kwargs = ((payload.get("config") or {}).get("runner_kwargs") or {}) if isinstance(payload, dict) else {}
            metadata = metadata if isinstance(metadata, dict) else {}
            batch_probe = batch_probe if isinstance(batch_probe, dict) else {}
            job_index_rows.append(
                {
                    "mode": mode,
                    "job_id": row.get("job_id"),
                    "status": row.get("status"),
                    "baseline_model_id": row.get("baseline_model_id"),
                    "task_type": payload.get("task_type") if isinstance(payload, dict) else None,
                    "status_reason": payload.get("status_reason") if isinstance(payload, dict) else None,
                    "branch_name": metadata.get("branch_name"),
                    "model_family": metadata.get("model_family"),
                    "profile_namespace": batch_probe.get("profile_namespace") or metadata.get("batch_probe_profile_namespace"),
                    "branch_profile_key": metadata.get("branch_profile_key") or metadata.get("model_family_profile_key"),
                    "batch_probe_enabled": batch_probe.get("enabled"),
                    "batch_probe_supported": metadata.get("batch_probe_supported"),
                    "batch_probe_disabled_reason": metadata.get("batch_probe_disabled_reason"),
                    "detected_batch_size": metadata.get("detected_batch_size"),
                    "proposed_batch_size": metadata.get("proposed_batch_size"),
                    "resolved_batch_size": metadata.get("resolved_batch_size"),
                    "batch_probe_key": metadata.get("batch_probe_key"),
                    "batch_probe_source": metadata.get("batch_probe_source"),
                    "peak_vram_mib": metadata.get("runtime_peak_torch_reserved_mib") or metadata.get("runtime_peak_vram_mb"),
                    "placement_backend": metadata.get("placement_backend"),
                    "probe_max_epochs": runner_kwargs.get("probe_max_epochs"),
                    "normal_timeout": runner_kwargs.get("timeout"),
                    "scheduler_probe_failure_kind": metadata.get("scheduler_probe_failure_kind"),
                    "scheduler_probe_failure_reason": metadata.get("scheduler_probe_failure_reason"),
                    "submitted_at": row.get("submitted_at"),
                    "updated_at": row.get("updated_at"),
                }
            )
    log_dir = output_dir / "log_excerpts"
    _write_json(log_dir / "scheduler_db_summary.json", summary)
    if job_index_rows:
        csv_path = log_dir / "scheduler_job_failure_index.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in job_index_rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(job_index_rows)
        _write_json(log_dir / "scheduler_job_failure_index.json", job_index_rows)


def _copy_run_artifacts(run_root: Path, output_dir: Path) -> None:
    artifacts = output_dir / "config_overlays"
    for relative in ("manifest.txt", "scheduler_off/command.txt", "scheduler_on/command.txt"):
        source = run_root / relative
        if source.exists():
            target_name = relative.replace("/", "_")
            _write_text(artifacts / target_name, source.read_text(encoding="utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--phase", choices=("preflight", "postrun"), required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "preflight":
        _capture_environment(output_dir, config_path)
    else:
        _copy_run_artifacts(run_root, output_dir)
        _collect_node_index(run_root, output_dir)
        _collect_scheduler_db_summary(run_root, output_dir)


if __name__ == "__main__":
    main()
