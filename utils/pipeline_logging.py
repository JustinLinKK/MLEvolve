"""SQLite action log for MLEvolve pipeline runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import sqlite3
import threading


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, sort_keys=True, default=str)


def _bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


class PipelineActionLogger:
    """Small append-friendly SQLite log used for experiment analysis."""

    def __init__(self, db_path: str | Path, *, run_id: str, mode: str):
        self.db_path = Path(db_path)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    node_id TEXT,
                    parent_node_id TEXT,
                    job_id TEXT,
                    stage TEXT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pipeline_events_run_type
                    ON pipeline_events(run_id, event_type);
                CREATE INDEX IF NOT EXISTS idx_pipeline_events_node
                    ON pipeline_events(run_id, node_id);
                CREATE INDEX IF NOT EXISTS idx_pipeline_events_job
                    ON pipeline_events(run_id, job_id);

                CREATE TABLE IF NOT EXISTS node_actions (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    parent_node_id TEXT,
                    branch_id INTEGER,
                    stage TEXT,
                    action_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metric REAL,
                    is_buggy INTEGER,
                    is_valid INTEGER,
                    exec_time REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_node_actions_run_node
                    ON node_actions(run_id, node_id);
                CREATE INDEX IF NOT EXISTS idx_node_actions_run_type
                    ON node_actions(run_id, action_type);

                CREATE TABLE IF NOT EXISTS job_packets (
                    job_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    node_id TEXT,
                    scheduler_mode TEXT,
                    placement_mode TEXT,
                    placement_backend TEXT,
                    status TEXT,
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_seconds REAL,
                    detected_batch_size INTEGER,
                    resolved_batch_size INTEGER,
                    proposed_epochs INTEGER,
                    model_key TEXT,
                    framework TEXT,
                    uses_amp INTEGER,
                    requires_gpu INTEGER,
                    script_signature TEXT,
                    metric REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_packets_run
                    ON job_packets(run_id, node_id);

                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                );
                """
            )

    def emit(
        self,
        event_type: str,
        *,
        node_id: str | None = None,
        parent_node_id: str | None = None,
        job_id: str | None = None,
        stage: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_events
                (run_id, mode, event_type, node_id, parent_node_id, job_id, stage, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    self.mode,
                    event_type,
                    node_id,
                    parent_node_id,
                    job_id,
                    stage,
                    utc_now_iso(),
                    _json(payload),
                ),
            )

    def record_node_action(
        self,
        *,
        node_id: str,
        action_type: str,
        parent_node_id: str | None = None,
        branch_id: int | None = None,
        stage: str | None = None,
        metric: float | None = None,
        is_buggy: bool | None = None,
        is_valid: bool | None = None,
        exec_time: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO node_actions
                (run_id, mode, node_id, parent_node_id, branch_id, stage, action_type, created_at,
                 metric, is_buggy, is_valid, exec_time, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    self.mode,
                    node_id,
                    parent_node_id,
                    branch_id,
                    stage,
                    action_type,
                    utc_now_iso(),
                    metric,
                    _bool_int(is_buggy),
                    _bool_int(is_valid),
                    exec_time,
                    _json(payload),
                ),
            )

    def upsert_job_packet(self, job_id: str, **fields: Any) -> None:
        payload = dict(fields.pop("payload", {}) or {})
        payload.update({key: value for key, value in fields.items() if key not in _JOB_COLUMNS and value is not None})
        row = {key: fields.get(key) for key in _JOB_COLUMNS}
        row["job_id"] = str(job_id)
        if row.get("run_id") is None:
            row["run_id"] = self.run_id
        if row.get("mode") is None:
            row["mode"] = self.mode
        if row.get("created_at") is None:
            row["created_at"] = utc_now_iso()
        row["payload_json"] = _json(payload)
        row["uses_amp"] = _bool_int(row.get("uses_amp"))
        row["requires_gpu"] = _bool_int(row.get("requires_gpu"))
        columns = [
            "job_id",
            "run_id",
            "mode",
            "node_id",
            "scheduler_mode",
            "placement_mode",
            "placement_backend",
            "status",
            "created_at",
            "submitted_at",
            "started_at",
            "finished_at",
            "duration_seconds",
            "detected_batch_size",
            "resolved_batch_size",
            "proposed_epochs",
            "model_key",
            "framework",
            "uses_amp",
            "requires_gpu",
            "script_signature",
            "metric",
            "payload_json",
        ]
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "job_id")
        with self._lock, self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO job_packets ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(job_id) DO UPDATE SET {updates}
                """,
                tuple(row.get(column) for column in columns),
            )

    def record_run_metrics(self, metrics: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_metrics(run_id, mode, created_at, metrics_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    mode=excluded.mode,
                    created_at=excluded.created_at,
                    metrics_json=excluded.metrics_json
                """,
                (self.run_id, self.mode, utc_now_iso(), _json(metrics)),
            )

    def update_job_packet_for_node(self, node_id: str, **fields: Any) -> None:
        allowed = {"status", "duration_seconds", "resolved_batch_size", "metric", "finished_at"}
        updates = {key: value for key, value in fields.items() if key in allowed and value is not None}
        if not updates:
            return
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE job_packets SET {assignments} WHERE run_id=? AND node_id=?",
                tuple(updates.values()) + (self.run_id, str(node_id)),
            )

    def close(self) -> None:
        return None


_JOB_COLUMNS = {
    "job_id",
    "run_id",
    "mode",
    "node_id",
    "scheduler_mode",
    "placement_mode",
    "placement_backend",
    "status",
    "created_at",
    "submitted_at",
    "started_at",
    "finished_at",
    "duration_seconds",
    "detected_batch_size",
    "resolved_batch_size",
    "proposed_epochs",
    "model_key",
    "framework",
    "uses_amp",
    "requires_gpu",
    "script_signature",
    "metric",
}


def pipeline_logger_for(owner: Any) -> PipelineActionLogger | None:
    logger = getattr(owner, "pipeline_logger", None)
    return logger if isinstance(logger, PipelineActionLogger) else None


def log_pipeline_event(
    owner: Any,
    event_type: str,
    *,
    node: Any | None = None,
    node_id: str | None = None,
    parent_node_id: str | None = None,
    job_id: str | None = None,
    stage: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    logger = pipeline_logger_for(owner)
    if logger is None:
        return
    if node is not None:
        node_id = node_id or str(getattr(node, "id", ""))
        parent = getattr(node, "parent", None)
        parent_node_id = parent_node_id or (str(getattr(parent, "id", "")) if parent is not None else None)
        stage = stage or getattr(node, "stage", None)
    try:
        logger.emit(
            event_type,
            node_id=node_id,
            parent_node_id=parent_node_id,
            job_id=job_id,
            stage=stage,
            payload=payload,
        )
    except Exception:
        return


def record_pipeline_node_action(
    owner: Any,
    node: Any,
    action_type: str,
    *,
    payload: dict[str, Any] | None = None,
) -> None:
    logger = pipeline_logger_for(owner)
    if logger is None or node is None:
        return
    parent = getattr(node, "parent", None)
    metric = getattr(getattr(node, "metric", None), "value", None)
    try:
        logger.record_node_action(
            node_id=str(getattr(node, "id", "")),
            parent_node_id=str(getattr(parent, "id", "")) if parent is not None else None,
            branch_id=getattr(node, "branch_id", None),
            stage=getattr(node, "stage", None),
            action_type=action_type,
            metric=float(metric) if metric is not None else None,
            is_buggy=getattr(node, "is_buggy", None),
            is_valid=getattr(node, "is_valid", None),
            exec_time=getattr(node, "exec_time", None),
            payload=payload,
        )
    except Exception:
        return
