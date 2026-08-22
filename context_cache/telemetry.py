"""Normalized cache timing and usage persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Mapping
import uuid

from .models import CacheFamily, NormalizedCacheUsage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump())
    dictionary = getattr(value, "dict", None)
    if callable(dictionary):
        return _jsonable(dictionary())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


_SECRET_TEXT = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"\b(sk-[A-Za-z0-9_-]{12,})\b|"
    r"((?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*)[^\s,;]+"
)
_SENSITIVE_FIELD_NAMES = {
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "credentials",
    "client_secret",
    "private_key",
}


def sanitize_prompt(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).lower() in _SENSITIVE_FIELD_NAMES
                else sanitize_prompt(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_prompt(item) for item in value]
    if isinstance(value, str):
        return _SECRET_TEXT.sub(
            lambda match: (match.group(1) or match.group(3) or "") + "<redacted>",
            value,
        )
    return _jsonable(value)


@dataclass
class CacheTelemetryEvent:
    timestamp: str
    run_id: str | None
    request_id: str
    provider: str
    upstream_provider: str | None
    api_family: str
    model: str
    agent_role: str
    cache_family_id: str | None
    stable_prefix_hash: str | None
    common_pack_hash: str | None
    role_pack_hash: str | None
    tool_schema_hash: str | None
    reasoning_config_hash: str | None
    local_pack_cache_hit: bool | None
    db_retrieval_ms: float | None
    pack_build_ms: float | None
    request_prepare_ms: float | None
    ttft_ms: float | None
    total_request_ms: float | None
    end_to_end_ms: float | None
    prompt_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cache_miss_tokens: int | None
    output_tokens: int | None
    cache_hit_ratio: float | None
    cost_usd: float | None
    finish_reason: str | None
    error_type: str | None
    raw_usage_json: str | None
    prompt_snapshot_json: str | None


class CacheTelemetryStore:
    def __init__(self, registry_path: str | Path) -> None:
        self.path = Path(registry_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS context_cache_events (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    run_id TEXT,
                    provider TEXT NOT NULL,
                    upstream_provider TEXT,
                    api_family TEXT NOT NULL,
                    model TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    cache_family_id TEXT,
                    stable_prefix_hash TEXT,
                    common_pack_hash TEXT,
                    role_pack_hash TEXT,
                    tool_schema_hash TEXT,
                    reasoning_config_hash TEXT,
                    local_pack_cache_hit INTEGER,
                    db_retrieval_ms REAL,
                    pack_build_ms REAL,
                    request_prepare_ms REAL,
                    ttft_ms REAL,
                    total_request_ms REAL,
                    end_to_end_ms REAL,
                    prompt_tokens INTEGER,
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    cache_miss_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_hit_ratio REAL,
                    cost_usd REAL,
                    finish_reason TEXT,
                    error_type TEXT,
                    raw_usage_json TEXT,
                    prompt_snapshot_json TEXT
                )
                """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_context_cache_events_family ON context_cache_events(cache_family_id, timestamp)"
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(context_cache_events)"
                ).fetchall()
            }
            if "prompt_snapshot_json" not in columns:
                connection.execute(
                    "ALTER TABLE context_cache_events ADD COLUMN prompt_snapshot_json TEXT"
                )

    def record(self, event: CacheTelemetryEvent) -> None:
        payload = asdict(event)
        payload["local_pack_cache_hit"] = (
            None
            if event.local_pack_cache_hit is None
            else int(event.local_pack_cache_hit)
        )
        columns = list(payload)
        placeholders = ",".join("?" for _ in columns)
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO context_cache_events ({','.join(columns)}) VALUES ({placeholders})",
                tuple(payload[column] for column in columns),
            )

    def rows(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM context_cache_events"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY timestamp, request_id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def export_jsonl(self, path: str | Path, *, run_id: str | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in self.rows(run_id=run_id):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return destination

    def export_csv(self, path: str | Path, *, run_id: str | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows(run_id=run_id)
        columns = list(CacheTelemetryEvent.__dataclass_fields__)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return destination


class RequestTelemetry:
    """Monotonic request timer that emits exactly one terminal event."""

    def __init__(
        self,
        store: CacheTelemetryStore,
        *,
        run_id: str | None,
        provider: str,
        api_family: str,
        model: str,
        agent_role: str,
        family: CacheFamily | None,
        stable_prefix_hash: str | None,
        local_pack_cache_hit: bool | None,
        expected_stable_prefix_tokens: int | None,
        db_retrieval_ms: float | None,
        pack_build_ms: float | None,
        started_at: float | None = None,
        prompt_snapshot: Any = None,
    ) -> None:
        self.store = store
        self.timestamp = _utc_now()
        self.run_id = run_id
        self.request_id = uuid.uuid4().hex
        self.provider = provider
        self.api_family = api_family
        self.model = model
        self.agent_role = agent_role
        self.family = family
        self.stable_prefix_hash = stable_prefix_hash
        self.local_pack_cache_hit = local_pack_cache_hit
        self.expected_stable_prefix_tokens = expected_stable_prefix_tokens
        self.db_retrieval_ms = db_retrieval_ms
        self.pack_build_ms = pack_build_ms
        self.prompt_snapshot = prompt_snapshot
        self.t0 = time.monotonic() if started_at is None else started_at
        self.t1: float | None = None
        self.t2: float | None = None
        self.t3: float | None = None
        self.t4: float | None = None
        self._emitted = False
        self._lock = threading.Lock()

    def pack_ready(self) -> None:
        if self.t1 is None:
            self.t1 = time.monotonic()

    def request_started(self) -> None:
        self.pack_ready()
        if self.t2 is None:
            self.t2 = time.monotonic()

    def first_meaningful_delta(self) -> None:
        if self.t3 is None:
            self.t3 = time.monotonic()

    def finish(
        self,
        *,
        usage: NormalizedCacheUsage | None = None,
        raw_usage: Any = None,
        upstream_provider: str | None = None,
        finish_reason: str | None = None,
        cost_usd: float | None = None,
        error_type: str | None = None,
    ) -> CacheTelemetryEvent | None:
        with self._lock:
            if self._emitted:
                return None
            self._emitted = True
            self.t4 = time.monotonic()
            if self.t2 is not None and self.t3 is None:
                self.t3 = self.t4
            normalized = usage or NormalizedCacheUsage()
            ratio = None
            if (
                normalized.cache_read_tokens is not None
                and self.expected_stable_prefix_tokens
            ):
                ratio = (
                    normalized.cache_read_tokens / self.expected_stable_prefix_tokens
                )
            raw_json = None
            if raw_usage is not None:
                raw_json = json.dumps(
                    _jsonable(raw_usage),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                raw_json = raw_json[:16384]
            prompt_json = None
            if self.prompt_snapshot is not None:
                prompt_json = json.dumps(
                    sanitize_prompt(self.prompt_snapshot),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )[:65536]
            family = self.family
            event = CacheTelemetryEvent(
                timestamp=self.timestamp,
                run_id=self.run_id,
                request_id=self.request_id,
                provider=self.provider,
                upstream_provider=upstream_provider,
                api_family=self.api_family,
                model=self.model,
                agent_role=self.agent_role,
                cache_family_id=family.id if family else None,
                stable_prefix_hash=self.stable_prefix_hash,
                common_pack_hash=family.common_pack_hash if family else None,
                role_pack_hash=family.role_pack_hash if family else None,
                tool_schema_hash=family.tool_schema_hash if family else None,
                reasoning_config_hash=family.reasoning_config_hash if family else None,
                local_pack_cache_hit=self.local_pack_cache_hit,
                db_retrieval_ms=self.db_retrieval_ms,
                pack_build_ms=self.pack_build_ms,
                request_prepare_ms=(
                    (self.t2 - self.t1) * 1000
                    if self.t1 is not None and self.t2 is not None
                    else None
                ),
                ttft_ms=(
                    (self.t3 - self.t2) * 1000
                    if self.t2 is not None and self.t3 is not None
                    else None
                ),
                total_request_ms=(
                    (self.t4 - self.t2) * 1000 if self.t2 is not None else None
                ),
                end_to_end_ms=(self.t4 - self.t0) * 1000,
                prompt_tokens=normalized.prompt_tokens,
                cache_read_tokens=normalized.cache_read_tokens,
                cache_write_tokens=normalized.cache_write_tokens,
                cache_miss_tokens=normalized.cache_miss_tokens,
                output_tokens=normalized.output_tokens,
                cache_hit_ratio=ratio,
                cost_usd=cost_usd,
                finish_reason=finish_reason,
                error_type=error_type,
                raw_usage_json=raw_json,
                prompt_snapshot_json=prompt_json,
            )
            self.store.record(event)
            return event
