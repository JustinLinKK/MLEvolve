"""SQLite authority for lesson observations, revisions, and durable jobs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .config import LessonProfileSettings
from .models import LessonRecord, ProfileIdentity


SCHEMA_VERSION = 1


_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    profile_key TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    identity_json TEXT NOT NULL,
    model_family TEXT NOT NULL,
    architecture_type TEXT NOT NULL,
    hardware_key TEXT NOT NULL,
    accelerator_key TEXT NOT NULL,
    resource_slice_key TEXT NOT NULL,
    runtime_class TEXT NOT NULL,
    framework_major TEXT NOT NULL,
    cuda_major TEXT NOT NULL,
    backend_class TEXT NOT NULL,
    workload_bucket TEXT NOT NULL,
    active_revision INTEGER,
    maturity TEXT NOT NULL DEFAULT 'provisional',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(schema_version, model_family, architecture_type, hardware_key,
           accelerator_key, resource_slice_key, runtime_class, backend_class,
           workload_bucket)
);

CREATE INDEX IF NOT EXISTS idx_profiles_compatibility ON profiles(
    model_family, architecture_type, accelerator_key, resource_slice_key,
    backend_class, workload_bucket, framework_major, cuda_major
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    profile_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    processing_state TEXT NOT NULL DEFAULT 'queued',
    created_at REAL NOT NULL,
    processed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_observations_profile ON observations(profile_key, created_at);
CREATE INDEX IF NOT EXISTS idx_observations_runs ON observations(profile_key, run_id, outcome);

CREATE TABLE IF NOT EXISTS builder_jobs (
    job_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL UNIQUE REFERENCES observations(observation_id),
    state TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at REAL NOT NULL,
    lease_owner TEXT,
    lease_expires_at REAL,
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_builder_jobs_claim ON builder_jobs(state, available_at, lease_expires_at);

CREATE TABLE IF NOT EXISTS profile_revisions (
    revision_id TEXT PRIMARY KEY,
    profile_key TEXT NOT NULL REFERENCES profiles(profile_key),
    revision_number INTEGER NOT NULL,
    source_observation_id TEXT NOT NULL UNIQUE REFERENCES observations(observation_id),
    state TEXT NOT NULL DEFAULT 'pending',
    maturity TEXT NOT NULL,
    baseline_json TEXT NOT NULL,
    trust_json TEXT NOT NULL,
    builder_model TEXT,
    builder_prompt_version TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    activated_at REAL,
    UNIQUE(profile_key, revision_number)
);

CREATE INDEX IF NOT EXISTS idx_revisions_active ON profile_revisions(profile_key, state, revision_number);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    profile_key TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    lesson_type TEXT NOT NULL,
    audiences_json TEXT NOT NULL,
    content_json TEXT NOT NULL,
    change_signature TEXT NOT NULL DEFAULT '',
    change_scope TEXT NOT NULL,
    change_action TEXT NOT NULL,
    layer_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY(profile_key, revision_number)
        REFERENCES profile_revisions(profile_key, revision_number)
);

CREATE INDEX IF NOT EXISTS idx_lessons_retrieval ON lessons(profile_key, active, lesson_type, confidence);

CREATE TABLE IF NOT EXISTS qdrant_outbox (
    outbox_id TEXT PRIMARY KEY,
    profile_key TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    source_observation_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(profile_key, revision_number)
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending ON qdrant_outbox(state, updated_at);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    profile_key TEXT NOT NULL,
    claim_key TEXT NOT NULL,
    left_observation_id TEXT NOT NULL,
    right_observation_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open',
    created_at REAL NOT NULL,
    resolved_at REAL,
    UNIQUE(profile_key, claim_key, left_observation_id, right_observation_id)
);

CREATE INDEX IF NOT EXISTS idx_conflicts_profile ON conflicts(profile_key, state, created_at);
"""


class LessonProfileRegistry:
    def __init__(self, settings: LessonProfileSettings | str | Path):
        if isinstance(settings, LessonProfileSettings):
            self.settings = settings
            self.path = settings.database_path
        else:
            self.settings = LessonProfileSettings(sqlite_path=str(settings))
            self.path = Path(settings).expanduser().resolve()

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.executescript(_DDL)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, time.time()),
            )
        return {"ok": True, "schema_version": SCHEMA_VERSION, "sqlite_path": str(self.path)}

    @staticmethod
    def _loads(value: Any, default: Any) -> Any:
        try:
            return json.loads(str(value))
        except Exception:
            return default

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _valid_evidence_refs(value: Any) -> bool:
        refs = list(value or [])
        return bool(refs) and all(
            isinstance(ref, str) and ":" in ref and bool(ref.split(":", 1)[1].strip())
            for ref in refs
        )

    def enqueue_observation(
        self,
        *,
        identity: ProfileIdentity | Mapping[str, Any],
        evidence: Mapping[str, Any],
        outcome: str,
        run_id: str,
        node_id: str,
        extractor_version: str,
    ) -> dict[str, Any]:
        identity_dict = identity.to_dict() if isinstance(identity, ProfileIdentity) else dict(identity)
        if not self._valid_evidence_refs(evidence.get("evidence_refs")):
            raise ValueError("Observation evidence_refs must contain resolvable typed references")
        profile_key = str(identity_dict["profile_key"])
        idempotency_key = hashlib.sha256(
            f"{profile_key}:{node_id}:{extractor_version}".encode("utf-8")
        ).hexdigest()
        now = time.time()
        observation_id = uuid.uuid5(uuid.NAMESPACE_URL, f"mlevolve-observation:{idempotency_key}").hex
        job_id = uuid.uuid5(uuid.NAMESPACE_URL, f"mlevolve-builder-job:{observation_id}").hex
        with self.transaction() as connection:
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO observations(
                    observation_id, idempotency_key, profile_key, run_id, node_id,
                    outcome, evidence_json, processing_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    observation_id,
                    idempotency_key,
                    profile_key,
                    str(run_id),
                    str(node_id),
                    str(outcome),
                    self._dumps(dict(evidence)),
                    now,
                ),
            ).rowcount
            connection.execute(
                """
                INSERT OR IGNORE INTO builder_jobs(
                    job_id, observation_id, state, attempts, available_at,
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?)
                """,
                (job_id, observation_id, now, now, now),
            )
        return {
            "ok": True,
            "inserted": bool(inserted),
            "observation_id": observation_id,
            "job_id": job_id,
            "profile_key": profile_key,
        }

    def lease_next_job(self, *, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        now = time.time()
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM builder_jobs
                WHERE available_at <= ?
                  AND (state = 'queued' OR (state = 'leased' AND lease_expires_at <= ?))
                ORDER BY created_at, job_id LIMIT 1
                """,
                (now, now),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE builder_jobs SET state='leased', attempts=attempts+1,
                    lease_owner=?, lease_expires_at=?, updated_at=?
                WHERE job_id=?
                """,
                (worker_id, now + max(1, int(lease_seconds)), now, row["job_id"]),
            )
            job = dict(row)
            job["attempts"] = int(job["attempts"]) + 1
            job["state"] = "leased"
            job["lease_owner"] = worker_id
            job["lease_expires_at"] = now + max(1, int(lease_seconds))
            return job

    def observation(self, observation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM observations WHERE observation_id=?", (observation_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["evidence"] = self._loads(result.pop("evidence_json"), {})
        return result

    def observations_for_profile(self, profile_key: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM observations WHERE profile_key=? ORDER BY created_at DESC LIMIT ?",
                (profile_key, max(1, int(limit))),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = self._loads(item.pop("evidence_json"), {})
            result.append(item)
        return result

    def complete_job(self, job_id: str, *, observation_id: str, ignored: bool = False) -> None:
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE builder_jobs SET state='completed', lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=?",
                (now, job_id),
            )
            connection.execute(
                "UPDATE observations SET processing_state=?, processed_at=? WHERE observation_id=?",
                ("ignored" if ignored else "published", now, observation_id),
            )

    def complete_publication_for_observation(self, observation_id: str) -> None:
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE builder_jobs SET state='completed', lease_owner=NULL,
                    lease_expires_at=NULL, updated_at=? WHERE observation_id=?
                """,
                (now, observation_id),
            )
            connection.execute(
                "UPDATE observations SET processing_state='published', processed_at=? WHERE observation_id=?",
                (now, observation_id),
            )

    def fail_job(self, job_id: str, *, error: str, max_retries: int, retry_delay_seconds: float) -> None:
        now = time.time()
        with self.transaction() as connection:
            row = connection.execute("SELECT attempts FROM builder_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return
            state = "failed" if int(row["attempts"]) >= max(1, int(max_retries)) else "queued"
            connection.execute(
                """
                UPDATE builder_jobs SET state=?, available_at=?, lease_owner=NULL,
                    lease_expires_at=NULL, last_error=?, updated_at=? WHERE job_id=?
                """,
                (state, now + max(0.0, float(retry_delay_seconds)), str(error)[:4000], now, job_id),
            )

    def profile(self, profile_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM profiles WHERE profile_key=?", (profile_key,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["identity"] = self._loads(result.pop("identity_json"), {})
        return result

    def find_compatible_profiles(self, identity: Mapping[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
        fields = (
            "model_family",
            "architecture_type",
            "accelerator_key",
            "resource_slice_key",
            "backend_class",
            "workload_bucket",
            "framework_major",
            "cuda_major",
        )
        query = " AND ".join(f"{field}=?" for field in fields)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM profiles WHERE {query} AND active_revision IS NOT NULL ORDER BY updated_at DESC LIMIT ?",
                tuple(str(identity.get(field) or "") for field in fields) + (max(1, int(limit)),),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["identity"] = self._loads(item.pop("identity_json"), {})
            results.append(item)
        return results

    def active_revision(self, profile_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT r.* FROM profile_revisions r
                JOIN profiles p ON p.profile_key=r.profile_key AND p.active_revision=r.revision_number
                WHERE r.profile_key=? AND r.state='active'
                """,
                (profile_key,),
            ).fetchone()
        return self._revision_dict(row) if row else None

    def _revision_dict(self, row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["baseline"] = self._loads(item.pop("baseline_json"), {})
        item["trust"] = self._loads(item.pop("trust_json"), {})
        return item

    def lessons_for_revision(self, profile_key: str, revision_number: int, *, role: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM lessons WHERE profile_key=? AND revision_number=? ORDER BY confidence DESC, lesson_id",
                (profile_key, int(revision_number)),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            audiences = self._loads(item.pop("audiences_json"), [])
            if role and role not in audiences:
                continue
            item["agent_audiences"] = audiences
            item["content"] = self._loads(item.pop("content_json"), {})
            item["evidence_refs"] = self._loads(item.pop("evidence_refs_json"), [])
            item["warnings"] = self._loads(item.pop("warnings_json"), [])
            result.append(item)
        return result

    def search_active_lessons(self, *, role: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Deterministic lexical fallback used only when Qdrant is unavailable."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT l.*, p.maturity FROM lessons l
                JOIN profiles p ON p.profile_key=l.profile_key
                WHERE l.active=1 AND p.active_revision=l.revision_number
                ORDER BY l.confidence DESC, l.created_at DESC LIMIT 500
                """
            ).fetchall()
        query_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", str(query).lower()))
        ranked = []
        for row in rows:
            item = self._lesson_row_payload(row)
            if role not in item.get("agent_audiences", []):
                continue
            text = self._dumps({
                "content": item.get("content"),
                "warnings": item.get("warnings"),
                "lesson_type": item.get("lesson_type"),
            }).lower()
            score = len(query_tokens & set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)))
            item["record_id"] = item["lesson_id"]
            item["revision"] = item["revision_number"]
            item["score"] = float(score) + float(item.get("confidence") or 0.0)
            ranked.append(item)
        ranked.sort(key=lambda item: (item["score"], item["confidence"]), reverse=True)
        return ranked[: max(1, int(limit))]

    def pending_publication(self, observation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM qdrant_outbox WHERE source_observation_id=?", (observation_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = self._loads(item.pop("payload_json"), {})
        return item

    def prepare_revision(
        self,
        *,
        identity: Mapping[str, Any],
        observation_id: str,
        baseline: Mapping[str, Any],
        trust: Mapping[str, Any],
        maturity: str,
        lessons: Iterable[LessonRecord | Mapping[str, Any]],
        builder_model: str | None,
        builder_prompt_version: str,
        extractor_version: str,
    ) -> dict[str, Any]:
        existing = self.pending_publication(observation_id)
        if existing is not None:
            return existing
        identity_dict = dict(identity)
        profile_key = str(identity_dict["profile_key"])
        normalized_lessons = [lesson.to_dict() if isinstance(lesson, LessonRecord) else dict(lesson) for lesson in lessons]
        if not self._valid_evidence_refs(trust.get("evidence_refs")):
            raise ValueError("Revision trust requires resolvable evidence references")
        for lesson in normalized_lessons:
            if not self._valid_evidence_refs(lesson.get("evidence_refs")):
                raise ValueError("Every lesson requires resolvable evidence references")
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO profiles(
                    profile_key, schema_version, identity_json, model_family,
                    architecture_type, hardware_key, accelerator_key,
                    resource_slice_key, runtime_class, framework_major, cuda_major,
                    backend_class, workload_bucket, maturity, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_key,
                    identity_dict["schema_version"],
                    self._dumps(identity_dict),
                    identity_dict["model_family"],
                    identity_dict["architecture_type"],
                    identity_dict["hardware_key"],
                    identity_dict["accelerator_key"],
                    identity_dict["resource_slice_key"],
                    identity_dict["runtime_class"],
                    identity_dict["framework_major"],
                    identity_dict["cuda_major"],
                    identity_dict["backend_class"],
                    identity_dict["workload_bucket"],
                    maturity,
                    now,
                    now,
                ),
            )
            prior = connection.execute(
                "SELECT COALESCE(MAX(revision_number), 0) AS value FROM profile_revisions WHERE profile_key=?",
                (profile_key,),
            ).fetchone()
            revision_number = int(prior["value"]) + 1
            revision_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"mlevolve-profile-revision:{profile_key}:{revision_number}"
            ).hex
            connection.execute(
                """
                INSERT INTO profile_revisions(
                    revision_id, profile_key, revision_number, source_observation_id,
                    state, maturity, baseline_json, trust_json, builder_model,
                    builder_prompt_version, extractor_version, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    profile_key,
                    revision_number,
                    observation_id,
                    maturity,
                    self._dumps(dict(baseline)),
                    self._dumps(dict(trust)),
                    builder_model,
                    builder_prompt_version,
                    extractor_version,
                    now,
                ),
            )
            for position, lesson in enumerate(normalized_lessons):
                lesson_id = str(lesson.get("lesson_id") or uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"mlevolve-lesson:{profile_key}:{revision_number}:{position}:{lesson.get('lesson_type')}",
                ).hex)
                connection.execute(
                    """
                    INSERT INTO lessons(
                        lesson_id, profile_key, revision_number, lesson_type,
                        audiences_json, content_json, change_signature, change_scope,
                        change_action, layer_type, confidence, evidence_refs_json,
                        warnings_json, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        lesson_id,
                        profile_key,
                        revision_number,
                        str(lesson.get("lesson_type") or "modification"),
                        self._dumps(list(lesson.get("agent_audiences") or [])),
                        self._dumps(dict(lesson.get("content") or {})),
                        str(lesson.get("change_signature") or ""),
                        str(lesson.get("change_scope") or "training_only"),
                        str(lesson.get("change_action") or "other"),
                        str(lesson.get("layer_type") or "other"),
                        max(0.0, min(1.0, float(lesson.get("confidence") or 0.0))),
                        self._dumps(list(lesson.get("evidence_refs") or [])),
                        self._dumps(list(lesson.get("warnings") or [])),
                        now,
                    ),
                )
            lesson_rows = connection.execute(
                "SELECT * FROM lessons WHERE profile_key=? AND revision_number=? ORDER BY lesson_id",
                (profile_key, revision_number),
            ).fetchall()
            payload = {
                "profile_key": profile_key,
                "revision_number": revision_number,
                "identity": identity_dict,
                "baseline": dict(baseline),
                "trust": dict(trust),
                "maturity": maturity,
                "lessons": [self._lesson_row_payload(row) for row in lesson_rows],
            }
            outbox_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"mlevolve-qdrant-outbox:{profile_key}:{revision_number}"
            ).hex
            connection.execute(
                """
                INSERT INTO qdrant_outbox(
                    outbox_id, profile_key, revision_number, source_observation_id,
                    state, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (outbox_id, profile_key, revision_number, observation_id, self._dumps(payload), now, now),
            )
        return {
            "outbox_id": outbox_id,
            "profile_key": profile_key,
            "revision_number": revision_number,
            "source_observation_id": observation_id,
            "state": "pending",
            "payload": payload,
        }

    def _lesson_row_payload(self, row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["agent_audiences"] = self._loads(item.pop("audiences_json"), [])
        item["content"] = self._loads(item.pop("content_json"), {})
        item["evidence_refs"] = self._loads(item.pop("evidence_refs_json"), [])
        item["warnings"] = self._loads(item.pop("warnings_json"), [])
        return item

    def record_outbox_failure(self, outbox_id: str, error: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE qdrant_outbox SET attempts=attempts+1, last_error=?, updated_at=? WHERE outbox_id=?",
                (str(error)[:4000], time.time(), outbox_id),
            )

    def activate_publication(self, outbox_id: str) -> dict[str, Any]:
        now = time.time()
        with self.transaction() as connection:
            outbox = connection.execute("SELECT * FROM qdrant_outbox WHERE outbox_id=?", (outbox_id,)).fetchone()
            if outbox is None:
                raise KeyError(f"Unknown outbox record: {outbox_id}")
            profile_key = str(outbox["profile_key"])
            revision_number = int(outbox["revision_number"])
            revision = connection.execute(
                "SELECT maturity FROM profile_revisions WHERE profile_key=? AND revision_number=?",
                (profile_key, revision_number),
            ).fetchone()
            if revision is None:
                raise RuntimeError("Outbox points to a missing revision")
            current_profile = connection.execute(
                "SELECT active_revision, maturity FROM profiles WHERE profile_key=?",
                (profile_key,),
            ).fetchone()
            if current_profile is not None and int(current_profile["active_revision"] or 0) > revision_number:
                connection.execute(
                    "UPDATE profile_revisions SET state='superseded' WHERE profile_key=? AND revision_number=?",
                    (profile_key, revision_number),
                )
                connection.execute(
                    "UPDATE qdrant_outbox SET state='published', updated_at=? WHERE outbox_id=?",
                    (now, outbox_id),
                )
                return {
                    "profile_key": profile_key,
                    "revision_number": int(current_profile["active_revision"]),
                    "maturity": current_profile["maturity"],
                    "superseded_publication": revision_number,
                }
            connection.execute(
                "UPDATE profile_revisions SET state='superseded' WHERE profile_key=? AND state='active'",
                (profile_key,),
            )
            connection.execute(
                "UPDATE lessons SET active=0 WHERE profile_key=?",
                (profile_key,),
            )
            connection.execute(
                "UPDATE profile_revisions SET state='active', activated_at=? WHERE profile_key=? AND revision_number=?",
                (now, profile_key, revision_number),
            )
            connection.execute(
                "UPDATE lessons SET active=1 WHERE profile_key=? AND revision_number=?",
                (profile_key, revision_number),
            )
            connection.execute(
                "UPDATE profiles SET active_revision=?, maturity=?, updated_at=? WHERE profile_key=?",
                (revision_number, revision["maturity"], now, profile_key),
            )
            connection.execute(
                "UPDATE qdrant_outbox SET state='published', updated_at=? WHERE outbox_id=?",
                (now, outbox_id),
            )
        return {"profile_key": profile_key, "revision_number": revision_number, "maturity": revision["maturity"]}

    def distinct_successful_runs(self, profile_key: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT run_id) AS value FROM observations WHERE profile_key=? AND outcome='valid'",
                (profile_key,),
            ).fetchone()
        return int(row["value"] if row else 0)

    def distinct_failed_runs(self, profile_key: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT run_id) AS value FROM observations WHERE profile_key=? AND outcome!='valid'",
                (profile_key,),
            ).fetchone()
        return int(row["value"] if row else 0)

    def add_conflict(
        self,
        *,
        profile_key: str,
        claim_key: str,
        left_observation_id: str,
        right_observation_id: str,
        details: Mapping[str, Any],
    ) -> str:
        ordered = sorted((left_observation_id, right_observation_id))
        conflict_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"mlevolve-conflict:{profile_key}:{claim_key}:{':'.join(ordered)}"
        ).hex
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conflicts(
                    conflict_id, profile_key, claim_key, left_observation_id,
                    right_observation_id, details_json, state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (conflict_id, profile_key, claim_key, ordered[0], ordered[1], self._dumps(dict(details)), time.time()),
            )
        return conflict_id

    def open_conflict_count(self, profile_key: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS value FROM conflicts WHERE profile_key=? AND state='open'", (profile_key,)
            ).fetchone()
        return int(row["value"] if row else 0)

    def list_conflicts(self, profile_key: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if profile_key:
                rows = connection.execute(
                    "SELECT * FROM conflicts WHERE profile_key=? ORDER BY created_at DESC", (profile_key,)
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM conflicts ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = self._loads(item.pop("details_json"), {})
            result.append(item)
        return result

    def list_revisions(self, profile_key: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM profile_revisions WHERE profile_key=? ORDER BY revision_number DESC",
                (profile_key,),
            ).fetchall()
        return [self._revision_dict(row) for row in rows]

    def rollback(self, profile_key: str, revision_number: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT revision_id FROM profile_revisions WHERE profile_key=? AND revision_number=?",
                (profile_key, int(revision_number)),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown revision {profile_key}@{revision_number}")
        # Rollback creates a new immutable revision with the old content.
        old = self.list_revisions(profile_key)
        source = next(item for item in old if int(item["revision_number"]) == int(revision_number))
        profile = self.profile(profile_key)
        if profile is None:
            raise KeyError(profile_key)
        lessons = self.lessons_for_revision(profile_key, revision_number)
        for lesson in lessons:
            lesson.pop("lesson_id", None)
        synthetic_observation = f"rollback:{profile_key}:{revision_number}:{time.time_ns()}"
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO observations(
                    observation_id, idempotency_key, profile_key, run_id, node_id,
                    outcome, evidence_json, processing_state, created_at, processed_at
                ) VALUES (?, ?, ?, 'operator', ?, 'rollback', '{}', 'queued', ?, NULL)
                """,
                (synthetic_observation, synthetic_observation, profile_key, synthetic_observation, now),
            )
            job_id = uuid.uuid5(uuid.NAMESPACE_URL, f"mlevolve-builder-job:{synthetic_observation}").hex
            connection.execute(
                """
                INSERT INTO builder_jobs(
                    job_id, observation_id, state, attempts, available_at,
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?)
                """,
                (job_id, synthetic_observation, now, now, now),
            )
        prepared = self.prepare_revision(
            identity=profile["identity"],
            observation_id=synthetic_observation,
            baseline=source["baseline"],
            trust=source["trust"],
            maturity=source["maturity"],
            lessons=lessons,
            builder_model="operator-rollback",
            builder_prompt_version=source["builder_prompt_version"],
            extractor_version=source["extractor_version"],
        )
        return prepared

    def retry_failed_jobs(self) -> int:
        now = time.time()
        with self.transaction() as connection:
            count = connection.execute(
                """
                UPDATE builder_jobs SET state='queued', attempts=0, available_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, last_error=NULL, updated_at=?
                WHERE state='failed'
                """,
                (now, now),
            ).rowcount
        return int(count)

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            counts = {}
            for table in ("profiles", "observations", "profile_revisions", "lessons", "conflicts"):
                counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            jobs = {
                row["state"]: int(row["value"])
                for row in connection.execute("SELECT state, COUNT(*) AS value FROM builder_jobs GROUP BY state")
            }
            outbox = {
                row["state"]: int(row["value"])
                for row in connection.execute("SELECT state, COUNT(*) AS value FROM qdrant_outbox GROUP BY state")
            }
        return {"sqlite_path": str(self.path), "counts": counts, "builder_jobs": jobs, "qdrant_outbox": outbox}
