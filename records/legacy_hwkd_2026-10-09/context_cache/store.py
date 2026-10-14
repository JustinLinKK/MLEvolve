"""Multi-process-safe immutable knowledge-pack storage.

SQLite is the coordination/control plane, immutable JSON files are the durable
objects, and a bounded in-process dictionary is the RAM read-through layer.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from typing import Any, Iterator

from .canonicalize import (
    canonical_json_bytes,
    canonical_sha256,
    canonicalize,
    sensitive_paths,
    stable_json_bytes,
)
from .models import KnowledgePackRef, PackBuild, PackLoadResult


class PackStoreError(RuntimeError):
    pass


class CorruptPackError(PackStoreError):
    pass


class KnowledgePackStore:
    """Content-addressed files backed by a SQLite alias and run registry."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_pack_bytes: int = 16 * 1024 * 1024,
        ram_entries: int = 64,
    ):
        self.root = Path(root).expanduser().resolve()
        self.objects_dir = self.root / "objects"
        self.registry_path = self.root / "cache-registry.sqlite3"
        self.max_pack_bytes = int(max_pack_bytes)
        self.ram_entries = max(1, int(ram_entries))
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._memory_lock = threading.RLock()
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.registry_path, timeout=120.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=120000")
        # Concurrent first-open processes can race while SQLite changes the
        # journal mode. This pragma does not consistently honor busy_timeout.
        for attempt in range(100):
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 99:
                    connection.close()
                    raise
                time.sleep(0.01)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS pack_objects (
                    content_sha256 TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    knowledge_version TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    sources_json TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pack_aliases (
                    role TEXT NOT NULL,
                    knowledge_version TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL REFERENCES pack_objects(content_sha256),
                    active INTEGER NOT NULL DEFAULT 1,
                    published_at TEXT NOT NULL,
                    PRIMARY KEY (role, knowledge_version)
                );
                CREATE TABLE IF NOT EXISTS run_pack_refs (
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    knowledge_version TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL REFERENCES pack_objects(content_sha256),
                    frozen_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, role)
                );
                CREATE INDEX IF NOT EXISTS idx_pack_alias_hash ON pack_aliases(content_sha256);
                CREATE INDEX IF NOT EXISTS idx_run_pack_hash ON run_pack_refs(content_sha256);
                """)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def object_path(self, content_sha256: str) -> Path:
        if len(content_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in content_sha256
        ):
            raise ValueError("content hash must be a lowercase SHA-256 hex digest")
        return self.objects_dir / f"{content_sha256}.json"

    def _remember(self, content_sha256: str, envelope: dict[str, Any]) -> None:
        with self._memory_lock:
            self._memory[content_sha256] = envelope
            self._memory.move_to_end(content_sha256)
            while len(self._memory) > self.ram_entries:
                self._memory.popitem(last=False)

    def _cached(self, content_sha256: str) -> dict[str, Any] | None:
        with self._memory_lock:
            value = self._memory.get(content_sha256)
            if value is not None:
                self._memory.move_to_end(content_sha256)
            return value

    def _validate_envelope(
        self, envelope: Mapping[str, Any], expected_hash: str
    ) -> dict[str, Any]:
        if not isinstance(envelope.get("content"), Mapping):
            raise CorruptPackError("knowledge-pack object has no mapping content")
        actual = canonical_sha256(envelope["content"])
        declared = str(envelope.get("content_sha256") or "")
        if actual != expected_hash or declared != expected_hash:
            raise CorruptPackError(
                f"knowledge-pack hash mismatch: expected {expected_hash}, declared {declared}, actual {actual}"
            )
        return dict(envelope)

    def load_object(
        self, content_sha256: str, *, update_access: bool = True
    ) -> dict[str, Any]:
        cached = self._cached(content_sha256)
        if cached is not None:
            return cached
        path = self.object_path(content_sha256)
        try:
            size = path.stat().st_size
            if size > self.max_pack_bytes:
                raise CorruptPackError(
                    f"knowledge pack exceeds {self.max_pack_bytes} bytes"
                )
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptPackError(
                f"cannot read knowledge-pack object {path}: {exc}"
            ) from exc
        validated = self._validate_envelope(envelope, content_sha256)
        self._remember(content_sha256, validated)
        if update_access:
            try:
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE pack_objects SET last_accessed_at = ? WHERE content_sha256 = ?",
                        (self._now(), content_sha256),
                    )
            except sqlite3.Error:
                pass
        return validated

    def resolve(
        self, role: str, knowledge_version: str, *, active_only: bool = True
    ) -> KnowledgePackRef | None:
        query = (
            "SELECT a.role, o.schema_version, a.knowledge_version, o.content_sha256, o.path "
            "FROM pack_aliases a JOIN pack_objects o ON o.content_sha256 = a.content_sha256 "
            "WHERE a.role = ? AND a.knowledge_version = ?"
        )
        if active_only:
            query += " AND a.active = 1"
        with self._connect() as connection:
            row = connection.execute(query, (role, knowledge_version)).fetchone()
        return self._ref_from_row(row) if row else None

    @staticmethod
    def _ref_from_row(row: Mapping[str, Any]) -> KnowledgePackRef:
        return KnowledgePackRef(
            role=str(row["role"]),
            schema_version=str(row["schema_version"]),
            knowledge_version=str(row["knowledge_version"]),
            content_sha256=str(row["content_sha256"]),
            path=str(row["path"]),
        )

    def _write_object(
        self, envelope: Mapping[str, Any], content_sha256: str
    ) -> tuple[Path, int]:
        payload = stable_json_bytes(envelope)
        if len(payload) > self.max_pack_bytes:
            raise PackStoreError(f"knowledge pack exceeds {self.max_pack_bytes} bytes")
        destination = self.object_path(content_sha256)
        if destination.exists():
            try:
                self.load_object(content_sha256)
                return destination, destination.stat().st_size
            except CorruptPackError:
                pass
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{content_sha256}.", suffix=".tmp", dir=self.objects_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._remember(content_sha256, dict(envelope))
        return destination, len(payload)

    def _publish_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        role: str,
        schema_version: str,
        knowledge_version: str,
        build: PackBuild,
    ) -> KnowledgePackRef:
        semantic_content = canonicalize(build.content)
        secrets = sensitive_paths(semantic_content)
        if secrets:
            raise PackStoreError(
                "sensitive fields are forbidden in knowledge packs: "
                + ", ".join(secrets)
            )
        content_sha256 = canonical_sha256(semantic_content)
        compiled_at = build.compiled_at or self._now()
        sources = canonicalize(list(build.sources), parent_key="sources")
        envelope = {
            "schema_version": str(schema_version),
            "knowledge_version": str(knowledge_version),
            "role": str(role),
            "content_sha256": content_sha256,
            "compiled_at": compiled_at,
            "sources": sources,
            "content": semantic_content,
        }
        path, byte_size = self._write_object(envelope, content_sha256)
        now = self._now()
        connection.execute(
            """
            INSERT INTO pack_objects (
                content_sha256, role, schema_version, knowledge_version, path,
                sources_json, byte_size, created_at, last_accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_sha256) DO UPDATE SET last_accessed_at = excluded.last_accessed_at
            """,
            (
                content_sha256,
                role,
                schema_version,
                knowledge_version,
                str(path),
                canonical_json_bytes(sources).decode("utf-8"),
                byte_size,
                now,
                now,
            ),
        )
        existing = connection.execute(
            "SELECT content_sha256 FROM pack_aliases WHERE role = ? AND knowledge_version = ?",
            (role, knowledge_version),
        ).fetchone()
        if existing and existing["content_sha256"] != content_sha256:
            raise PackStoreError(
                f"knowledge version {knowledge_version!r} for role {role!r} is immutable; publish a new version"
            )
        connection.execute(
            """
            INSERT INTO pack_aliases(role, knowledge_version, content_sha256, active, published_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(role, knowledge_version) DO UPDATE SET active = 1
            """,
            (role, knowledge_version, content_sha256, now),
        )
        return KnowledgePackRef(
            role, schema_version, knowledge_version, content_sha256, str(path)
        )

    def get_or_compile(
        self,
        *,
        role: str,
        schema_version: str,
        knowledge_version: str,
        builder: Callable[[], PackBuild],
    ) -> PackLoadResult:
        """Resolve an alias or compile it once across all local subprocesses."""

        started = time.monotonic()
        ref = self.resolve(role, knowledge_version)
        if ref is not None:
            try:
                envelope = self.load_object(ref.content_sha256)
                return PackLoadResult(
                    ref, envelope, True, (time.monotonic() - started) * 1000
                )
            except CorruptPackError:
                pass

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT a.role, o.schema_version, a.knowledge_version, o.content_sha256, o.path
                FROM pack_aliases a JOIN pack_objects o ON o.content_sha256 = a.content_sha256
                WHERE a.role = ? AND a.knowledge_version = ? AND a.active = 1
                """,
                (role, knowledge_version),
            ).fetchone()
            if row:
                locked_ref = self._ref_from_row(row)
                try:
                    envelope = self.load_object(
                        locked_ref.content_sha256, update_access=False
                    )
                    connection.commit()
                    return PackLoadResult(
                        locked_ref, envelope, True, (time.monotonic() - started) * 1000
                    )
                except CorruptPackError:
                    connection.execute(
                        "DELETE FROM pack_aliases WHERE role = ? AND knowledge_version = ?",
                        (role, knowledge_version),
                    )

            build_started = time.monotonic()
            build = builder()
            build_ms = (time.monotonic() - build_started) * 1000
            ref = self._publish_with_connection(
                connection,
                role=role,
                schema_version=schema_version,
                knowledge_version=knowledge_version,
                build=build,
            )
            connection.commit()
        envelope = self.load_object(ref.content_sha256)
        return PackLoadResult(
            ref,
            envelope,
            False,
            (time.monotonic() - started) * 1000,
            build_ms=build_ms,
        )

    def freeze(self, run_id: str, ref: KnowledgePackRef) -> KnowledgePackRef:
        """Freeze one role/version reference for the lifetime of a run."""

        now = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT r.role, o.schema_version, r.knowledge_version, o.content_sha256, o.path
                FROM run_pack_refs r JOIN pack_objects o ON o.content_sha256 = r.content_sha256
                WHERE r.run_id = ? AND r.role = ?
                """,
                (run_id, ref.role),
            ).fetchone()
            if existing:
                connection.commit()
                return self._ref_from_row(existing)
            connection.execute(
                """INSERT INTO run_pack_refs(run_id, role, knowledge_version, content_sha256, frozen_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, ref.role, ref.knowledge_version, ref.content_sha256, now),
            )
            connection.commit()
        return ref

    def list(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        query = (
            "SELECT a.role, a.knowledge_version, a.active, a.published_at, o.content_sha256, "
            "o.schema_version, o.path, o.byte_size, o.last_accessed_at "
            "FROM pack_aliases a JOIN pack_objects o ON o.content_sha256 = a.content_sha256"
        )
        if not include_inactive:
            query += " WHERE a.active = 1"
        query += " ORDER BY a.role, a.knowledge_version"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query).fetchall()]

    def verify(self, *, knowledge_version: str | None = None) -> list[dict[str, Any]]:
        rows = self.list(include_inactive=True)
        if knowledge_version is not None:
            rows = [
                row for row in rows if row["knowledge_version"] == knowledge_version
            ]
        results = []
        for row in rows:
            try:
                self.load_object(row["content_sha256"])
                results.append({**row, "valid": True, "error": None})
            except PackStoreError as exc:
                results.append({**row, "valid": False, "error": str(exc)})
        return results

    def retire(self, role: str, knowledge_version: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE pack_aliases SET active = 0 WHERE role = ? AND knowledge_version = ?",
                (role, knowledge_version),
            )
        return bool(result.rowcount)

    def cleanup(self, *, dry_run: bool = True) -> list[str]:
        """Find or delete objects not referenced by aliases or frozen runs."""

        with self._connect() as connection:
            rows = connection.execute("""
                SELECT content_sha256, path FROM pack_objects o
                WHERE NOT EXISTS (SELECT 1 FROM pack_aliases a WHERE a.content_sha256 = o.content_sha256)
                  AND NOT EXISTS (SELECT 1 FROM run_pack_refs r WHERE r.content_sha256 = o.content_sha256)
                ORDER BY content_sha256
                """).fetchall()
            paths = [str(row["path"]) for row in rows]
            if not dry_run:
                for row in rows:
                    try:
                        Path(row["path"]).unlink(missing_ok=True)
                    except OSError as exc:
                        raise PackStoreError(
                            f"cannot remove unreferenced object {row['path']}: {exc}"
                        ) from exc
                    connection.execute(
                        "DELETE FROM pack_objects WHERE content_sha256 = ?",
                        (row["content_sha256"],),
                    )
                    with self._memory_lock:
                        self._memory.pop(row["content_sha256"], None)
        return paths
