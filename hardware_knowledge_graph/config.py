"""Standalone hardware-knowledge configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import sys


@dataclass(slots=True)
class HardwareKnowledgeRedisCacheSettings:
    enabled: bool = False
    url: str = "redis://127.0.0.1:6379/0"
    url_env: str = "LOCALML_HARDWARE_KNOWLEDGE_REDIS_URL"
    key_prefix: str = "localml_hardware_knowledge"
    ttl_seconds: int | None = 300
    max_entries: int | None = 4096
    socket_timeout_seconds: float = 0.2
    cache_graph_queries: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.url = str(self.url or "redis://127.0.0.1:6379/0").strip()
        self.url_env = str(
            self.url_env or "LOCALML_HARDWARE_KNOWLEDGE_REDIS_URL"
        ).strip()
        self.key_prefix = (
            str(self.key_prefix or "localml_hardware_knowledge").strip().strip(":")
            or "localml_hardware_knowledge"
        )
        if self.ttl_seconds is not None:
            self.ttl_seconds = max(1, int(self.ttl_seconds))
        if self.max_entries is not None:
            self.max_entries = max(0, int(self.max_entries))
        self.socket_timeout_seconds = max(0.0, float(self.socket_timeout_seconds))
        self.cache_graph_queries = bool(self.cache_graph_queries)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None
    ) -> "HardwareKnowledgeRedisCacheSettings":
        return cls(**(payload or {}))

    def resolved_url(self) -> str:
        if self.url_env:
            env_value = os.getenv(self.url_env)
            if env_value:
                return env_value
        return self.url

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url": self.url,
            "url_env": self.url_env,
            "key_prefix": self.key_prefix,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "socket_timeout_seconds": self.socket_timeout_seconds,
            "cache_graph_queries": self.cache_graph_queries,
        }


@dataclass(slots=True)
class HardwareKnowledgeGraphSettings:
    enabled: bool = True
    provider: str = "neo4j"
    uri: str = "bolt://127.0.0.1:7688"
    username: str = "neo4j"
    password_env: str = "HARDWARE_KNOWLEDGE_NEO4J_PASSWORD"
    database: str = "neo4j"

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.provider = str(self.provider or "neo4j").strip().lower().replace("-", "_")
        self.uri = str(self.uri or "").strip()
        self.username = str(self.username or "").strip()
        self.password_env = str(self.password_env or "").strip()
        self.database = str(self.database or "").strip()

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None
    ) -> "HardwareKnowledgeGraphSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "uri": self.uri,
            "username": self.username,
            "password_env": self.password_env,
            "database": self.database,
        }


@dataclass(slots=True)
class _HardwareGpuMemorySettings:
    predicted_budget_fraction: float = 0.95

    def budget_mb(self, total_vram_mb: int | float | None) -> float:
        try:
            total = float(total_vram_mb) if total_vram_mb is not None else 0.0
        except (TypeError, ValueError):
            total = 0.0
        return max(0.0, total * float(self.predicted_budget_fraction))


@dataclass(slots=True)
class _HardwareGpuSettings:
    device_index: int = 0
    mode: str = "parallel_time_aware"
    parallel_job_cap: int | None = None
    packing_backend: str = "mps_process"
    exclusive_fallback_enabled: bool = True
    memory: _HardwareGpuMemorySettings = field(
        default_factory=_HardwareGpuMemorySettings
    )


@dataclass(slots=True)
class HardwareKnowledgeSettings:
    runtime_root: Path = Path("hardware_knowledge_graph/runtime")
    graph: HardwareKnowledgeGraphSettings | dict[str, Any] = field(
        default_factory=HardwareKnowledgeGraphSettings
    )
    redis_cache: HardwareKnowledgeRedisCacheSettings | dict[str, Any] = field(
        default_factory=HardwareKnowledgeRedisCacheSettings
    )
    device_index: int = 0
    sqlite_busy_timeout_ms: int = 10_000
    python_executable: str = field(default_factory=lambda: sys.executable)

    db_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    gpu_scheduler: _HardwareGpuSettings = field(init=False)

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = HardwareKnowledgeGraphSettings()
        if isinstance(self.graph, dict):
            self.graph = HardwareKnowledgeGraphSettings.from_dict(self.graph)
        if self.redis_cache is None:
            self.redis_cache = HardwareKnowledgeRedisCacheSettings()
        if isinstance(self.redis_cache, dict):
            self.redis_cache = HardwareKnowledgeRedisCacheSettings.from_dict(
                self.redis_cache
            )
        self.runtime_root = Path(self.runtime_root).resolve()
        self.db_dir = self.runtime_root / "db"
        self.db_path = self.db_dir / "hardware_knowledge.sqlite3"
        self.gpu_scheduler = _HardwareGpuSettings(
            device_index=int(self.device_index or 0)
        )

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None = None, **overrides: Any
    ) -> "HardwareKnowledgeSettings":
        data = dict(payload or {})
        data.update(overrides)
        return cls(**data)

    def ensure_runtime_layout(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_root": str(self.runtime_root),
            "graph": self.graph.to_dict(),
            "redis_cache": self.redis_cache.to_dict(),
            "device_index": self.device_index,
            "sqlite_busy_timeout_ms": self.sqlite_busy_timeout_ms,
            "python_executable": self.python_executable,
        }
