"""Configuration for the independent lesson-profile knowledge domain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from localml_scheduler.redis_cache import RedisCacheSettings


@dataclass
class QdrantSettings:
    enabled: bool = True
    url: str = "http://127.0.0.1:6333"
    api_key_env: str = "MLEVOLVE_LESSON_QDRANT_API_KEY"
    collection_name: str = "lesson_profile_records_v1"
    embedding_model_type: str = "local"
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dimension: int | None = 768
    distance: str = "Cosine"


@dataclass
class BuilderSettings:
    enabled: bool = True
    concurrency: int = 1
    poll_interval_seconds: float = 0.5
    lease_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    model: str | None = None
    prompt_version: str = "lesson-profile-builder-v1"
    extractor_version: str = "lesson-profile-extractor-v1"
    max_summary_chars: int = 1200


def _lesson_redis_defaults() -> RedisCacheSettings:
    return RedisCacheSettings(
        enabled=True,
        url="redis://127.0.0.1:6379/1",
        url_env="MLEVOLVE_LESSON_REDIS_URL",
        key_prefix="mlevolve_lesson_profiles",
        ttl_seconds=300,
        max_entries=4096,
        socket_timeout_seconds=0.2,
        cache_graph_queries=True,
        cache_vector_queries=True,
    )


@dataclass
class LessonProfileSettings:
    """Top-level ``lesson_profiles`` settings.

    SQLite remains authoritative regardless of whether either acceleration
    service is available. Qdrant and Redis are deliberately configured under
    this subsystem so records cannot be mixed with hardware knowledge.
    """

    enabled: bool = True
    read_enabled: bool = True
    write_enabled: bool = True
    enable_in_baseline_modes: bool = False
    runtime_root: str = "lesson_profile_database/runtime"
    sqlite_path: str | None = None
    stability_threshold: int = 3
    minimum_family_confidence: float = 0.75
    max_lessons: int = 3
    max_prompt_chars: int = 2500
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    redis_cache: RedisCacheSettings = field(default_factory=_lesson_redis_defaults)
    builder: BuilderSettings = field(default_factory=BuilderSettings)

    def __post_init__(self) -> None:
        self.stability_threshold = max(1, int(self.stability_threshold))
        self.minimum_family_confidence = max(0.0, min(1.0, float(self.minimum_family_confidence)))
        self.max_lessons = max(0, int(self.max_lessons))
        self.max_prompt_chars = max(512, int(self.max_prompt_chars))

    @property
    def database_path(self) -> Path:
        if self.sqlite_path:
            return Path(self.sqlite_path).expanduser().resolve()
        return (Path(self.runtime_root).expanduser().resolve() / "db" / "lesson_profiles.sqlite3")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | Any | None) -> "LessonProfileSettings":
        if isinstance(payload, cls):
            return payload
        if payload is None:
            return cls()
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(payload):
                payload = OmegaConf.to_container(payload, resolve=True)
        except Exception:
            pass
        values = dict(payload or {})
        qdrant = values.get("qdrant")
        builder = values.get("builder")
        redis_cache = values.get("redis_cache")
        if not isinstance(qdrant, QdrantSettings):
            values["qdrant"] = QdrantSettings(**dict(qdrant or {}))
        if not isinstance(builder, BuilderSettings):
            values["builder"] = BuilderSettings(**dict(builder or {}))
        if not isinstance(redis_cache, RedisCacheSettings):
            defaults = asdict(_lesson_redis_defaults())
            defaults.update(dict(redis_cache or {}))
            values["redis_cache"] = RedisCacheSettings.from_dict(defaults)
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sqlite_path"] = str(self.database_path)
        return result


def lesson_profile_settings_from_config(cfg: Any) -> LessonProfileSettings:
    payload = getattr(cfg, "lesson_profiles", None)
    if is_dataclass(payload):
        payload = asdict(payload)
    return LessonProfileSettings.from_mapping(payload)
