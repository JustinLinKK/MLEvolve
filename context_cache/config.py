"""Validated configuration for the context-cache subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping


def _bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass
class ContextCacheSettings:
    enabled: bool = False
    local_pack_cache_enabled: bool = True
    provider_prompt_cache_enabled: bool = False
    provider_prompt_cache_roles: list[str] = field(default_factory=list)
    provider_prompt_cache_models: list[str] = field(default_factory=list)
    cache_dir: str = "var/context-cache"
    policy: str = "auto"
    ttl: str | None = None
    prewarm: bool = False
    telemetry: bool = True
    verify_prefix: bool = False
    capture_prompts: bool = False
    shadow: bool = False
    knowledge_version: str = "k1"
    openrouter_sticky_routing: bool = True
    openrouter_routing_shards: int = 1
    openrouter_upstream: str | None = None
    openrouter_allow_fallbacks: bool = True
    max_pack_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.policy not in {"auto", "explicit", "none"}:
            raise ValueError(
                "context cache policy must be one of: auto, explicit, none"
            )
        if self.openrouter_routing_shards < 1 or self.openrouter_routing_shards > 64:
            raise ValueError("openrouter routing shards must be between 1 and 64")
        if self.max_pack_bytes < 1024:
            raise ValueError("context cache max_pack_bytes must be at least 1024")
        if not str(self.knowledge_version or "").strip():
            raise ValueError("context cache knowledge_version must not be empty")
        if self.ttl is not None and not str(self.ttl).strip():
            self.ttl = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | Any | None
    ) -> "ContextCacheSettings":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            value = {
                name: getattr(value, name)
                for name in cls.__dataclass_fields__
                if hasattr(value, name)
            }
        payload = dict(value)
        for name in (
            "enabled",
            "local_pack_cache_enabled",
            "provider_prompt_cache_enabled",
            "prewarm",
            "telemetry",
            "verify_prefix",
            "capture_prompts",
            "shadow",
            "openrouter_sticky_routing",
            "openrouter_allow_fallbacks",
        ):
            if name in payload:
                payload[name] = _bool(payload[name], name=name)
        for name in ("openrouter_routing_shards", "max_pack_bytes"):
            if name in payload:
                payload[name] = int(payload[name])
        for name in ("provider_prompt_cache_roles", "provider_prompt_cache_models"):
            if name in payload:
                raw = payload[name]
                if isinstance(raw, str):
                    payload[name] = [
                        item.strip() for item in raw.split(",") if item.strip()
                    ]
                else:
                    payload[name] = [
                        str(item).strip() for item in (raw or []) if str(item).strip()
                    ]
        return cls(
            **{key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        )

    @property
    def directory(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()


_ENV_FIELDS: dict[str, tuple[str, Any]] = {
    "MLEVOLVE_CONTEXT_CACHE_ENABLED": ("enabled", bool),
    "MLEVOLVE_LOCAL_PACK_CACHE_ENABLED": ("local_pack_cache_enabled", bool),
    "MLEVOLVE_PROVIDER_PROMPT_CACHE_ENABLED": ("provider_prompt_cache_enabled", bool),
    "MLEVOLVE_PROVIDER_PROMPT_CACHE_ROLES": ("provider_prompt_cache_roles", list),
    "MLEVOLVE_PROVIDER_PROMPT_CACHE_MODELS": ("provider_prompt_cache_models", list),
    "MLEVOLVE_CONTEXT_CACHE_DIR": ("cache_dir", str),
    "MLEVOLVE_CONTEXT_CACHE_POLICY": ("policy", str),
    "MLEVOLVE_CONTEXT_CACHE_TTL": ("ttl", str),
    "MLEVOLVE_CONTEXT_CACHE_PREWARM": ("prewarm", bool),
    "MLEVOLVE_CONTEXT_CACHE_TELEMETRY": ("telemetry", bool),
    "MLEVOLVE_CONTEXT_CACHE_VERIFY_PREFIX": ("verify_prefix", bool),
    "MLEVOLVE_CONTEXT_CACHE_CAPTURE_PROMPTS": ("capture_prompts", bool),
    "MLEVOLVE_CONTEXT_CACHE_SHADOW": ("shadow", bool),
    "MLEVOLVE_CONTEXT_CACHE_KNOWLEDGE_VERSION": ("knowledge_version", str),
    "MLEVOLVE_OPENROUTER_STICKY_ROUTING": ("openrouter_sticky_routing", bool),
    "MLEVOLVE_OPENROUTER_ROUTING_SHARDS": ("openrouter_routing_shards", int),
    "MLEVOLVE_OPENROUTER_UPSTREAM": ("openrouter_upstream", str),
    "MLEVOLVE_OPENROUTER_ALLOW_FALLBACKS": ("openrouter_allow_fallbacks", bool),
}


def environment_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = os.environ if environ is None else environ
    result: dict[str, Any] = {}
    for env_name, (field_name, field_type) in _ENV_FIELDS.items():
        if env_name not in source:
            continue
        raw = source[env_name]
        if field_type is bool:
            result[field_name] = _bool(raw, name=env_name)
        elif field_type is int:
            result[field_name] = int(raw)
        elif field_type is list:
            result[field_name] = [
                item.strip() for item in raw.split(",") if item.strip()
            ]
        else:
            result[field_name] = raw or None if field_name == "ttl" else raw
    return result
