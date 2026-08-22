"""Provider-neutral data contracts for MLEvolve context caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Literal, Mapping

CacheMode = Literal["auto", "explicit", "none"]


@dataclass(frozen=True)
class CachePolicy:
    mode: CacheMode = "auto"
    ttl: str | None = None
    scope: str = "role"
    prewarm: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "explicit", "none"}:
            raise ValueError("cache policy mode must be one of: auto, explicit, none")
        if not str(self.scope or "").strip():
            raise ValueError("cache policy scope must not be empty")


@dataclass(frozen=True)
class CacheCapabilities:
    explicit_breakpoints: bool
    ttl_values: tuple[str, ...] = ()
    supports_prewarm: bool = False
    metrics_mapping: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgePackRef:
    role: str
    schema_version: str
    knowledge_version: str
    content_sha256: str
    path: str


@dataclass(frozen=True)
class CacheFamily:
    provider: str
    model: str
    common_pack_hash: str
    role_pack_hash: str
    tool_schema_hash: str
    reasoning_config_hash: str
    api_family: str = "chat_completions"
    upstream_constraints_hash: str = ""
    system_instructions_hash: str = ""

    @property
    def id(self) -> str:
        from .canonicalize import canonical_json_bytes

        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class NormalizedCacheUsage:
    prompt_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_miss_tokens: int | None = None
    output_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class AssembledPrompt:
    """A provider-neutral prompt with an explicit stable/dynamic boundary."""

    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    stable_prefix: str
    dynamic_suffix: tuple[Mapping[str, Any], ...]
    stable_prefix_hash: str
    component_hashes: Mapping[str, str]
    tool_schema_hash: str
    reasoning_config_hash: str
    stable_message_index: int | None
    expected_stable_prefix_tokens: int | None


@dataclass(frozen=True)
class PackBuild:
    content: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...] = ()
    compiled_at: str | None = None


@dataclass(frozen=True)
class PackLoadResult:
    ref: KnowledgePackRef
    envelope: Mapping[str, Any]
    cache_hit: bool
    elapsed_ms: float
    build_ms: float | None = None
    retrieval_ms: float | None = None


@dataclass
class PreparedCacheRequest:
    """Result returned to the shared LLM backend."""

    params: dict[str, Any]
    active: bool = False
    family: CacheFamily | None = None
    assembled: AssembledPrompt | None = None
    adapter: Any = None
    telemetry: Any = None
    local_pack_cache_hit: bool | None = None
    fallback_reason: str | None = None
    request_gate: Any = None
