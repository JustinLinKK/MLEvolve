"""Deterministic knowledge-pack compiler and manifest loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from .canonicalize import canonical_sha256, canonicalize
from .models import KnowledgePackRef, PackBuild, PackLoadResult
from .store import KnowledgePackStore

SourceResolver = Callable[[Mapping[str, Any]], Any]
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "manifests"


@dataclass(frozen=True)
class PackManifest:
    role: str
    schema_version: str
    knowledge_version: str
    content: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PackManifest":
        role = str(payload.get("role") or "").strip()
        if not role:
            raise ValueError("knowledge-pack manifest requires role")
        content = payload.get("content", {"sections": []})
        if not isinstance(content, Mapping):
            raise ValueError("knowledge-pack manifest content must be a mapping")
        sources = payload.get("sources") or ()
        if not isinstance(sources, (list, tuple)):
            raise ValueError("knowledge-pack manifest sources must be a list")
        return cls(
            role=role,
            schema_version=str(payload.get("schema_version") or "1"),
            knowledge_version=str(payload.get("knowledge_version") or "k1"),
            content=content,
            sources=tuple(dict(item) for item in sources),
        )

    @classmethod
    def load(
        cls, path: str | Path, *, knowledge_version: str | None = None
    ) -> "PackManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if knowledge_version is not None:
            payload["knowledge_version"] = knowledge_version
        return cls.from_mapping(payload)


class KnowledgePackCompiler:
    def __init__(
        self,
        store: KnowledgePackStore,
        *,
        source_resolvers: Mapping[str, SourceResolver] | None = None,
    ) -> None:
        self.store = store
        self.source_resolvers = dict(source_resolvers or {})

    def _build(self, manifest: PackManifest) -> tuple[PackBuild, float]:
        content = dict(manifest.content)
        sections = list(content.get("sections") or [])
        retrieval_started = time.monotonic()
        source_diagnostics: list[dict[str, Any]] = []
        for source in manifest.sources:
            descriptor = dict(source)
            kind = str(descriptor.get("kind") or "manifest")
            resolver = self.source_resolvers.get(kind)
            if resolver is None:
                if descriptor.get("required"):
                    raise ValueError(
                        f"no source resolver registered for required source kind {kind!r}"
                    )
                source_diagnostics.append(descriptor)
                continue
            result = resolver(descriptor)
            section = {
                "stable_id": descriptor.get("stable_id")
                or f"{kind}:{descriptor.get('snapshot', 'unknown')}",
                "kind": kind,
                "content": (
                    {"records": list(result)}
                    if isinstance(result, (list, tuple))
                    else result
                ),
            }
            if "order" in descriptor:
                section["order"] = descriptor["order"]
            sections.append(section)
            source_diagnostics.append(
                {
                    key: value
                    for key, value in descriptor.items()
                    if key != "credentials"
                }
            )
        retrieval_ms = (time.monotonic() - retrieval_started) * 1000
        content["sections"] = sections
        compiled_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return (
            PackBuild(canonicalize(content), tuple(source_diagnostics), compiled_at),
            retrieval_ms,
        )

    def compile(self, manifest: PackManifest) -> PackLoadResult:
        retrieval_ms: float | None = None

        def builder() -> PackBuild:
            nonlocal retrieval_ms
            build, retrieval_ms = self._build(manifest)
            return build

        result = self.store.get_or_compile(
            role=manifest.role,
            schema_version=manifest.schema_version,
            knowledge_version=manifest.knowledge_version,
            builder=builder,
        )
        if result.cache_hit:
            return result
        return PackLoadResult(
            ref=result.ref,
            envelope=result.envelope,
            cache_hit=False,
            elapsed_ms=result.elapsed_ms,
            build_ms=result.build_ms,
            retrieval_ms=retrieval_ms,
        )


def default_manifest_path(role: str) -> Path:
    role_name = str(role).strip().lower().replace("-", "_")
    path = DEFAULT_MANIFEST_DIR / f"{role_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no built-in context-cache manifest for role {role_name!r}"
        )
    return path


def load_default_manifest(role: str, knowledge_version: str) -> PackManifest:
    return PackManifest.load(
        default_manifest_path(role), knowledge_version=knowledge_version
    )


def transient_ref(manifest: PackManifest) -> tuple[KnowledgePackRef, dict[str, Any]]:
    """Compile semantic content without persistence for provider-only trials."""

    content = canonicalize(manifest.content)
    digest = canonical_sha256(content)
    envelope = {
        "schema_version": manifest.schema_version,
        "knowledge_version": manifest.knowledge_version,
        "role": manifest.role,
        "content_sha256": digest,
        "compiled_at": None,
        "sources": canonicalize(list(manifest.sources), parent_key="sources"),
        "content": content,
    }
    return (
        KnowledgePackRef(
            manifest.role,
            manifest.schema_version,
            manifest.knowledge_version,
            digest,
            "",
        ),
        envelope,
    )
