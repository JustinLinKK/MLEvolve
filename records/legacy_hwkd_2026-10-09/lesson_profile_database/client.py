"""Public cache-aside API for lesson profile writes and role-scoped reads."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from typing import Any, Mapping

from localml_scheduler.hardware import detect_hardware_profile
from localml_scheduler.redis_cache import RedisLRUCache

from .builder import LessonBuilder, SummaryGenerator
from .config import LessonProfileSettings, lesson_profile_settings_from_config
from .evidence import bounded_parent_diff, build_evidence_packet
from .identity import build_profile_identity
from .models import ProfileIdentity, empty_profile_view
from .registry import LessonProfileRegistry
from .vector_store import LessonVectorStore
from .worker import LessonBuilderWorker


LOGGER = logging.getLogger("MLEvolve")
_ROLE_ALIASES = {"code_review": "review", "fusion_draft": "aggregation"}
_RETRIEVAL_POLICY_VERSION = "lesson-retrieval-v1"
_COMPLETE_IDENTITY_FIELDS = {
    "schema_version",
    "profile_key",
    "model_family",
    "architecture_type",
    "hardware_key",
    "accelerator_key",
    "resource_slice_key",
    "runtime_class",
    "framework_major",
    "cuda_major",
    "backend_class",
    "workload_bucket",
}
_ROLE_LESSON_TYPES = {
    "draft": {"family_baseline"},
    "improve": {"modification"},
    "debug": {"verified_fix", "failure_warning"},
    "evolution": {"branch_trajectory"},
    "fusion": {"transfer"},
    "aggregation": {"cross_branch_consensus"},
    "review": {"implementation_contract"},
}


def normalize_agent_role(role: str) -> str:
    normalized = str(role or "draft").strip().lower().replace("-", "_")
    return _ROLE_ALIASES.get(normalized, normalized)


def _signature(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _strip_advisory_unsafe(value: Any, *, parent_key: str = "") -> Any:
    """Similar-only records cannot advertise safe numbers or reusable code."""
    forbidden = {
        "implementation_example",
        "safe_training_defaults",
        "physical_batch_size",
        "batch_size",
        "peak_vram_mb",
        "observed_runtime_seconds",
        "runtime_seconds",
        "throughput",
        "code",
    }
    if isinstance(value, Mapping):
        return {
            key: _strip_advisory_unsafe(item, parent_key=str(key))
            for key, item in value.items()
            if key not in forbidden
        }
    if isinstance(value, list):
        return [_strip_advisory_unsafe(item, parent_key=parent_key) for item in value]
    if isinstance(value, str) and parent_key in {"lesson", "model_summary", "summary", "advice", "recommended_start"}:
        return re.sub(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", "[numeric value omitted]", value)
    return value


class LessonProfileClient:
    def __init__(
        self,
        settings: LessonProfileSettings,
        *,
        cfg: Any | None = None,
        registry: LessonProfileRegistry | None = None,
        vector_store: LessonVectorStore | None = None,
        redis_client: Any | None = None,
        qdrant_client: Any | None = None,
        qdrant_models: Any | None = None,
        embedding_model: Any | None = None,
        summary_generator: SummaryGenerator | None = None,
    ):
        self.settings = settings
        self.cfg = cfg
        self.registry = registry or LessonProfileRegistry(settings)
        self.vector_store = vector_store or LessonVectorStore(
            settings,
            qdrant_client=qdrant_client,
            qdrant_models=qdrant_models,
            embedding_model=embedding_model,
        )
        self.cache = RedisLRUCache(settings.redis_cache, redis_client=redis_client)
        self.builder = LessonBuilder(
            settings,
            self.registry,
            cfg=cfg,
            summary_generator=summary_generator,
        )
        self.worker = LessonBuilderWorker(
            settings,
            self.registry,
            self.builder,
            self.vector_store,
            invalidator=self.invalidate_profile,
        )

    @classmethod
    def from_config(cls, cfg: Any, **kwargs: Any) -> "LessonProfileClient":
        return cls(lesson_profile_settings_from_config(cfg), cfg=cfg, **kwargs)

    def initialize(self, *, initialize_qdrant: bool = True) -> dict[str, Any]:
        sqlite_result = self.registry.initialize()
        qdrant_result: dict[str, Any]
        if initialize_qdrant and self.settings.qdrant.enabled:
            try:
                qdrant_result = self.vector_store.ensure_collection()
            except Exception as exc:
                qdrant_result = {"ok": False, "error": str(exc)}
        else:
            qdrant_result = {"ok": False, "skipped": True}
        return {"ok": True, "sqlite": sqlite_result, "qdrant": qdrant_result}

    def start_worker(self) -> None:
        if self.settings.enabled and self.settings.write_enabled:
            self.worker.start()

    def stop_worker(self) -> None:
        self.worker.stop()

    def invalidate_profile(self, profile_key: str) -> None:
        self.cache.invalidate_namespace(f"profile:{profile_key}")
        self.cache.invalidate_namespace("search")

    @staticmethod
    def _agent_hardware(agent: Any) -> dict[str, Any]:
        scheduler = getattr(agent, "scheduler_client", None)
        try:
            if scheduler is not None and getattr(scheduler, "store", None) is not None:
                return dict(scheduler.store.hardware_profile().to_dict())
        except Exception:
            pass
        return detect_hardware_profile().to_dict()

    @staticmethod
    def _job_packet(agent: Any, node: Any) -> dict[str, Any]:
        pipeline_logger = getattr(agent, "pipeline_logger", None)
        if pipeline_logger is not None and hasattr(pipeline_logger, "latest_job_packet"):
            try:
                return dict(pipeline_logger.latest_job_packet(str(node.id)) or {})
            except Exception:
                return {}
        return {}

    def identity_for_node(self, agent: Any, node: Any, *, code: str | None = None) -> ProfileIdentity | None:
        packet = self._job_packet(agent, node)
        packet_payload = dict(packet.get("payload") or {})
        scheduler_enabled = bool(getattr(getattr(agent, "cfg", None), "scheduler", None) and getattr(agent.cfg.scheduler, "enabled", False))
        backend = packet.get("placement_backend") or getattr(node, "backend_name", None)
        if not backend and not scheduler_enabled:
            backend = "local_process"
        return build_profile_identity(
            code=code if code is not None else str(getattr(node, "code", "") or ""),
            hardware=self._agent_hardware(agent),
            backend=backend,
            task_description=str(getattr(agent, "task_desc", "") or ""),
            model_family_hint=getattr(node, "model_family", None),
            resource_slice=(
                packet.get("resource_slice_key")
                or packet.get("resource_slice")
                or packet_payload.get("resource_slice_key")
                or packet_payload.get("resource_slice")
                or packet_payload.get("mig_profile")
            ),
            minimum_confidence=self.settings.minimum_family_confidence,
        )

    def enqueue_validated_node(self, agent: Any, node: Any, *, outcome: str) -> dict[str, Any]:
        """Freeze and enqueue one final-validation result; never invokes an LLM."""
        if not (self.settings.enabled and self.settings.write_enabled):
            return {"ok": False, "reason": "lesson profile writes disabled"}
        if outcome == "valid":
            metric_value = getattr(getattr(node, "metric", None), "value", None)
            try:
                metric_is_real = metric_value is not None and math.isfinite(float(metric_value))
            except (TypeError, ValueError):
                metric_is_real = False
            if bool(getattr(node, "is_buggy", False)) or not metric_is_real:
                return {"ok": False, "reason": "validated successes require a real metric and non-buggy node"}
        identity = self.identity_for_node(agent, node)
        if identity is None:
            return {"ok": False, "reason": "uncertain or incomplete profile identity; cold start"}
        packet = self._job_packet(agent, node)
        if getattr(node, "exec_time", None) is not None and "runtime_seconds" not in packet:
            packet["runtime_seconds"] = getattr(node, "exec_time")
        for attr in ("resolved_batch_size", "peak_vram_mb", "estimated_runtime_seconds"):
            value = getattr(node, attr, None)
            if value is not None:
                key = "runtime_seconds" if attr == "estimated_runtime_seconds" else attr
                packet.setdefault(key, value)
        pipeline_logger = getattr(agent, "pipeline_logger", None)
        prompt_ref = None
        if pipeline_logger is not None and hasattr(pipeline_logger, "latest_prompt_reference"):
            try:
                prompt_ref = pipeline_logger.latest_prompt_reference(str(node.id))
            except Exception:
                prompt_ref = None
        if prompt_ref and not getattr(node, "prompt_snapshot_path", None):
            setattr(node, "prompt_snapshot_path", prompt_ref.get("prompt_path"))
        if prompt_ref and not getattr(node, "prompt_snapshot_sha256", None):
            setattr(node, "prompt_snapshot_sha256", prompt_ref.get("prompt_sha256"))
        run_id = str(getattr(getattr(agent, "cfg", None), "exp_name", "") or "unknown-run")
        evidence = build_evidence_packet(
            node=node,
            identity=identity.to_dict(),
            outcome=str(outcome),
            run_id=run_id,
            task_description=str(getattr(agent, "task_desc", "") or ""),
            scheduler_measurements=packet,
            evidence_refs=[
                f"job:{packet['job_id']}" for _ in [0] if packet.get("job_id")
            ],
        )
        self.registry.initialize()
        return self.registry.enqueue_observation(
            identity=identity,
            evidence=evidence,
            outcome=str(outcome),
            run_id=run_id,
            node_id=str(node.id),
            extractor_version=self.settings.builder.extractor_version,
        )

    @staticmethod
    def _query_text(role: str, *, code: str, parent_code: str, error: str) -> tuple[str, str]:
        delta = bounded_parent_diff(parent_code, code)
        query = "\n".join([
            f"agent role: {role}",
            f"change scope: {delta.get('change_scope')}",
            str(delta.get("unified_diff") or "")[-4000:],
            str(error or "")[-1500:],
        ])
        return query, _signature(delta, str(error or "")[-1500:])

    @staticmethod
    def _lesson_public(item: Mapping[str, Any]) -> dict[str, Any]:
        content = dict(item.get("content") or {})
        return {
            "lesson_id": str(item.get("lesson_id") or item.get("record_id") or ""),
            "lesson_type": str(item.get("lesson_type") or ""),
            **content,
            "warnings": list(item.get("warnings") or []),
            "confidence": float(item.get("confidence") or 0.0),
            "evidence_refs": list(item.get("evidence_refs") or []),
        }

    def _compose(
        self,
        *,
        profile: Mapping[str, Any],
        revision: Mapping[str, Any],
        match_level: str,
        role: str,
        lessons: list[Mapping[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        trust = dict(revision.get("trust") or {})
        warnings = list((revision.get("baseline") or {}).get("warnings") or [])
        maturity = str(revision.get("maturity") or profile.get("maturity") or "provisional")
        if maturity == "provisional":
            warnings.insert(0, "Provisional profile: fewer than the configured distinct-run stability threshold support it.")
        elif maturity == "conflicted":
            warnings.insert(0, "Conflicting controlled observations lower confidence; re-measure disputed assumptions.")
        if match_level == "compatible":
            warnings.insert(0, "Advisory compatible match: the strict runtime/hardware key differs; revalidate defaults.")
        elif match_level == "similar":
            warnings.insert(0, "Inspiration-only semantic match: numeric defaults and reusable code were omitted.")
        relevant = [self._lesson_public(item) for item in lessons[: self.settings.max_lessons]]
        baseline: Any = copy.deepcopy(dict(revision.get("baseline") or {}))
        if match_level == "similar":
            baseline = _strip_advisory_unsafe(baseline)
            relevant = _strip_advisory_unsafe(relevant)
        result = {
            "family_hardware_profile": {
                "profile_key": str(profile.get("profile_key") or revision.get("profile_key") or ""),
                "revision": int(revision.get("revision_number") or 0),
                "match_level": match_level,
                "maturity": maturity,
                "baseline": baseline,
                "relevant_lessons": relevant,
                "warnings": list(dict.fromkeys(warnings)),
                "evidence_refs": list(trust.get("evidence_refs") or []),
                "confidence": float(trust.get("confidence") or 0.0),
                "source": source,
            }
        }
        return self._fit_budget(result)

    def _fit_budget(self, result: dict[str, Any]) -> dict[str, Any]:
        view = result["family_hardware_profile"]
        while len(json.dumps(result, default=str)) > self.settings.max_prompt_chars and view["relevant_lessons"]:
            view["relevant_lessons"].pop()
        while len(json.dumps(result, default=str)) > self.settings.max_prompt_chars and view["warnings"]:
            view["warnings"].pop()
        if len(json.dumps(result, default=str)) > self.settings.max_prompt_chars:
            baseline = view.get("baseline") or {}
            view["baseline"] = {
                "model_summary": str(baseline.get("model_summary") or "")[:500],
                "safe_training_defaults": baseline.get("safe_training_defaults") or {},
            }
        if len(json.dumps(result, default=str)) > self.settings.max_prompt_chars:
            view["evidence_refs"] = view["evidence_refs"][:5]
        if len(json.dumps(result, default=str)) > self.settings.max_prompt_chars:
            view["baseline"] = {}
            view["evidence_refs"] = []
        while len(json.dumps(result, default=str)) > self.settings.max_prompt_chars and view["warnings"]:
            view["warnings"].pop()
        return result

    def _rank_exact_or_compatible(
        self,
        *,
        profile: Mapping[str, Any],
        revision: Mapping[str, Any],
        role: str,
        query: str,
    ) -> tuple[list[dict[str, Any]], str]:
        sqlite_lessons = self.registry.lessons_for_revision(
            str(profile["profile_key"]), int(revision["revision_number"]), role=role
        )
        expected_types = _ROLE_LESSON_TYPES.get(role)
        if expected_types:
            sqlite_lessons = [item for item in sqlite_lessons if item.get("lesson_type") in expected_types]
        if role == "improve":
            sqlite_lessons = [item for item in sqlite_lessons if item.get("change_scope") != "multi_change"]
        query_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query.lower()))
        sqlite_lessons.sort(
            key=lambda item: (
                len(query_tokens & set(re.findall(
                    r"[a-zA-Z_][a-zA-Z0-9_]{2,}",
                    json.dumps({
                        "content": item.get("content"),
                        "warnings": item.get("warnings"),
                        "change_signature": item.get("change_signature"),
                    }, sort_keys=True, default=str).lower(),
                ))),
                float(item.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        if not sqlite_lessons:
            return [], "sqlite_fallback"
        try:
            vector_lessons = self.vector_store.search(
                query=query,
                filters={
                    "profile_key": str(profile["profile_key"]),
                    "revision": int(revision["revision_number"]),
                    "record_kind": "lesson",
                    "agent_audiences": role,
                    "active": True,
                },
                limit=max(self.settings.max_lessons * 2, 3),
            )
            if vector_lessons:
                by_id = {str(item["lesson_id"]): item for item in sqlite_lessons}
                ranked = [by_id[str(item.get("record_id"))] for item in vector_lessons if str(item.get("record_id")) in by_id]
                ranked.extend(item for item in sqlite_lessons if item not in ranked)
                return ranked, "sqlite_qdrant"
            return sqlite_lessons, "sqlite_fallback"
        except Exception as exc:
            LOGGER.debug("Lesson Qdrant search unavailable; using SQLite: %s", exc)
            return sqlite_lessons, "sqlite_fallback"

    def _similar_match(
        self, *, role: str, query: str
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str] | None:
        filters: dict[str, Any] = {"record_kind": "lesson", "agent_audiences": role, "active": True}
        expected_types = _ROLE_LESSON_TYPES.get(role)
        if expected_types:
            filters["lesson_type"] = sorted(expected_types)
        source = "sqlite_qdrant"
        try:
            records = self.vector_store.search(
                query=query,
                filters=filters,
                limit=max(12, self.settings.max_lessons * 4),
            )
        except Exception as exc:
            LOGGER.debug("Lesson similar-only search unavailable: %s", exc)
            records = []
        if not records:
            records = self.registry.search_active_lessons(
                role=role,
                query=query,
                limit=max(12, self.settings.max_lessons * 4),
            )
            source = "sqlite_fallback"
        for candidate in records:
            profile = self.registry.profile(str(candidate.get("profile_key") or ""))
            if profile is None:
                continue
            revision_number = int(candidate.get("revision") or 0)
            if int(profile.get("active_revision") or 0) != revision_number:
                continue
            revision = self.registry.active_revision(str(profile["profile_key"]))
            if revision is None:
                continue
            if role == "aggregation" and str(revision.get("maturity") or "") != "stable":
                continue
            sqlite_lessons = self.registry.lessons_for_revision(
                str(profile["profile_key"]), revision_number, role=role
            )
            by_id = {str(item["lesson_id"]): item for item in sqlite_lessons}
            ranked = [
                by_id[str(item.get("record_id"))]
                for item in records
                if str(item.get("profile_key") or "") == str(profile["profile_key"])
                and int(item.get("revision") or 0) == revision_number
                and str(item.get("record_id") or "") in by_id
            ]
            if role == "improve":
                ranked = [item for item in ranked if item.get("change_scope") != "multi_change"]
            return profile, revision, ranked, source
        return None

    def get_family_hardware_profile(
        self,
        *,
        agent_role: str,
        identity: ProfileIdentity | Mapping[str, Any] | None = None,
        code: str = "",
        parent_code: str = "",
        error: str = "",
        hardware: Mapping[str, Any] | None = None,
        backend: str | None = None,
        task_description: str = "",
        model_family_hint: str | None = None,
    ) -> dict[str, Any]:
        if not (self.settings.enabled and self.settings.read_enabled):
            return empty_profile_view()
        role = normalize_agent_role(agent_role)
        if identity is None and hardware is not None:
            identity = build_profile_identity(
                code=code,
                hardware=hardware,
                backend=backend,
                task_description=task_description,
                model_family_hint=model_family_hint,
                minimum_confidence=self.settings.minimum_family_confidence,
            )
        if identity is None:
            return empty_profile_view()
        identity_dict = identity.to_dict() if isinstance(identity, ProfileIdentity) else dict(identity)
        if any(not str(identity_dict.get(field) or "").strip() for field in _COMPLETE_IDENTITY_FIELDS):
            return empty_profile_view()
        profile_key = str(identity_dict["profile_key"])
        query, delta_signature = self._query_text(role, code=code, parent_code=parent_code, error=error)
        cache_payload = {
            "role": role,
            "profile_scope": profile_key,
            "delta_error_signature": delta_signature,
            "retrieval_policy": _RETRIEVAL_POLICY_VERSION,
            "max_lessons": self.settings.max_lessons,
            "max_chars": self.settings.max_prompt_chars,
        }
        cached = self.cache.get(f"profile:{profile_key}", cache_payload)
        if not isinstance(cached, Mapping):
            cached = self.cache.get("search", cache_payload)
        if isinstance(cached, Mapping):
            result = copy.deepcopy(dict(cached))
            result["family_hardware_profile"]["source"] = "redis"
            return result

        profile = self.registry.profile(profile_key)
        match_level = "exact"
        if profile is None or profile.get("active_revision") is None:
            compatible = self.registry.find_compatible_profiles(identity_dict, limit=1)
            if compatible:
                profile = compatible[0]
                match_level = "compatible"
        if profile is None or profile.get("active_revision") is None:
            similar = self._similar_match(role=role, query=query)
            if similar is not None:
                similar_profile, similar_revision, similar_lessons, similar_source = similar
                result = self._compose(
                    profile=similar_profile,
                    revision=similar_revision,
                    match_level="similar",
                    role=role,
                    lessons=similar_lessons,
                    source=similar_source,
                )
            else:
                result = empty_profile_view()
            self.cache.set("search", cache_payload, result)
            return result
        revision = self.registry.active_revision(str(profile["profile_key"]))
        if revision is None:
            result = empty_profile_view()
            self.cache.set("search" if match_level != "exact" else f"profile:{profile_key}", cache_payload, result)
            return result
        if role == "aggregation" and str(revision.get("maturity") or "") != "stable":
            result = self._compose(
                profile=profile,
                revision={**revision, "baseline": {}},
                match_level=match_level,
                role=role,
                lessons=[],
                source="sqlite_fallback",
            )
            result["family_hardware_profile"]["warnings"].insert(
                0, "Aggregation lessons are withheld until the profile is stable across distinct runs."
            )
            self.cache.set("search" if match_level != "exact" else f"profile:{profile_key}", cache_payload, result)
            return result
        lessons, source = self._rank_exact_or_compatible(
            profile=profile,
            revision=revision,
            role=role,
            query=query,
        )
        result = self._compose(
            profile=profile,
            revision=revision,
            match_level=match_level,
            role=role,
            lessons=lessons,
            source=source,
        )
        self.cache.set("search" if match_level != "exact" else f"profile:{profile_key}", cache_payload, result)
        return result

    def profile_for_agent(
        self,
        agent: Any,
        *,
        agent_role: str,
        node: Any | None = None,
        code: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        anchor = node
        if anchor is None:
            return empty_profile_view()
        identity = self.identity_for_node(agent, anchor, code=code or getattr(anchor, "code", ""))
        parent = getattr(anchor, "parent", None)
        return self.get_family_hardware_profile(
            agent_role=agent_role,
            identity=identity,
            code=code or str(getattr(anchor, "code", "") or ""),
            parent_code=str(getattr(parent, "code", "") or ""),
            error=error,
        )

    def search_lesson_profiles(
        self,
        *,
        query: str,
        agent_role: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read-only inspiration search; every result is similar-only."""
        if not (self.settings.enabled and self.settings.read_enabled):
            return []
        role = normalize_agent_role(agent_role)
        cache_payload = {
            "role": role,
            "query_signature": _signature(query, filters or {}),
            "retrieval_policy": _RETRIEVAL_POLICY_VERSION,
            "limit": limit or self.settings.max_lessons,
        }
        cached = self.cache.get("search", cache_payload)
        if isinstance(cached, list):
            return copy.deepcopy(cached)
        merged_filters = {"record_kind": "lesson", "agent_audiences": role, "active": True, **dict(filters or {})}
        try:
            records = self.vector_store.search(
                query=query,
                filters=merged_filters,
                limit=max(1, int(limit or self.settings.max_lessons)),
            )
        except Exception as exc:
            LOGGER.debug("Lesson semantic search unavailable: %s", exc)
            records = []
        source = "sqlite_qdrant"
        if not records:
            records = self.registry.search_active_lessons(
                role=role,
                query=query,
                limit=max(1, int(limit or self.settings.max_lessons)),
            )
            source = "sqlite_fallback"
        results = []
        for record in records:
            profile = self.registry.profile(str(record.get("profile_key") or ""))
            if profile is None or int(profile.get("active_revision") or 0) != int(record.get("revision") or -1):
                continue
            results.append(_strip_advisory_unsafe({
                "profile_key": record.get("profile_key"),
                "revision": record.get("revision"),
                "match_level": "similar",
                "maturity": record.get("maturity"),
                "lesson": self._lesson_public(record),
                "warning": "Inspiration-only semantic match; revalidate all assumptions.",
                "source": source,
            }))
        self.cache.set("search", cache_payload, results)
        return results

    def status(self) -> dict[str, Any]:
        result = self.registry.status()
        result["worker_running"] = self.worker.running
        try:
            redis_available = self.cache.client is not None if self.settings.redis_cache.enabled else None
        except Exception:
            redis_available = False
        try:
            qdrant_available = self.vector_store._collection_exists() if self.settings.qdrant.enabled else None
        except Exception:
            qdrant_available = False
        result["services"] = {
            "redis_available": redis_available,
            "qdrant_collection_available": qdrant_available,
        }
        result["settings"] = {
            "read_enabled": self.settings.read_enabled,
            "write_enabled": self.settings.write_enabled,
            "qdrant_collection": self.settings.qdrant.collection_name,
            "redis_prefix": self.settings.redis_cache.key_prefix,
        }
        return result
