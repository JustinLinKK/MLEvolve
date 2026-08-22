"""Run-level pack freezing, request assembly, fail-open adapters, and singleflight."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .assembler import DeterministicPromptAssembler
from .canonicalize import canonical_sha256
from .compiler import KnowledgePackCompiler, load_default_manifest, transient_ref
from .config import ContextCacheSettings
from .models import CacheFamily, CachePolicy, PackLoadResult, PreparedCacheRequest
from .providers import adapter_for
from .store import KnowledgePackStore
from .telemetry import CacheTelemetryStore, RequestTelemetry

logger = logging.getLogger("MLEvolve")
ROLE_NAMES = ("model_generator", "analysis", "result_parser", "reviewer", "supervisor")


class _Flight:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class SingleFlight:
    """Deduplicate identical work in one process and release waiters on error."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[tuple[Any, ...], _Flight] = {}

    def run(
        self,
        key: tuple[Any, ...],
        function: Callable[[], Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        with self._lock:
            flight = self._flights.get(key)
            leader = flight is None
            if leader:
                flight = _Flight()
                self._flights[key] = flight
        assert flight is not None
        if leader:
            try:
                flight.result = function()
            except BaseException as exc:
                flight.error = exc
            finally:
                flight.event.set()
                with self._lock:
                    self._flights.pop(key, None)
        elif not flight.event.wait(timeout):
            raise TimeoutError(f"singleflight timed out for {key!r}")
        if flight.error is not None:
            raise flight.error
        return flight.result


class WarmFirstGate:
    """Let one real request complete before a same-family fanout proceeds."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[tuple[Any, ...], threading.Event] = {}
        self._warmed: set[tuple[Any, ...]] = set()

    @contextmanager
    def hold(self, key: tuple[Any, ...], *, timeout: float = 30.0):
        with self._lock:
            if key in self._warmed:
                event = None
                leader = False
            else:
                event = self._flights.get(key)
                leader = event is None
                if leader:
                    event = threading.Event()
                    self._flights[key] = event
        if event is None:
            yield
            return
        if not leader:
            event.wait(timeout)
            yield
            return
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            with self._lock:
                self._flights.pop(key, None)
                if succeeded:
                    self._warmed.add(key)
                event.set()


class ContextCacheCoordinator:
    def __init__(
        self, settings: ContextCacheSettings, *, run_id: str | None = None
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        self.store = KnowledgePackStore(
            settings.directory, max_pack_bytes=settings.max_pack_bytes
        )
        self.compiler = KnowledgePackCompiler(self.store)
        self.telemetry_store = CacheTelemetryStore(self.store.registry_path)
        self.assembler = DeterministicPromptAssembler(
            verify_prefix=settings.verify_prefix
        )
        self.singleflight = SingleFlight()
        self.warm_first = WarmFirstGate()

    def _pack(self, role: str) -> PackLoadResult:
        manifest = load_default_manifest(role, self.settings.knowledge_version)
        result = self.compiler.compile(manifest)
        if self.run_id:
            frozen = self.store.freeze(self.run_id, result.ref)
            if frozen.content_sha256 != result.ref.content_sha256:
                envelope = self.store.load_object(frozen.content_sha256)
                return PackLoadResult(frozen, envelope, True, result.elapsed_ms)
        return result

    def _load_packs(self, role: str) -> tuple[PackLoadResult, PackLoadResult]:
        if self.settings.local_pack_cache_enabled:
            common = self._pack("common")
            specific = self._pack(role)
            return common, specific
        common_manifest = load_default_manifest(
            "common", self.settings.knowledge_version
        )
        role_manifest = load_default_manifest(role, self.settings.knowledge_version)
        common_ref, common_envelope = transient_ref(common_manifest)
        role_ref, role_envelope = transient_ref(role_manifest)
        return (
            PackLoadResult(
                common_ref, common_envelope, False, 0.0, build_ms=0.0, retrieval_ms=0.0
            ),
            PackLoadResult(
                role_ref, role_envelope, False, 0.0, build_ms=0.0, retrieval_ms=0.0
            ),
        )

    def prepare_run(self, roles: Sequence[str] = ROLE_NAMES) -> dict[str, str]:
        """Compile and freeze all packs before the first agent fanout."""

        hashes: dict[str, str] = {}
        if not self.settings.enabled or not self.settings.local_pack_cache_enabled:
            return hashes
        common = self._pack("common")
        hashes["common"] = common.ref.content_sha256
        for role in roles:
            result = self._pack(role)
            hashes[role] = result.ref.content_sha256
        return hashes

    def prepare_request(
        self,
        params: Mapping[str, Any],
        *,
        provider: str,
        model: str,
        agent_role: str,
        api_family: str = "chat_completions",
        stable_system_instructions: str | None = None,
        dynamic_messages_override: Sequence[Mapping[str, Any]] | None = None,
        reasoning_config: Mapping[str, Any] | None = None,
    ) -> PreparedCacheRequest:
        original = copy.deepcopy(dict(params))
        role = agent_role if agent_role in ROLE_NAMES else "model_generator"
        preparation_started = time.monotonic()
        try:
            common, specific = self._load_packs(role)
            common_envelope = dict(common.envelope)
            common_envelope["role"] = common.ref.role
            role_envelope = dict(specific.envelope)
            role_envelope["role"] = specific.ref.role
            dynamic_messages = list(
                dynamic_messages_override
                if dynamic_messages_override is not None
                else (original.get("messages") or ())
            )
            if (
                stable_system_instructions
                and dynamic_messages
                and dynamic_messages[0].get("role") == "system"
                and dynamic_messages[0].get("content") == stable_system_instructions
            ):
                dynamic_messages = dynamic_messages[1:]
            assembled = self.assembler.assemble(
                dynamic_messages=dynamic_messages,
                tools=list(original.get("tools") or ()),
                common_pack=common_envelope,
                role_pack=role_envelope,
                stable_system_instructions=stable_system_instructions,
                reasoning_config=reasoning_config,
            )
            upstream_constraints = {
                "request": (original.get("extra_body") or {}).get("provider", {}),
                "configured_upstream": self.settings.openrouter_upstream,
                "allow_fallbacks": self.settings.openrouter_allow_fallbacks,
            }
            family = CacheFamily(
                provider=provider,
                model=model,
                common_pack_hash=common.ref.content_sha256,
                role_pack_hash=specific.ref.content_sha256,
                tool_schema_hash=assembled.tool_schema_hash,
                reasoning_config_hash=assembled.reasoning_config_hash,
                api_family=api_family,
                upstream_constraints_hash=canonical_sha256(upstream_constraints),
                system_instructions_hash=assembled.component_hashes[
                    "system_instructions"
                ],
            )
            prepared_params = copy.deepcopy(original)
            prepared_params["messages"] = list(assembled.messages)
            if "tools" in prepared_params:
                prepared_params["tools"] = list(assembled.tools)
            adapter = adapter_for(
                provider,
                sticky_routing=self.settings.openrouter_sticky_routing,
                routing_shards=self.settings.openrouter_routing_shards,
                upstream=self.settings.openrouter_upstream,
                allow_fallbacks=self.settings.openrouter_allow_fallbacks,
            )
            policy = CachePolicy(
                mode=self.settings.policy,
                ttl=self.settings.ttl,
                prewarm=self.settings.prewarm,
            )
            role_allowed = (
                not self.settings.provider_prompt_cache_roles
                or role in self.settings.provider_prompt_cache_roles
            )
            model_allowed = (
                not self.settings.provider_prompt_cache_models
                or model in self.settings.provider_prompt_cache_models
            )
            provider_cache_active = (
                self.settings.provider_prompt_cache_enabled
                and role_allowed
                and model_allowed
                and not self.settings.shadow
            )
            if provider_cache_active:
                prepared_params = adapter.apply_cache_policy(
                    prepared_params, assembled, family, policy
                )
            if self.settings.shadow:
                prepared_params = original
            local_hit = (
                bool(common.cache_hit and specific.cache_hit)
                if self.settings.local_pack_cache_enabled
                else False
            )
            telemetry = None
            if self.settings.telemetry:
                retrieval_values = [
                    value
                    for value in (common.retrieval_ms, specific.retrieval_ms)
                    if value is not None
                ]
                build_values = [
                    value
                    for value in (common.build_ms, specific.build_ms)
                    if value is not None
                ]
                telemetry = RequestTelemetry(
                    self.telemetry_store,
                    run_id=self.run_id,
                    provider=provider,
                    api_family=api_family,
                    model=model,
                    agent_role=role,
                    family=family,
                    stable_prefix_hash=assembled.stable_prefix_hash,
                    local_pack_cache_hit=local_hit,
                    expected_stable_prefix_tokens=assembled.expected_stable_prefix_tokens,
                    db_retrieval_ms=sum(retrieval_values) if retrieval_values else None,
                    pack_build_ms=sum(build_values) if build_values else None,
                    started_at=preparation_started,
                    prompt_snapshot=(
                        {
                            "messages": prepared_params.get("messages"),
                            "tools": prepared_params.get("tools"),
                        }
                        if self.settings.capture_prompts
                        else None
                    ),
                )
                telemetry.pack_ready()
            request_gate = None
            if provider_cache_active and self.settings.prewarm:
                gate_key = (
                    provider,
                    self.settings.openrouter_upstream,
                    model,
                    family.id,
                )
                request_gate = lambda: self.warm_first.hold(gate_key)
            return PreparedCacheRequest(
                params=prepared_params,
                active=True,
                family=family,
                assembled=assembled,
                adapter=adapter,
                telemetry=telemetry,
                local_pack_cache_hit=local_hit,
                request_gate=request_gate,
            )
        except Exception as exc:
            logger.warning(
                "Context-cache preparation failed; using the original request: %s", exc
            )
            telemetry = None
            if self.settings.telemetry:
                telemetry = RequestTelemetry(
                    self.telemetry_store,
                    run_id=self.run_id,
                    provider=provider,
                    api_family=api_family,
                    model=model,
                    agent_role=role,
                    family=None,
                    stable_prefix_hash=None,
                    local_pack_cache_hit=None,
                    expected_stable_prefix_tokens=None,
                    db_retrieval_ms=None,
                    pack_build_ms=None,
                    started_at=preparation_started,
                    prompt_snapshot=None,
                )
                telemetry.pack_ready()
            return PreparedCacheRequest(
                params=original,
                active=False,
                telemetry=telemetry,
                fallback_reason=type(exc).__name__,
            )


_COORDINATORS: dict[tuple[str, str | None, str], ContextCacheCoordinator] = {}
_COORDINATORS_LOCK = threading.Lock()


def settings_from_cfg(cfg: Any) -> ContextCacheSettings:
    return ContextCacheSettings.from_mapping(getattr(cfg, "context_cache", None))


def get_coordinator(cfg: Any) -> ContextCacheCoordinator | None:
    settings = settings_from_cfg(cfg)
    if not settings.enabled:
        return None
    run_id = str(getattr(cfg, "exp_name", "") or "") or None
    key = (str(settings.directory), run_id, settings.knowledge_version)
    with _COORDINATORS_LOCK:
        coordinator = _COORDINATORS.get(key)
        if coordinator is None:
            coordinator = ContextCacheCoordinator(settings, run_id=run_id)
            _COORDINATORS[key] = coordinator
    return coordinator


def prepare_llm_request(
    params: Mapping[str, Any],
    *,
    cfg: Any,
    provider: str,
    model: str,
    agent_role: str,
    api_family: str = "chat_completions",
    stable_system_instructions: str | None = None,
    dynamic_messages_override: Sequence[Mapping[str, Any]] | None = None,
    reasoning_config: Mapping[str, Any] | None = None,
) -> PreparedCacheRequest:
    coordinator = get_coordinator(cfg)
    if coordinator is None:
        # Deliberately retain the original object graph on the global-kill-switch path.
        return PreparedCacheRequest(params=dict(params), active=False)
    return coordinator.prepare_request(
        params,
        provider=provider,
        model=model,
        agent_role=agent_role,
        api_family=api_family,
        stable_system_instructions=stable_system_instructions,
        dynamic_messages_override=dynamic_messages_override,
        reasoning_config=reasoning_config,
    )


def prepare_run_context_cache(cfg: Any) -> dict[str, str]:
    coordinator = get_coordinator(cfg)
    if coordinator is None:
        return {}
    try:
        return coordinator.prepare_run()
    except Exception as exc:
        logger.warning(
            "Context-cache run preparation failed; continuing uncached: %s", exc
        )
        return {}
