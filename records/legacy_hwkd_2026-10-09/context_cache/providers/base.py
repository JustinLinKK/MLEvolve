"""Provider-neutral prompt-cache adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, Mapping

from ..models import (
    AssembledPrompt,
    CacheCapabilities,
    CacheFamily,
    CachePolicy,
    NormalizedCacheUsage,
    NormalizedRequestMetrics,
)


def field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    marker = object()
    direct = getattr(value, name, marker)
    if direct is not marker:
        return direct
    # OpenAI's Pydantic response models retain fields introduced by newer
    # servers in model_extra even when the installed SDK has no typed member.
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, Mapping):
        return extra.get(name, default)
    return default


def path(value: Any, *names: str) -> Any:
    current = value
    for name in names:
        current = field(current, name)
        if current is None:
            return None
    return current


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


class ProviderCacheAdapter(ABC):
    provider = "unknown"

    @abstractmethod
    def capabilities(self, model: str) -> CacheCapabilities:
        raise NotImplementedError

    @abstractmethod
    def apply_cache_policy(
        self,
        params: Mapping[str, Any],
        assembled: AssembledPrompt,
        family: CacheFamily,
        policy: CachePolicy,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def extract_cache_usage(self, raw_response: Any) -> NormalizedCacheUsage:
        raise NotImplementedError

    def extract_upstream_provider(self, raw_response: Any) -> str | None:
        return None

    def extract_request_metrics(self, raw_response: Any) -> NormalizedRequestMetrics:
        metrics = field(raw_response, "metrics")
        if metrics is None:
            extra = field(raw_response, "model_extra") or {}
            metrics = field(extra, "metrics")
        return NormalizedRequestMetrics(
            server_ttft_ms=optional_float(field(metrics, "time_to_first_token_ms")),
            server_queue_ms=optional_float(field(metrics, "queue_time_ms")),
            server_generation_ms=optional_float(field(metrics, "generation_time_ms")),
            server_mean_itl_ms=optional_float(field(metrics, "mean_itl_ms")),
            server_tokens_per_second=optional_float(field(metrics, "tokens_per_second")),
        )

    def supports_prewarm(self, model: str, policy: CachePolicy) -> bool:
        capabilities = self.capabilities(model)
        return capabilities.supports_prewarm and policy.prewarm


class NoOpCacheAdapter(ProviderCacheAdapter):
    def __init__(self, provider: str = "unknown", **_: Any) -> None:
        self.provider = provider

    def capabilities(self, model: str) -> CacheCapabilities:
        del model
        return CacheCapabilities(False, (), False, {})

    def apply_cache_policy(
        self,
        params: Mapping[str, Any],
        assembled: AssembledPrompt,
        family: CacheFamily,
        policy: CachePolicy,
    ) -> dict[str, Any]:
        del assembled, family, policy
        return copy.deepcopy(dict(params))

    def extract_cache_usage(self, raw_response: Any) -> NormalizedCacheUsage:
        usage = field(raw_response, "usage")
        return NormalizedCacheUsage(
            prompt_tokens=optional_int(
                first_not_none(
                    field(usage, "prompt_tokens"), field(usage, "input_tokens")
                )
            ),
            output_tokens=optional_int(
                first_not_none(
                    field(usage, "completion_tokens"), field(usage, "output_tokens")
                )
            ),
        )
