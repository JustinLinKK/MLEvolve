"""DeepSeek automatic context-cache adapter."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .base import ProviderCacheAdapter, field, optional_int
from ..models import (
    AssembledPrompt,
    CacheCapabilities,
    CacheFamily,
    CachePolicy,
    NormalizedCacheUsage,
)


class DeepSeekCacheAdapter(ProviderCacheAdapter):
    provider = "deepseek"

    def __init__(self, **_: Any) -> None:
        pass

    def capabilities(self, model: str) -> CacheCapabilities:
        del model
        return CacheCapabilities(
            explicit_breakpoints=False,
            ttl_values=(),
            supports_prewarm=False,
            metrics_mapping={
                "prompt_cache_hit_tokens": "cache_read_tokens",
                "prompt_cache_miss_tokens": "cache_miss_tokens",
            },
        )

    def apply_cache_policy(
        self,
        params: Mapping[str, Any],
        assembled: AssembledPrompt,
        family: CacheFamily,
        policy: CachePolicy,
    ) -> dict[str, Any]:
        del assembled, family
        # DeepSeek caching is automatic. Explicit controls are intentionally not invented.
        if policy.mode == "explicit" or policy.ttl:
            return copy.deepcopy(dict(params))
        return copy.deepcopy(dict(params))

    def extract_cache_usage(self, raw_response: Any) -> NormalizedCacheUsage:
        usage = field(raw_response, "usage")
        return NormalizedCacheUsage(
            prompt_tokens=optional_int(field(usage, "prompt_tokens")),
            cache_read_tokens=optional_int(field(usage, "prompt_cache_hit_tokens")),
            cache_miss_tokens=optional_int(field(usage, "prompt_cache_miss_tokens")),
            output_tokens=optional_int(field(usage, "completion_tokens")),
        )

