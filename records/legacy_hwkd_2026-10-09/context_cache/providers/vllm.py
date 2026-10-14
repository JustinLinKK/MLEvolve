"""vLLM automatic-prefix-cache adapter.

The adapter deliberately uses only stable OpenAI-compatible fields plus the
router header. Cache salt is injected by ``llm.vllm`` even when the context
cache feature is disabled, so an enabled vLLM server never receives an
accidentally unsalted request.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .base import ProviderCacheAdapter, field, first_not_none, optional_int
from ..models import (
    AssembledPrompt,
    CacheCapabilities,
    CacheFamily,
    CachePolicy,
    NormalizedCacheUsage,
)


class VLLMCacheAdapter(ProviderCacheAdapter):
    provider = "vllm"

    def __init__(self, *, session_affinity: bool = True, **_: Any) -> None:
        self.session_affinity = bool(session_affinity)

    def capabilities(self, model: str) -> CacheCapabilities:
        del model
        return CacheCapabilities(
            explicit_breakpoints=False,
            ttl_values=(),
            supports_prewarm=True,
            metrics_mapping={
                "prompt_tokens_details.cached_tokens": "cache_read_tokens",
                "prompt_tokens_details.created_cache_tokens": "cache_write_tokens",
            },
        )

    def apply_cache_policy(
        self,
        params: Mapping[str, Any],
        assembled: AssembledPrompt,
        family: CacheFamily,
        policy: CachePolicy,
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(params))
        if policy.mode == "none" or assembled.stable_message_index is None:
            return result
        if self.session_affinity:
            headers = dict(result.get("extra_headers") or {})
            headers["X-Session-ID"] = f"mlevolve:{family.id}"
            result["extra_headers"] = headers
        return result

    def extract_cache_usage(self, raw_response: Any) -> NormalizedCacheUsage:
        usage = field(raw_response, "usage")
        details = field(usage, "prompt_tokens_details") or field(
            usage, "input_tokens_details"
        )
        read_tokens = optional_int(
            first_not_none(
                field(details, "cached_tokens"),
                field(usage, "cached_tokens"),
            )
        )
        prompt_tokens = optional_int(
            first_not_none(field(usage, "prompt_tokens"), field(usage, "input_tokens"))
        )
        return NormalizedCacheUsage(
            prompt_tokens=prompt_tokens,
            cache_read_tokens=read_tokens,
            cache_write_tokens=optional_int(
                first_not_none(
                    field(details, "created_cache_tokens"),
                    field(details, "cache_write_tokens"),
                    field(usage, "created_cache_tokens"),
                )
            ),
            cache_miss_tokens=(
                max(0, prompt_tokens - read_tokens)
                if prompt_tokens is not None and read_tokens is not None
                else None
            ),
            output_tokens=optional_int(
                first_not_none(
                    field(usage, "completion_tokens"), field(usage, "output_tokens")
                )
            ),
        )
