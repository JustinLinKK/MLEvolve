"""OpenRouter prompt-cache and sticky-routing adapter."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from .base import ProviderCacheAdapter, field, first_not_none, optional_int
from .openai import _explicit_capable, _mark_breakpoint
from ..models import (
    AssembledPrompt,
    CacheCapabilities,
    CacheFamily,
    CachePolicy,
    NormalizedCacheUsage,
)

_QWEN_EXPLICIT = (
    "qwen/qwen3-max",
    "qwen/qwen-plus",
    "qwen/qwen3.6-plus",
    "qwen/qwen3-coder-plus",
    "qwen/qwen3-coder-flash",
    "deepseek/deepseek-v3.2",
)


class OpenRouterCacheAdapter(ProviderCacheAdapter):
    provider = "openrouter"

    def __init__(
        self,
        *,
        sticky_routing: bool = True,
        routing_shards: int = 1,
        upstream: str | None = None,
        allow_fallbacks: bool = True,
        **_: Any,
    ) -> None:
        self.sticky_routing = bool(sticky_routing)
        self.routing_shards = max(1, min(64, int(routing_shards)))
        self.upstream = str(upstream).strip() if upstream else None
        self.allow_fallbacks = bool(allow_fallbacks)

    def capabilities(self, model: str) -> CacheCapabilities:
        lowered = model.lower().lstrip("~")
        explicit = (
            lowered.startswith(("anthropic/", "google/gemini"))
            or lowered in _QWEN_EXPLICIT
            or _explicit_capable(lowered)
        )
        if lowered.startswith("anthropic/"):
            ttls = ("5m", "1h")
        elif _explicit_capable(lowered):
            ttls = ("30m",)
        elif explicit:
            ttls = ("5m",)
        else:
            ttls = ()
        return CacheCapabilities(
            explicit_breakpoints=explicit,
            ttl_values=ttls,
            supports_prewarm=False,
            metrics_mapping={
                "prompt_tokens_details.cached_tokens": "cache_read_tokens",
                "prompt_tokens_details.cache_write_tokens": "cache_write_tokens",
            },
        )

    def _session_id(self, family: CacheFamily) -> str:
        if self.routing_shards == 1:
            shard = 0
        else:
            shard = (
                int(hashlib.sha256(family.id.encode("ascii")).hexdigest()[:8], 16)
                % self.routing_shards
            )
        model = family.model.replace(" ", "_")[:96]
        return f"mlevolve:{model}:{family.common_pack_hash}:{shard}"

    def apply_cache_policy(
        self,
        params: Mapping[str, Any],
        assembled: AssembledPrompt,
        family: CacheFamily,
        policy: CachePolicy,
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(params))
        if policy.mode == "none":
            return result
        capabilities = self.capabilities(family.model)
        if policy.mode == "explicit" and not capabilities.explicit_breakpoints:
            return result
        if assembled.stable_message_index is None:
            return result
        if policy.ttl and policy.ttl not in capabilities.ttl_values:
            return result
        extra = dict(result.get("extra_body") or {})
        if self.sticky_routing:
            extra["session_id"] = self._session_id(family)
        if self.upstream:
            provider = dict(extra.get("provider") or {})
            provider["order"] = [self.upstream]
            provider["allow_fallbacks"] = self.allow_fallbacks
            extra["provider"] = provider
        if policy.mode == "explicit" and assembled.stable_message_index is not None:
            messages = copy.deepcopy(list(result.get("messages") or assembled.messages))
            lowered = family.model.lower().lstrip("~")
            if _explicit_capable(lowered):
                _mark_breakpoint(messages, assembled.stable_message_index)
                options: dict[str, Any] = {"mode": "explicit"}
                if policy.ttl:
                    options["ttl"] = policy.ttl
                extra["prompt_cache_options"] = options
                extra["prompt_cache_key"] = family.id
            else:
                message = messages[assembled.stable_message_index]
                content = message.get("content")
                control: dict[str, str] = {"type": "ephemeral"}
                if policy.ttl and policy.ttl != "5m":
                    control["ttl"] = policy.ttl
                if isinstance(content, str):
                    message["content"] = [
                        {"type": "text", "text": content, "cache_control": control}
                    ]
                elif isinstance(content, list) and content:
                    last = dict(content[-1])
                    last["cache_control"] = control
                    message["content"] = [*content[:-1], last]
            result["messages"] = messages
        result["extra_body"] = extra
        return result

    def extract_cache_usage(self, raw_response: Any) -> NormalizedCacheUsage:
        usage = field(raw_response, "usage")
        details = field(usage, "prompt_tokens_details") or field(
            usage, "input_tokens_details"
        )
        return NormalizedCacheUsage(
            prompt_tokens=optional_int(
                first_not_none(
                    field(usage, "prompt_tokens"), field(usage, "input_tokens")
                )
            ),
            cache_read_tokens=optional_int(field(details, "cached_tokens")),
            cache_write_tokens=optional_int(field(details, "cache_write_tokens")),
            cache_miss_tokens=optional_int(field(usage, "prompt_cache_miss_tokens")),
            output_tokens=optional_int(
                first_not_none(
                    field(usage, "completion_tokens"), field(usage, "output_tokens")
                )
            ),
        )

    def extract_upstream_provider(self, raw_response: Any) -> str | None:
        for name in ("provider", "upstream_provider"):
            value = field(raw_response, name)
            if value:
                return str(value)
        metadata = field(raw_response, "openrouter_metadata") or field(
            raw_response, "metadata"
        )
        value = field(metadata, "provider_name") or field(metadata, "provider")
        return str(value) if value else None

