"""OpenAI direct prompt-cache adapter."""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

from .base import ProviderCacheAdapter, field, first_not_none, optional_int
from ..models import (
    AssembledPrompt,
    CacheCapabilities,
    CacheFamily,
    CachePolicy,
    NormalizedCacheUsage,
)


def _explicit_capable(model: str) -> bool:
    normalized = model.lower().split("/")[-1]
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?", normalized)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (5, 6)


def _mark_breakpoint(messages: list[dict[str, Any]], stable_index: int | None) -> None:
    if stable_index is None:
        return
    message = messages[stable_index]
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
    elif isinstance(content, list) and content:
        last = dict(content[-1])
        last["prompt_cache_breakpoint"] = {"mode": "explicit"}
        message["content"] = [*content[:-1], last]


class OpenAICacheAdapter(ProviderCacheAdapter):
    provider = "openai"

    def __init__(self, **_: Any) -> None:
        pass

    def capabilities(self, model: str) -> CacheCapabilities:
        explicit = _explicit_capable(model)
        return CacheCapabilities(
            explicit_breakpoints=explicit,
            ttl_values=("30m",) if explicit else (),
            supports_prewarm=False,
            metrics_mapping={
                "prompt_tokens_details.cached_tokens": "cache_read_tokens",
                "prompt_tokens_details.cache_write_tokens": "cache_write_tokens",
                "input_tokens_details.cached_tokens": "cache_read_tokens",
                "input_tokens_details.cache_write_tokens": "cache_write_tokens",
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
        extra["prompt_cache_key"] = family.id
        if capabilities.explicit_breakpoints:
            options: dict[str, Any] = {}
            if policy.mode == "explicit":
                options["mode"] = "explicit"
                messages = copy.deepcopy(
                    list(result.get("messages") or assembled.messages)
                )
                _mark_breakpoint(messages, assembled.stable_message_index)
                result["messages"] = messages
            if policy.ttl:
                options["ttl"] = policy.ttl
            if options:
                extra["prompt_cache_options"] = options
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
            output_tokens=optional_int(
                first_not_none(
                    field(usage, "completion_tokens"), field(usage, "output_tokens")
                )
            ),
        )
