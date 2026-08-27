from __future__ import annotations

from context_cache.assembler import DeterministicPromptAssembler
from context_cache.models import CacheFamily, CachePolicy
from context_cache.providers.base import NoOpCacheAdapter
from context_cache.providers.deepseek import DeepSeekCacheAdapter
from context_cache.providers.openai import OpenAICacheAdapter
from context_cache.providers.openrouter import OpenRouterCacheAdapter


def _assembled():
    return DeterministicPromptAssembler().assemble(
        dynamic_messages=[{"role": "user", "content": "dynamic"}],
        tools=[],
        common_pack={
            "role": "common",
            "content": {"sections": [{"stable_id": "c", "content": "common"}]},
        },
        role_pack={
            "role": "reviewer",
            "content": {"sections": [{"stable_id": "r", "content": "review"}]},
        },
    )


def _family(model="openai/gpt-5.6"):
    assembled = _assembled()
    return CacheFamily(
        "openrouter",
        model,
        "c",
        "r",
        assembled.tool_schema_hash,
        assembled.reasoning_config_hash,
    )


def test_noop_adapter_preserves_request_and_missing_metrics_are_null() -> None:
    params = {"model": "custom", "messages": [{"role": "user", "content": "unchanged"}]}
    adapter = NoOpCacheAdapter()

    assert (
        adapter.apply_cache_policy(
            params, _assembled(), _family("custom"), CachePolicy()
        )
        == params
    )
    usage = adapter.extract_cache_usage({"usage": {}})
    assert usage.cache_read_tokens is None
    assert usage.prompt_tokens is None
    zero = adapter.extract_cache_usage(
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    )
    assert zero.prompt_tokens == 0
    assert zero.output_tokens == 0


def test_openrouter_uses_stable_bounded_session_and_normalizes_usage() -> None:
    adapter = OpenRouterCacheAdapter(sticky_routing=True, routing_shards=4)
    assembled = _assembled()
    family = _family()
    params = {"model": family.model, "messages": list(assembled.messages)}

    first = adapter.apply_cache_policy(
        params, assembled, family, CachePolicy(mode="explicit", ttl="30m")
    )
    second = adapter.apply_cache_policy(
        params, assembled, family, CachePolicy(mode="explicit", ttl="30m")
    )

    assert first["extra_body"]["session_id"] == second["extra_body"]["session_id"]
    assert len(first["extra_body"]["session_id"]) <= 256
    assert first["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    usage = adapter.extract_cache_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 0},
            }
        }
    )
    assert usage.cache_read_tokens == 80
    assert usage.cache_write_tokens == 0


def test_openrouter_can_pin_upstream_and_disable_fallbacks() -> None:
    adapter = OpenRouterCacheAdapter(
        upstream="OpenAI",
        allow_fallbacks=False,
    )
    assembled = _assembled()
    result = adapter.apply_cache_policy(
        {"model": "openai/gpt-5.6", "messages": list(assembled.messages)},
        assembled,
        _family(),
        CachePolicy(),
    )

    assert result["extra_body"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
    }


def test_provider_controls_are_suppressed_without_a_stable_boundary() -> None:
    assembled = DeterministicPromptAssembler().assemble(
        dynamic_messages=[{"role": "user", "content": "dynamic first"}],
        tools=[],
        common_pack={"role": "common", "content": {"sections": []}},
        role_pack={"role": "reviewer", "content": {"sections": []}},
    )
    params = {"model": "openai/gpt-5.6", "messages": list(assembled.messages)}

    assert (
        OpenRouterCacheAdapter().apply_cache_policy(
            params, assembled, _family(), CachePolicy()
        )
        == params
    )
    assert (
        OpenAICacheAdapter().apply_cache_policy(
            params, assembled, _family("gpt-5.6"), CachePolicy()
        )
        == params
    )


def test_openai_explicit_controls_only_on_capable_models() -> None:
    adapter = OpenAICacheAdapter()
    assembled = _assembled()
    capable = _family("gpt-5.6")
    old = _family("gpt-5.5")
    params = {"model": "gpt", "messages": list(assembled.messages)}

    explicit = adapter.apply_cache_policy(
        params, assembled, capable, CachePolicy(mode="explicit", ttl="30m")
    )
    unsupported = adapter.apply_cache_policy(
        params, assembled, old, CachePolicy(mode="explicit")
    )

    assert explicit["extra_body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    assert "extra_body" not in unsupported


def test_deepseek_does_not_invent_controls_and_maps_hit_miss() -> None:
    adapter = DeepSeekCacheAdapter()
    assembled = _assembled()
    params = {"model": "deepseek-chat", "messages": list(assembled.messages)}

    assert (
        adapter.apply_cache_policy(
            params, assembled, _family("deepseek-chat"), CachePolicy()
        )
        == params
    )
    usage = adapter.extract_cache_usage(
        {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "prompt_cache_hit_tokens": 12,
                "prompt_cache_miss_tokens": 8,
            }
        }
    )
    assert usage.cache_read_tokens == 12
    assert usage.cache_miss_tokens == 8

