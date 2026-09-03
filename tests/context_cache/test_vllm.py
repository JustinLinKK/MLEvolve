from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm
from context_cache.models import (
    AssembledPrompt,
    CacheFamily,
    CachePolicy,
)
from context_cache.providers.vllm import VLLMCacheAdapter
from context_cache.config import ContextCacheSettings
from context_cache.telemetry import CacheTelemetryStore
from llm import FunctionSpec
from llm import vllm


def _assembled() -> AssembledPrompt:
    return AssembledPrompt(
        messages=(
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "dynamic"},
        ),
        tools=(),
        stable_prefix="stable",
        dynamic_suffix=({"role": "user", "content": "dynamic"},),
        stable_prefix_hash="prefix",
        component_hashes={},
        tool_schema_hash="tools",
        reasoning_config_hash="reasoning",
        stable_message_index=0,
        expected_stable_prefix_tokens=10,
    )


def _family(**overrides) -> CacheFamily:
    values = dict(
        provider="vllm",
        model="qwen3.8-27b-int8-w8a16",
        common_pack_hash="common",
        role_pack_hash="role",
        tool_schema_hash="tools",
        reasoning_config_hash="reasoning",
        api_family="chat_completions",
        system_instructions_hash="system",
    )
    values.update(overrides)
    return CacheFamily(**values)


def _cfg(*, require_salt: bool = True):
    code = SimpleNamespace(
        model="same-model",
        provider="vllm",
        base_url="http://code:8000/v1",
        api_key="",
    )
    feedback = SimpleNamespace(
        model="same-model",
        provider="openai",
        base_url="http://feedback:8000/v1",
        api_key="feedback-key",
    )
    return SimpleNamespace(
        agent=SimpleNamespace(code=code, feedback=feedback),
        context_cache=SimpleNamespace(enabled=False),
        vllm_client=SimpleNamespace(
            cache_salt_env="TEST_VLLM_CACHE_SALT",
            require_cache_salt=require_salt,
            session_affinity=True,
        ),
        exp_name="vllm-test",
    )


def test_adapter_uses_only_session_header_and_normalizes_optional_usage_fields():
    adapter = VLLMCacheAdapter(session_affinity=True)
    family = _family()
    params = {
        "model": family.model,
        "messages": list(_assembled().messages),
        "extra_body": {"enable_thinking": True},
    }

    result = adapter.apply_cache_policy(
        params, _assembled(), family, CachePolicy(mode="auto", prewarm=True)
    )

    assert result["extra_headers"] == {"X-Session-ID": f"mlevolve:{family.id}"}
    serialized = repr(result)
    for unsupported in (
        "prompt_cache_breakpoint",
        "ttl",
        "kv_transfer_params",
        "cache_control",
    ):
        assert unsupported not in serialized

    read_only = adapter.extract_cache_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 80},
            }
        }
    )
    modern = adapter.extract_cache_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "created_cache_tokens": 20,
                },
            }
        }
    )
    assert (read_only.cache_read_tokens, read_only.cache_write_tokens) == (80, None)
    assert (modern.cache_read_tokens, modern.cache_write_tokens) == (80, 20)
    assert modern.cache_miss_tokens == 20


def test_modern_metrics_are_optional_and_normalized():
    adapter = VLLMCacheAdapter()
    assert adapter.extract_request_metrics({}).server_ttft_ms is None
    metrics = adapter.extract_request_metrics(
        {
            "metrics": {
                "time_to_first_token_ms": 12.5,
                "queue_time_ms": 3,
                "generation_time_ms": 40,
                "mean_itl_ms": 2.25,
                "tokens_per_second": 44.4,
            }
        }
    )
    assert metrics.server_ttft_ms == 12.5
    assert metrics.server_queue_ms == 3.0
    assert metrics.server_generation_ms == 40.0
    assert metrics.server_mean_itl_ms == 2.25
    assert metrics.server_tokens_per_second == 44.4


def test_salt_is_enforced_before_transport_creation(monkeypatch):
    cfg = _cfg()
    monkeypatch.delenv("TEST_VLLM_CACHE_SALT", raising=False)
    monkeypatch.setattr(
        vllm,
        "_client_for",
        lambda stage: pytest.fail("transport must not be created before salt validation"),
    )

    with pytest.raises(ValueError, match="32 bytes"):
        vllm.query("system", "user", cfg=cfg, model="same-model", stage_name="code")


def test_stage_disambiguation_and_provider_dispatch_for_identical_models():
    cfg = _cfg(require_salt=False)
    assert llm._provider("same-model", cfg, "code") == "vllm"
    assert llm._provider("same-model", cfg, "feedback") == "openai"


def test_endpoint_pool_reuses_connections_and_separates_endpoints(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(vllm, "OpenAI", FakeClient)
    vllm.close_clients()
    first_stage = SimpleNamespace(base_url="http://one:8000/v1/", api_key="")
    same_endpoint = SimpleNamespace(base_url="http://one:8000/v1", api_key="")
    second_stage = SimpleNamespace(base_url="http://two:8000/v1", api_key="")

    first = vllm._client_for(first_stage)
    assert vllm._client_for(same_endpoint) is first
    assert vllm._client_for(second_stage) is not first
    assert len(created) == 2
    assert created[0].kwargs["api_key"] == "EMPTY"

    vllm.close_clients()
    assert all(client.closed for client in created)


def test_vllm_uses_lightweight_http_transport_without_sdk(monkeypatch):
    monkeypatch.setattr(vllm, "OpenAI", None)
    vllm.close_clients()

    client = vllm._client_for(
        SimpleNamespace(base_url="http://vllm:8000/v1", api_key="EMPTY")
    )

    assert type(client).__name__ == "_VLLMHttpClient"
    vllm.close_clients()


def test_cache_family_changes_with_reasoning_system_tools_and_api_family():
    base = _family()
    variants = [
        _family(reasoning_config_hash="other"),
        _family(system_instructions_hash="other"),
        _family(tool_schema_hash="other"),
        _family(api_family="responses"),
        _family(model="another-model"),
    ]
    assert all(item.id != base.id for item in variants)


def test_nonstreaming_named_tool_request_sends_salt_affinity_and_metrics(
    monkeypatch, tmp_path
):
    capture = {}
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="report", arguments='{\"score\":7}')
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    details = SimpleNamespace(cached_tokens=80, created_cache_tokens=20)
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=5,
        prompt_tokens_details=details,
    )
    completion = SimpleNamespace(
        choices=[choice],
        usage=usage,
        model="same-model",
        created=1,
        model_extra={
            "metrics": {
                "time_to_first_token_ms": 12.5,
                "queue_time_ms": 2.0,
                "generation_time_ms": 8.0,
                "mean_itl_ms": 1.5,
                "tokens_per_second": 50.0,
            }
        },
    )

    class Completions:
        def create(self, **kwargs):
            capture.update(kwargs)
            return completion

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    stage = SimpleNamespace(
        model="same-model",
        provider="vllm",
        base_url="http://vllm:8000/v1",
        api_key="",
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=stage, feedback=stage),
        context_cache=ContextCacheSettings(
            enabled=True,
            provider_prompt_cache_enabled=True,
            prewarm=True,
            cache_dir=str(tmp_path),
            telemetry=True,
        ),
        vllm_client=SimpleNamespace(
            cache_salt_env="TEST_VLLM_CACHE_SALT",
            require_cache_salt=True,
            session_affinity=True,
        ),
        exp_name="vllm-nonstream",
    )
    monkeypatch.setenv("TEST_VLLM_CACHE_SALT", "s" * 32)
    monkeypatch.setattr(vllm, "_client_for", lambda stage: client)
    spec = FunctionSpec(
        name="report",
        description="Report a score",
        json_schema={
            "type": "object",
            "properties": {"score": {"type": "integer"}},
            "required": ["score"],
        },
    )

    output, _, _, _, info = vllm.query(
        "stable system",
        "dynamic task",
        func_spec=spec,
        cfg=cfg,
        model="same-model",
        stage_name="code",
        context_cache_role="reviewer",
        context_cache_stable_prefix="stable system",
    )

    assert output == {"score": 7}
    assert capture["extra_body"]["cache_salt"] == "s" * 32
    assert capture["extra_headers"]["X-Session-ID"] == (
        f"mlevolve:{info['cache_family_id']}"
    )
    for unsupported in ("ttl", "kv_transfer_params", "prompt_cache_options"):
        assert unsupported not in repr(capture)
    row = CacheTelemetryStore(tmp_path / "cache-registry.sqlite3").rows()[0]
    assert row["cache_read_tokens"] == 80
    assert row["cache_write_tokens"] == 20
    assert row["server_ttft_ms"] == 12.5
    assert row["server_tokens_per_second"] == 50.0


def test_streaming_text_requests_usage_and_accepts_minimal_shape(monkeypatch, tmp_path):
    capture = {}
    empty = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)],
        usage=None,
        model_extra={},
    )
    token = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"), finish_reason=None)],
        usage=None,
        model_extra={},
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=1,
        prompt_tokens_details=SimpleNamespace(cached_tokens=8),
    )
    final = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")],
        usage=usage,
        model_extra={},
    )

    class Completions:
        def create(self, **kwargs):
            capture.update(kwargs)
            return iter([empty, token, final])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    stage = SimpleNamespace(
        model="same-model",
        provider="vllm",
        base_url="http://vllm:8000/v1",
        api_key="",
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=stage, feedback=stage),
        context_cache=ContextCacheSettings(
            enabled=True,
            provider_prompt_cache_enabled=True,
            prewarm=True,
            cache_dir=str(tmp_path),
            telemetry=True,
        ),
        vllm_client=SimpleNamespace(
            cache_salt_env="TEST_VLLM_CACHE_SALT",
            require_cache_salt=True,
            session_affinity=True,
        ),
        exp_name="vllm-stream",
    )
    monkeypatch.setenv("TEST_VLLM_CACHE_SALT", "x" * 32)
    monkeypatch.setattr(vllm, "_client_for", lambda stage: client)

    output = vllm.generate(
        {"system": "stable", "user": "dynamic"},
        cfg,
        max_retries=1,
        context_cache_stable_prefix="stable",
    )

    assert output == "hello"
    assert capture["stream"] is True
    assert capture["stream_options"] == {"include_usage": True}
    row = CacheTelemetryStore(tmp_path / "cache-registry.sqlite3").rows()[0]
    assert row["cache_read_tokens"] == 8
    assert row["server_ttft_ms"] is None
