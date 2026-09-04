"""Regression coverage for the local Qwen vLLM request budget."""

from types import SimpleNamespace

from context_cache.config import ContextCacheSettings
from llm.openai import (
    _context_safe_max_tokens,
    _default_max_tokens,
    _local_vllm_completion_budget,
    _use_thinking_for_request,
)


def test_qwen_default_generation_budget_uses_the_verified_32k_context_window():
    assert _default_max_tokens("qwen3.8-27b-int8-l40s") == 8192


def test_non_qwen_default_generation_budget_is_unchanged():
    assert _default_max_tokens("openai/gpt-5-nano") == 16384


def test_local_qwen_requests_disable_thinking_to_protect_agent_latency():
    stage = SimpleNamespace(provider="vllm", base_url="http://local-qwen:8000/v1")
    assert _use_thinking_for_request("qwen3.8-27b-int8-l40s", None, stage) is False


def test_local_vllm_qwen_keeps_context_independent_from_completion_budget():
    stage = SimpleNamespace(provider="vllm", base_url="http://127.0.0.1:8010/v1")
    cfg = SimpleNamespace(vllm_client=SimpleNamespace(default_completion_tokens=16384))
    assert _local_vllm_completion_budget("qwen3.8-27b-int8-a100", stage, cfg) == 16384


def test_local_vllm_qwen_can_explicitly_leave_completion_uncapped():
    stage = SimpleNamespace(provider="vllm", base_url="http://127.0.0.1:8010/v1")
    cfg = SimpleNamespace(vllm_client=SimpleNamespace(default_completion_tokens=None))
    assert _local_vllm_completion_budget("qwen3.8-27b-int8-a100", stage, cfg) is None


def test_local_vllm_qwen_request_uses_completion_budget_without_capping_context():
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **params):
            captured.update(params)
            choice = SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[]), finish_reason="stop"
            )
            return SimpleNamespace(
                choices=[choice], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)
            )

    stage = SimpleNamespace(
        model="qwen3.8-27b-int8-a100", provider="vllm", base_url="http://local/v1", api_key=""
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=stage, feedback=stage),
        context_cache=ContextCacheSettings(enabled=False),
        vllm_client=SimpleNamespace(default_completion_tokens=16384),
        exp_name="qwen-budget-test",
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    from llm.openai import query

    output, *_ = query("system", "user", cfg=cfg, model=stage.model, _client=client)

    assert output == "ok"
    assert captured["max_tokens"] == 16384


def test_context_error_uses_available_context_instead_of_fixed_2k_fallback():
    error = (
        "This model's maximum context length is 8192 tokens. However, you "
        "requested 4096 output tokens and your prompt contains at least 4097 "
        "input tokens, for a total of at least 8193 tokens."
    )
    assert _context_safe_max_tokens(error, requested_tokens=4096) == 3072


def test_32k_context_retry_preserves_enough_budget_for_full_training_code():
    error = (
        "This model's maximum context length is 32768 tokens. However, you "
        "requested 8192 output tokens and your prompt contains at least 25000 "
        "input tokens, for a total of at least 33192 tokens."
    )
    assert _context_safe_max_tokens(error, requested_tokens=8192) == 7168


def test_unstructured_context_error_falls_back_once_by_halving_the_budget():
    error = "maximum context length exceeded"
    assert _context_safe_max_tokens(error, requested_tokens=8192) == 4096
    assert _context_safe_max_tokens(error, requested_tokens=2048) is None
