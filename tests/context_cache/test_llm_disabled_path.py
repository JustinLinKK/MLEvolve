from __future__ import annotations

import json
from types import SimpleNamespace

from context_cache.config import ContextCacheSettings
from llm import openai as backend


class _Completions:
    def __init__(self, capture):
        self.capture = capture

    def create(self, **params):
        self.capture.update(params)
        message = SimpleNamespace(content="ok", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=1)
        return SimpleNamespace(
            choices=[choice], usage=usage, model=params["model"], created=1
        )


class _FakeOpenAI:
    capture = None

    def __init__(self, **kwargs):
        del kwargs
        self.chat = SimpleNamespace(completions=_Completions(self.capture))


def test_disabled_path_serializes_legacy_request_byte_for_byte(monkeypatch) -> None:
    capture = {}
    _FakeOpenAI.capture = capture
    monkeypatch.setattr(backend, "OpenAI", _FakeOpenAI)
    stage = SimpleNamespace(
        model="custom-model",
        api_key="secret",
        base_url="https://example.invalid",
        provider="openrouter",
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=stage, feedback=stage),
        context_cache=ContextCacheSettings(enabled=False),
        exp_name="disabled-test",
    )

    output, _, _, _, _ = backend.query(
        "system",
        "user",
        cfg=cfg,
        model="custom-model",
        temperature=0.25,
        max_tokens=99,
    )

    assert output == "ok"
    assert json.dumps(capture, sort_keys=True, separators=(",", ":")) == (
        '{"max_tokens":99,"messages":[{"content":"system","role":"system"},'
        '{"content":"user","role":"user"}],"model":"custom-model","stream":false,'
        '"temperature":0.25}'
    ).replace(',"stream":false', "")


def test_enabled_shared_backend_sends_stable_then_dynamic_without_duplication(
    monkeypatch, tmp_path
) -> None:
    capture = {}
    _FakeOpenAI.capture = capture
    monkeypatch.setattr(backend, "OpenAI", _FakeOpenAI)
    stage = SimpleNamespace(
        model="openai/gpt-5.6",
        api_key="secret",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=stage, feedback=stage),
        context_cache=ContextCacheSettings(
            enabled=True,
            cache_dir=str(tmp_path),
            provider_prompt_cache_enabled=True,
        ),
        exp_name="enabled-test",
    )

    backend.query(
        "legacy stable and dynamic",
        None,
        cfg=cfg,
        model=stage.model,
        context_cache_role="reviewer",
        context_cache_stable_prefix="stable rules",
        context_cache_dynamic_system_message="dynamic candidate",
    )

    serialized_messages = json.dumps(capture["messages"], sort_keys=True)
    assert "stable rules" in serialized_messages
    assert "dynamic candidate" in serialized_messages
    assert "legacy stable and dynamic" not in serialized_messages
    assert capture["extra_body"]["session_id"].startswith("mlevolve:")

