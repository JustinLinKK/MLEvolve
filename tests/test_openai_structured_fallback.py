from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm.gemini import FunctionSpec
from llm.openai import query


SPEC = FunctionSpec(
    name="submit_code_review",
    description="Submit code review for search node solution.",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {"type": "boolean"},
            "reasoning": {"type": "string"},
            "revised_code": {"type": ["string", "null"]},
        },
        "required": ["needs_revision", "reasoning"],
    },
)


def test_structured_query_retries_direct_json_when_tool_call_content_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    completions = [
        _completion(content="", tool_calls=[]),
        _completion(content='{"needs_revision": false, "reasoning": "Looks valid."}', tool_calls=[]),
    ]

    monkeypatch.setattr("llm.openai.OpenAI", _fake_openai_factory(calls, completions))

    output, _req_time, _in_tok, _out_tok, _info = query(
        system_message="Review this code.",
        user_message=None,
        func_spec=SPEC,
        cfg=_cfg(),
        model="google/gemini-2.5-pro",
        temperature=1,
    )

    assert output == {"needs_revision": False, "reasoning": "Looks valid."}
    assert len(calls) == 2
    assert "tools" in calls[0]
    assert calls[0]["tool_choice"]["function"]["name"] == "submit_code_review"
    assert "tools" not in calls[1]
    assert calls[1]["response_format"]["type"] == "json_schema"
    assert calls[1]["messages"][-1]["content"].startswith("The previous response did not include")


def test_structured_query_parses_plain_json_content_without_second_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    completions = [
        _completion(
            content='```json\n{"needs_revision": true, "reasoning": "Missing metric.", "revised_code": "patch"}\n```',
            tool_calls=[],
        )
    ]

    monkeypatch.setattr("llm.openai.OpenAI", _fake_openai_factory(calls, completions))

    output, _req_time, _in_tok, _out_tok, _info = query(
        system_message="Review this code.",
        user_message=None,
        func_spec=SPEC,
        cfg=_cfg(),
        model="google/gemini-2.5-pro",
        temperature=1,
    )

    assert output["needs_revision"] is True
    assert output["reasoning"] == "Missing metric."
    assert output["revised_code"] == "patch"
    assert len(calls) == 1


def test_structured_query_retries_without_response_format_when_provider_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    completions = [
        _completion(content="", tool_calls=[]),
        RuntimeError("response_format unsupported"),
        _completion(content='{"needs_revision": false, "reasoning": "No issues."}', tool_calls=[]),
    ]

    monkeypatch.setattr("llm.openai.OpenAI", _fake_openai_factory(calls, completions))

    output, _req_time, _in_tok, _out_tok, _info = query(
        system_message="Review this code.",
        user_message=None,
        func_spec=SPEC,
        cfg=_cfg(),
        model="google/gemini-2.5-pro",
        temperature=1,
    )

    assert output == {"needs_revision": False, "reasoning": "No issues."}
    assert len(calls) == 3
    assert "response_format" in calls[1]
    assert "response_format" not in calls[2]


def _cfg() -> SimpleNamespace:
    stage = SimpleNamespace(
        model="google/gemini-2.5-pro",
        temp=1,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
    )
    return SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))


def _completion(content: str, tool_calls: list | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        model="fake-model",
        created=123,
    )


def _fake_openai_factory(calls: list[dict], completions: list[SimpleNamespace | Exception]):
    class FakeCompletions:
        def create(self, **params):
            calls.append(dict(params))
            response = completions.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = FakeChat()

    return FakeOpenAI
