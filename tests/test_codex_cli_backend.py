from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import llm
from llm import codex_cli
from llm.gemini import FunctionSpec


def _cfg(*, api_key: str = "") -> SimpleNamespace:
    stage = SimpleNamespace(
        provider="codex",
        model="gpt-5.5",
        temp=1.0,
        base_url="",
        api_key=api_key,
        executable="codex",
        reasoning_effort="low",
        timeout_seconds=30,
        ephemeral=True,
    )
    return SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))


def _jsonl(message: str) -> str:
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item-1", "type": "agent_message", "text": message},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 123, "cached_input_tokens": 20, "output_tokens": 45},
                }
            ),
        ]
    )


def test_codex_query_uses_stdin_read_only_ephemeral_and_parses_usage(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(codex_cli.shutil, "which", lambda executable: "/opt/bin/codex")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=_jsonl("done"), stderr="")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    output, request_time, input_tokens, output_tokens, info = codex_cli.query(
        system_message="Be precise.",
        user_message="Inspect the repository.",
        model="gpt-5.5",
        cfg=_cfg(),
    )

    command = captured["command"]
    assert output == "done"
    assert request_time >= 0
    assert input_tokens == 123
    assert output_tokens == 45
    assert info["cached_input_tokens"] == 20
    assert command[0] == "/opt/bin/codex"
    assert command[-1] == "-"
    assert "--json" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert 'model_reasoning_effort="low"' in command
    assert "Inspect the repository." not in command
    assert "# System instructions" in captured["input"]
    assert "# User request" in captured["input"]
    assert "CODEX_API_KEY" not in captured["env"]
    assert "CODEX_HOME" in captured["env"]
    assert captured["cwd"] == codex_cli.REPO_ROOT


def test_codex_query_passes_output_schema_and_parses_structured_response(monkeypatch) -> None:
    captured_schema_path = None
    monkeypatch.setattr(codex_cli.shutil, "which", lambda executable: "/opt/bin/codex")
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {"source": {"type": "string"}},
            },
        },
        "required": ["answer"],
    }
    func_spec = FunctionSpec(name="answer", description="Return an answer", json_schema=schema)

    def fake_run(command, **kwargs):
        nonlocal captured_schema_path
        captured_schema_path = command[command.index("--output-schema") + 1]
        written_schema = json.loads(Path(captured_schema_path).read_text(encoding="utf-8"))
        assert written_schema["additionalProperties"] is False
        assert written_schema["properties"]["details"]["additionalProperties"] is False
        assert written_schema["required"] == ["answer", "details"]
        assert written_schema["properties"]["details"]["type"] == ["object", "null"]
        return subprocess.CompletedProcess(command, 0, stdout=_jsonl('{"answer":"yes"}'), stderr="")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    output, *_ = codex_cli.query(
        system_message=None,
        user_message="Answer.",
        func_spec=func_spec,
        model="gpt-5.5",
        cfg=_cfg(api_key="dedicated-test-key"),
    )

    assert output == {"answer": "yes"}
    assert captured_schema_path is not None
    assert not Path(captured_schema_path).exists()


def test_llm_dispatches_explicit_codex_provider(monkeypatch) -> None:
    calls = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return "codex-result", 0.1, 1, 2, {"provider": "codex-cli"}

    monkeypatch.setattr(llm._codex_cli, "query", fake_query)

    result = llm.query(
        system_message="system",
        user_message="user",
        model="gpt-5.5",
        cfg=_cfg(),
    )

    assert result == "codex-result"
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-5.5"
