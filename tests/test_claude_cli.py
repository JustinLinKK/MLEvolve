from types import SimpleNamespace
import json

import llm
from llm.gemini import FunctionSpec


def test_explicit_claude_cli_stage_selects_claude_cli_provider() -> None:
    """Changing a configured Claude CLI stage to another backend must change routing."""
    stage = SimpleNamespace(model="sonnet", provider="claude_cli")
    cfg = SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))

    assert llm._provider("sonnet", cfg, "code") == "claude_cli"


def test_claude_cli_query_uses_subscription_cli_without_tools(monkeypatch) -> None:
    from llm import claude_cli

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "result": "agent answer",
                    "duration_ms": 125,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                    "modelUsage": {"claude-sonnet-5": {"canonicalModel": "claude-sonnet-5"}},
                    "is_error": False,
                }
            ),
            stderr="",
        )

    monkeypatch.setenv("MLEVOLVE_CLAUDE_CLI_COMMAND", "/opt/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)

    output, _, input_tokens, output_tokens, info = claude_cli.query(
        system_message="System rule",
        user_message="User task",
        model="sonnet",
    )

    assert output == "agent answer"
    assert input_tokens == 11
    assert output_tokens == 7
    assert info["canonical_model"] == "claude-sonnet-5"
    assert captured["command"] == [
        "/opt/claude",
        "-p",
        "User task",
        "--system-prompt",
        "System rule",
        "--model",
        "sonnet",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--tools",
        "",
    ]


def test_claude_cli_query_decodes_schema_validated_feedback(monkeypatch) -> None:
    from llm import claude_cli

    spec = FunctionSpec(
        name="review",
        description="Review a run.",
        json_schema={
            "type": "object",
            "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"],
            "additionalProperties": False,
        },
    )
    payload = {"result": "{\"approved\": true}", "usage": {"input_tokens": 3, "output_tokens": 2}}

    monkeypatch.setenv("MLEVOLVE_CLAUDE_CLI_COMMAND", "/opt/claude")
    monkeypatch.setattr(
        claude_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    output, *_ = claude_cli.query(
        system_message="Return a decision.",
        user_message=None,
        model="sonnet",
        func_spec=spec,
    )

    assert output == {"approved": True}
