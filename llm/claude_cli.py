"""Claude Code subscription backend for MLEvolve.

The backend uses the locally authenticated Claude Code command-line interface
rather than an Anthropic API key.  It intentionally disables Claude tools: the
MLEvolve interpreter, not the LLM process, owns all workspace execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import jsonschema

from .gemini import FunctionSpec, OutputType, compile_prompt_to_md


def _command_path() -> str:
    configured = os.getenv("MLEVOLVE_CLAUDE_CLI_COMMAND", "").strip()
    if configured:
        return configured
    discovered = shutil.which("claude")
    if discovered:
        return discovered
    local_install = Path.home() / ".local" / "bin" / "claude"
    if local_install.is_file():
        return str(local_install)
    raise FileNotFoundError(
        "Claude Code CLI was not found. Install it or set MLEVOLVE_CLAUDE_CLI_COMMAND."
    )


def _prompt(user_message: str | None) -> str:
    if user_message and user_message.strip():
        return user_message.strip()
    return "Proceed according to the system instructions."


def _run(
    prompt: str,
    *,
    model: str,
    json_schema: dict[str, Any] | None,
    timeout_seconds: float,
    system_prompt: str | None = None,
) -> tuple[str, float, int, int, dict[str, Any]]:
    if not prompt.strip():
        raise ValueError("Claude CLI query needs a system_message or user_message")
    command = [
        _command_path(),
        "-p",
        prompt,
    ]
    if system_prompt and system_prompt.strip():
        command.extend(["--system-prompt", system_prompt.strip()])
    command.extend([
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--tools",
        "",
    ])
    if json_schema is not None:
        command.extend(["--json-schema", json.dumps(json_schema, separators=(",", ":"))])
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"Claude Code CLI failed ({result.returncode}): {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Claude Code CLI returned invalid JSON") from exc
    if payload.get("is_error"):
        raise RuntimeError(f"Claude Code CLI error: {payload.get('result') or payload}")
    usage = payload.get("usage") or {}
    model_usage = payload.get("modelUsage") or {}
    canonical_model = next(
        (
            usage_info.get("canonicalModel")
            for usage_info in model_usage.values()
            if isinstance(usage_info, dict) and usage_info.get("canonicalModel")
        ),
        model,
    )
    info = {
        "model": model,
        "canonical_model": canonical_model,
        "duration_ms": payload.get("duration_ms"),
        "ttft_ms": payload.get("ttft_ms"),
    }
    return (
        str(payload.get("result") or ""),
        elapsed,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        info,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg=None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict[str, Any]]:
    del cfg
    model = str(model_kwargs.get("model") or "sonnet")
    max_tokens = model_kwargs.get("max_tokens")
    timeout_seconds = max(1.0, float(model_kwargs.get("timeout_seconds") or 1200.0))
    schema = func_spec.json_schema if func_spec is not None else None
    prompt = _prompt(user_message)
    if max_tokens is not None:
        prompt = f"{prompt}\n\nKeep the response within {int(max_tokens)} tokens."
    if func_spec is not None:
        prompt = (
            f"{prompt}\n\nReturn only the JSON arguments for function {func_spec.name!r}; "
            "they must validate against the supplied JSON schema."
        )
    response, elapsed, input_tokens, output_tokens, info = _run(
        prompt,
        model=model,
        json_schema=schema,
        timeout_seconds=timeout_seconds,
        system_prompt=system_message,
    )
    output: OutputType = response
    if func_spec is not None:
        try:
            output = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Claude CLI structured response was not valid JSON") from exc
        jsonschema.Draft7Validator(func_spec.json_schema).validate(output)
    return output, elapsed, input_tokens, output_tokens, info


def generate(
    prompt,
    cfg,
    temperature=None,
    max_tokens=None,
    stop_tokens=None,
    json_schema=None,
    max_retries=20,
    retry_delay=3,
    **_unused,
) -> str:
    del temperature, stop_tokens, max_retries, retry_delay
    text = compile_prompt_to_md(prompt) if not isinstance(prompt, str) else prompt.strip()
    response, *_ = _run(
        text,
        model=str(getattr(cfg.agent.code, "model", "sonnet") or "sonnet"),
        json_schema=json_schema,
        timeout_seconds=1200.0,
    )
    return response
