"""Codex subscription-CLI backend for MLEvolve."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import jsonschema

from .common import FunctionSpec, OutputType, compile_prompt_to_md


def _command_path() -> str:
    configured = os.getenv("MLEVOLVE_CODEX_CLI_COMMAND", "").strip()
    if configured:
        return configured
    discovered = shutil.which("codex")
    if discovered:
        return discovered
    raise FileNotFoundError("Codex CLI was not found. Install it or set MLEVOLVE_CODEX_CLI_COMMAND.")


def _prompt(system_message: str | None, user_message: str | None) -> str:
    parts = [
        "Do not modify files or execute experiment code. Return only the requested response; "
        "MLEvolve owns all candidate execution.",
    ]
    if system_message and system_message.strip():
        parts.append(f"System instructions:\n{system_message.strip()}")
    if user_message and user_message.strip():
        parts.append(f"User request:\n{user_message.strip()}")
    if len(parts) == 1:
        raise ValueError("Codex CLI query needs a system_message or user_message")
    return "\n\n".join(parts)


def _strict_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize an arbitrary JSON Schema for Codex strict structured output.

    Codex requires every object property to appear in ``required`` and rejects
    stale entries that are not declared in ``properties``.  MLEvolve's planning
    schemas can evolve their property set between stages, so normalize each
    object recursively instead of forwarding its historical ``required`` list.
    """
    result = copy.deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if value.get("type") == "object" or isinstance(properties, dict):
                value["additionalProperties"] = False
                if isinstance(properties, dict):
                    value["required"] = list(properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def _supports_strict_schema(schema: dict[str, Any]) -> bool:
    """Return whether a schema can be represented by Codex strict JSON output.

    Codex strict mode rejects dynamic object maps, while MLEvolve's planner uses
    one for module-name-to-plan entries.  Sending such a schema causes a server
    error before the model can answer, so the caller must use prompt-guided JSON
    for that one case.
    """
    def visit(value: Any) -> bool:
        if isinstance(value, dict):
            if isinstance(value.get("additionalProperties"), dict):
                return False
            return all(visit(child) for child in value.values())
        if isinstance(value, list):
            return all(visit(child) for child in value)
        return True

    return visit(schema)


def _dynamic_schema_guidance(schema: dict[str, Any]) -> str:
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    return (
        "Return a single top-level JSON object matching the requested schema. "
        f"Its top-level required fields: {json.dumps(required)}. "
        "Do not return only the contents of a nested field."
    )


def _run(prompt: str, *, model: str, json_schema: dict[str, Any] | None,
         timeout_seconds: float) -> tuple[str, float, int, int, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="mlevolve-codex-") as temp_dir:
        output_path = Path(temp_dir) / "last-message.txt"
        command = [
            _command_path(), "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--model", model, "--output-last-message",
            str(output_path), "-",
        ]
        if json_schema is not None:
            schema_path = Path(temp_dir) / "response-schema.json"
            schema_path.write_text(json.dumps(json_schema), encoding="utf-8")
            command.extend(["--output-schema", str(schema_path)])
        started = time.monotonic()
        result = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                timeout=timeout_seconds, check=False)
        elapsed = time.monotonic() - started
        if result.returncode:
            raise RuntimeError(f"Codex CLI failed ({result.returncode}): {result.stderr.strip()}")
        if not output_path.is_file():
            raise RuntimeError("Codex CLI did not write its final response")
        response = output_path.read_text(encoding="utf-8").strip()
    return response, elapsed, 0, 0, {"model": model, "provider": "codex_cli"}


def query(system_message: str | None, user_message: str | None,
          func_spec: FunctionSpec | None = None, cfg=None, **model_kwargs):
    del cfg
    model = str(model_kwargs.get("model") or "gpt-5.6-terra")
    timeout_seconds = max(1.0, float(model_kwargs.get("timeout_seconds") or 1200.0))
    prompt = _prompt(system_message, user_message)
    if func_spec is not None:
        prompt += f"\n\nReturn JSON arguments for function {func_spec.name!r}."
        if not _supports_strict_schema(func_spec.json_schema):
            prompt += "\n" + _dynamic_schema_guidance(func_spec.json_schema)
    response, elapsed, input_tokens, output_tokens, info = _run(
        prompt, model=model,
        json_schema=(
            _strict_response_schema(func_spec.json_schema)
            if func_spec and _supports_strict_schema(func_spec.json_schema)
            else None
        ),
        timeout_seconds=timeout_seconds,
    )
    output: OutputType = response
    if func_spec is not None:
        try:
            output = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Codex CLI structured response was not valid JSON") from exc
        jsonschema.Draft7Validator(func_spec.json_schema).validate(output)
    return output, elapsed, input_tokens, output_tokens, info


def generate(prompt, cfg, temperature=None, max_tokens=None, stop_tokens=None,
             json_schema=None, max_retries=20, retry_delay=3, **_unused) -> str:
    del temperature, max_tokens, stop_tokens, max_retries, retry_delay
    text = compile_prompt_to_md(prompt) if not isinstance(prompt, str) else prompt.strip()
    if json_schema is not None and not _supports_strict_schema(json_schema):
        text += "\n\n" + _dynamic_schema_guidance(json_schema)
    response, *_ = _run(
        _prompt(None, text),
        model=str(getattr(cfg.agent.code, "model", "gpt-5.6-terra") or "gpt-5.6-terra"),
        json_schema=(
            _strict_response_schema(json_schema)
            if json_schema is not None and _supports_strict_schema(json_schema)
            else None
        ),
        timeout_seconds=1200.0,
    )
    return response
