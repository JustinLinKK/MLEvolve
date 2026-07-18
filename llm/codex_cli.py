"""Codex CLI backend for repository-wide agent stress testing."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from config import Config
from .gemini import FunctionSpec, compile_prompt_to_md

logger = logging.getLogger("MLEvolve")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _stage_config_for_model(cfg: Config, model: str):
    if cfg.agent.code.model == model:
        return cfg.agent.code
    return cfg.agent.feedback


def _prompt_text(system_message: str | None, user_message: str | None) -> str:
    sections: list[str] = []
    if system_message:
        sections.append(f"# System instructions\n\n{system_message.strip()}")
    if user_message:
        sections.append(f"# User request\n\n{user_message.strip()}")
    if not sections:
        raise ValueError("Either system_message or user_message must be provided")
    return "\n\n".join(sections) + "\n"


def _resolve_executable(stage: Any) -> str:
    configured = str(getattr(stage, "executable", "codex") or "codex")
    if Path(configured).is_file():
        return configured
    resolved = shutil.which(configured)
    if resolved:
        return resolved
    if Path(configured).name == configured:
        user_local = Path.home() / ".local" / "bin" / configured
        if user_local.is_file():
            return str(user_local)
    raise FileNotFoundError(
        f"Codex CLI executable `{configured}` was not found. Install the Codex CLI and authenticate it before "
        "using provider `codex`."
    )


def _parse_jsonl(stdout: str) -> tuple[str, int, int, dict[str, Any]]:
    final_message = ""
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    thread_id = None
    event_types: list[str] = []
    malformed_lines = 0

    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        if event_type:
            event_types.append(event_type)
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                final_message = str(item.get("text") or "")
        elif event_type == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
                cached_input_tokens = int(usage.get("cached_input_tokens") or 0)

    if not final_message:
        raise RuntimeError("Codex CLI JSONL output did not contain a completed agent message")
    return final_message, input_tokens, output_tokens, {
        "thread_id": thread_id,
        "cached_input_tokens": cached_input_tokens,
        "event_types": event_types,
        "malformed_jsonl_lines": malformed_lines,
    }


def _parse_structured_output(text: str, func_spec: FunctionSpec) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    output = json.loads(cleaned)
    if not isinstance(output, dict):
        raise ValueError(f"Structured response for {func_spec.name} must be a JSON object")
    required = func_spec.json_schema.get("required") or []
    missing = [field for field in required if field not in output]
    if missing:
        raise ValueError(f"Structured response for {func_spec.name} missing required fields: {missing}")
    return output


def _nullable_schema(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        normalized["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list) and "null" not in schema_type:
        normalized["type"] = [*schema_type, "null"]
    elif isinstance(normalized.get("anyOf"), list):
        normalized["anyOf"] = [*normalized["anyOf"], {"type": "null"}]
    return normalized


def _normalize_output_schema(value: Any) -> Any:
    """Adapt ordinary Draft 7 schemas to Codex's strict object-schema requirement."""
    if isinstance(value, list):
        return [_normalize_output_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _normalize_output_schema(item) for key, item in value.items()}
    if normalized.get("type") == "object" or "properties" in normalized:
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            originally_required = set(normalized.get("required") or [])
            normalized["properties"] = {
                key: item if key in originally_required else _nullable_schema(item)
                for key, item in properties.items()
            }
            normalized["required"] = list(properties)
        normalized["additionalProperties"] = False
    return normalized


def _has_dynamic_object_keys(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_dynamic_object_keys(item) for item in value)
    if not isinstance(value, dict):
        return False
    additional = value.get("additionalProperties")
    if additional is True or isinstance(additional, dict):
        return True
    return any(_has_dynamic_object_keys(item) for item in value.values())


def _run_codex(
    *,
    prompt: str,
    model: str,
    stage: Any,
    output_schema: dict[str, Any] | None = None,
) -> tuple[str, float, int, int, dict[str, Any]]:
    schema_mode = "none"
    if output_schema is not None and _has_dynamic_object_keys(output_schema):
        prompt = (
            f"{prompt.rstrip()}\n\n# Required output\n\nReturn only one JSON value matching this schema exactly:\n"
            f"{json.dumps(output_schema, indent=2, sort_keys=True)}\n"
        )
        output_schema = None
        schema_mode = "prompt"
    elif output_schema is not None:
        schema_mode = "cli"
    executable = _resolve_executable(stage)
    effort = str(getattr(stage, "reasoning_effort", "low") or "low")
    timeout = float(getattr(stage, "timeout_seconds", 1200.0) or 1200.0)
    command = [
        executable,
        "-c",
        'approval_policy="never"',
        "-c",
        f'model_reasoning_effort="{effort}"',
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        "--model",
        model,
    ]
    if bool(getattr(stage, "ephemeral", True)):
        command.append("--ephemeral")
    if bool(getattr(stage, "ignore_user_config", True)):
        command.append("--ignore-user-config")

    schema_path: str | None = None
    isolated_home: tempfile.TemporaryDirectory[str] | None = None
    try:
        if output_schema is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="mlevolve-codex-schema-",
                encoding="utf-8",
                delete=False,
            ) as schema_file:
                json.dump(_normalize_output_schema(output_schema), schema_file)
                schema_path = schema_file.name
            command.extend(["--output-schema", schema_path])
        command.append("-")

        env = os.environ.copy()
        configured_api_key = str(getattr(stage, "api_key", "") or "").strip()
        if configured_api_key:
            env["CODEX_API_KEY"] = configured_api_key
        if bool(getattr(stage, "isolated_home", True)):
            source_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
            source_home.mkdir(parents=True, exist_ok=True)
            isolated_home = tempfile.TemporaryDirectory(prefix=".mlevolve-home-", dir=source_home)
            isolated_home_path = Path(isolated_home.name)
            isolated_home_path.chmod(0o700)
            source_auth = source_home / "auth.json"
            if source_auth.is_file() and not configured_api_key:
                shutil.copy2(source_auth, isolated_home_path / "auth.json")
            env["CODEX_HOME"] = str(isolated_home_path)

        started_at = time.time()
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Codex CLI exceeded its {timeout:g}s timeout") from exc
        request_time = time.time() - started_at
        if completed.returncode != 0:
            stderr = str(completed.stderr or "").strip()
            stdout = str(completed.stdout or "").strip()
            stderr_excerpt = stderr[-1600:] if stderr else "no stderr output"
            stdout_excerpt = stdout[-1600:] if stdout else "no stdout output"
            raise RuntimeError(
                f"Codex CLI exited with code {completed.returncode}. "
                f"stderr: {stderr_excerpt}; stdout: {stdout_excerpt}"
            )

        output, input_tokens, output_tokens, info = _parse_jsonl(completed.stdout)
        info.update(
            {
                "provider": "codex-cli",
                "model": model,
                "reasoning_effort": effort,
                "ephemeral": bool(getattr(stage, "ephemeral", True)),
                "isolated_home": bool(getattr(stage, "isolated_home", True)),
                "schema_mode": schema_mode,
            }
        )
        return output, request_time, input_tokens, output_tokens, info
    finally:
        if schema_path:
            Path(schema_path).unlink(missing_ok=True)
        if isolated_home is not None:
            isolated_home.cleanup()


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs: Any,
) -> tuple[str | dict, float, int, int, dict[str, Any]]:
    if cfg is None:
        raise ValueError("cfg is required for Codex CLI backend")
    model = str(model_kwargs.get("model") or "")
    stage = _stage_config_for_model(cfg, model)
    output, request_time, input_tokens, output_tokens, info = _run_codex(
        prompt=_prompt_text(system_message, user_message),
        model=model,
        stage=stage,
        output_schema=func_spec.json_schema if func_spec else None,
    )
    parsed: str | dict = _parse_structured_output(output, func_spec) if func_spec else output
    return parsed, request_time, input_tokens, output_tokens, info


def generate(
    prompt: Any,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict[str, Any] | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    del temperature, max_tokens, stop_tokens
    stage = cfg.agent.code
    prompt_text = prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            output, _, _, _, _ = _run_codex(
                prompt=prompt_text,
                model=stage.model,
                stage=stage,
                output_schema=json_schema,
            )
            return output
        except (OSError, RuntimeError, TimeoutError):
            if attempt + 1 >= attempts:
                raise
            logger.warning("Codex CLI generation attempt %s/%s failed", attempt + 1, attempts)
            time.sleep(max(0.0, float(retry_delay)))
    raise RuntimeError("Codex CLI generation exhausted all retries")
