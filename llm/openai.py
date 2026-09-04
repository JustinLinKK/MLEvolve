"""OpenAI-compatible API backend (query + generate). Supports Qwen API and any OpenAI-compatible endpoint."""

import json
import logging
import re
import time
from contextlib import nullcontext
from typing import Any

from openai import OpenAI

from config import Config
from .common import FunctionSpec, compile_prompt_to_md
from .model_profiles import get_profile, supports_json_schema, thinking_json_incompatible

logger = logging.getLogger("MLEvolve")


def _default_max_tokens(model: str) -> int:
    """Retain legacy output defaults for non-local-vLLM requests."""
    return 8192 if (model or "").lower().startswith("qwen") else 16384


def _is_local_vllm_qwen(model: str, stage: Any) -> bool:
    return (model or "").lower().startswith("qwen") and (
        str(getattr(stage, "provider", "") or "").strip().lower() == "vllm"
    )


def _local_vllm_completion_budget(model: str, stage: Any, cfg: Config) -> int | None:
    """Return the independent completion budget for a local Qwen vLLM request.

    The Agent's context window is deliberately left to the server.  That must
    not make a single streamed completion unbounded: a model that never emits
    its closing fence otherwise prevents the scheduler from receiving a node.
    A null setting preserves an explicitly requested uncapped completion.
    """
    if not _is_local_vllm_qwen(model, stage):
        return None
    configured = getattr(getattr(cfg, "vllm_client", None), "default_completion_tokens", 16384)
    if configured is None:
        return None
    budget = int(configured)
    if budget <= 0:
        raise ValueError("vllm_client.default_completion_tokens must be positive or null")
    return budget


def _context_safe_max_tokens(error: Exception | str, requested_tokens: int) -> int | None:
    """Retry with the largest useful output budget that fits the context."""
    message = str(error).lower()
    if "maximum context length" not in message:
        return None
    requested = int(requested_tokens)
    if requested <= 2048:
        return None

    context_match = re.search(r"maximum context length is\s+(\d+)\s+tokens", message)
    prompt_match = re.search(
        r"prompt contains at least\s+(\d+)\s+input tokens",
        message,
    )
    if context_match is not None and prompt_match is not None:
        context_tokens = int(context_match.group(1))
        prompt_tokens = int(prompt_match.group(1))
        available = context_tokens - prompt_tokens - 512
        safe = (min(requested - 1, available) // 512) * 512
        return safe if 512 <= safe < requested else None

    safe = (requested // 2 // 512) * 512
    return safe if 2048 <= safe < requested else None


def _use_thinking_for_request(model: str, func_spec: FunctionSpec | None, stage: Any) -> bool:
    """Disable hidden reasoning on the local Qwen endpoint to protect code budget."""
    provider = str(getattr(stage, "provider", "") or "").strip().lower()
    local_qwen = (model or "").lower().startswith("qwen") and provider in {
        "vllm",
        "openai-compatible",
    }
    return func_spec is None and not local_qwen


def _strip_markdown_fences(args: str) -> str:
    """Remove markdown code fences that LLMs sometimes append inside JSON string values."""
    cleaned = re.sub(r'\\n```[a-z]*\s*("?\s*\}?\s*)$', r'\1', args.rstrip())
    cleaned = cleaned.rstrip()
    if not cleaned.endswith('}'):
        if not cleaned.endswith('"'):
            cleaned += '"'
        cleaned += '}'
    return cleaned


def _parse_json_args(args: str) -> dict:
    """Parse function call arguments, tolerating Python literals and markdown fences."""
    # 1. Fast path: valid JSON as-is
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        pass

    # 2. Try stripping markdown fences
    try:
        cleaned = _strip_markdown_fences(args)
        if cleaned != args:
            result = json.loads(cleaned)
            logger.warning("Fixed malformed function args by stripping markdown code fences")
            return result
    except json.JSONDecodeError:
        pass

    # 3. Normalize Python literals (None/True/False) outside quoted strings
    parts = re.split(r'("(?:[^"\\]|\\.)*")', args)
    normalized = []
    for part in parts:
        if part.startswith('"'):
            normalized.append(part)
        else:
            part = re.sub(r'\bNone\b', 'null', part)
            part = re.sub(r'\bTrue\b', 'true', part)
            part = re.sub(r'\bFalse\b', 'false', part)
            normalized.append(part)
    normalized_str = ''.join(normalized)

    try:
        return json.loads(normalized_str)
    except json.JSONDecodeError:
        pass

    # 4. Normalized + strip markdown fences
    cleaned = _strip_markdown_fences(normalized_str)
    return json.loads(cleaned)


def _extract_json_object(text: str) -> str:
    """Extract the first top-level JSON object from plain-text model output."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty content; cannot extract JSON object")

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in assistant content")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Unterminated JSON object in assistant content")

# Return type aligned with gemini.query
OutputType = str | dict


def _stage_config_for_model(cfg: Config, model: str, stage_name: str | None = None):
    """Return code or feedback config depending on which model is being used."""
    if stage_name in {"code", "feedback"}:
        return getattr(cfg.agent, stage_name)
    if cfg.agent.code.model == model:
        return cfg.agent.code
    return cfg.agent.feedback


def _is_openrouter_stage(stage) -> bool:
    return (getattr(stage, "provider", "") or "").lower() == "openrouter"


def _provider_name(stage) -> str:
    provider = (getattr(stage, "provider", "") or "").strip().lower()
    if provider in {"openrouter", "openai", "deepseek"}:
        return provider
    return provider or "openai-compatible"


def _prepare_context_cache(
    params: dict[str, Any],
    *,
    cfg: Config,
    stage,
    model: str,
    role: str,
    stable_prefix: str | None,
    dynamic_messages_override: list[dict[str, str]] | None,
    reasoning_config: dict[str, Any],
    provider_override: str | None = None,
):
    from context_cache.coordinator import prepare_llm_request

    return prepare_llm_request(
        params,
        cfg=cfg,
        provider=provider_override or _provider_name(stage),
        model=model,
        agent_role=role,
        stable_system_instructions=stable_prefix,
        dynamic_messages_override=dynamic_messages_override,
        reasoning_config=reasoning_config,
    )


def _finish_telemetry(prepared, **kwargs) -> None:
    telemetry = getattr(prepared, "telemetry", None)
    if telemetry is None:
        return
    try:
        telemetry.finish(**kwargs)
    except Exception as exc:
        logger.warning("Context-cache telemetry persistence failed: %s", exc)


def _cache_response_details(prepared, raw_response, *, prompt_tokens=None, output_tokens=None):
    from context_cache.models import NormalizedCacheUsage, NormalizedRequestMetrics

    adapter = getattr(prepared, "adapter", None)
    if adapter is None:
        return NormalizedCacheUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        ), None, NormalizedRequestMetrics()
    try:
        usage = adapter.extract_cache_usage(raw_response)
        upstream = adapter.extract_upstream_provider(raw_response)
        metrics = adapter.extract_request_metrics(raw_response)
        return usage, upstream, metrics
    except Exception as exc:
        logger.warning("Context-cache usage normalization failed; retaining response: %s", exc)
        return NormalizedCacheUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        ), None, NormalizedRequestMetrics()


def _build_messages(system_message: str | None, user_message: str | None) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """OpenAI-compatible query (chat completions, optional function calling). Same return shape as gemini.query."""
    if cfg is None:
        raise ValueError("cfg is required for OpenAI backend")
    stage_name = model_kwargs.pop("stage_name", None)
    client_override = model_kwargs.pop("_client", None)
    provider_override = model_kwargs.pop("_provider_override", None)
    vllm_cache_salt = model_kwargs.pop("_vllm_cache_salt", None)
    context_cache_role = str(model_kwargs.pop("context_cache_role", "analysis"))
    context_cache_stable_prefix = model_kwargs.pop("context_cache_stable_prefix", None)
    context_cache_dynamic_system_message = model_kwargs.pop(
        "context_cache_dynamic_system_message", None
    )
    filtered = {k: v for k, v in model_kwargs.items() if v is not None}
    model = filtered.get("model", "")
    stage = _stage_config_for_model(cfg, model, stage_name)
    client = client_override or OpenAI(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        timeout=1200.0,
    )
    messages = _build_messages(system_message, user_message)
    if not messages:
        raise ValueError("Either system_message or user_message must be provided")
    dynamic_messages_override = None
    if context_cache_dynamic_system_message is not None:
        dynamic_messages_override = [
            {"role": "system", "content": str(context_cache_dynamic_system_message)}
        ]
        if user_message:
            dynamic_messages_override.append({"role": "user", "content": user_message})

    # Function calling requires non_thinking mode, otherwise Qwen API errors:
    # "tool_choice does not support required/object in thinking mode"
    use_thinking = _use_thinking_for_request(model, func_spec, stage)
    profile = get_profile(model, use_thinking=use_thinking)

    extra_body: dict[str, Any] = {}
    if "top_k" in profile:
        extra_body["top_k"] = profile["top_k"]
    if "enable_thinking" in profile:
        extra_body["enable_thinking"] = profile["enable_thinking"]

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": profile.get("temperature", filtered.get("temperature", 1.0)),
    }
    requested_max_tokens = filtered.get("max_tokens")
    if requested_max_tokens is not None:
        params["max_tokens"] = requested_max_tokens
    else:
        local_budget = _local_vllm_completion_budget(model, stage, cfg)
        if local_budget is not None:
            params["max_tokens"] = local_budget
        elif not _is_local_vllm_qwen(model, stage):
            params["max_tokens"] = _default_max_tokens(model)
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if "presence_penalty" in profile:
        params["presence_penalty"] = profile["presence_penalty"]
    if extra_body:
        params["extra_body"] = extra_body
    if func_spec is not None:
        tool_dict = func_spec.as_openai_tool_dict
        if _is_openrouter_stage(stage) or not supports_json_schema(model):
            tool_dict.pop("strict", None)
        params["tools"] = [tool_dict]
        params["tool_choice"] = func_spec.openai_tool_choice_dict

    prepared = _prepare_context_cache(
        params,
        cfg=cfg,
        stage=stage,
        model=model,
        role=context_cache_role,
        stable_prefix=context_cache_stable_prefix,
        dynamic_messages_override=dynamic_messages_override,
        reasoning_config={
            "profile": {key: profile[key] for key in sorted(profile)},
            "response_format": params.get("response_format"),
            "tool_choice": params.get("tool_choice"),
        },
        provider_override=provider_override,
    )
    params = prepared.params
    if vllm_cache_salt is not None:
        extra_body = dict(params.get("extra_body") or {})
        extra_body["cache_salt"] = vllm_cache_salt
        params["extra_body"] = extra_body

    t0 = time.time()
    logger.info(f"Querying OpenAI-compatible API with model: {model}")
    try:
        if prepared.telemetry is not None:
            prepared.telemetry.request_started()
        gate = prepared.request_gate() if prepared.request_gate else nullcontext()
        with gate as lease:
            completion = client.chat.completions.create(**params)
    except Exception as e:
        _finish_telemetry(prepared, error_type=type(e).__name__)
        logger.error(f"Error calling OpenAI-compatible API: {e}")
        raise
    req_time = time.time() - t0
    choice = completion.choices[0]
    message = choice.message
    if prepared.telemetry is not None:
        prepared.telemetry.first_meaningful_delta()
    in_tok = getattr(completion.usage, "prompt_tokens", 0) or 0
    out_tok = getattr(completion.usage, "completion_tokens", 0) or 0
    cache_usage, upstream_provider, server_metrics = _cache_response_details(
        prepared,
        completion,
        prompt_tokens=in_tok,
        output_tokens=out_tok,
    )
    _finish_telemetry(
        prepared,
        usage=cache_usage,
        raw_usage=getattr(completion, "usage", None),
        upstream_provider=upstream_provider,
        finish_reason=getattr(choice, "finish_reason", None),
        cost_usd=getattr(getattr(completion, "usage", None), "cost", None),
        server_metrics=server_metrics,
    )

    if getattr(choice, "finish_reason", None) == "length":
        logger.warning(f"Response truncated by max_tokens ({params.get('max_tokens')}), consider increasing it")

    if func_spec is None:
        output = message.content or ""
        logger.info(f"OpenAI response: {output}", extra={"verbose": True})
    else:
        if message.tool_calls:
            tc = message.tool_calls[0]
            if tc.function.name != func_spec.name:
                raise ValueError(f"Function name mismatch: expected {func_spec.name}, got {tc.function.name}")
            try:
                output = _parse_json_args(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid function arguments: {tc.function.arguments}")
                raise e
            logger.info(f"OpenAI function call response: {output}", extra={"verbose": True})
        else:
            logger.warning("Expected function call, got no tool_calls; attempting JSON content fallback")
            raw_content = message.content or ""
            json_payload = _extract_json_object(raw_content)
            output = _parse_json_args(json_payload)
            logger.info(f"OpenAI JSON content fallback response: {output}", extra={"verbose": True})

    info = {
        "model": getattr(completion, "model", model),
        "created": getattr(completion, "created", int(time.time())),
    }
    if prepared.family is not None:
        info["cache_family_id"] = prepared.family.id
        info["stable_prefix_hash"] = prepared.assembled.stable_prefix_hash if prepared.assembled else None
    return output, req_time, in_tok, out_tok, info


def _prompt_to_messages(prompt: str | dict | list, model: str = "") -> list[dict[str, str]]:
    """Convert prompt to chat messages. Supports Qwen/OpenAI chat format: {system, user, assistant}.

    For GPT models, assistant content is appended to the user message instead of
    being sent as a separate assistant message, because GPT models may return
    empty responses when they see a trailing assistant prefill.
    """
    if isinstance(prompt, dict) and ("system" in prompt or "user" in prompt or "assistant" in prompt):
        messages = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": str(prompt["system"])})

        is_gpt = (model or "").lower().startswith("gpt")
        user_content = str(prompt["user"]) if prompt.get("user") else ""
        assistant_content = str(prompt["assistant"]) if prompt.get("assistant") else ""

        if is_gpt and assistant_content:
            # GPT: merge assistant prefill into user message
            combined = f"{user_content}\n\n{assistant_content}" if user_content else assistant_content
            messages.append({"role": "user", "content": combined})
        else:
            if user_content:
                messages.append({"role": "user", "content": user_content})
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

        if not messages:
            raise ValueError("Chat dict must have at least one of: system, user, assistant")
        return messages
    content = prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)
    return [{"role": "user", "content": content}]


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
    context_cache_role: str = "model_generator",
    context_cache_stable_prefix: str | None = None,
    _client: Any = None,
    _provider_override: str | None = None,
    _vllm_cache_salt: str | None = None,
) -> str:
    """Streaming text generation via OpenAI-compatible Chat API. Supports chat format {system, user, assistant} for Qwen."""
    stage = cfg.agent.code
    model = stage.model
    messages = _prompt_to_messages(prompt, model=model)
    client = _client or OpenAI(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        timeout=1200.0,
    )
    # Qwen: thinking + json_schema are mutually exclusive — drop schema, keep thinking.
    if json_schema is not None and thinking_json_incompatible(model):
        json_schema = None
    use_thinking = json_schema is None and _use_thinking_for_request(model, None, stage)
    profile = get_profile(model, use_thinking=use_thinking)

    extra_body: dict[str, Any] = {}
    if "top_k" in profile:
        extra_body["top_k"] = profile["top_k"]
    if "enable_thinking" in profile:
        extra_body["enable_thinking"] = profile["enable_thinking"]

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": profile.get("temperature", temperature if temperature is not None else 1.0),
        "stream": True,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    else:
        local_budget = _local_vllm_completion_budget(model, stage, cfg)
        if local_budget is not None:
            params["max_tokens"] = local_budget
        elif not _is_local_vllm_qwen(model, stage):
            params["max_tokens"] = _default_max_tokens(model)
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if "presence_penalty" in profile:
        params["presence_penalty"] = profile["presence_penalty"]
    if extra_body:
        params["extra_body"] = extra_body
    if stop_tokens:
        params["stop"] = stop_tokens
    if json_schema is not None:
        if supports_json_schema(model):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "strict": False, "schema": json_schema},
            }
        else:
            params["response_format"] = {"type": "json_object"}

    prepared = _prepare_context_cache(
        params,
        cfg=cfg,
        stage=stage,
        model=model,
        role=context_cache_role,
        stable_prefix=context_cache_stable_prefix,
        dynamic_messages_override=None,
        reasoning_config={
            "profile": {key: profile[key] for key in sorted(profile)},
            "response_format": params.get("response_format"),
        },
        provider_override=_provider_override,
    )
    params = prepared.params
    if _vllm_cache_salt is not None:
        extra_body = dict(params.get("extra_body") or {})
        extra_body["cache_salt"] = _vllm_cache_salt
        params["extra_body"] = extra_body
    if prepared.telemetry is not None:
        params.setdefault("stream_options", {"include_usage": True})

    logger.info(f"generate messages: {len(messages)} turns", extra={"verbose": True})
    for attempt in range(max_retries):
        try:
            if prepared.telemetry is not None:
                prepared.telemetry.request_started()
            gate = prepared.request_gate() if prepared.request_gate else nullcontext()
            with gate as lease:
                stream = client.chat.completions.create(**params)
                full_text = ""
                stream_usage = None
                finish_reason = None
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    if chunk.choices and chunk.choices[0].delta.content:
                        if prepared.telemetry is not None:
                            prepared.telemetry.first_meaningful_delta()
                        marker = getattr(lease, "mark_warm", None)
                        if marker is not None:
                            marker()
                        full_text += chunk.choices[0].delta.content
                    if chunk.choices and getattr(chunk.choices[0], "finish_reason", None):
                        finish_reason = chunk.choices[0].finish_reason
                    if getattr(chunk, "usage", None) is not None:
                        stream_usage = chunk.usage
            if "</think>" in full_text:
                full_text = full_text[full_text.find("</think>") + 8:]
            logger.info(f"generate response: {full_text}", extra={"verbose": True})
            raw_response = {"usage": stream_usage}
            if last_chunk is not None and getattr(last_chunk, "provider", None):
                raw_response["provider"] = last_chunk.provider
            if last_chunk is not None:
                extra = getattr(last_chunk, "model_extra", None) or {}
                metrics = extra.get("metrics") if isinstance(extra, dict) else None
                if metrics is not None:
                    raw_response["metrics"] = metrics
            cache_usage, upstream_provider, server_metrics = _cache_response_details(
                prepared, raw_response
            )
            _finish_telemetry(
                prepared,
                usage=cache_usage,
                raw_usage=stream_usage,
                upstream_provider=upstream_provider,
                finish_reason=finish_reason,
                cost_usd=getattr(stream_usage, "cost", None),
                server_metrics=server_metrics,
            )
            return full_text
        except Exception as e:
            requested_max_tokens = params.get("max_tokens")
            safe_max_tokens = (
                _context_safe_max_tokens(e, int(requested_max_tokens))
                if requested_max_tokens is not None
                else None
            )
            if safe_max_tokens is not None:
                params["max_tokens"] = safe_max_tokens
                logger.warning(
                    "vLLM context rejection: retrying with max_tokens=%s",
                    safe_max_tokens,
                )
                continue
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                _finish_telemetry(prepared, error_type=type(e).__name__)
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
    return ""
