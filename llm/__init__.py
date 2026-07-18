import logging
import time
from . import gemini as _gemini
from . import openai as _openai
from . import codex_cli as _codex_cli
from .gemini import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from config import Config
logger = logging.getLogger("MLEvolve")


def _emit_llm_timing(event_type: str, payload: dict) -> None:
    try:
        from utils.pipeline_logging import get_process_pipeline_logger

        pipeline_logger = get_process_pipeline_logger()
        if pipeline_logger is not None:
            pipeline_logger.emit(event_type, stage="llm", payload=payload)
    except Exception:
        return


def _stage_config_for_model(cfg: Config | None, model: str):
    if cfg is None:
        return None
    if getattr(cfg.agent.code, "model", None) == model:
        return cfg.agent.code
    if getattr(cfg.agent.feedback, "model", None) == model:
        return cfg.agent.feedback
    return None


def _provider(model: str, cfg: Config | None = None) -> str:
    """Select LLM backend from explicit provider first, then legacy model-name routing."""
    stage = _stage_config_for_model(cfg, model)
    provider = (getattr(stage, "provider", "") or "").lower()
    if provider in {"openrouter", "openai", "openai-compatible"}:
        return "openai"
    if provider in {"gemini", "google"}:
        return "gemini"
    if provider in {"codex", "codex-cli"}:
        return "codex"
    return "gemini" if (model or "").lower().startswith("gemini") else "openai"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gemini-3-pro-preview")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    started_at = time.time()
    provider = _provider(model, cfg)
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        if len(system_message) > 1000:
            logger.info(f"system: {system_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        if len(user_message) > 1000:
            logger.info(f"user: {user_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    try:
        if provider == "openai":
            output, req_time, in_tok_count, out_tok_count, info = _openai.query(
                system_message=system_message,
                user_message=user_message,
                func_spec=func_spec,
                cfg=cfg,
                **model_kwargs,
            )
        elif provider == "codex":
            output, req_time, in_tok_count, out_tok_count, info = _codex_cli.query(
                system_message=system_message,
                user_message=user_message,
                func_spec=func_spec,
                cfg=cfg,
                **model_kwargs,
            )
        else:
            output, req_time, in_tok_count, out_tok_count, info = _gemini.query(
                system_message=system_message,
                user_message=user_message,
                func_spec=func_spec,
                cfg=cfg,
                **model_kwargs,
            )
    except Exception as exc:
        _emit_llm_timing(
            "llm_call_failed",
            {
                "provider": provider,
                "model": model,
                "interface": "query",
                "wall_time_seconds": time.time() - started_at,
                "system_prompt_chars": len(system_message or ""),
                "user_prompt_chars": len(user_message or ""),
                "has_function_spec": func_spec is not None,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc)[:500],
            },
        )
        raise
    _emit_llm_timing(
        "llm_call_completed",
        {
            "provider": provider,
            "model": model,
            "interface": "query",
            "wall_time_seconds": time.time() - started_at,
            "request_time_seconds": req_time,
            "input_tokens": in_tok_count,
            "output_tokens": out_tok_count,
            "system_prompt_chars": len(system_message or ""),
            "user_prompt_chars": len(user_message or ""),
            "output_chars": len(str(output)) if output is not None else 0,
            "has_function_spec": func_spec is not None,
            "info": info,
        },
    )
    logger.info("---Query complete---", extra={"verbose": True})

    return output


def generate(
    prompt,
    cfg,
    temperature=None,
    max_tokens=None,
    stop_tokens=None,
    json_schema=None,
    max_retries=20,
    retry_delay=3,
):
    """Streaming text generation. Dispatches to Gemini or OpenAI-compatible backend by cfg.agent.code.model."""
    model = getattr(cfg.agent.code, "model", "") or ""
    provider = _provider(model, cfg)
    started_at = time.time()
    try:
        if provider == "openai":
            output = _openai.generate(
                prompt=prompt,
                cfg=cfg,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_tokens=stop_tokens,
                json_schema=json_schema,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        elif provider == "codex":
            output = _codex_cli.generate(
                prompt=prompt,
                cfg=cfg,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_tokens=stop_tokens,
                json_schema=json_schema,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        else:
            output = _gemini.generate(
                prompt=prompt,
                cfg=cfg,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_tokens=stop_tokens,
                json_schema=json_schema,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
    except Exception as exc:
        _emit_llm_timing(
            "llm_call_failed",
            {
                "provider": provider,
                "model": model,
                "interface": "generate",
                "wall_time_seconds": time.time() - started_at,
                "prompt_chars": len(compile_prompt_to_md(prompt) if not isinstance(prompt, str) else prompt),
                "max_tokens": max_tokens,
                "max_retries": max_retries,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc)[:500],
            },
        )
        raise
    _emit_llm_timing(
        "llm_call_completed",
        {
            "provider": provider,
            "model": model,
            "interface": "generate",
            "wall_time_seconds": time.time() - started_at,
            "prompt_chars": len(compile_prompt_to_md(prompt) if not isinstance(prompt, str) else prompt),
            "output_chars": len(output or ""),
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "has_json_schema": json_schema is not None,
        },
    )
    return output
