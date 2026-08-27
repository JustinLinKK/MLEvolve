import logging
from . import gemini as _gemini
from . import openai as _openai
from . import vllm as _vllm
from .gemini import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from config import Config
logger = logging.getLogger("MLEvolve")


def _stage_config_for_model(
    cfg: Config | None, model: str, stage_name: str | None = None
):
    if cfg is None:
        return None
    if stage_name in {"code", "feedback"}:
        return getattr(cfg.agent, stage_name)
    if getattr(cfg.agent.code, "model", None) == model:
        return cfg.agent.code
    if getattr(cfg.agent.feedback, "model", None) == model:
        return cfg.agent.feedback
    return None


def _provider(
    model: str, cfg: Config | None = None, stage_name: str | None = None
) -> str:
    """Select LLM backend from explicit provider first, then legacy model-name routing."""
    stage = _stage_config_for_model(cfg, model, stage_name)
    provider = (getattr(stage, "provider", "") or "").lower()
    if provider == "vllm":
        return "vllm"
    if provider in {"openrouter", "openai", "openai-compatible", "deepseek"}:
        return "openai"
    if provider in {"gemini", "google"}:
        return "gemini"
    return "gemini" if (model or "").lower().startswith("gemini") else "openai"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    context_cache_role: str = "analysis",
    context_cache_stable_prefix: PromptType | None = None,
    context_cache_dynamic_system_message: PromptType | None = None,
    stage_name: str | None = None,
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

    cache_dynamic_system_message = (
        compile_prompt_to_md(context_cache_dynamic_system_message)
        if context_cache_dynamic_system_message
        else None
    )
    cache_stable_prefix = (
        compile_prompt_to_md(context_cache_stable_prefix)
        if context_cache_stable_prefix
        else None
    )

    if stage_name not in {None, "code", "feedback"}:
        raise ValueError("stage_name must be 'code', 'feedback', or None")
    provider = _provider(model, cfg, stage_name)
    if provider == "openai":
        output, req_time, in_tok_count, out_tok_count, info = _openai.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            context_cache_role=context_cache_role,
            context_cache_stable_prefix=cache_stable_prefix,
            context_cache_dynamic_system_message=cache_dynamic_system_message,
            stage_name=stage_name,
            **model_kwargs,
        )
    elif provider == "vllm":
        output, req_time, in_tok_count, out_tok_count, info = _vllm.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            context_cache_role=context_cache_role,
            context_cache_stable_prefix=cache_stable_prefix,
            context_cache_dynamic_system_message=cache_dynamic_system_message,
            stage_name=stage_name,
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
    context_cache_role="model_generator",
    context_cache_stable_prefix=None,
):
    """Streaming text generation. Dispatches to Gemini or OpenAI-compatible backend by cfg.agent.code.model."""
    model = getattr(cfg.agent.code, "model", "") or ""
    provider = _provider(model, cfg, "code")
    if provider == "openai":
        return _openai.generate(
            prompt=prompt,
            cfg=cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context_cache_role=context_cache_role,
            context_cache_stable_prefix=context_cache_stable_prefix,
        )
    if provider == "vllm":
        return _vllm.generate(
            prompt=prompt,
            cfg=cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context_cache_role=context_cache_role,
            context_cache_stable_prefix=context_cache_stable_prefix,
        )
    return _gemini.generate(
        prompt=prompt,
        cfg=cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        json_schema=json_schema,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
