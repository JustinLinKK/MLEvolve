"""Dedicated, cache-aware client for an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import atexit
import os
import threading
from typing import Any

from openai import OpenAI

from config import Config
from .gemini import FunctionSpec
from . import openai as _openai

_CLIENTS: dict[str, tuple[str, OpenAI]] = {}
_CLIENTS_LOCK = threading.RLock()


def _stage(cfg: Config, model: str, stage_name: str | None):
    if stage_name in {"code", "feedback"}:
        return getattr(cfg.agent, stage_name)
    if getattr(cfg.agent.code, "model", None) == model:
        return cfg.agent.code
    return cfg.agent.feedback


def _cache_salt(cfg: Config) -> str | None:
    settings = getattr(cfg, "vllm_client", None)
    env_name = str(
        getattr(settings, "cache_salt_env", "MLEVOLVE_VLLM_CACHE_SALT") or ""
    ).strip()
    required = bool(getattr(settings, "require_cache_salt", True))
    salt = os.getenv(env_name, "") if env_name else ""
    if required and len(salt.encode("utf-8")) < 32:
        label = env_name or "vllm_client.cache_salt_env"
        raise ValueError(
            f"{label} must contain at least 32 bytes before a vLLM request is sent"
        )
    return salt or None


def _client_for(stage: Any) -> OpenAI:
    endpoint = str(getattr(stage, "base_url", "") or "").rstrip("/")
    if not endpoint:
        raise ValueError("A vLLM stage requires a base_url ending in /v1")
    api_key = str(getattr(stage, "api_key", "") or "EMPTY")
    with _CLIENTS_LOCK:
        pooled = _CLIENTS.get(endpoint)
        if pooled is not None and pooled[0] == api_key:
            return pooled[1]
        if pooled is not None:
            # Never reuse a transport carrying stale endpoint credentials.
            try:
                pooled[1].close()
            except Exception:
                pass
        client = OpenAI(api_key=api_key, base_url=endpoint, timeout=1200.0)
        _CLIENTS[endpoint] = (api_key, client)
        return client


def close_clients() -> None:
    with _CLIENTS_LOCK:
        clients = [client for _, client in _CLIENTS.values()]
        _CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


atexit.register(close_clients)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs: Any,
):
    if cfg is None:
        raise ValueError("cfg is required for vLLM backend")
    stage_name = model_kwargs.get("stage_name")
    model = str(model_kwargs.get("model", "") or "")
    stage = _stage(cfg, model, stage_name)
    salt = _cache_salt(cfg)
    return _openai.query(
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        cfg=cfg,
        _client=_client_for(stage),
        _provider_override="vllm",
        _vllm_cache_salt=salt,
        **model_kwargs,
    )


def generate(prompt: Any, cfg: Config, **kwargs: Any) -> str:
    stage = cfg.agent.code
    salt = _cache_salt(cfg)
    return _openai.generate(
        prompt=prompt,
        cfg=cfg,
        _client=_client_for(stage),
        _provider_override="vllm",
        _vllm_cache_salt=salt,
        **kwargs,
    )
