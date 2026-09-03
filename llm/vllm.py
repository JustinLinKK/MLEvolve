"""Dedicated, cache-aware client for an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import atexit
import json
import os
import threading
from types import SimpleNamespace
from typing import Any

import httpx

from config import Config
from .common import FunctionSpec
from . import openai as _openai

# The official OpenAI SDK imports its complete Responses API schema on startup.
# That is unnecessarily expensive for a local vLLM Chat Completions endpoint,
# especially when Python packages reside on a network volume.  Keep the symbol
# injectable for compatibility tests, but use the small HTTP transport by
# default.
OpenAI = None


class _ResponseObject(SimpleNamespace):
    """Attribute-shaped view of an OpenAI-compatible JSON response."""

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(**{key: self._convert(value) for key, value in raw.items()})
        self.model_extra = raw

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value


class _VLLMHttpClient:
    """Minimal persistent client for vLLM's OpenAI-compatible endpoint."""

    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def _request_parts(self, params: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        payload = dict(params)
        headers = dict(self._headers)
        headers.update(payload.pop("extra_headers", {}) or {})
        payload.update(payload.pop("extra_body", {}) or {})
        return headers, payload

    def create(self, **params: Any) -> Any:
        headers, payload = self._request_parts(params)
        if not payload.get("stream"):
            response = self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            return _ResponseObject(response.json())

        def stream_response():
            with self._client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        return
                    if data:
                        yield _ResponseObject(json.loads(data))

        return stream_response()

    def close(self) -> None:
        self._client.close()


_CLIENTS: dict[str, tuple[str, Any]] = {}
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


def _client_for(stage: Any) -> Any:
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
        client = (
            OpenAI(api_key=api_key, base_url=endpoint, timeout=1200.0)
            if OpenAI is not None
            else _VLLMHttpClient(api_key=api_key, base_url=endpoint, timeout=1200.0)
        )
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
