"""Provider adapter registry."""

from .base import NoOpCacheAdapter, ProviderCacheAdapter
from .deepseek import DeepSeekCacheAdapter
from .openai import OpenAICacheAdapter
from .openrouter import OpenRouterCacheAdapter
from .vllm import VLLMCacheAdapter


def adapter_for(provider: str, **options) -> ProviderCacheAdapter:
    normalized = str(provider or "").strip().lower().replace("_", "-")
    if normalized == "openrouter":
        return OpenRouterCacheAdapter(**options)
    if normalized == "openai":
        return OpenAICacheAdapter(**options)
    if normalized == "deepseek":
        return DeepSeekCacheAdapter(**options)
    if normalized == "vllm":
        return VLLMCacheAdapter(**options)
    return NoOpCacheAdapter(provider=normalized or "unknown")


__all__ = [
    "DeepSeekCacheAdapter",
    "NoOpCacheAdapter",
    "OpenAICacheAdapter",
    "OpenRouterCacheAdapter",
    "ProviderCacheAdapter",
    "VLLMCacheAdapter",
    "adapter_for",
]
