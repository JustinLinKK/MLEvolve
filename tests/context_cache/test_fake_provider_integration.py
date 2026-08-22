from __future__ import annotations

from context_cache.providers.openrouter import OpenRouterCacheAdapter


class FakeCachingProvider:
    """Small stateful fake for cold write, warm read, eviction, and omitted metrics."""

    def __init__(self) -> None:
        self.cached = False
        self.omit_metrics = False

    def call(self, tokens: int = 100):
        details = None
        if not self.omit_metrics:
            details = (
                {"cached_tokens": tokens, "cache_write_tokens": 0}
                if self.cached
                else {"cached_tokens": 0, "cache_write_tokens": tokens}
            )
        self.cached = True
        usage = {"prompt_tokens": tokens, "completion_tokens": 2}
        if details is not None:
            usage["prompt_tokens_details"] = details
        return {"provider": "fake-upstream", "usage": usage}

    def evict(self) -> None:
        self.cached = False


def test_fake_provider_cold_warm_evict_and_missing_metrics() -> None:
    provider = FakeCachingProvider()
    adapter = OpenRouterCacheAdapter()

    cold = adapter.extract_cache_usage(provider.call())
    warm = adapter.extract_cache_usage(provider.call())
    provider.evict()
    evicted = adapter.extract_cache_usage(provider.call())
    provider.omit_metrics = True
    missing = adapter.extract_cache_usage(provider.call())

    assert (cold.cache_read_tokens, cold.cache_write_tokens) == (0, 100)
    assert (warm.cache_read_tokens, warm.cache_write_tokens) == (100, 0)
    assert (evicted.cache_read_tokens, evicted.cache_write_tokens) == (0, 100)
    assert missing.cache_read_tokens is None
    assert missing.cache_write_tokens is None
    assert adapter.extract_upstream_provider(provider.call()) == "fake-upstream"
