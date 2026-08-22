"""Anthropic extension point; direct SDK wiring is intentionally deferred."""

from .base import NoOpCacheAdapter


class AnthropicCacheAdapter(NoOpCacheAdapter):
    def __init__(self, **kwargs):
        super().__init__(provider="anthropic", **kwargs)
