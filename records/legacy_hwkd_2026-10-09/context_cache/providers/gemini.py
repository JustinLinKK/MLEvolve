"""Gemini extension point; explicit cache-resource management is deferred."""

from .base import NoOpCacheAdapter


class GeminiCacheAdapter(NoOpCacheAdapter):
    def __init__(self, **kwargs):
        super().__init__(provider="gemini", **kwargs)

