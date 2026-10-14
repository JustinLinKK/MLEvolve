"""Optional cold/warm retrieval latency benchmark."""

from __future__ import annotations

import statistics
import time
from typing import Any, Mapping

from .client import LessonProfileClient


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def benchmark_retrieval(
    client: LessonProfileClient,
    *,
    identity: Mapping[str, Any],
    agent_role: str = "improve",
    query: str = "benchmark",
    iterations: int = 20,
) -> dict[str, Any]:
    """Report cold and Redis-warm p50/p95 plus deterministic I/O counters."""
    iterations = max(2, int(iterations))
    profile_key = str(identity["profile_key"])
    cold_ms: list[float] = []
    for _ in range(iterations):
        client.invalidate_profile(profile_key)
        started = time.perf_counter()
        client.get_family_hardware_profile(
            agent_role=agent_role,
            identity=identity,
            code=query,
        )
        cold_ms.append((time.perf_counter() - started) * 1000.0)

    client.get_family_hardware_profile(agent_role=agent_role, identity=identity, code=query)
    counters = {"sqlite": 0, "embedding": 0, "qdrant": 0}
    originals = {
        "profile": client.registry.profile,
        "active_revision": client.registry.active_revision,
        "compatible": client.registry.find_compatible_profiles,
        "vector_search": client.vector_store.search,
    }

    def count(name, function):
        def wrapped(*args, **kwargs):
            counters[name] += 1
            return function(*args, **kwargs)
        return wrapped

    client.registry.profile = count("sqlite", originals["profile"])
    client.registry.active_revision = count("sqlite", originals["active_revision"])
    client.registry.find_compatible_profiles = count("sqlite", originals["compatible"])
    client.vector_store.search = count("qdrant", originals["vector_search"])
    embedder = client.vector_store._embedding_model
    original_encode = getattr(embedder, "encode", None) if embedder is not None else None
    if original_encode is not None:
        embedder.encode = count("embedding", original_encode)
    warm_ms: list[float] = []
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            client.get_family_hardware_profile(
                agent_role=agent_role,
                identity=identity,
                code=query,
            )
            warm_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        client.registry.profile = originals["profile"]
        client.registry.active_revision = originals["active_revision"]
        client.registry.find_compatible_profiles = originals["compatible"]
        client.vector_store.search = originals["vector_search"]
        if original_encode is not None:
            embedder.encode = original_encode
    return {
        "iterations": iterations,
        "cold_sqlite_qdrant_ms": {
            "p50": _percentile(cold_ms, 0.50),
            "p95": _percentile(cold_ms, 0.95),
            "mean": statistics.fmean(cold_ms),
        },
        "warm_redis_ms": {
            "p50": _percentile(warm_ms, 0.50),
            "p95": _percentile(warm_ms, 0.95),
            "mean": statistics.fmean(warm_ms),
        },
        "warm_io_calls": counters,
        "acceptance": all(value == 0 for value in counters.values()),
    }
