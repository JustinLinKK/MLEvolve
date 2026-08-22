"""Opt-in live provider checks.

Set MLEVOLVE_RUN_PROVIDER_CACHE_INTEGRATION=1 plus the provider-specific API key
and model variables. These tests are intentionally skipped in normal CI because
they issue billable requests.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from context_cache.config import ContextCacheSettings
from context_cache.telemetry import CacheTelemetryStore
from llm import openai as backend


def _enabled() -> bool:
    return os.getenv("MLEVOLVE_RUN_PROVIDER_CACHE_INTEGRATION") == "1"


def _run_calls(
    tmp_path: Path,
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    repetitions: int,
):
    if not _enabled():
        pytest.skip("live context-cache integration tests are opt-in")
    if not api_key or not model:
        pytest.skip(f"{provider} integration credentials/model are not configured")
    stage = SimpleNamespace(
        model=model,
        api_key=api_key,
        base_url=base_url or "",
        provider=provider,
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=stage, feedback=stage),
        context_cache=ContextCacheSettings(
            enabled=True,
            local_pack_cache_enabled=True,
            provider_prompt_cache_enabled=True,
            cache_dir=str(tmp_path),
            telemetry=True,
        ),
        exp_name=f"live-{provider}",
    )
    stable = "Stable integration-test reference.\n" * 1400
    for index in range(repetitions):
        backend.query(
            stable,
            f"Return the word OK. Trial {index}.",
            cfg=cfg,
            model=model,
            temperature=0,
            max_tokens=8,
            context_cache_role="reviewer",
            context_cache_stable_prefix=stable,
        )
    rows = CacheTelemetryStore(tmp_path / "cache-registry.sqlite3").rows()
    assert len(rows) == repetitions
    return rows


def test_openrouter_live_cache_metrics_and_upstream(tmp_path: Path) -> None:
    rows = _run_calls(
        tmp_path,
        provider="openrouter",
        model=os.getenv("MLEVOLVE_OPENROUTER_INTEGRATION_MODEL", ""),
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        repetitions=2,
    )
    assert any(
        row["upstream_provider"] or row["cache_read_tokens"] is not None for row in rows
    )


def test_openai_live_request_schema(tmp_path: Path) -> None:
    rows = _run_calls(
        tmp_path,
        provider="openai",
        model=os.getenv("MLEVOLVE_OPENAI_INTEGRATION_MODEL", ""),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=None,
        repetitions=2,
    )
    assert all(row["provider"] == "openai" for row in rows)


def test_deepseek_live_repeats_for_observable_metrics(tmp_path: Path) -> None:
    rows = _run_calls(
        tmp_path,
        provider="deepseek",
        model=os.getenv("MLEVOLVE_DEEPSEEK_INTEGRATION_MODEL", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        repetitions=3,
    )
    assert len(rows) == 3
