from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from context_cache.config import ContextCacheSettings, environment_overrides
from context_cache.coordinator import (
    ContextCacheCoordinator,
    SingleFlight,
    WarmFirstGate,
    prepare_llm_request,
)
from context_cache.models import CacheFamily, NormalizedCacheUsage
from context_cache.telemetry import CacheTelemetryStore, RequestTelemetry


def _cfg(tmp_path: Path, **overrides):
    values = {
        "enabled": True,
        "cache_dir": str(tmp_path),
        "local_pack_cache_enabled": True,
        "provider_prompt_cache_enabled": False,
        "telemetry": True,
    }
    values.update(overrides)
    return SimpleNamespace(
        context_cache=ContextCacheSettings(**values), exp_name=f"run-{tmp_path.name}"
    )


def test_global_kill_switch_returns_baseline_request_without_runtime_files(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, enabled=False)
    params = {
        "model": "x",
        "messages": [{"role": "user", "content": "exact"}],
        "temperature": 1.0,
    }

    prepared = prepare_llm_request(
        params,
        cfg=cfg,
        provider="openrouter",
        model="x",
        agent_role="reviewer",
    )

    assert prepared.params == params
    assert prepared.active is False
    assert list(tmp_path.iterdir()) == []


def test_all_roles_use_same_shared_assembler_and_freeze_packs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    family_ids = set()
    for role in (
        "model_generator",
        "analysis",
        "result_parser",
        "reviewer",
        "supervisor",
    ):
        prepared = prepare_llm_request(
            {"model": "x", "messages": [{"role": "user", "content": role}]},
            cfg=cfg,
            provider="openrouter",
            model="x",
            agent_role=role,
        )
        assert prepared.active is True
        assert prepared.assembled is not None
        assert prepared.assembled.dynamic_suffix[0]["content"] == role
        family_ids.add(prepared.family.id)

    # Role packs add no prompt text, but their semantic contracts split role families.
    assert len(family_ids) == 5
    with CacheTelemetryStore(
        tmp_path / "cache-registry.sqlite3"
    )._connect() as connection:
        frozen_roles = {
            row[0] for row in connection.execute("SELECT role FROM run_pack_refs")
        }
    assert frozen_roles == {
        "common",
        "model_generator",
        "analysis",
        "result_parser",
        "reviewer",
        "supervisor",
    }


def test_extracting_an_existing_system_prefix_preserves_enabled_message_semantics(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    params = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "stable instructions"},
            {"role": "user", "content": "dynamic task"},
        ],
    }

    prepared = prepare_llm_request(
        params,
        cfg=cfg,
        provider="openrouter",
        model="x",
        agent_role="supervisor",
        stable_system_instructions="stable instructions",
    )

    assert prepared.params["messages"] == params["messages"]


def test_dynamic_override_never_enters_stable_prefix(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    prepared = prepare_llm_request(
        {
            "model": "x",
            "messages": [{"role": "system", "content": "combined prompt"}],
        },
        cfg=cfg,
        provider="openrouter",
        model="x",
        agent_role="reviewer",
        stable_system_instructions="stable reviewer rules",
        dynamic_messages_override=[
            {"role": "system", "content": "candidate code and current trace"}
        ],
    )

    assert "stable reviewer rules" in prepared.assembled.stable_prefix
    assert "candidate code" not in prepared.assembled.stable_prefix
    assert (
        prepared.params["messages"][-1]["content"] == "candidate code and current trace"
    )


def test_provider_cache_and_local_cache_are_independent(tmp_path: Path) -> None:
    params = {
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "dynamic"}],
    }
    local_only = prepare_llm_request(
        params,
        cfg=_cfg(tmp_path / "local", provider_prompt_cache_enabled=False),
        provider="openrouter",
        model=params["model"],
        agent_role="reviewer",
    )
    provider_only = prepare_llm_request(
        params,
        cfg=_cfg(
            tmp_path / "provider",
            local_pack_cache_enabled=False,
            provider_prompt_cache_enabled=True,
        ),
        provider="openrouter",
        model=params["model"],
        agent_role="reviewer",
        stable_system_instructions="stable reviewer instructions",
    )

    assert "extra_body" not in local_only.params
    assert provider_only.params["extra_body"]["session_id"].startswith("mlevolve:")
    assert provider_only.local_pack_cache_hit is False


def test_provider_role_allowlist_supports_staged_rollout(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        provider_prompt_cache_enabled=True,
        provider_prompt_cache_roles=["reviewer"],
    )
    params = {
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "dynamic"}],
    }
    analysis = prepare_llm_request(
        params,
        cfg=cfg,
        provider="openrouter",
        model=params["model"],
        agent_role="analysis",
        stable_system_instructions="stable",
    )
    reviewer = prepare_llm_request(
        params,
        cfg=cfg,
        provider="openrouter",
        model=params["model"],
        agent_role="reviewer",
        stable_system_instructions="stable",
    )

    assert "extra_body" not in analysis.params
    assert reviewer.params["extra_body"]["session_id"].startswith("mlevolve:")


def test_adapter_failure_keeps_complete_dynamic_request(
    monkeypatch, tmp_path: Path
) -> None:
    class BrokenAdapter:
        def apply_cache_policy(self, *args, **kwargs):
            raise RuntimeError("adapter broke")

    monkeypatch.setattr(
        "context_cache.coordinator.adapter_for", lambda *args, **kwargs: BrokenAdapter()
    )
    coordinator = ContextCacheCoordinator(
        ContextCacheSettings(
            enabled=True,
            cache_dir=str(tmp_path),
            provider_prompt_cache_enabled=True,
        )
    )
    original = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "dynamic must survive"},
        ],
    }

    prepared = coordinator.prepare_request(
        original,
        provider="openrouter",
        model="x",
        agent_role="reviewer",
        stable_system_instructions="stable",
    )

    assert prepared.active is False
    assert prepared.params == original
    assert prepared.fallback_reason == "RuntimeError"


def test_request_timing_math_and_null_usage(monkeypatch, tmp_path: Path) -> None:
    values = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr("context_cache.telemetry.time.monotonic", lambda: next(values))
    store = CacheTelemetryStore(tmp_path / "registry.sqlite3")
    family = CacheFamily("openai", "gpt", "c", "r", "t", "reason")
    timer = RequestTelemetry(
        store,
        run_id="run",
        provider="openai",
        api_family="chat_completions",
        model="gpt",
        agent_role="analysis",
        family=family,
        stable_prefix_hash="prefix",
        local_pack_cache_hit=True,
        expected_stable_prefix_tokens=100,
        db_retrieval_ms=None,
        pack_build_ms=None,
    )
    timer.pack_ready()
    timer.request_started()
    timer.first_meaningful_delta()
    event = timer.finish(usage=NormalizedCacheUsage(cache_read_tokens=None))

    assert event is not None
    assert event.request_prepare_ms == 1000
    assert event.ttft_ms == 1000
    assert event.total_request_ms == 2000
    assert event.end_to_end_ms == 4000
    assert event.cache_hit_ratio is None
    assert timer.finish() is None
    assert len(store.rows()) == 1


def test_singleflight_failure_releases_waiters() -> None:
    flight = SingleFlight()
    leader_started = threading.Event()
    release_leader = threading.Event()
    errors = []

    def failing():
        leader_started.set()
        release_leader.wait(2)
        raise RuntimeError("leader failed")

    def call():
        try:
            flight.run(("key",), failing, timeout=2)
        except RuntimeError as exc:
            errors.append(str(exc))

    first = threading.Thread(target=call)
    second = threading.Thread(target=call)
    first.start()
    assert leader_started.wait(1)
    second.start()
    release_leader.set()
    first.join(3)
    second.join(3)

    assert errors == ["leader failed", "leader failed"]
    assert flight.run(("key",), lambda: "recovered") == "recovered"


def test_warm_first_gate_releases_fanout_after_leader_completes() -> None:
    gate = WarmFirstGate()
    leader_inside = threading.Event()
    release = threading.Event()
    order = []

    def leader():
        with gate.hold(("family",)):
            order.append("leader-start")
            leader_inside.set()
            release.wait(2)
            order.append("leader-end")

    def follower():
        assert leader_inside.wait(1)
        with gate.hold(("family",)):
            order.append("follower")

    first = threading.Thread(target=leader)
    second = threading.Thread(target=follower)
    first.start()
    second.start()
    assert leader_inside.wait(1)
    release.set()
    first.join(2)
    second.join(2)

    assert order == ["leader-start", "leader-end", "follower"]


def test_warm_first_stream_leader_releases_at_first_token() -> None:
    gate = WarmFirstGate()
    leader_inside = threading.Event()
    token_sent = threading.Event()
    finish_leader = threading.Event()
    order = []

    def leader():
        with gate.hold(("stream-family",)) as lease:
            order.append("leader-start")
            leader_inside.set()
            lease.mark_warm()
            token_sent.set()
            finish_leader.wait(2)
            order.append("leader-end")

    def follower():
        assert leader_inside.wait(1)
        with gate.hold(("stream-family",)):
            order.append("follower")
            finish_leader.set()

    first = threading.Thread(target=leader)
    second = threading.Thread(target=follower)
    first.start()
    second.start()
    assert token_sent.wait(1)
    first.join(2)
    second.join(2)
    assert order == ["leader-start", "follower", "leader-end"]


def test_warm_first_failure_releases_without_marking_family_warm() -> None:
    gate = WarmFirstGate()
    follower_released = threading.Event()

    def follower():
        with gate.hold(("failed-family",)):
            follower_released.set()

    with pytest.raises(RuntimeError):
        with gate.hold(("failed-family",)):
            thread = threading.Thread(target=follower)
            thread.start()
            raise RuntimeError("leader failed")
    assert follower_released.wait(1)
    thread.join(1)
    assert ("failed-family",) not in gate._warmed


def test_opt_in_prompt_snapshot_is_sanitized(tmp_path: Path) -> None:
    store = CacheTelemetryStore(tmp_path / "registry.sqlite3")
    timer = RequestTelemetry(
        store,
        run_id="run",
        provider="openai",
        api_family="chat_completions",
        model="gpt",
        agent_role="reviewer",
        family=None,
        stable_prefix_hash=None,
        local_pack_cache_hit=None,
        expected_stable_prefix_tokens=None,
        db_retrieval_ms=None,
        pack_build_ms=None,
        prompt_snapshot={
            "api_key": "top-secret",
            "cache_salt": "never-record-this-salt",
            "messages": [{"content": "Authorization: Bearer abcdefghijklmnop"}],
        },
    )
    timer.finish()

    snapshot = store.rows()[0]["prompt_snapshot_json"]
    assert "top-secret" not in snapshot
    assert "never-record-this-salt" not in snapshot
    assert "abcdefghijklmnop" not in snapshot
    assert snapshot.count("<redacted>") == 3


def test_environment_validation_rejects_invalid_policy_and_boolean() -> None:
    with pytest.raises(ValueError, match="policy"):
        ContextCacheSettings(policy="sometimes")
    with pytest.raises(ValueError, match="boolean"):
        environment_overrides({"MLEVOLVE_CONTEXT_CACHE_ENABLED": "perhaps"})
    assert environment_overrides(
        {"MLEVOLVE_PROVIDER_PROMPT_CACHE_ROLES": "reviewer, supervisor"}
    )["provider_prompt_cache_roles"] == ["reviewer", "supervisor"]
