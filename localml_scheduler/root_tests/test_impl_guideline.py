from __future__ import annotations

from types import SimpleNamespace

from agents.prompts import impl_guideline


def _agent(
    *,
    exec_timeout: int | None,
    scheduler: bool = False,
    preflight: bool = True,
    prediction_mode: str = "branch_profile",
    prompt_compatibility_profile: str = "current",
    preflight_prompt_contract: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        acfg=SimpleNamespace(
            time_limit=60,
            steps=5,
            prompt_compatibility_profile=prompt_compatibility_profile,
        ),
        cfg=SimpleNamespace(
            exec=SimpleNamespace(timeout=exec_timeout),
            pretrain_model_dir="",
            scheduler=SimpleNamespace(
                enabled=scheduler,
                settings={"prediction": {"mode": prediction_mode}},
            ),
            preflight=SimpleNamespace(
                enabled=preflight,
                prompt_contract_enabled=preflight_prompt_contract,
            ),
        ),
        current_step=2,
        start_time=100.0,
    )


def test_baseline_guideline_omits_scheduler_and_preflight_contracts(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(
        _agent(exec_timeout=5, scheduler=False, preflight=False)
    )

    text = "\n".join(guideline["Implementation guideline"])
    assert "Scheduler Model Family Contract" not in text
    assert "CPU Model-Preflight Adapter Contract" not in text
    assert "CandidateAdapter" not in text


def test_preflight_only_guideline_omits_scheduler_contract(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(
        _agent(exec_timeout=5, scheduler=False, preflight=True)
    )

    text = "\n".join(guideline["Implementation guideline"])
    assert "CPU Model-Preflight Adapter Contract" in text
    assert "CandidateAdapter" in text
    assert "Scheduler Model Family Contract" not in text


def test_post_generation_preflight_guideline_matches_baseline(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(
        _agent(
            exec_timeout=5,
            scheduler=False,
            preflight=True,
            preflight_prompt_contract=False,
        )
    )

    text = "\n".join(guideline["Implementation guideline"])
    assert "CPU Model-Preflight Adapter Contract" not in text
    assert "CandidateAdapter" not in text
    assert "Scheduler Model Family Contract" not in text


def test_frozen_profile_uses_baseline_budget_surface(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 104.0)
    agent = _agent(
        exec_timeout=5,
        scheduler=False,
        preflight=True,
        preflight_prompt_contract=False,
        prompt_compatibility_profile="original_mlevolve_20260924",
    )
    agent.current_step = 0

    guideline = impl_guideline.get_impl_guideline_from_agent(agent)

    first = guideline["Implementation guideline"][0]
    assert "Time left ≈ 0h 0m 56s" in first
    assert "Steps left = 50" in first


def test_ml_predictor_guideline_keeps_adapter_without_preflight(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(
        _agent(
            exec_timeout=5,
            scheduler=True,
            preflight=False,
            prediction_mode="ml_predictor",
        )
    )

    text = "\n".join(guideline["Implementation guideline"])
    assert "Scheduler Model Family Contract" in text
    assert "CPU Model-Preflight Adapter Contract" in text


def test_impl_guideline_uses_remaining_time_when_exec_timeout_is_none(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=None))

    text = "\n".join(guideline["Implementation guideline"])
    assert "Time left" in text
    assert "Steps left = 3" in text
    assert "Max execution time per run = 50 seconds" in text


def test_impl_guideline_caps_configured_exec_timeout_by_remaining_time(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "Max execution time per run = 5 seconds" in text


def test_impl_guideline_requires_partial_preflight_context_support(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "caller-supplied `context` as a partial mapping" in text
    assert "context mutations do not persist" in text
    assert "real criterion" in text
    assert "merge it over adapter defaults" in text


def test_impl_guideline_requires_adapter_build_model_to_return_module(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "build_model(context) MUST return the real torch.nn.Module" in text


def test_petfinder_guideline_names_its_multimodal_preflight_batch(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)
    agent = _agent(exec_timeout=5)
    agent.cfg.exp_id = "petfinder-pawpularity-score"

    guideline = impl_guideline.get_impl_guideline_from_agent(agent)

    text = "\n".join(guideline["Implementation guideline"])
    assert "image [B, 3, 256, 256]" in text
    assert "tabular [B, 12]" in text
    assert "Subject Focus" in text


def test_original_mlevolve_profile_preserves_baseline_core_plus_preflight(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(
        _agent(
            exec_timeout=5,
            scheduler=False,
            preflight=True,
            prompt_compatibility_profile="original_mlevolve_20260924",
        )
    )

    text = "\n".join(guideline["Implementation guideline"])
    assert "Preserve Task Semantics in Local Validation" in text
    assert "Make Validation Auditable" in text
    assert "numpy, pandas, sklearn, torch, transformers, timm" in text
    assert "CPU Model-Preflight Adapter Contract" in text
    assert "Scheduler Model Family Contract" not in text
    assert "Guaranteed installed packages" not in text
