from __future__ import annotations

from types import SimpleNamespace

from agents.prompts import impl_guideline


def _agent(*, exec_timeout: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        acfg=SimpleNamespace(
            time_limit=60,
            steps=5,
        ),
        cfg=SimpleNamespace(
            exec=SimpleNamespace(timeout=exec_timeout),
            pretrain_model_dir="",
        ),
        current_step=2,
        start_time=100.0,
    )


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


def test_impl_guideline_requires_extracted_zip_layout_checks(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "archives may store files directly at the root or inside a nested folder" in text
    assert "`rglob`/fallback checks" in text


def test_impl_guideline_requires_low_precision_export_boundary(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "Low-precision metric/export boundary" in text
    assert "tensor.detach().to(torch.float32).cpu().numpy()" in text


def test_impl_guideline_contains_complete_scheduler_script_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "top-level integer literal such as `batch_size = 32`" in text
    assert "Never pass `batch_size=`" in text
    assert "session.register_training_state" in text
    assert "progress = session.restore_if_present()" in text
    assert "same accumulation condition as the optimizer step" in text
    assert "batch_index=batch_index" in text
    assert "never `batch_idx`" in text
    assert "must use a checkpointable PyTorch model and optimizer" in text
