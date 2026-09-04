from __future__ import annotations

from types import SimpleNamespace

from agents.prompts import impl_guideline
from agents.prompts.shared import get_internet_clarification


def test_generated_code_prompt_requires_offline_weight_resolution(monkeypatch) -> None:
    """Generation guidance must agree with the network-disabled worker."""

    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(time_limit=60, steps=5),
        cfg=SimpleNamespace(
            exec=SimpleNamespace(timeout=5),
            pretrain_model_dir="/models/approved",
        ),
        current_step=2,
        start_time=100.0,
    )

    implementation_text = "\n".join(
        impl_guideline.get_impl_guideline_from_agent(agent)["Implementation guideline"]
    )
    review_text = "\n".join(get_internet_clarification("/models/approved"))

    assert "generated code runs offline" in implementation_text.lower()
    assert "do not download weights" in implementation_text.lower()
    assert "generated candidate runs offline" in review_text.lower()
    assert "do not use torch.hub.load" in review_text.lower()
    assert "all standard ml libraries and pretrained models are available" not in review_text.lower()
