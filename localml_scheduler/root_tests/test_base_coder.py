from __future__ import annotations

from types import SimpleNamespace

from agents.coder import base_coder


def test_fixed_seed_retry_changes_the_prompt_after_missing_code(monkeypatch) -> None:
    """A deterministic local LLM must not receive an identical retry prompt."""

    prompts: list[str] = []
    responses = iter(
        [
            "I will first inspect the dataset.",
            "Plan: train a small model.\n```python\nvalue = 42\n```",
        ]
    )

    def generate(*, prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(base_coder, "generate", generate)
    agent = SimpleNamespace(acfg=SimpleNamespace(code=SimpleNamespace(temp=0.7)), cfg=object())

    plan, code = base_coder.plan_and_code_query(agent, "original task", retries=2)

    assert plan == "Plan: train a small model."
    assert code.strip() == "value = 42"
    assert prompts[0] == "original task"
    assert prompts[1] != prompts[0]
    assert "previous response did not contain a complete Python markdown code block" in prompts[1]
