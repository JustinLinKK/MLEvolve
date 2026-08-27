from __future__ import annotations

import json
from pathlib import Path

from context_cache.assembler import DeterministicPromptAssembler

SNAPSHOT = Path(__file__).parent / "snapshots" / "role_prompts.json"


def test_all_sanitized_role_snapshots_keep_dynamic_state_after_prefix() -> None:
    snapshots = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert set(snapshots) == {
        "model_generator",
        "analysis",
        "result_parser",
        "reviewer",
        "supervisor",
    }
    assembler = DeterministicPromptAssembler(verify_prefix=True)
    for role, snapshot in snapshots.items():
        assembled = assembler.assemble(
            dynamic_messages=snapshot["dynamic_messages"],
            tools=[],
            common_pack={"role": "common", "content": {"sections": []}},
            role_pack={"role": role, "content": {"sections": []}},
            stable_system_instructions=snapshot["stable_system_instructions"],
        )
        dynamic_marker = snapshot["dynamic_messages"][0]["content"]
        assert dynamic_marker not in assembled.stable_prefix
        assert assembled.messages[-1]["content"] == dynamic_marker

