from __future__ import annotations

from context_cache.assembler import DeterministicPromptAssembler
from context_cache.models import CacheFamily


def _pack(role: str, text: str):
    return {
        "role": role,
        "content": {"sections": [{"stable_id": "rules", "content": text}]},
    }


def test_dynamic_task_does_not_change_prefix_hash() -> None:
    assembler = DeterministicPromptAssembler(verify_prefix=True)
    common = _pack("common", "shared facts")
    role = _pack("reviewer", "review rules")
    tools = [{"type": "function", "function": {"name": "review", "parameters": {}}}]

    first = assembler.assemble(
        dynamic_messages=[{"role": "user", "content": "candidate A"}],
        tools=tools,
        common_pack=common,
        role_pack=role,
        reasoning_config={"effort": "low"},
    )
    second = assembler.assemble(
        dynamic_messages=[{"role": "user", "content": "candidate B"}],
        tools=tools,
        common_pack=common,
        role_pack=role,
        reasoning_config={"effort": "low"},
    )

    assert first.stable_prefix_hash == second.stable_prefix_hash
    assert first.messages[0]["role"] == "system"
    assert first.dynamic_suffix != second.dynamic_suffix
    assert "candidate A" not in first.stable_prefix


def test_volatile_pack_fields_are_absent_before_breakpoint() -> None:
    assembled = DeterministicPromptAssembler().assemble(
        dynamic_messages=[{"role": "user", "content": "trace: dynamic"}],
        tools=[],
        common_pack={
            "role": "common",
            "content": {"timestamp": "volatile", "sections": []},
        },
        role_pack={
            "role": "analysis",
            "content": {"run_id": "volatile", "sections": []},
        },
    )

    assert "volatile" not in assembled.stable_prefix
    assert "trace: dynamic" in assembled.dynamic_suffix[0]["content"]


def test_cache_family_invalidation_covers_all_routing_inputs() -> None:
    base = dict(
        provider="openrouter",
        model="openai/gpt-5.6",
        common_pack_hash="common",
        role_pack_hash="role",
        tool_schema_hash="tools",
        reasoning_config_hash="reasoning",
        api_family="chat_completions",
        upstream_constraints_hash="upstream",
        system_instructions_hash="system",
    )
    baseline = CacheFamily(**base).id

    for field in base:
        changed = dict(base)
        changed[field] += "-changed"
        assert CacheFamily(**changed).id != baseline
