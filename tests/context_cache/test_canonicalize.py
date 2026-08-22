from __future__ import annotations

import math

import pytest

from context_cache.canonicalize import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    canonicalize,
    canonicalize_tools,
)


def test_reordered_records_and_volatile_metadata_have_the_same_hash() -> None:
    first = {
        "records": [
            {"stable_id": "b", "value": 2.0, "similarity_score": 0.5},
            {"stable_id": "a", "value": 1.0, "timestamp": "today"},
        ]
    }
    second = {
        "records": [
            {"value": 1.0, "stable_id": "a", "timestamp": "tomorrow"},
            {"value": 2.0, "stable_id": "b", "similarity_score": 0.9},
        ]
    }

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_semantic_one_byte_change_changes_hash() -> None:
    assert canonical_sha256({"text": "alpha"}) != canonical_sha256({"text": "alphb"})


def test_unicode_and_line_endings_are_normalized() -> None:
    assert canonical_json_bytes({"text": "e\u0301\r\nline"}) == canonical_json_bytes(
        {"text": "é\nline"}
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonicalize({"value": value})


def test_tool_schemas_are_sorted_by_stable_function_name() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "zeta", "parameters": {"b": 2, "a": 1}},
        },
        {"type": "function", "function": {"name": "alpha", "parameters": {}}},
    ]

    normalized = canonicalize_tools(tools)

    assert [tool["function"]["name"] for tool in normalized] == ["alpha", "zeta"]
    assert list(normalized[1]["function"]["parameters"]) == ["a", "b"]
