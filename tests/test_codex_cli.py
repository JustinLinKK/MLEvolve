from llm.codex_cli import _strict_response_schema


def test_strict_schema_replaces_stale_required_with_declared_properties():
    schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["obsolete"],
            },
        },
        "required": ["plan"],
    }

    normalized = _strict_response_schema(schema)

    assert normalized["additionalProperties"] is False
    assert normalized["required"] == ["analysis", "nested"]
    assert normalized["properties"]["nested"]["required"] == ["code"]
    assert normalized["properties"]["nested"]["additionalProperties"] is False
