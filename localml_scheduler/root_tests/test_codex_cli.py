from llm.common import FunctionSpec
from llm.codex_cli import _strict_response_schema, _supports_strict_schema, query


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


def test_dynamic_property_map_skips_strict_schema_mode():
    schema = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            }
        },
        "required": ["plan"],
    }

    assert _supports_strict_schema(schema) is False


def test_dynamic_schema_query_names_required_root_fields(monkeypatch):
    captured = {}

    def fake_run(prompt, **_kwargs):
        captured["prompt"] = prompt
        return '{"plan": {"alpha": "test"}}', 0.0, 0, 0, {}

    monkeypatch.setattr("llm.codex_cli._run", fake_run)
    spec = FunctionSpec(
        name="dynamic_plan",
        json_schema={
            "type": "object",
            "properties": {"plan": {"type": "object", "additionalProperties": {"type": "string"}}},
            "required": ["plan"],
        },
        description="",
    )

    output, *_ = query(None, "Return a plan.", func_spec=spec, model="gpt-5.6-terra")

    assert output["plan"] == {"alpha": "test"}
    assert 'top-level required fields: ["plan"]' in captured["prompt"]
