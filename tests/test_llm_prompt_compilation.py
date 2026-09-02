from llm.gemini import compile_prompt_to_md


def test_compile_prompt_to_md_renders_nested_scalar_evidence_values():
    prompt = {
        "Deterministic facts": {
            "validation": {"metric": 19.649798039376417, "is_valid": True},
            "measurements": {"runtime_seconds": None},
        }
    }

    rendered = compile_prompt_to_md(prompt)

    assert "metric" in rendered
    assert "19.649798039376417" in rendered
    assert "True" in rendered
    assert "None" in rendered
