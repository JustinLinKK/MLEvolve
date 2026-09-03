"""Provider-neutral prompt and function-call types.

Kept separate from provider SDK modules so a local vLLM run does not import
unrelated remote-provider clients at process startup.
"""

from __future__ import annotations

from dataclasses import dataclass

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    if isinstance(prompt, list):
        return "\n".join([f"- {str(item).strip()}" for item in prompt] + ["\n"])
    if not isinstance(prompt, dict):
        return str(prompt).strip() + "\n"

    out = []
    header_prefix = "#" * _header_depth
    for key, value in prompt.items():
        out.append(f"{header_prefix} {key}\n")
        out.append(compile_prompt_to_md(value, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict
    description: str

    def __post_init__(self):
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {"type": "function", "function": {"name": self.name}}
