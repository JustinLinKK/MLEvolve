"""Deterministic stable-prefix and dynamic-suffix prompt assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import re
from typing import Any

from .canonicalize import (
    canonical_json_bytes,
    canonical_sha256,
    canonicalize,
    canonicalize_tools,
)
from .models import AssembledPrompt

FORBIDDEN_STABLE_KEYS = frozenset(
    {
        "timestamp",
        "compiled_at",
        "created_at",
        "updated_at",
        "run_id",
        "request_id",
        "trace_id",
        "similarity_score",
        "execution_trace",
        "queue_state",
        "perfseer_prediction",
    }
)


class UnstablePrefixError(ValueError):
    pass


def _render_content(content: Mapping[str, Any], heading: str) -> str:
    sections = content.get("sections")
    if not sections:
        remainder = {
            key: value
            for key, value in content.items()
            if key not in {"sections", "pack_contract"}
        }
        if not remainder:
            return ""
        body = canonical_json_bytes(remainder).decode("utf-8")
        return f"## {heading}\n{body}"
    rendered: list[str] = [f"## {heading}"]
    for index, section in enumerate(sections, 1):
        if isinstance(section, Mapping):
            title = (
                section.get("title")
                or section.get("name")
                or section.get("stable_id")
                or f"Section {index}"
            )
            body_value = section.get(
                "content",
                {
                    key: value
                    for key, value in section.items()
                    if key not in {"title", "name", "stable_id", "order"}
                },
            )
        else:
            title = f"Section {index}"
            body_value = section
        if isinstance(body_value, str):
            body = body_value.strip()
        else:
            body = canonical_json_bytes(body_value).decode("utf-8")
        rendered.extend((f"### {title}", body))
    return "\n\n".join(part for part in rendered if part)


def _find_forbidden(value: Any, path: str = "stable_prefix") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            child = f"{path}.{key}"
            if key_text in FORBIDDEN_STABLE_KEYS:
                found.append(child)
            found.extend(_find_forbidden(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_forbidden(item, f"{path}[{index}]"))
    return found


class DeterministicPromptAssembler:
    """Build one stable prefix without moving or rewriting dynamic messages."""

    def __init__(self, *, verify_prefix: bool = False) -> None:
        self.verify_prefix = verify_prefix

    def assemble(
        self,
        *,
        dynamic_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        common_pack: Mapping[str, Any],
        role_pack: Mapping[str, Any],
        stable_system_instructions: str | None = None,
        reasoning_config: Mapping[str, Any] | None = None,
    ) -> AssembledPrompt:
        common_content = canonicalize(common_pack.get("content", common_pack))
        role_content = canonicalize(role_pack.get("content", role_pack))
        forbidden = _find_forbidden({"common": common_content, "role": role_content})
        if forbidden:
            raise UnstablePrefixError(
                "volatile fields found in stable packs: " + ", ".join(forbidden)
            )

        canonical_tools = canonicalize_tools(tools)
        stable_instructions = (
            (stable_system_instructions or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )
        parts = [stable_instructions]
        parts.append(_render_content(common_content, "MLEvolve Common Knowledge"))
        role_name = str(role_pack.get("role") or "Role").replace("_", " ").title()
        parts.append(_render_content(role_content, f"MLEvolve {role_name} Knowledge"))
        stable_prefix = "\n\n".join(part for part in parts if part)

        component_hashes = {
            "tools": canonical_sha256(canonical_tools),
            "system_instructions": canonical_sha256(stable_instructions),
            "common_pack": canonical_sha256(common_content),
            "role_pack": canonical_sha256(role_content),
        }
        reasoning_hash = canonical_sha256(reasoning_config or {})
        logical_prefix = {
            "tools": canonical_tools,
            "system_instructions": stable_instructions,
            "common_pack": common_content,
            "role_pack": role_content,
        }
        prefix_hash = canonical_sha256(logical_prefix)

        messages = [copy.deepcopy(dict(message)) for message in dynamic_messages]
        stable_index: int | None = None
        if stable_prefix:
            messages.insert(0, {"role": "system", "content": stable_prefix})
            stable_index = 0

        if self.verify_prefix:
            second_hash = canonical_sha256(logical_prefix)
            if prefix_hash != second_hash:
                raise UnstablePrefixError(
                    "stable prefix changed during deterministic verification"
                )

        expected_tokens = (
            max(1, len(stable_prefix.encode("utf-8")) // 4) if stable_prefix else None
        )
        return AssembledPrompt(
            messages=tuple(messages),
            tools=canonical_tools,
            stable_prefix=stable_prefix,
            dynamic_suffix=tuple(
                copy.deepcopy(dict(message)) for message in dynamic_messages
            ),
            stable_prefix_hash=prefix_hash,
            component_hashes=component_hashes,
            tool_schema_hash=component_hashes["tools"],
            reasoning_config_hash=reasoning_hash,
            stable_message_index=stable_index,
            expected_stable_prefix_tokens=expected_tokens,
        )

