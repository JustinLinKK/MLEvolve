"""Canonical semantic serialization used by packs, tools, and cache families."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

VOLATILE_KEYS = frozenset(
    {
        "compiled_at",
        "created_at",
        "updated_at",
        "timestamp",
        "retrieved_at",
        "accessed_at",
        "last_accessed_at",
        "request_id",
        "run_id",
        "trace_id",
        "database_id",
        "db_id",
        "similarity_score",
        "volatile_score",
    }
)
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "refresh_token",
        "authorization",
        "password",
        "credentials",
        "client_secret",
        "private_key",
    }
)
_ORDER_INSENSITIVE_LIST_KEYS = frozenset({"records", "sources", "tools"})


class CanonicalizationError(ValueError):
    pass


def sensitive_paths(value: Any, path: str = "content") -> list[str]:
    found: list[str] = []
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in SENSITIVE_KEYS:
                found.append(child)
            found.extend(sensitive_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(sensitive_paths(item, f"{path}[{index}]"))
    return found


def _stable_sort_key(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonicalize(value: Any, *, parent_key: str | None = None) -> Any:
    """Return a JSON-compatible value with volatile metadata removed.

    List order is retained unless the field is explicitly a set-like collection.
    Sections may opt into an integer ``order`` and otherwise sort by a stable
    semantic identifier.
    """

    if is_dataclass(value):
        value = asdict(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return unicodedata.normalize(
            "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(
                "non-finite numbers are not valid in canonical content"
            )
        return 0.0 if value == 0 else value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CanonicalizationError(
                "non-finite decimals are not valid in canonical content"
            )
        return format(value.normalize(), "f")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = unicodedata.normalize("NFC", str(raw_key))
            if key.lower() in VOLATILE_KEYS:
                continue
            normalized[key] = canonicalize(raw_value, parent_key=key)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item, parent_key=parent_key) for item in value]
        return sorted(items, key=_stable_sort_key)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [canonicalize(item, parent_key=parent_key) for item in value]
        if parent_key in _ORDER_INSENSITIVE_LIST_KEYS:
            items.sort(key=_stable_sort_key)
        elif parent_key == "sections":

            def section_key(item: Any) -> tuple[Any, ...]:
                if isinstance(item, Mapping):
                    if "order" in item:
                        return (0, item["order"], _stable_sort_key(item))
                    for key in ("stable_id", "name", "title", "kind"):
                        if key in item:
                            return (1, str(item[key]), _stable_sort_key(item))
                return (2, _stable_sort_key(item))

            items.sort(key=section_key)
        return items
    raise CanonicalizationError(
        f"unsupported canonical value type: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_json_bytes(value: Any) -> bytes:
    """Serialize validated JSON without stripping diagnostic envelope fields."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonicalize_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    normalized = [canonicalize(tool) for tool in (tools or ())]

    def name(tool: Mapping[str, Any]) -> tuple[str, bytes]:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        function_name = (
            function.get("name", "")
            if isinstance(function, Mapping)
            else tool.get("name", "")
        )
        return str(function_name), _stable_sort_key(tool)

    normalized.sort(key=name)
    return tuple(normalized)
