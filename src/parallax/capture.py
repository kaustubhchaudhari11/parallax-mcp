"""Captured tool results and the label-field scan.

The raw dump is the audit record. The visible string is the exact text
a prompt is allowed to contain. Label fields are rejected when they
appear as keys in an observation object.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult

from parallax.observe import model_visible_observation, raw_tool_result

FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "decision",
        "label",
        "oracle",
        "gold",
        "expected_decision",
        "oracle_decision",
    }
)


@dataclass(frozen=True, slots=True)
class CapturedObservation:
    """One server result, split into the audit dump and the visible text."""

    raw: dict[str, Any]
    visible: str
    is_error: bool


def sha256_text(value: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def capture_result(result: CallToolResult) -> CapturedObservation:
    """Store the raw dump and the canonical visible string."""

    return CapturedObservation(
        raw=raw_tool_result(result),
        visible=model_visible_observation(result),
        is_error=bool(result.is_error),
    )


def label_keys(value: object) -> set[str]:
    """Return every dict key in a nested JSON-like value."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(label_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(label_keys(child))
    return found


def label_key_overlap(value: object) -> set[str]:
    """Return forbidden label keys present in ``value``."""

    return label_keys(value) & FORBIDDEN_LABEL_KEYS
