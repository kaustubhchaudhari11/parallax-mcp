"""Canonical serializer for the model-visible MCP observation.

The agent, when it exists, inserts one string: the compact JSON of
``structured_content``. The raw ``CallToolResult`` is kept separately
for trace capture. Decision fields are not added here.
"""

import json
from typing import Any

from mcp.types import CallToolResult, TextContent


def canonical_json(value: Any) -> str:
    """Serialize ``value`` with sorted keys and compact separators.

    Array order is preserved. Object key order and insignificant
    whitespace are not.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def model_visible_observation(result: CallToolResult) -> str:
    """Return the string a later prompt builder must insert.

    A dict ``structured_content`` is canonicalized. When that field is
    missing, the archived text content is returned instead so an SDK
    exception path remains visible. That fallback is not used for an
    unknown incident, which always carries structured content.
    """

    structured = result.structured_content
    if structured is not None:
        if not isinstance(structured, dict):
            raise TypeError("structured_content must be a JSON object when present")
        return canonical_json(structured)
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    if not texts:
        raise ValueError("tool result has no structured content and no text content")
    return "\n".join(texts)


def raw_tool_result(result: CallToolResult) -> dict[str, Any]:
    """Audit dump of the whole tool result. This dump is not model-visible."""

    dumped: dict[str, Any] = result.model_dump(mode="json")
    return dumped
