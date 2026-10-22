"""Canonical observation strings, independent of a live tool call."""

import json

import pytest
from mcp.types import CallToolResult, TextContent

from parallax.observe import canonical_json, model_visible_observation, raw_tool_result


def test_canonical_json_sorts_keys_and_drops_insignificant_whitespace() -> None:
    left = {"b": 1, "a": {"z": 2, "y": [1, 2]}}
    right = {"a": {"y": [1, 2], "z": 2}, "b": 1}
    expected = '{"a":{"y":[1,2],"z":2},"b":1}'
    assert canonical_json(left) == expected
    assert canonical_json(right) == expected
    pretty = json.dumps(left, indent=4)
    assert pretty != expected
    assert canonical_json(json.loads(pretty)) == expected
    assert canonical_json({"note": "café"}) == '{"note":"café"}'


def test_array_order_remains_visible() -> None:
    forward = canonical_json({"lines": [{"message": "a"}, {"message": "b"}]})
    backward = canonical_json({"lines": [{"message": "b"}, {"message": "a"}]})
    assert forward != backward


def test_model_visible_observation_uses_structured_content() -> None:
    payload = {"incident_id": "F000", "error": "unknown_incident"}
    result = CallToolResult(
        content=[TextContent(type="text", text=canonical_json(payload))],
        structured_content=payload,
        is_error=True,
    )
    visible = model_visible_observation(result)
    assert visible == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert visible == '{"error":"unknown_incident","incident_id":"F000"}'
    parsed = json.loads(visible)
    assert "is_error" not in parsed
    assert "decision" not in parsed


def test_missing_structured_content_falls_back_to_text() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="Error executing tool get_metrics")],
        is_error=True,
    )
    assert result.structured_content is None
    assert model_visible_observation(result) == "Error executing tool get_metrics"


def test_non_object_structured_content_is_rejected() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="1")],
        structured_content=["not", "an", "object"],
        is_error=True,
    )
    with pytest.raises(TypeError):
        model_visible_observation(result)


def test_raw_dump_keeps_the_whole_result() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text='{"a":1}')],
        structured_content={"a": 1},
        is_error=False,
    )
    raw = raw_tool_result(result)
    assert raw["is_error"] is False
    assert raw["structured_content"] == {"a": 1}
    assert raw["content"][0]["text"] == '{"a":1}'
    assert "is_error" not in json.loads(model_visible_observation(result))
