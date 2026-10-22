"""MCP client tests against the installed server, in-process and over stdio."""

import ast
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from mcp.types import TextContent
from mcp_types.version import KNOWN_PROTOCOL_VERSIONS

from parallax.fixtures import FIXTURE_IDS, FIXTURES
from parallax.observe import model_visible_observation, raw_tool_result
from parallax.schemas import (
    DeploymentInfo,
    IncidentLogs,
    ServiceMetrics,
    UnknownIncidentError,
)
from parallax.server import create_server

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = ("get_metrics", "get_logs", "get_deployment")
TOOL_MODELS = {
    "get_metrics": ServiceMetrics,
    "get_logs": IncidentLogs,
    "get_deployment": DeploymentInfo,
}
FORBIDDEN_KEYS = frozenset(
    {"decision", "label", "oracle", "gold", "expected_decision", "oracle_decision"}
)
DECISION_TOKENS = ("MONITOR", "INVESTIGATE", "ESCALATE")
_REFUSALS: list[str] = []


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def _assert_unlabeled(value: object) -> None:
    assert _keys(value).isdisjoint(FORBIDDEN_KEYS)
    text = json.dumps(value, ensure_ascii=False)
    for token in DECISION_TOKENS:
        assert token not in text
    assert "oracle" not in text.lower()


async def _refuse_sampling(context: object, params: object) -> object:
    _REFUSALS.append("sampling")
    raise AssertionError("server requested sampling")


async def _refuse_elicitation(context: object, params: object) -> object:
    _REFUSALS.append("elicitation")
    raise AssertionError("server requested elicitation")


def _client(server: object) -> Client:
    return Client(
        server,  # type: ignore[arg-type]
        read_timeout_seconds=15,
        sampling_callback=_refuse_sampling,
        elicitation_callback=_refuse_elicitation,
    )


def _text(result: object) -> str:
    block = result.content[0]  # type: ignore[attr-defined]
    assert isinstance(block, TextContent)
    return block.text


def _assert_success(result: Any, model: type[ServiceMetrics | IncidentLogs | DeploymentInfo]) -> None:
    assert result.is_error is False
    assert isinstance(result.structured_content, dict)
    validated = model.model_validate(result.structured_content)
    assert validated.model_dump(mode="json") == result.structured_content
    visible = model_visible_observation(result)
    assert visible == json.dumps(
        result.structured_content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    archived = _text(result)
    assert archived != visible
    assert "\n" in archived
    assert json.loads(archived) == result.structured_content
    reordered = {key: result.structured_content[key] for key in reversed(result.structured_content)}
    assert list(reordered) != list(result.structured_content)
    assert json.dumps(reordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == visible
    pretty = json.dumps(result.structured_content, indent=4)
    assert pretty != visible
    assert json.dumps(json.loads(pretty), sort_keys=True, separators=(",", ":"), ensure_ascii=False) == visible
    _assert_unlabeled(result.structured_content)
    _assert_unlabeled(json.loads(visible))
    raw = raw_tool_result(result)
    assert raw["is_error"] is False
    assert raw["structured_content"] == result.structured_content
    assert raw["content"][0]["text"] == archived
    assert "is_error" not in json.loads(visible)


def _assert_unknown(result: Any, incident_id: str) -> None:
    assert result.is_error is True
    payload = {"error": "unknown_incident", "incident_id": incident_id}
    assert result.structured_content == payload
    assert UnknownIncidentError.model_validate(result.structured_content) == UnknownIncidentError(
        error="unknown_incident",
        incident_id=incident_id,
    )
    visible = model_visible_observation(result)
    archived = _text(result)
    assert archived == visible
    assert visible == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert "Traceback" not in archived
    assert 'File "' not in archived
    _assert_unlabeled(result.structured_content)


async def _assert_tool_contract(client: Client) -> None:
    listed = await client.list_tools()
    names = tuple(tool.name for tool in listed.tools)
    assert names == TOOL_NAMES
    assert "propose_action" not in names
    for tool in listed.tools:
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["properties"]["incident_id"]["type"] == "string"
        assert "incident_id" in tool.input_schema["required"]
        assert set(tool.input_schema["properties"]) == {"incident_id"}
        assert tool.output_schema is not None
        schema_text = json.dumps(tool.output_schema)
        if tool.name == "get_metrics":
            assert "error_rate_bps" in schema_text
            assert "integer" in schema_text
        if tool.name == "get_logs":
            assert "error_count" in schema_text
            assert "highest_severity" in schema_text
        if tool.name == "get_deployment":
            assert "deployed_within_window" in schema_text
        dumped = tool.model_dump(mode="json")
        _assert_unlabeled(dumped)
    version = client.protocol_version
    assert version in KNOWN_PROTOCOL_VERSIONS, version
    session = client.session
    assert session.initialize_result is not None or session.discover_result is not None


async def _inprocess_contract() -> None:
    _REFUSALS.clear()
    async with _client(create_server()) as client:
        await _assert_tool_contract(client)
        for incident_id in FIXTURE_IDS:
            fixture = FIXTURES[incident_id]
            expected = {
                "get_metrics": fixture.metrics.model_dump(mode="json"),
                "get_logs": fixture.logs.model_dump(mode="json"),
                "get_deployment": fixture.deployment.model_dump(mode="json"),
            }
            for name in TOOL_NAMES:
                first = await client.call_tool(name, {"incident_id": incident_id})
                second = await client.call_tool(name, {"incident_id": incident_id})
                assert first.structured_content == expected[name]
                assert second.structured_content == first.structured_content
                assert model_visible_observation(second) == model_visible_observation(first)
                assert raw_tool_result(second) == raw_tool_result(first)
                _assert_success(first, TOOL_MODELS[name])
                if name == "get_metrics":
                    assert type(first.structured_content["error_rate_bps"]) is int
                    assert first.structured_content["error_rate_bps"] == fixture.metrics.error_rate_bps
        for name in TOOL_NAMES:
            unknown = await client.call_tool(name, {"incident_id": "missing-incident"})
            _assert_unknown(unknown, "missing-incident")
            lower = await client.call_tool(name, {"incident_id": "f000"})
            _assert_unknown(lower, "f000")
        invalid = await client.call_tool("get_metrics", {})
        assert invalid.is_error is True
        assert invalid.structured_content is None
        assert "Traceback" not in _text(invalid)
        assert "unknown_incident" not in _text(invalid)
        missing_tool = await client.call_tool("propose_action", {"incident_id": "F000"})
        assert missing_tool.is_error is True
        assert missing_tool.structured_content is None
        assert "Traceback" not in _text(missing_tool)
    assert _REFUSALS == []


async def _stdio_contract() -> None:
    _REFUSALS.clear()
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "parallax.server"],
        cwd=ROOT,
    )
    with anyio.fail_after(30):
        async with _client(parameters) as client:
            await _assert_tool_contract(client)
            fixture = FIXTURES["F111"]
            expected: Mapping[str, object] = {
                "get_metrics": fixture.metrics.model_dump(mode="json"),
                "get_logs": fixture.logs.model_dump(mode="json"),
                "get_deployment": fixture.deployment.model_dump(mode="json"),
            }
            for name in TOOL_NAMES:
                first = await client.call_tool(name, {"incident_id": "F111"})
                second = await client.call_tool(name, {"incident_id": "F111"})
                assert first.structured_content == expected[name]
                assert second.structured_content == first.structured_content
                _assert_success(first, TOOL_MODELS[name])
                unknown = await client.call_tool(name, {"incident_id": "missing-incident"})
                _assert_unknown(unknown, "missing-incident")
    assert _REFUSALS == []


def test_inprocess_client_observes_the_fixture_store() -> None:
    anyio.run(_inprocess_contract, backend="asyncio")


def test_stdio_client_launches_the_local_interpreter() -> None:
    anyio.run(_stdio_contract, backend="asyncio")


def test_server_source_does_not_import_the_oracle() -> None:
    path = ROOT / "src" / "parallax" / "server.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: set[str] = set()
    prints = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            prints += 1
    assert "parallax.oracle" not in modules
    assert prints == 0
    for token in ("MONITOR", "INVESTIGATE", "ESCALATE", "oracle", "expected_decision"):
        assert token not in source
