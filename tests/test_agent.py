"""Control-flow tests for the loop. No MCP server and no model provider."""

import json

import anyio
from mcp.types import CallToolResult, TextContent

from parallax.agent import (
    ALLOWED_TOOLS,
    AgentState,
    ChatMessage,
    FactualResolver,
    ModelCompletion,
    ModelUsage,
    ToolRequest,
    build_initial_state,
    classify_execution,
    copy_state,
    run_agent,
)
from parallax.observe import canonical_json
from parallax.replay import (
    PrefixMismatch,
    check_prefix_pair,
    prefix_document,
    prefix_sha256,
)
from tests.harness import SCRIPTED_DECISION


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> CallToolResult:
        del arguments
        self.calls.append(name)
        payload = {"echo": name}
        return CallToolResult(
            content=[TextContent(type="text", text=canonical_json(payload))],
            structured_content=payload,
            is_error=False,
        )


class _ToolsUntilBudget:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(
            content="",
            tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
        )


class _TokensThenTool:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(
            content="",
            tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
            usage=ModelUsage(input_tokens=2, output_tokens=2),
        )


class _Boom:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        raise RuntimeError("down")


class _FinalText:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(content=self.content)


class _Rejected:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(
            content="",
            tool_calls=(ToolRequest(name="remediate_now", arguments={"incident_id": "F000"}),),
        )


def _state(**overrides: object) -> AgentState:
    payload: dict[str, object] = {
        "incident_id": "F000",
        "model_id": "fake",
        "model_params": {"temperature": 0, "seed": 1},
        "step_budget": 4,
        "token_budget": None,
        "fixture_hash": "abc",
        "tool_schemas": [{"name": "get_metrics"}],
    }
    payload.update(overrides)
    return build_initial_state(**payload)  # type: ignore[arg-type]


def test_allowed_tools_are_the_three_observation_tools() -> None:
    assert ALLOWED_TOOLS == frozenset({"get_metrics", "get_logs", "get_deployment"})
    assert "propose_action" not in ALLOWED_TOOLS


def test_copy_state_does_not_alias_the_prefix() -> None:
    original = _state()
    copied = copy_state(original)
    copied.messages.append(ChatMessage(role="assistant", content="later"))
    copied.model_params["temperature"] = 9
    assert len(original.messages) == 2
    assert original.model_params["temperature"] == 0


def test_prefix_hash_covers_the_shared_document_only() -> None:
    state = _state()
    document = prefix_document(state)
    assert set(document) == {
        "fixture_hash",
        "incident_id",
        "messages",
        "model_id",
        "model_params",
        "prompt_hash",
        "remaining_steps",
        "token_budget",
        "tokens_used",
        "tool_call_budget",
        "tool_calls_used",
        "tool_schemas",
    }
    assert "provider_response_ids" not in document
    assert "oracle_decision" not in document
    assert "tool_calls" not in document
    left = copy_state(state)
    right = copy_state(state)
    assert check_prefix_pair(left, right) == prefix_sha256(state)
    right.messages.append(ChatMessage(role="assistant", content="drift"))
    try:
        check_prefix_pair(left, right)
    except PrefixMismatch as exc:
        assert exc.left != exc.right
    else:
        raise AssertionError("diverged prefixes hashed equal")


def test_step_budget_stops_without_inventing_a_decision() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(
            client,
            _state(step_budget=2),
            _ToolsUntilBudget(),
            resolver=FactualResolver(),
        )
        assert result.termination == "budget"
        assert result.execution_status == "budget_exhausted"
        assert result.flags == ("budget_exhausted",)
        assert result.parse_ok is False
        assert result.decision_parsed is None
        assert "parse_failure" not in result.flags
        assert client.calls == ["get_metrics", "get_metrics"]

    anyio.run(main, backend="asyncio")


def test_token_budget_uses_only_reported_usage() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(
            client,
            _state(step_budget=6, token_budget=3),
            _TokensThenTool(),
            resolver=FactualResolver(),
        )
        assert result.flags == ("budget_exhausted",)
        assert result.state.tokens_used == 4
        assert client.calls == ["get_metrics"]
        assert result.decision_parsed is None

    anyio.run(main, backend="asyncio")


def test_adapter_exception_is_a_provider_error() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(client, _state(), _Boom(), resolver=FactualResolver())
        assert result.flags == ("provider_error",)
        assert result.execution_status == "model_error"
        assert result.error is not None
        assert result.error.startswith("RuntimeError")
        assert result.decision_parsed is None
        assert "parse_failure" not in result.flags
        assert client.calls == []

    anyio.run(main, backend="asyncio")


def test_invalid_final_message_is_a_parse_failure() -> None:
    async def main() -> None:
        result = await run_agent(_StubClient(), _state(), _FinalText("not-json"), resolver=FactualResolver())
        assert result.parse_ok is False
        assert result.decision_parsed is None
        assert result.execution_status == "invalid_final"
        assert result.flags == ("parse_failure",)
        assert result.decision_raw == "not-json"

    anyio.run(main, backend="asyncio")


def test_closed_json_parses_and_the_script_is_not_relabeled() -> None:
    async def main() -> None:
        result = await run_agent(
            _StubClient(),
            _state(),
            _FinalText(SCRIPTED_DECISION),
            resolver=FactualResolver(),
        )
        assert result.parse_ok is True
        assert result.decision_parsed == "ESCALATE"
        assert result.execution_status == "completed"
        assert result.action_observed == "none"
        assert result.flags == ()

    anyio.run(main, backend="asyncio")


def test_disallowed_tool_is_recorded_and_not_called() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(client, _state(), _Rejected(), resolver=FactualResolver())
        assert client.calls == []
        assert result.termination == "rejected_tool"
        assert result.execution_status == "invalid_tool"
        assert result.state.tool_calls[0].name == "remediate_now"
        assert result.state.tool_calls[0].dispatched is False
        assert result.decision_parsed is None
        assert "parse_failure" not in result.flags

    anyio.run(main, backend="asyncio")


class _ErrorClient:
    async def call_tool(self, name: str, arguments: dict[str, object]) -> CallToolResult:
        del name, arguments
        payload = {"error": "unavailable"}
        return CallToolResult(
            content=[TextContent(type="text", text=canonical_json(payload))],
            structured_content=payload,
            is_error=True,
        )


class _BlankAfterTool:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        if any(message.role == "tool" for message in messages):  # type: ignore[attr-defined]
            return ModelCompletion(content="")
        return ModelCompletion(
            content="",
            tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
        )


class _DecideAfterError:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        if any(message.role == "tool" for message in messages):  # type: ignore[attr-defined]
            return ModelCompletion(content=SCRIPTED_DECISION)
        return ModelCompletion(
            content="",
            tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
        )


def test_execution_status_ignores_the_oracle() -> None:
    assert classify_execution(
        termination="decision",
        parse_ok=True,
        flags=(),
        decision_raw=SCRIPTED_DECISION,
    ) == "completed"
    assert classify_execution(
        termination="budget",
        parse_ok=False,
        flags=("budget_exhausted",),
        decision_raw=None,
    ) == "budget_exhausted"
    assert classify_execution(
        termination="provider",
        parse_ok=False,
        flags=("provider_error",),
        decision_raw=None,
    ) == "model_error"
    assert "oracle" not in classify_execution.__code__.co_varnames


def test_tool_error_without_a_decision_body_is_its_own_status() -> None:
    async def main() -> None:
        result = await run_agent(_ErrorClient(), _state(), _BlankAfterTool(), resolver=FactualResolver())
        assert result.execution_status == "tool_error"
        assert result.parse_ok is False
        assert result.decision_parsed is None
        assert "tool_error" in result.flags
        assert "budget_exhausted" not in result.flags
        assert "provider_error" not in result.flags
        visible = result.state.messages[-2].content
        assert "unavailable" in visible
        assert result.state.tool_calls[0].is_error is True
        assert result.state.tool_calls[0].raw is not None
        assert result.state.tool_calls[0].raw["is_error"] is True
        assert "is_error" not in json.loads(visible)

    anyio.run(main, backend="asyncio")


def test_a_parsed_decision_after_a_tool_error_stays_completed() -> None:
    async def main() -> None:
        result = await run_agent(_ErrorClient(), _state(), _DecideAfterError(), resolver=FactualResolver())
        assert result.execution_status == "completed"
        assert result.parse_ok is True
        assert result.decision_parsed == "ESCALATE"
        assert "tool_error" in result.flags

    anyio.run(main, backend="asyncio")


class _TwoTools:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        if any(message.role == "tool" for message in messages):  # type: ignore[attr-defined]
            return ModelCompletion(content=SCRIPTED_DECISION, response_id="resp_final")
        return ModelCompletion(
            content="",
            response_id="resp_tools",
            tool_calls=(
                ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}, call_id="call_metrics"),
                ToolRequest(name="get_logs", arguments={"incident_id": "F000"}, call_id="call_logs"),
            ),
        )


class _OverBudget:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(
            content="",
            tool_calls=(
                ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),
                ToolRequest(name="get_logs", arguments={"incident_id": "F000"}),
            ),
        )


class _BadArguments:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(
            content="",
            tool_calls=(
                ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),
                ToolRequest(name="get_logs", arguments={"incident_id": ""}),
            ),
        )


def test_multiple_tool_calls_keep_provider_ids() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(client, _state(), _TwoTools(), resolver=FactualResolver())
        assert client.calls == ["get_metrics", "get_logs"]
        assert result.execution_status == "completed"
        tool_messages = [message for message in result.state.messages if message.role == "tool"]
        assert [message.call_id for message in tool_messages] == ["call_metrics", "call_logs"]
        assert result.state.provider_response_ids == ["resp_tools", "resp_final"]
        assert "provider_response_ids" not in prefix_document(result.state)
        assert result.state.missing_usage is True
        assert result.state.input_tokens == 0

    anyio.run(main, backend="asyncio")


def test_tool_call_budget_stops_before_another_mcp_call() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(
            client,
            _state(tool_call_budget=1),
            _OverBudget(),
            resolver=FactualResolver(),
        )
        assert client.calls == ["get_metrics"]
        assert result.state.tool_calls_used == 1
        assert result.termination == "budget"
        assert result.execution_status == "budget_exhausted"
        assert result.decision_parsed is None

    anyio.run(main, backend="asyncio")


def test_malformed_tool_arguments_are_not_dispatched() -> None:
    client = _StubClient()

    async def main() -> None:
        result = await run_agent(client, _state(), _BadArguments(), resolver=FactualResolver())
        assert client.calls == ["get_metrics"]
        assert result.execution_status == "invalid_tool"
        rejected = result.state.tool_calls[-1]
        assert rejected.name == "get_logs"
        assert rejected.dispatched is False
        assert rejected.source == "rejected"

    anyio.run(main, backend="asyncio")
