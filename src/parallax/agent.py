"""One agent control loop.

The loop dispatches the three observation tools and stops on a closed
decision, a rejected tool name, a budget, or an adapter error. It does
not import the decision rule. A scripted adapter used by tests is not a
measured model.
"""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict, Field

from parallax.capture import capture_result, sha256_text
from parallax.observe import model_visible_observation
from parallax.prompt import POLICY, prompt_hash, user_message
from parallax.trace import ExecutionStatus, ToolCallRecord

ALLOWED_TOOLS = frozenset({"get_metrics", "get_logs", "get_deployment"})
OUTPUT_DECISIONS = frozenset({"MONITOR", "INVESTIGATE", "ESCALATE"})
EVIDENCE_NAMES = frozenset({"metrics", "logs", "deployment"})
OUTPUT_FIELDS = frozenset({"decision", "action", "evidence"})


class ToolRequest(BaseModel):
    """One tool call requested by the adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    """One transcript message. Tool content is the visible observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str
    tool_calls: tuple[ToolRequest, ...] = ()
    tool_name: str | None = None


class ModelUsage(BaseModel):
    """Provider-reported token counts. Missing fields stay null."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelCompletion(BaseModel):
    """Adapter output. Usage is null when the adapter does not report it."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: tuple[ToolRequest, ...] = ()
    usage: ModelUsage | None = None


class AgentState(BaseModel):
    """Loop state. The prefix hash covers the fields replay selects."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model_id: str
    model_params: dict[str, Any]
    step_budget: int
    steps_used: int = 0
    token_budget: int | None
    tokens_used: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reported_usage: bool = False
    missing_usage: bool = False
    incident_id: str
    prompt_hash: str
    fixture_hash: str
    tool_schemas: list[dict[str, Any]]
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ResolvedObservation(BaseModel):
    """Visible text for one tool result after overlay resolution."""

    model_config = ConfigDict(extra="forbid")

    content: str
    is_error: bool
    source: str
    flag: str | None = None


class ObservationResolver(Protocol):
    """Turn a server result into the string the transcript stores."""

    def resolve(self, call: ToolRequest, result: CallToolResult) -> ResolvedObservation:
        """Return the visible observation for this call."""


class ModelAdapter(Protocol):
    """Narrow completion interface. Milestone 2A has no network adapter."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
    ) -> ModelCompletion:
        """Return the next assistant message."""


class CheckpointSink(Protocol):
    """Observe a tool request before its result is appended."""

    def consider(
        self,
        state: AgentState,
        call: ToolRequest,
        result: CallToolResult,
        latency_ms: float,
    ) -> None:
        """Record the request when it is an eligible branch point."""


@dataclass(frozen=True, slots=True)
class OutputInspection:
    """Parse result for a final assistant message."""

    parse_ok: bool
    decision: str | None
    action: str | None
    raw: str | None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Terminal loop state. ``decision_parsed`` is set only when parse_ok."""

    state: AgentState
    decision_raw: str | None
    decision_parsed: str | None
    action_observed: str | None
    parse_ok: bool
    flags: tuple[str, ...]
    error: str | None
    termination: str
    execution_status: ExecutionStatus


def build_initial_state(
    *,
    incident_id: str,
    model_id: str,
    model_params: Mapping[str, Any],
    step_budget: int,
    token_budget: int | None,
    fixture_hash: str,
    tool_schemas: Sequence[Mapping[str, Any]],
) -> AgentState:
    """Build S0. The messages are the policy and the incident task."""

    messages = [
        ChatMessage(role="system", content=POLICY),
        ChatMessage(role="user", content=user_message(incident_id)),
    ]
    return AgentState(
        messages=messages,
        model_id=model_id,
        model_params=dict(model_params),
        step_budget=step_budget,
        token_budget=token_budget,
        incident_id=incident_id,
        prompt_hash=prompt_hash(incident_id),
        fixture_hash=fixture_hash,
        tool_schemas=[dict(schema) for schema in tool_schemas],
    )


def copy_state(state: AgentState) -> AgentState:
    """Deep copy through JSON so a branch cannot alias the prefix."""

    return AgentState.model_validate_json(state.model_dump_json())


def classify_execution(
    *,
    termination: str,
    parse_ok: bool,
    flags: Sequence[str],
    decision_raw: str | None,
) -> ExecutionStatus:
    """Classify how the loop stopped.

    The result is not an oracle score. A parsed decision is ``completed``
    even when a later comparison disagrees with the oracle. Budget
    exhaustion and adapter failures are not parse failures. ``tool_error``
    applies when an MCP result failed and the loop produced no decision
    body. A non-empty unparsed body is ``invalid_final``.
    """

    if termination == "provider":
        return "model_error"
    if termination == "budget":
        return "budget_exhausted"
    if termination == "rejected_tool":
        return "invalid_tool"
    if parse_ok:
        return "completed"
    if "tool_error" in flags and (decision_raw is None or decision_raw.strip() == ""):
        return "tool_error"
    return "invalid_final"


def factual_resolution(result: CallToolResult) -> ResolvedObservation:
    """Canonicalize a server result. Errors stay visible and are flagged."""

    visible = model_visible_observation(result)
    if result.is_error or result.structured_content is None:
        return ResolvedObservation(content=visible, is_error=True, source="base", flag="tool_error")
    return ResolvedObservation(content=visible, is_error=False, source="base")


class FactualResolver:
    """Read the base server result. This resolver has an empty overlay."""

    def resolve(self, call: ToolRequest, result: CallToolResult) -> ResolvedObservation:
        del call
        return factual_resolution(result)


def inspect_output(raw: str | None) -> OutputInspection:
    """Parse the closed decision object.

    A usable decision requires the three fields, a known decision, action
    ``none``, and evidence drawn from the three observation names. Any
    other object is unparsed, and this function does not substitute a
    decision for it. A string action other than ``none`` is still
    returned so instruction adoption can be scored.
    """

    if raw is None:
        return OutputInspection(parse_ok=False, decision=None, action=None, raw=None)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return OutputInspection(parse_ok=False, decision=None, action=None, raw=raw)
    if not isinstance(value, dict):
        return OutputInspection(parse_ok=False, decision=None, action=None, raw=raw)
    action = value.get("action")
    action_observed = action if isinstance(action, str) else None
    decision = value.get("decision")
    evidence = value.get("evidence")
    decision_ok = isinstance(decision, str) and decision in OUTPUT_DECISIONS
    evidence_ok = (
        isinstance(evidence, list)
        and len(set(evidence)) == len(evidence)
        and all(isinstance(item, str) and item in EVIDENCE_NAMES for item in evidence)
    )
    parse_ok = decision_ok and action == "none" and evidence_ok and set(value) <= OUTPUT_FIELDS
    return OutputInspection(
        parse_ok=parse_ok,
        decision=decision if parse_ok and isinstance(decision, str) else None,
        action=action_observed,
        raw=raw,
    )


async def run_agent(
    client: Any,
    state: AgentState,
    adapter: ModelAdapter,
    *,
    resolver: ObservationResolver,
    checkpoint: CheckpointSink | None = None,
) -> AgentResult:
    """Run the loop from ``state`` until it stops.

    ``parse_failure`` is set only when the adapter returns a final
    message and that message is not the closed JSON object. A budget
    stop, an adapter exception, or a rejected tool name with no decision
    body does not also invent a parse failure.
    """

    flags: list[str] = []
    error: str | None = None
    termination = "decision"
    final_content: str | None = None
    produced_final = False
    while True:
        if state.steps_used >= state.step_budget:
            flags.append("budget_exhausted")
            termination = "budget"
            break
        if _token_budget_spent(state):
            flags.append("budget_exhausted")
            termination = "budget"
            break
        try:
            completion = await adapter.complete(state.messages, state.tool_schemas, state.model_params)
            if not isinstance(completion, ModelCompletion):
                raise TypeError("adapter returned a value that is not a completion")
        except Exception as exc:  # noqa: BLE001 — an adapter failure is a ledger row, not a crash
            flags.append("provider_error")
            error = f"{type(exc).__name__}: {exc}"
            termination = "provider"
            break
        state.steps_used += 1
        _record_usage(state, completion.usage)
        state.messages.append(
            ChatMessage(
                role="assistant",
                content=completion.content,
                tool_calls=completion.tool_calls,
            )
        )
        if not completion.tool_calls:
            final_content = completion.content
            produced_final = True
            termination = "decision"
            break
        rejected = await _dispatch_tools(
            client,
            state,
            completion,
            resolver=resolver,
            checkpoint=checkpoint,
            flags=flags,
        )
        if rejected:
            final_content = completion.content or None
            produced_final = bool(completion.content)
            termination = "rejected_tool"
            break
    inspection = inspect_output(final_content if produced_final else None)
    if produced_final and not inspection.parse_ok:
        flags.append("parse_failure")
    decision_raw = inspection.raw if produced_final else None
    unique_flags = _unique(flags)
    return AgentResult(
        state=state,
        decision_raw=decision_raw,
        decision_parsed=inspection.decision,
        action_observed=inspection.action if produced_final else None,
        parse_ok=inspection.parse_ok,
        flags=unique_flags,
        error=error,
        termination=termination,
        execution_status=classify_execution(
            termination=termination,
            parse_ok=inspection.parse_ok,
            flags=unique_flags,
            decision_raw=decision_raw,
        ),
    )


async def _dispatch_tools(
    client: Any,
    state: AgentState,
    completion: ModelCompletion,
    *,
    resolver: ObservationResolver,
    checkpoint: CheckpointSink | None,
    flags: list[str],
) -> bool:
    for call in completion.tool_calls:
        arguments = dict(call.arguments)
        if call.name not in ALLOWED_TOOLS:
            state.tool_calls.append(
                ToolCallRecord(
                    name=call.name,
                    arguments=arguments,
                    dispatched=False,
                    source="rejected",
                )
            )
            return True
        started = _now()
        result = await client.call_tool(call.name, arguments)
        latency_ms = (_now() - started) * 1000
        if not isinstance(result, CallToolResult):
            raise TypeError("client returned a value that is not a tool result")
        if checkpoint is not None:
            checkpoint.consider(copy_state(state), call, result, latency_ms)
        resolved = resolver.resolve(call, result)
        if resolved.flag is not None:
            flags.append(resolved.flag)
        visible = resolved.content
        state.messages.append(ChatMessage(role="tool", content=visible, tool_name=call.name))
        captured = capture_result(result)
        state.tool_calls.append(
            ToolCallRecord(
                name=call.name,
                arguments=arguments,
                latency_ms=latency_ms,
                is_error=captured.is_error,
                raw=captured.raw,
                model_visible_observation=visible,
                model_visible_sha256=sha256_text(visible),
                dispatched=True,
                source=resolved.source,
            )
        )
    return False


def _record_usage(state: AgentState, usage: ModelUsage | None) -> None:
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        state.missing_usage = True
        return
    state.reported_usage = True
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    state.tokens_used = state.input_tokens + state.output_tokens


def _token_budget_spent(state: AgentState) -> bool:
    if state.missing_usage or not state.reported_usage:
        return False
    return (
        state.token_budget is not None
        and state.tokens_used is not None
        and state.tokens_used >= state.token_budget
    )


def _unique(flags: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.append(flag)
    return tuple(seen)


def _now() -> float:
    return time.perf_counter()
