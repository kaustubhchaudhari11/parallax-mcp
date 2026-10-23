"""Snapshot boundary and paired replay.

A baseline runs on the factual server. The first eligible request for
the intervention's tool is copied before that result is appended. The
factual arm and the mutant arm each receive a copy of that prefix.
Their hashes are compared before either arm continues. The mutant arm
keeps one branch-local overlay. The fixture store is not written.

This module scores a finished transcript. It does not put that score
into the messages. Milestone 2A injects an adapter. It does not open a
network connection to a model provider.

``run_pair`` creates one baseline for that pair. A scheduler that reuses
one checkpoint across scenarios is planned for Milestone 2B and is not
implemented here.
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from mcp import Client
from mcp.types import CallToolResult
from pydantic import ValidationError

from parallax.agent import (
    AgentResult,
    AgentState,
    ChatMessage,
    FactualResolver,
    ModelAdapter,
    ResolvedObservation,
    ToolRequest,
    build_initial_state,
    copy_state,
    factual_resolution,
    run_agent,
)
from parallax.capture import capture_result, sha256_text
from parallax.fixtures import IncidentFixture, get_fixture
from parallax.mutate import (
    INTERVENTIONS,
    TOOL_MODELS,
    apply_intervention,
    intervention_source_sha256,
)
from parallax.observe import canonical_json
from parallax.oracle import decide, predicate_d, predicate_l, predicate_m
from parallax.server import create_server
from parallax.trace import (
    Arm,
    Phase,
    ToolCallRecord,
    TraceRecord,
    append_trace,
    utc_now,
)

TOOL_ORDER = ("get_metrics", "get_logs", "get_deployment")


class PrefixMismatch(Exception):
    """The two arm prefixes hashed differently."""

    def __init__(self, left: str, right: str) -> None:
        self.left = left
        self.right = right
        super().__init__("prefix hashes differ")


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Eligible branch point. The tool result is not in ``state`` yet."""

    state: AgentState
    tool_name: str
    arguments: dict[str, Any]
    raw: dict[str, Any]
    visible: str
    latency_ms: float


@dataclass
class PairRun:
    """Rows and transcripts from one paired attempt."""

    baseline: TraceRecord
    factual: TraceRecord | None
    mutant: TraceRecord
    baseline_messages: list[ChatMessage]
    factual_messages: list[ChatMessage] | None
    mutant_messages: list[ChatMessage] | None
    snapshot: Checkpoint | None
    prefix_sha256: str | None


class CheckpointRecorder:
    """Save the first successful factual request for the intervention tool."""

    def __init__(self, tool_name: str, incident_id: str) -> None:
        self._tool_name = tool_name
        self._incident_id = incident_id
        self.checkpoint: Checkpoint | None = None

    def consider(
        self,
        state: AgentState,
        call: ToolRequest,
        result: CallToolResult,
        latency_ms: float,
    ) -> None:
        if self.checkpoint is not None:
            return
        if call.name != self._tool_name:
            return
        if call.arguments.get("incident_id") != self._incident_id:
            return
        if result.is_error or not isinstance(result.structured_content, dict):
            return
        try:
            TOOL_MODELS[call.name].model_validate(result.structured_content)
        except ValidationError:
            return
        if not _prefix_is_factual(state):
            return
        captured = capture_result(result)
        self.checkpoint = Checkpoint(
            state=state,
            tool_name=call.name,
            arguments=dict(call.arguments),
            raw=captured.raw,
            visible=captured.visible,
            latency_ms=latency_ms,
        )


class OverlayResolver:
    """Serve one stored observation for a successful read of one tool.

    An MCP error stays an error. The stored string is not substituted
    for it. A later successful read of the same tool and incident still
    returns the stored string. The loop records the server payload
    separately from the text this resolver chooses.
    """

    def __init__(self, visible_by_tool: Mapping[str, str], *, incident_id: str | None = None) -> None:
        self._visible_by_tool = dict(visible_by_tool)
        self._incident_id = incident_id

    def resolve(self, call: ToolRequest, result: CallToolResult) -> ResolvedObservation:
        factual = factual_resolution(result)
        if factual.is_error:
            return factual
        stored = self._visible_by_tool.get(call.name)
        if stored is None:
            return factual
        requested = call.arguments.get("incident_id")
        if self._incident_id is not None and requested != self._incident_id:
            return factual
        return ResolvedObservation(content=stored, is_error=False, source="overlay")


def first_difference(left: Sequence[ChatMessage], right: Sequence[ChatMessage]) -> int | None:
    """Return the first index where two transcripts differ."""

    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index].model_dump(mode="json") != right[index].model_dump(mode="json"):
            return index
    if len(left) != len(right):
        return limit
    return None


def first_action_divergence(left: Sequence[ChatMessage], right: Sequence[ChatMessage]) -> int | None:
    """Return the first assistant message that differs.

    An injected tool observation is not an agent action. A pair whose
    assistant messages match has no action divergence.
    """

    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index].role != "assistant" or right[index].role != "assistant":
            continue
        if left[index].model_dump(mode="json") != right[index].model_dump(mode="json"):
            return index
    longer = left if len(left) > len(right) else right
    for index in range(limit, len(longer)):
        if longer[index].role == "assistant":
            return index
    return None


def prefix_document(state: AgentState) -> dict[str, Any]:
    """The prefix that both arms must share.

    Tool-call audit records, latency, and the decision score are not
    part of this document.
    """

    return {
        "fixture_hash": state.fixture_hash,
        "incident_id": state.incident_id,
        "messages": [message.model_dump(mode="json") for message in state.messages],
        "model_id": state.model_id,
        "model_params": state.model_params,
        "prompt_hash": state.prompt_hash,
        "remaining_steps": state.step_budget - state.steps_used,
        "token_budget": state.token_budget,
        "tokens_used": state.tokens_used,
        "tool_schemas": state.tool_schemas,
    }


def prefix_sha256(state: AgentState) -> str:
    """Hash the prefix document."""

    return sha256_text(canonical_json(prefix_document(state)))


def check_prefix_pair(left: AgentState, right: AgentState) -> str:
    """Return the shared hash, or raise when the copies differ."""

    left_hash = prefix_sha256(left)
    right_hash = prefix_sha256(right)
    if left_hash != right_hash:
        raise PrefixMismatch(left_hash, right_hash)
    return left_hash


def restore(
    state: AgentState,
    checkpoint: Checkpoint,
    *,
    content: str,
    source: Literal["base", "overlay"],
) -> None:
    """Append one selected observation. This does not read a trace label."""

    state.messages.append(ChatMessage(role="tool", content=content, tool_name=checkpoint.tool_name))
    is_error = checkpoint.raw.get("is_error")
    state.tool_calls.append(
        ToolCallRecord(
            name=checkpoint.tool_name,
            arguments=dict(checkpoint.arguments),
            latency_ms=checkpoint.latency_ms,
            is_error=is_error if isinstance(is_error, bool) else None,
            raw=checkpoint.raw,
            model_visible_observation=content,
            model_visible_sha256=sha256_text(content),
            dispatched=True,
            source=source,
        )
    )


def fixture_hash(fixture: IncidentFixture) -> str:
    """Hash the three tool objects and the incident id."""

    payload = {
        "deployment": fixture.deployment.model_dump(mode="json"),
        "incident_id": fixture.incident_id,
        "logs": fixture.logs.model_dump(mode="json"),
        "metrics": fixture.metrics.model_dump(mode="json"),
    }
    return sha256_text(canonical_json(payload))


def usage_payload(state: AgentState) -> dict[str, int] | None:
    """Trial usage when every completion reported both counts, else null."""

    if state.missing_usage or not state.reported_usage:
        return None
    return {"input_tokens": state.input_tokens, "output_tokens": state.output_tokens}


async def run_pair(
    *,
    incident_id: str,
    intervention_id: str,
    adapter: ModelAdapter,
    model_id: str,
    model_params: Mapping[str, Any],
    step_budget: int,
    token_budget: int | None,
    phase: Phase,
    scenario_id: str | None,
    repeat_index: int,
    trace_path: Path | None = None,
) -> PairRun:
    """Run one factual baseline and, when the branch is legal, both arms.

    Each call creates its own baseline. A shared-checkpoint scheduler is
    Milestone 2B and is not implemented here. ``phase="harness"`` marks a
    scripted control-flow trace. ``model_id="fake"`` must use that phase.
    Those rows are not evaluation trials.
    """

    if model_id == "fake" and phase != "harness":
        raise ValueError("a fake model trace must use phase harness")
    fixture = get_fixture(incident_id)
    if fixture is None:
        raise ValueError(f"unknown incident: {incident_id}")
    try:
        intervention = INTERVENTIONS[intervention_id]
    except KeyError as exc:
        raise ValueError(f"unknown intervention: {intervention_id}") from exc
    pair_id = uuid.uuid4().hex
    source_hash = intervention_source_sha256()
    sdk_version = version("mcp")
    async with Client(
        create_server(),
        read_timeout_seconds=15,
        sampling_callback=_refuse_sampling,
        elicitation_callback=_refuse_elicitation,
    ) as client:
        listed = await client.list_tools()
        names = tuple(tool.name for tool in listed.tools)
        if names != TOOL_ORDER:
            raise RuntimeError(f"unexpected tools: {names}")
        protocol = _protocol_version(client)
        schemas = _tool_schemas(listed.tools)
        state = build_initial_state(
            incident_id=incident_id,
            model_id=model_id,
            model_params=model_params,
            step_budget=step_budget,
            token_budget=token_budget,
            fixture_hash=fixture_hash(fixture),
            tool_schemas=schemas,
        )
        recorder = CheckpointRecorder(intervention.tool_name, incident_id)
        baseline_started = utc_now()
        baseline = await run_agent(
            client,
            state,
            adapter,
            resolver=FactualResolver(),
            checkpoint=recorder,
        )
        common = _Common(
            pair_id=pair_id,
            incident_id=incident_id,
            fixture_hash=state.fixture_hash,
            model_id=model_id,
            model_params=dict(model_params),
            prompt_hash=state.prompt_hash,
            sdk_version=sdk_version,
            protocol=protocol,
            phase=phase,
            scenario_id=scenario_id,
            repeat_index=repeat_index,
            source_hash=source_hash,
        )
        factual_bits = _fixture_bits(fixture)
        factual_oracle = decide(fixture.metrics, fixture.logs, fixture.deployment).value
        checkpoint = recorder.checkpoint
        baseline_row = _row_from_result(
            common,
            baseline,
            arm="baseline",
            intervention_id=None,
            snapshot_id=None,
            prefix=None,
            oracle=factual_oracle,
            bits=factual_bits,
            observation_original=checkpoint.visible if checkpoint is not None else None,
            observation_mutated=None,
            extra_flags=(),
            error=baseline.error,
            started_at=baseline_started,
            ended_at=utc_now(),
        )
        if checkpoint is None:
            mutant_row = await _unobserved_row(
                client,
                common,
                fixture,
                intervention_id,
                baseline.error,
            )
            run = PairRun(
                baseline=baseline_row,
                factual=None,
                mutant=mutant_row,
                baseline_messages=list(baseline.state.messages),
                factual_messages=None,
                mutant_messages=None,
                snapshot=None,
                prefix_sha256=None,
            )
            _write(trace_path, run)
            return run
        mutation = apply_intervention(
            fixture,
            intervention_id,
            captured=_structured(checkpoint),
        )
        if not mutation.ok or mutation.mutated_visible is None or mutation.original_visible != checkpoint.visible:
            reason = mutation.error or "captured observation did not match the mutated input"
            mutant_row = _blocked_row(
                common,
                intervention_id=intervention_id,
                flag="engine_error",
                error=reason,
                observation_original=checkpoint.visible,
                observation_mutated=None,
                oracle=None,
                bits=None,
            )
            run = PairRun(
                baseline=baseline_row,
                factual=None,
                mutant=mutant_row,
                baseline_messages=list(baseline.state.messages),
                factual_messages=None,
                mutant_messages=None,
                snapshot=checkpoint,
                prefix_sha256=None,
            )
            _write(trace_path, run)
            return run
        factual_state = copy_state(checkpoint.state)
        mutant_state = copy_state(checkpoint.state)
        try:
            shared_hash = check_prefix_pair(factual_state, mutant_state)
        except PrefixMismatch as exc:
            mutant_row = _blocked_row(
                common,
                intervention_id=intervention_id,
                flag="engine_error",
                error=f"prefix hashes differ: {exc.left} {exc.right}",
                observation_original=checkpoint.visible,
                observation_mutated=None,
                oracle=None,
                bits=None,
            )
            run = PairRun(
                baseline=baseline_row,
                factual=None,
                mutant=mutant_row,
                baseline_messages=list(baseline.state.messages),
                factual_messages=None,
                mutant_messages=None,
                snapshot=checkpoint,
                prefix_sha256=None,
            )
            _write(trace_path, run)
            return run
        restore(factual_state, checkpoint, content=checkpoint.visible, source="base")
        restore(mutant_state, checkpoint, content=mutation.mutated_visible, source="overlay")
        factual_started = utc_now()
        factual_result = await run_agent(client, factual_state, adapter, resolver=FactualResolver())
        factual_ended = utc_now()
        mutant_started = utc_now()
        mutant_result = await run_agent(
            client,
            mutant_state,
            adapter,
            resolver=OverlayResolver(
                {intervention.tool_name: mutation.mutated_visible},
                incident_id=incident_id,
            ),
        )
        mutant_ended = utc_now()
        factual_row = _row_from_result(
            common,
            factual_result,
            arm="factual_replay",
            intervention_id=intervention_id,
            snapshot_id=shared_hash,
            prefix=shared_hash,
            oracle=factual_oracle,
            bits=factual_bits,
            observation_original=checkpoint.visible,
            observation_mutated=None,
            extra_flags=(),
            error=factual_result.error,
            started_at=factual_started,
            ended_at=factual_ended,
        )
        mutant_bits = mutation.predicates_after
        mutant_row = _row_from_result(
            common,
            mutant_result,
            arm="mutant",
            intervention_id=intervention_id,
            snapshot_id=shared_hash,
            prefix=shared_hash,
            oracle=mutation.oracle_after,
            bits=mutant_bits,
            observation_original=checkpoint.visible,
            observation_mutated=mutation.mutated_visible,
            extra_flags=(),
            error=mutant_result.error,
            started_at=mutant_started,
            ended_at=mutant_ended,
        )
        run = PairRun(
            baseline=baseline_row,
            factual=factual_row,
            mutant=mutant_row,
            baseline_messages=list(baseline.state.messages),
            factual_messages=list(factual_result.state.messages),
            mutant_messages=list(mutant_result.state.messages),
            snapshot=checkpoint,
            prefix_sha256=shared_hash,
        )
        _write(trace_path, run)
        return run


@dataclass(frozen=True, slots=True)
class _Common:
    pair_id: str
    incident_id: str
    fixture_hash: str
    model_id: str
    model_params: dict[str, Any]
    prompt_hash: str
    sdk_version: str
    protocol: str | None
    phase: Phase
    scenario_id: str | None
    repeat_index: int
    source_hash: str


async def _unobserved_row(
    client: Client,
    common: _Common,
    fixture: IncidentFixture,
    intervention_id: str,
    baseline_error: str | None,
) -> TraceRecord:
    intervention = INTERVENTIONS[intervention_id]
    result = await client.call_tool(intervention.tool_name, {"incident_id": common.incident_id})
    if not isinstance(result, CallToolResult) or result.is_error or not isinstance(result.structured_content, dict):
        return _blocked_row(
            common,
            intervention_id=intervention_id,
            flag="engine_error",
            error="ledger capture of the intervention tool failed",
            observation_original=None,
            observation_mutated=None,
            oracle=None,
            bits=None,
        )
    captured = capture_result(result)
    mutation = apply_intervention(
        fixture,
        intervention_id,
        captured=result.structured_content,
    )
    if not mutation.ok or mutation.mutated_visible is None:
        return _blocked_row(
            common,
            intervention_id=intervention_id,
            flag="engine_error",
            error=mutation.error or "mutation failed",
            observation_original=captured.visible,
            observation_mutated=None,
            oracle=None,
            bits=None,
        )
    started = utc_now()
    return _blocked_row(
        common,
        intervention_id=intervention_id,
        flag="intervention_unobserved",
        error=baseline_error or "no eligible checkpoint",
        observation_original=captured.visible,
        observation_mutated=mutation.mutated_visible,
        oracle=mutation.oracle_after,
        bits=mutation.predicates_after,
        started_at=started,
    )


def _blocked_row(
    common: _Common,
    *,
    intervention_id: str,
    flag: str,
    error: str,
    observation_original: str | None,
    observation_mutated: str | None,
    oracle: str | None,
    bits: tuple[bool, bool, bool] | None,
    started_at: str | None = None,
) -> TraceRecord:
    started = started_at or utc_now()
    return TraceRecord(
        run_id=uuid.uuid4().hex,
        pair_id=common.pair_id,
        arm="mutant",
        incident_id=common.incident_id,
        fixture_hash=common.fixture_hash,
        intervention_id=intervention_id,
        model_id=common.model_id,
        model_params=common.model_params,
        prompt_hash=common.prompt_hash,
        sdk_version=common.sdk_version,
        negotiated_protocol_version=common.protocol,
        snapshot_id=None,
        phase=common.phase,
        scenario_id=common.scenario_id,
        repeat_index=common.repeat_index,
        tool_calls=[],
        observation_original=observation_original,
        observation_mutated=observation_mutated,
        model_visible_observations=[],
        usage=None,
        decision_raw=None,
        decision_parsed=None,
        action_observed=None,
        parse_ok=False,
        oracle_decision=oracle,
        predicate_m=None if bits is None else bits[0],
        predicate_l=None if bits is None else bits[1],
        predicate_d=None if bits is None else bits[2],
        flags=[flag],
        error=error,
        started_at=started,
        ended_at=utc_now(),
        prefix_sha256=None,
        intervention_source_sha256=common.source_hash,
    )


def _row_from_result(
    common: _Common,
    result: AgentResult,
    *,
    arm: Arm,
    intervention_id: str | None,
    snapshot_id: str | None,
    prefix: str | None,
    oracle: str | None,
    bits: tuple[bool, bool, bool] | None,
    observation_original: str | None,
    observation_mutated: str | None,
    extra_flags: Sequence[str],
    error: str | None,
    started_at: str,
    ended_at: str,
) -> TraceRecord:
    flags = _merge_flags(result.flags, extra_flags)
    return TraceRecord(
        run_id=uuid.uuid4().hex,
        pair_id=common.pair_id,
        arm=arm,
        incident_id=common.incident_id,
        fixture_hash=common.fixture_hash,
        intervention_id=intervention_id,
        model_id=common.model_id,
        model_params=common.model_params,
        prompt_hash=common.prompt_hash,
        sdk_version=common.sdk_version,
        negotiated_protocol_version=common.protocol,
        snapshot_id=snapshot_id,
        phase=common.phase,
        scenario_id=common.scenario_id,
        repeat_index=common.repeat_index,
        tool_calls=list(result.state.tool_calls),
        observation_original=observation_original,
        observation_mutated=observation_mutated,
        model_visible_observations=_visible(result.state),
        usage=usage_payload(result.state),
        decision_raw=result.decision_raw,
        decision_parsed=result.decision_parsed,
        action_observed=result.action_observed,
        parse_ok=result.parse_ok,
        oracle_decision=oracle,
        predicate_m=None if bits is None else bits[0],
        predicate_l=None if bits is None else bits[1],
        predicate_d=None if bits is None else bits[2],
        flags=flags,
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        prefix_sha256=prefix,
        intervention_source_sha256=common.source_hash,
        execution_status=result.execution_status,
    )


def _visible(state: AgentState) -> list[str]:
    return [message.content for message in state.messages if message.role == "tool"]


def _merge_flags(primary: Sequence[str], extra: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for flag in (*primary, *extra):
        if flag not in merged:
            merged.append(flag)
    return merged


def _fixture_bits(fixture: IncidentFixture) -> tuple[bool, bool, bool]:
    return (
        predicate_m(fixture.metrics),
        predicate_l(fixture.logs),
        predicate_d(fixture.deployment),
    )


def _structured(checkpoint: Checkpoint) -> dict[str, Any]:
    structured = checkpoint.raw.get("structured_content")
    if not isinstance(structured, dict):
        return {}
    return structured


def _prefix_is_factual(state: AgentState) -> bool:
    return all(record.source == "base" and record.is_error is False and record.dispatched for record in state.tool_calls)


def _tool_schemas(tools: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in tools:
        rows.append(
            {
                "description": tool.description,
                "input_schema": _json_ready(tool.input_schema),
                "name": tool.name,
                "output_schema": _json_ready(tool.output_schema),
            }
        )
    return rows


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _protocol_version(client: Client) -> str | None:
    version_value = client.protocol_version
    if isinstance(version_value, str) and version_value:
        return version_value
    return None


def _write(path: Path | None, run: PairRun) -> None:
    if path is None:
        return
    append_trace(path, run.baseline)
    if run.factual is not None:
        append_trace(path, run.factual)
    append_trace(path, run.mutant)


async def _refuse_sampling(context: object, params: object) -> object:
    del context, params
    raise AssertionError("server requested sampling")


async def _refuse_elicitation(context: object, params: object) -> object:
    del context, params
    raise AssertionError("server requested elicitation")
