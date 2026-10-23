"""One-incident live smoke runner.

The command loads ``<project-root>/.env`` and asks the OpenAI adapter for
completions. Tool calls still go through the in-process MCP client.
``--dry-run`` prints the plan and does not construct a client.
``--trace-summary`` prints the tool ledger for this process.
``--trace`` appends the same row to a JSONL file. Neither flag
reconstructs an earlier run.

This is not the paired-evaluation scheduler. A smoke row uses
``phase="harness"`` and does not score the oracle.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anyio
from mcp import Client

from parallax.agent import AgentResult, FactualResolver, build_initial_state, run_agent
from parallax.fixtures import get_fixture
from parallax.openai_adapter import (
    OpenAIAdapter,
    load_project_env,
    low_reasoning_effort_supported,
)
from parallax.replay import TOOL_ORDER, fixture_hash, usage_payload
from parallax.server import create_server
from parallax.trace import TraceRecord, append_trace, utc_now

_DEFAULT_MODEL_CALLS = 4
_DEFAULT_TOOL_CALLS = 6
_DEBUG_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_DEBUG_KEY = re.compile(r"^[a-z_]{1,32}$")
_SMOKE_SOURCE = hashlib.sha256(b"parallax-live-smoke").hexdigest()


def format_dry_run(
    *,
    incident_id: str,
    environ: Mapping[str, str],
    model_calls: int | None = None,
    tool_calls: int | None = None,
) -> str:
    """Describe one smoke run. The key value is not included."""

    key_state = "set" if environ.get("OPENAI_API_KEY", "").strip() else "missing"
    model = environ.get("PARALLAX_MODEL", "").strip() or "(unset)"
    lines = [
        "dry-run: no model request will be sent",
        f"incident_id={incident_id}",
        f"model={model}",
        f"OPENAI_API_KEY={key_state}",
        "model_call_budget="
        + _budget_label(model_calls, environ, "PARALLAX_MAX_MODEL_CALLS", _DEFAULT_MODEL_CALLS),
        "tool_call_budget="
        + _budget_label(tool_calls, environ, "PARALLAX_MAX_TOOL_CALLS", _DEFAULT_TOOL_CALLS),
        "tools=get_metrics,get_logs,get_deployment",
    ]
    return "\n".join(lines) + "\n"


def smoke_model_params(model: str) -> dict[str, Any]:
    """Parameters stored for one smoke run and passed to the adapter.

    GPT-6 accepts ``reasoning.effort="low"``. The model id is unchanged.
    Temperature, top_p, top_logprobs, and seed are not part of that request.
    """

    if low_reasoning_effort_supported(model):
        return {"reasoning_effort": "low"}
    return {"temperature": 0}


def configured_budget(explicit: int | None, environ: Mapping[str, str], name: str, default: int) -> int:
    """Resolve one budget. The error text does not include the raw value."""

    if explicit is not None:
        if explicit < 1:
            raise ValueError(f"{name} must be an integer of at least 1")
        return explicit
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer of at least 1") from exc
    if value < 1:
        raise ValueError(f"{name} must be an integer of at least 1")
    return value


def smoke_record(
    result: AgentResult,
    *,
    sdk_version: str,
    protocol: str | None,
    started_at: str,
    ended_at: str,
) -> TraceRecord:
    """Ledger row for one smoke attempt. Oracle fields stay empty."""

    state = result.state
    usage = usage_payload(state)
    return TraceRecord(
        run_id=uuid.uuid4().hex,
        pair_id=uuid.uuid4().hex,
        arm="baseline",
        incident_id=state.incident_id,
        fixture_hash=state.fixture_hash,
        intervention_id=None,
        model_id=state.model_id,
        model_params=dict(state.model_params),
        prompt_hash=state.prompt_hash,
        sdk_version=sdk_version,
        negotiated_protocol_version=protocol,
        snapshot_id=None,
        phase="harness",
        scenario_id=None,
        repeat_index=0,
        tool_calls=list(state.tool_calls),
        observation_original=None,
        observation_mutated=None,
        model_visible_observations=[message.content for message in state.messages if message.role == "tool"],
        usage=usage,
        decision_raw=result.decision_raw,
        decision_parsed=result.decision_parsed,
        action_observed=result.action_observed,
        parse_ok=result.parse_ok,
        oracle_decision=None,
        predicate_m=None,
        predicate_l=None,
        predicate_d=None,
        flags=list(result.flags),
        error=result.error,
        started_at=started_at,
        ended_at=ended_at,
        prefix_sha256=None,
        intervention_source_sha256=_SMOKE_SOURCE,
        execution_status=result.execution_status,
        provider_response_ids=list(state.provider_response_ids),
    )


def format_smoke_summary(result: AgentResult) -> str:
    """Tool ledger taken from this run's agent state.

    Observation text is omitted. The lines do not reconstruct a prior
    process that did not save a trace.
    """

    state = result.state
    dispatched = sum(1 for call in state.tool_calls if call.dispatched)
    names = ",".join(call.name for call in state.tool_calls) or "(none)"
    failed = [call.name for call in state.tool_calls if call.is_error]
    lines = [
        "phase=harness",
        f"model_calls={state.steps_used}",
        f"mcp_tool_calls={dispatched}",
        f"tool_names={names}",
        "tool_errors=" + (",".join(failed) if failed else "(none)"),
    ]
    for index, call in enumerate(state.tool_calls):
        error = "null" if call.is_error is None else str(call.is_error).lower()
        dispatched_label = str(call.dispatched).lower()
        lines.append(
            f"tool.{index} name={call.name} dispatched={dispatched_label} "
            f"is_error={error} source={call.source}"
        )
    return "\n".join(lines) + "\n"


async def run_incident(
    incident_id: str,
    model_calls: int,
    tool_calls: int,
    trace_path: Path | None,
    debug_errors: bool = False,
    trace_summary: bool = False,
) -> int:
    """Run one factual incident. The caller has already resolved the budgets."""

    fixture = get_fixture(incident_id)
    if fixture is None:
        print(f"unknown incident: {incident_id}", file=sys.stderr)
        return 2
    try:
        adapter = OpenAIAdapter.from_env()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    started = utc_now()
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
        protocol_value = client.protocol_version
        protocol = protocol_value if isinstance(protocol_value, str) and protocol_value else None
        state = build_initial_state(
            incident_id=incident_id,
            model_id=adapter.model,
            model_params=smoke_model_params(adapter.model),
            step_budget=model_calls,
            token_budget=None,
            fixture_hash=fixture_hash(fixture),
            tool_schemas=_tool_schemas(listed.tools),
            tool_call_budget=tool_calls,
        )
        result = await run_agent(client, state, adapter, resolver=FactualResolver())
    ended = utc_now()
    if trace_path is not None:
        append_trace(
            trace_path,
            smoke_record(
                result,
                sdk_version=version("mcp"),
                protocol=protocol,
                started_at=started,
                ended_at=ended,
            ),
        )
    _print_result(result, debug_errors=debug_errors)
    if trace_summary or trace_path is not None:
        sys.stdout.write(format_smoke_summary(result))
    if trace_path is not None:
        print(f"trace_path={trace_path}")
    return 0 if result.execution_status == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. ``--dry-run`` stops before a provider client exists."""

    parser = argparse.ArgumentParser(prog="python -m parallax.live")
    parser.add_argument("--incident", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-calls", type=int, default=None)
    parser.add_argument("--tool-calls", type=int, default=None)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--trace-summary", action="store_true")
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.dry_run:
        load_project_env()
        sys.stdout.write(
            format_dry_run(
                incident_id=args.incident,
                environ=dict(os_environ()),
                model_calls=args.model_calls,
                tool_calls=args.tool_calls,
            )
        )
        return 0
    try:
        model_calls = configured_budget(
            args.model_calls, os_environ(), "PARALLAX_MAX_MODEL_CALLS", _DEFAULT_MODEL_CALLS
        )
        tool_calls = configured_budget(
            args.tool_calls, os_environ(), "PARALLAX_MAX_TOOL_CALLS", _DEFAULT_TOOL_CALLS
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.trace is not None and args.trace.suffix.lower() != ".jsonl":
        print("trace path must end in .jsonl", file=sys.stderr)
        return 2
    return anyio.run(
        run_incident,
        args.incident,
        model_calls,
        tool_calls,
        args.trace,
        args.debug_errors,
        args.trace_summary,
    )


def os_environ() -> Mapping[str, str]:
    """Process environment after the caller has loaded the project file."""

    return os.environ


def format_debug_report(error: str | None) -> str:
    """One safe diagnostic line. Unstructured exception text is dropped."""

    if error and error.startswith("ProviderCallError: "):
        kept: list[str] = []
        for token in error.removeprefix("ProviderCallError: ").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if not _DEBUG_KEY.fullmatch(key) or not _DEBUG_TOKEN.fullmatch(value):
                continue
            if "sk-" in value.lower() or "bearer" in value.lower():
                continue
            kept.append(f"{key}={value}")
        if kept:
            return " ".join(kept)
    exception = "Exception"
    if error and ":" in error:
        name = error.split(":", 1)[0].strip()
        if name.isidentifier():
            exception = name
    elif error and error.isidentifier():
        exception = error
    return f"stage=agent_controller exception={exception}"


def _print_result(result: AgentResult, *, debug_errors: bool = False) -> None:
    state = result.state
    usage = usage_payload(state)
    print(f"execution_status={result.execution_status}")
    print(f"decision={result.decision_parsed}")
    print(f"model_id={state.model_id}")
    print("provider_response_ids=" + ",".join(state.provider_response_ids))
    if usage is None:
        print("usage=null")
    else:
        print(f"usage_input_tokens={usage['input_tokens']}")
        print(f"usage_output_tokens={usage['output_tokens']}")
    if debug_errors and result.execution_status == "model_error":
        print(format_debug_report(result.error))


def _budget_label(explicit: int | None, environ: Mapping[str, str], name: str, default: int) -> str:
    if explicit is not None:
        return str(explicit) if explicit >= 1 else "invalid"
    raw = environ.get(name, "").strip()
    if not raw:
        return str(default)
    try:
        value = int(raw)
    except ValueError:
        return "invalid"
    return str(value) if value >= 1 else "invalid"


def _tool_schemas(tools: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in tools:
        rows.append(
            {
                "description": tool.description,
                "input_schema": json.loads(json.dumps(tool.input_schema)),
                "name": tool.name,
                "output_schema": json.loads(json.dumps(tool.output_schema)),
            }
        )
    return rows


async def _refuse_sampling(context: object, params: object) -> object:
    del context, params
    raise AssertionError("server requested sampling")


async def _refuse_elicitation(context: object, params: object) -> object:
    del context, params
    raise AssertionError("server requested elicitation")


if __name__ == "__main__":
    raise SystemExit(main())
