"""Bounded counterfactual evidence search.

One baseline reads the source fixture. Each candidate continues from a
copy of the pre-fork checkpoint with one allowlisted integer field
edited. The oracle is not used to order candidates, stop the search, or
decide whether the model changed its decision. Minimality is a claim
about the enumerated candidate set only.

The search does not write fixture files, does not register scenarios,
and does not call the experiment runner. Rows use ``phase=harness`` so
they are not protocol rates.
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp import Client
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from parallax.agent import (
    OUTPUT_DECISIONS,
    AgentResult,
    AgentState,
    FactualResolver,
    LoopBudget,
    ModelAdapter,
    build_initial_state,
    copy_state,
    run_agent,
)
from parallax.capture import sha256_text
from parallax.fixtures import get_fixture
from parallax.mutate import TOOL_MODELS
from parallax.observe import canonical_json
from parallax.prompt import prompt_hash
from parallax.replay import (
    TOOL_ORDER,
    Checkpoint,
    CheckpointRecorder,
    OverlayResolver,
    _carry_same_turn_results,
    _protocol_version,
    _refuse_elicitation,
    _refuse_sampling,
    _tool_schemas,
    check_prefix_pair,
    fixture_hash,
    prefix_sha256,
    restore,
    usage_payload,
)
from parallax.server import create_server
from parallax.trace import (
    TRACE_SCHEMA_VERSION,
    TraceRecord,
    append_trace,
    drop_torn_jsonl_tail,
    utc_now,
)

XRAY_SCHEMA_VERSION = 2
DEFAULT_CANDIDATES = (499, 501, 498, 502, 495, 505)
DEFAULT_MAX_CANDIDATES = 6
DEFAULT_MODEL_CALLS = 4
DEFAULT_TOOL_CALLS = 6
DISTANCE_METRIC = "absolute_change_in_field_units"
SEARCHABLE_FIELDS = frozenset({("get_metrics", "error_rate_bps")})
MINIMALITY_DEFINITION = (
    "Distance is the absolute change in one allowlisted integer field. "
    "A minimum edit distance is proven over the enumerated candidate space only when "
    "a verified flip exists and every enumerated candidate at a smaller distance was "
    "a conclusive no-flip. Otherwise a flip is the smallest verified flip among tested "
    "edits. A finished search with no verified flip is no-flip when every candidate was "
    "conclusive, and inconclusive due to invalid candidates when any candidate was not. "
    "This is not a minimum over other fields, tools, or untested values."
)
_PROVEN_CLAIM = "minimum edit distance proven over the enumerated candidate space"
_TESTED_CLAIM = "smallest verified flip among tested edits"
_NO_FLIP_CLAIM = "no verified flip"
_INCONCLUSIVE_CLAIM = "inconclusive due to invalid candidates"
_SECRET_MARKERS = ("sk-", "api_key", "bearer ", "authorization")
Outcome = Literal[
    "verified_flip",
    "no_flip",
    "invalid_output",
    "provider_error",
    "mcp_error",
    "budget_exhausted",
    "intervention_missing",
    "observation_not_visible",
    "not_run",
]
Strategy = Literal["early_stop", "exhaustive"]


class SearchSpec(BaseModel):
    """Immutable validated search. Candidate values are in execution order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: str
    tool_name: str
    field_path: str
    candidate_values: tuple[int, ...]
    distances: tuple[int, ...]
    original_value: int
    max_candidates: int
    model_id: str
    model_params: dict[str, Any]
    model_call_budget: int
    tool_call_budget: int
    strategy: Strategy
    distance_metric: str


class SearchPlan(BaseModel):
    """Provider-independent order. Construction does not read the oracle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_value: int
    ordered_values: tuple[int, ...]
    distances: tuple[int, ...]
    strategy: Strategy
    distance_metric: str


class EditedEvidence(BaseModel):
    """One schema-valid field edit of a captured observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    field_path: str
    original_value: int
    candidate_value: int
    observation: str
    source_sha256: str
    evidence_sha256: str


class XRaySpec(BaseModel):
    """Versioned search inputs. The oracle decision is not an input."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    xray_id: str
    fixture_id: str
    tool: str
    field: str
    original_value: int
    candidates: list[int]
    distances: list[int]
    target_decision: str | None
    model_id: str
    model_params: dict[str, Any]
    prompt_hash: str
    fixture_hash: str
    model_call_budget: int
    tool_call_budget: int
    max_candidates: int
    strategy: Strategy
    distance_metric: str
    source_evidence_sha256: str
    git_revision: str
    worst_case_model_requests: int
    minimality_definition: str
    search_source_sha256: str
    software_version: str
    sdk_version: str
    python_version: str
    trace_schema_version: int
    phase: Literal["harness"] = "harness"
    notes: list[str] = Field(default_factory=list)


class StoredBaseline(BaseModel):
    """Durable factual execution. Later resumes do not repeat it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    xray_id: str
    checkpoint: dict[str, Any] | None
    final_state: dict[str, Any]
    decision_parsed: str | None
    parse_ok: bool
    execution_status: str | None
    flags: list[str]
    error: str | None
    observation_original: str | None
    observation_sha256: str | None
    captured_value: int | None
    prefix_sha256: str | None
    prefix_usage: dict[str, int] | None
    usage: dict[str, int] | None
    model_calls: int
    provider_response_ids: list[str]
    factual_valid: bool
    stop_reason: str | None


class CandidateRecord(BaseModel):
    """One executed candidate. A missing line was not durable."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    xray_id: str
    candidate_value: int
    distance: int
    order_index: int
    status: Outcome
    decision_parsed: str | None
    flipped: bool
    execution_valid: bool
    intervention_applied: bool
    model_visible: bool
    factual_decision: str | None
    evidence_sha256_original: str | None
    evidence_sha256_mutated: str | None
    checkpoint_id: str | None
    execution_status: str | None
    flags: list[str]
    error: str | None
    oracle_before: str | None
    oracle_after: str | None
    usage: dict[str, int] | None
    model_calls: int
    provider_response_ids: list[str]


class SearchRow(BaseModel):
    """One declared candidate, including those the search did not run."""

    model_config = ConfigDict(extra="forbid")

    candidate_value: int
    distance: int
    order_index: int
    status: Outcome
    decision: str | None
    flipped: bool
    execution_valid: bool
    evidence_sha256_original: str | None
    evidence_sha256_mutated: str | None
    checkpoint_id: str | None
    oracle_before: str | None
    oracle_after: str | None
    error: str | None
    model_calls: int


class XRayResult(BaseModel):
    """Assembled search record. Commits remain the durable source."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    xray_id: str
    original_decision: str | None
    termination_reason: str
    minimality_definition: str
    minimality_claim: str
    minimal_edit: dict[str, Any] | None
    proof_blockers: list[int]
    shared_checkpoint: bool
    checkpoint_id: str | None
    candidates: list[SearchRow]
    provider_requests: int
    usage_deduped: dict[str, int] | None
    target_decision: str | None
    target_reached: bool
    notes: list[str]


def field_distance(original: int, candidate: int) -> int:
    """Absolute change in the selected field's own units."""

    return abs(candidate - original)


def order_candidates(original: int, candidates: Sequence[int]) -> list[int]:
    """Nearest edits first. Equal distances use the smaller value."""

    return sorted(candidates, key=lambda value: (field_distance(original, value), value))


def request_ceiling(model_calls: int, candidate_count: int) -> int:
    """Worst-case provider requests for one shared checkpoint.

    The baseline may use the whole per-branch budget. A fork on the first
    call leaves one fewer call for every candidate continuation.
    """

    if model_calls < 1 or candidate_count < 1:
        return 0
    return model_calls + candidate_count * (model_calls - 1)


def classify_outcome(
    *,
    factual_status: str | None,
    factual_parsed: bool,
    factual_decision: str | None,
    mutant_status: str | None,
    mutant_parsed: bool,
    mutant_decision: str | None,
    intervention_applied: bool,
    model_visible: bool,
    factual_flags: Sequence[str],
    mutant_flags: Sequence[str],
) -> Outcome:
    """Decide a flip from executions only. This does not call the oracle."""

    flags = (*factual_flags, *mutant_flags)
    statuses = {factual_status, mutant_status}
    if "provider_error" in flags or "model_error" in statuses:
        return "provider_error"
    if "tool_error" in flags or "tool_error" in statuses or "mcp_error" in flags:
        return "mcp_error"
    if "budget_exhausted" in flags or "budget_exhausted" in statuses:
        return "budget_exhausted"
    if not intervention_applied:
        return "intervention_missing"
    if not model_visible:
        return "observation_not_visible"
    factual_ok = factual_parsed and factual_status == "completed" and factual_decision is not None
    mutant_ok = mutant_parsed and mutant_status == "completed" and mutant_decision is not None
    if not factual_ok or not mutant_ok:
        return "invalid_output"
    if factual_decision != mutant_decision:
        return "verified_flip"
    return "no_flip"


def plan_search(original: int, candidates: Sequence[int], strategy: Strategy) -> SearchPlan:
    """Order a finite set. This function does not call a provider or the oracle."""

    if strategy not in ("early_stop", "exhaustive"):
        raise ValueError("strategy must be early_stop or exhaustive")
    ordered = tuple(order_candidates(original, candidates))
    return SearchPlan(
        original_value=original,
        ordered_values=ordered,
        distances=tuple(field_distance(original, candidate) for candidate in ordered),
        strategy=strategy,
        distance_metric=DISTANCE_METRIC,
    )


def make_search_spec(
    *,
    fixture_id: str,
    tool_name: str,
    field_path: str,
    candidate_values: Sequence[int] | None,
    max_candidates: int,
    model_id: str,
    model_params: Mapping[str, Any],
    model_call_budget: int,
    tool_call_budget: int,
    strategy: Strategy = "early_stop",
) -> SearchSpec:
    """Validate one allowlisted field search and freeze its execution order."""

    if max_candidates < 1:
        raise ValueError("max candidates must be an integer of at least 1")
    if model_call_budget < 1 or tool_call_budget < 1:
        raise ValueError("budgets must be integers of at least 1")
    model = model_id.strip()
    if not model:
        raise ValueError("model id is required")
    fixture = get_fixture(fixture_id)
    if fixture is None:
        raise ValueError(f"unknown incident: {fixture_id}")
    model_type = _searchable_model(tool_name, field_path)
    payload = _tool_payload(fixture, tool_name)
    original = _stored_integer(payload, field_path)
    selected = list(DEFAULT_CANDIDATES if candidate_values is None else candidate_values)
    if len(selected) > max_candidates:
        raise ValueError("candidate count exceeds the maximum")
    if not selected:
        raise ValueError("at least one candidate is required")
    _reject_duplicate_candidates(selected)
    for candidate in selected:
        _validate_candidate(model_type, payload, field_path, original, candidate)
    plan = plan_search(original, selected, strategy)
    return SearchSpec(
        fixture_id=fixture_id,
        tool_name=tool_name,
        field_path=field_path,
        candidate_values=plan.ordered_values,
        distances=plan.distances,
        original_value=original,
        max_candidates=max_candidates,
        model_id=model,
        model_params=dict(model_params),
        model_call_budget=model_call_budget,
        tool_call_budget=tool_call_budget,
        strategy=plan.strategy,
        distance_metric=plan.distance_metric,
    )


def build_spec(
    *,
    fixture_id: str,
    tool: str,
    field: str,
    candidates: Sequence[int] | None,
    target_decision: str | None,
    model_id: str,
    model_params: Mapping[str, Any],
    model_call_budget: int,
    tool_call_budget: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    strategy: Strategy = "early_stop",
) -> XRaySpec:
    """Validate a finite search. Candidate order does not read the oracle."""

    if target_decision is not None and target_decision not in OUTPUT_DECISIONS:
        raise ValueError("target decision must be MONITOR, INVESTIGATE, or ESCALATE")
    search = make_search_spec(
        fixture_id=fixture_id,
        tool_name=tool,
        field_path=field,
        candidate_values=candidates,
        max_candidates=max_candidates,
        model_id=model_id,
        model_params=model_params,
        model_call_budget=model_call_budget,
        tool_call_budget=tool_call_budget,
        strategy=strategy,
    )
    fixture = get_fixture(fixture_id)
    if fixture is None:
        raise ValueError(f"unknown incident: {fixture_id}")
    payload = _tool_payload(fixture, search.tool_name)
    ordered = list(search.candidate_values)
    ceiling = request_ceiling(search.model_call_budget, len(ordered))
    params = dict(search.model_params)
    source_evidence_sha256 = sha256_text(canonical_json(payload))
    identity = {
        "candidates": ordered,
        "distance_metric": search.distance_metric,
        "field": search.field_path,
        "fixture_hash": fixture_hash(fixture),
        "fixture_id": fixture_id,
        "max_candidates": max_candidates,
        "model_call_budget": search.model_call_budget,
        "model_id": search.model_id,
        "model_params": params,
        "original_value": search.original_value,
        "phase": "harness",
        "prompt_hash": prompt_hash(fixture_id),
        "schema_version": XRAY_SCHEMA_VERSION,
        "search_source_sha256": _source_sha256(),
        "source_evidence_sha256": source_evidence_sha256,
        "strategy": search.strategy,
        "target_decision": target_decision,
        "tool": search.tool_name,
        "tool_call_budget": search.tool_call_budget,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }
    notes = [
        "The factual decision is the single baseline sample.",
        "Candidates share that pre-fork checkpoint.",
        "Post-fork samples are one draw each.",
        "The oracle is recorded after the flip decision and does not order the search.",
        "Early stop leaves later candidates untested. Exhaustive mode runs every declared candidate.",
        MINIMALITY_DEFINITION,
    ]
    return XRaySpec(
        schema_version=XRAY_SCHEMA_VERSION,
        xray_id="xray-" + sha256_text(canonical_json(identity))[:16],
        fixture_id=fixture_id,
        tool=search.tool_name,
        field=search.field_path,
        original_value=search.original_value,
        candidates=ordered,
        distances=list(search.distances),
        target_decision=target_decision,
        model_id=search.model_id,
        model_params=params,
        prompt_hash=str(identity["prompt_hash"]),
        fixture_hash=str(identity["fixture_hash"]),
        model_call_budget=search.model_call_budget,
        tool_call_budget=search.tool_call_budget,
        max_candidates=max_candidates,
        strategy=search.strategy,
        distance_metric=search.distance_metric,
        source_evidence_sha256=source_evidence_sha256,
        git_revision=_git_revision(),
        worst_case_model_requests=ceiling,
        minimality_definition=MINIMALITY_DEFINITION,
        search_source_sha256=str(identity["search_source_sha256"]),
        software_version=_package_version("parallax", "0.1.0"),
        sdk_version=_package_version("mcp", "unknown"),
        python_version=platform.python_version(),
        trace_schema_version=TRACE_SCHEMA_VERSION,
        notes=notes,
    )


def format_dry_run(spec: XRaySpec) -> str:
    """Print the plan. No provider client is constructed by this function."""

    lines = [
        "dry-run: no model request will be sent",
        f"xray_id={spec.xray_id}",
        f"fixture={spec.fixture_id}",
        f"tool={spec.tool}",
        f"field={spec.field}",
        f"original_value={spec.original_value}",
        f"model={spec.model_id}",
        f"phase={spec.phase}",
        "shared_checkpoint=true",
        "oracle=post-hoc",
        f"target_decision={spec.target_decision}",
        "target_source=operator" if spec.target_decision else "target_source=none",
        f"strategy={spec.strategy}",
        f"distance_metric={spec.distance_metric}",
        f"model_call_budget={spec.model_call_budget}",
        f"tool_call_budget={spec.tool_call_budget}",
        f"max_candidates={spec.max_candidates}",
        f"worst_case_model_requests={spec.worst_case_model_requests}",
        f"source_evidence_sha256={spec.source_evidence_sha256}",
        f"git_revision={spec.git_revision}",
        f"prompt_hash={spec.prompt_hash}",
        f"fixture_hash={spec.fixture_hash}",
        f"search_source_sha256={spec.search_source_sha256}",
        "minimality=" + spec.minimality_definition,
    ]
    for candidate, distance in zip(spec.candidates, spec.distances, strict=True):
        lines.append(f"candidate value={candidate} distance={distance}")
    return "\n".join(lines) + "\n"


def format_result(result: XRayResult) -> str:
    """Operator summary. Observation bodies are omitted."""

    minimal = "none"
    if result.minimal_edit is not None:
        edit = result.minimal_edit
        minimal = f"value={edit['value']} distance={edit['distance']} decision={edit['decision']}"
    lines = [
        f"xray_id={result.xray_id}",
        f"original_decision={result.original_decision}",
        f"termination_reason={result.termination_reason}",
        f"minimality_claim={result.minimality_claim}",
        f"proof_blockers={','.join(str(value) for value in result.proof_blockers) or 'none'}",
        f"minimal_edit={minimal}",
        f"shared_checkpoint={str(result.shared_checkpoint).lower()}",
        f"checkpoint_id={result.checkpoint_id}",
        f"provider_requests={result.provider_requests}",
        "usage_deduped=" + _usage_text(result.usage_deduped),
        f"target_reached={str(result.target_reached).lower()}",
    ]
    for row in result.candidates:
        lines.append(
            f"candidate value={row.candidate_value} distance={row.distance} "
            f"status={row.status} decision={row.decision} flipped={str(row.flipped).lower()}"
        )
    return "\n".join(lines) + "\n"


async def execute_xray(
    spec: XRaySpec,
    output_dir: Path,
    adapter: ModelAdapter,
    *,
    resume: bool,
) -> XRayResult:
    """Run the remaining candidates. Committed work is not repeated."""

    if spec.worst_case_model_requests < 1:
        raise ValueError("request ceiling is zero")
    _store_manifest(output_dir / "manifest.json", spec)
    baseline_path = output_dir / "baseline.json"
    candidate_path = output_dir / "candidates.jsonl"
    if _search_started(baseline_path, candidate_path) and not resume:
        raise ValueError("x-ray output exists; pass --resume to continue")
    spent = _spent_requests(baseline_path, candidate_path)
    limiter = _LimitedAdapter(adapter, spec.worst_case_model_requests, spent)
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
        if not baseline_path.is_file():
            await _run_baseline(spec, output_dir, client, limiter, listed.tools)
        baseline = _load_baseline(baseline_path)
        executed = _read_candidates(candidate_path)
        if baseline.stop_reason is None and baseline.factual_valid and baseline.checkpoint is not None:
            await _run_remaining(spec, output_dir, client, limiter, baseline, executed)
            executed = _read_candidates(candidate_path)
    result = _assemble(spec, _load_baseline(baseline_path), executed)
    _write_result(output_dir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Without ``--live`` this only prints the plan."""

    parser = argparse.ArgumentParser(prog="python -m parallax.xray")
    parser.add_argument("--fixture", default="F100")
    parser.add_argument("--tool", default="get_metrics")
    parser.add_argument("--field", default="error_rate_bps")
    parser.add_argument("--candidates", nargs="+", type=int, default=None)
    parser.add_argument("--target-decision", default=None)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--model-calls", type=int, default=None)
    parser.add_argument("--tool-calls", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strategy", choices=("early-stop", "exhaustive"), default="early-stop")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.live and args.dry_run:
        print("choose either --dry-run or --live", file=sys.stderr)
        return 2
    from parallax.live import configured_budget, os_environ
    from parallax.live_pair import pair_model_params
    from parallax.openai_adapter import load_project_env

    load_project_env()
    environ = dict(os_environ())
    model_id = environ.get("PARALLAX_MODEL", "").strip()
    if args.live and not model_id:
        print("PARALLAX_MODEL is not set", file=sys.stderr)
        return 2
    try:
        model_calls = configured_budget(
            args.model_calls, environ, "PARALLAX_MAX_MODEL_CALLS", DEFAULT_MODEL_CALLS
        )
        tool_calls = configured_budget(
            args.tool_calls, environ, "PARALLAX_MAX_TOOL_CALLS", DEFAULT_TOOL_CALLS
        )
        spec = build_spec(
            fixture_id=args.fixture,
            tool=args.tool,
            field=args.field,
            candidates=args.candidates,
            target_decision=args.target_decision,
            model_id=model_id or "(unset)",
            model_params=pair_model_params(model_id or "(unset)"),
            model_call_budget=model_calls,
            tool_call_budget=tool_calls,
            max_candidates=args.max_candidates,
            strategy=_strategy_name(args.strategy),
        )
    except (TypeError, ValueError) as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    if not args.live:
        sys.stdout.write(format_dry_run(spec))
        return 0
    if spec.worst_case_model_requests < 1:
        print("request ceiling is zero", file=sys.stderr)
        return 2
    directory = args.output or Path("runs") / "xray" / spec.xray_id
    try:
        _store_manifest(directory / "manifest.json", spec)
        if _search_started(directory / "baseline.json", directory / "candidates.jsonl") and not args.resume:
            raise ValueError("x-ray output exists; pass --resume to continue")
    except ValueError as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    try:
        from parallax.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.from_env()
    except RuntimeError as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    try:
        result = anyio.run(execute_xray, spec, directory, adapter, resume=args.resume)
    except (TypeError, ValueError) as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    sys.stdout.write(format_result(result))
    print(f"output_dir={directory}")
    return 0


def _assemble(spec: XRaySpec, baseline: StoredBaseline, executed: Sequence[CandidateRecord]) -> XRayResult:
    by_value = {row.candidate_value: row for row in executed}
    rows: list[SearchRow] = []
    for index, candidate in enumerate(spec.candidates):
        found = by_value.get(candidate)
        if found is None:
            rows.append(
                SearchRow(
                    candidate_value=candidate,
                    distance=spec.distances[index],
                    order_index=index,
                    status="not_run",
                    decision=None,
                    flipped=False,
                    execution_valid=False,
                    evidence_sha256_original=None,
                    evidence_sha256_mutated=None,
                    checkpoint_id=baseline.prefix_sha256,
                    oracle_before=None,
                    oracle_after=None,
                    error=None,
                    model_calls=0,
                )
            )
            continue
        rows.append(_search_row(found))
    ran = [row for row in executed if row.status != "not_run"]
    claim = classify_search(rows)
    flip = _minimal_flip(ran)
    minimal = None
    if flip is not None:
        minimal = {
            "decision": flip.decision_parsed,
            "distance": flip.distance,
            "value": flip.candidate_value,
        }
    reason = baseline.stop_reason or _termination(spec, ran, rows)
    target_reached = any(
        row.status == "verified_flip" and row.decision_parsed == spec.target_decision for row in ran
    )
    return XRayResult(
        schema_version=XRAY_SCHEMA_VERSION,
        xray_id=spec.xray_id,
        original_decision=baseline.decision_parsed,
        termination_reason=reason,
        minimality_definition=spec.minimality_definition,
        minimality_claim=claim,
        minimal_edit=minimal,
        proof_blockers=_proof_blockers(rows),
        shared_checkpoint=baseline.prefix_sha256 is not None,
        checkpoint_id=baseline.prefix_sha256,
        candidates=rows,
        provider_requests=baseline.model_calls + sum(row.model_calls for row in ran),
        usage_deduped=_deduped_usage(baseline, ran),
        target_decision=spec.target_decision,
        target_reached=target_reached,
        notes=list(spec.notes),
    )


async def _run_baseline(
    spec: XRaySpec,
    output_dir: Path,
    client: Client,
    limiter: "_LimitedAdapter",
    tools: Sequence[Any],
) -> None:
    fixture = get_fixture(spec.fixture_id)
    if fixture is None:
        raise ValueError(f"unknown incident: {spec.fixture_id}")
    state = build_initial_state(
        incident_id=spec.fixture_id,
        model_id=spec.model_id,
        model_params=spec.model_params,
        step_budget=spec.model_call_budget,
        token_budget=None,
        fixture_hash=fixture_hash(fixture),
        tool_schemas=_tool_schemas(tools),
        tool_call_budget=spec.tool_call_budget,
    )
    recorder = CheckpointRecorder(spec.tool, spec.fixture_id)
    before = limiter.used
    result = await run_agent(
        client,
        state,
        limiter,
        resolver=FactualResolver(),
        checkpoint=recorder,
    )
    calls = limiter.used - before
    checkpoint = recorder.checkpoint
    captured_value = None
    stop_reason = None
    observation = None
    prefix = None
    prefix_usage = None
    checkpoint_payload = None
    if checkpoint is None:
        if "provider_error" in result.flags or result.execution_status == "model_error":
            stop_reason = "factual_invalid"
        else:
            stop_reason = "intervention_unobserved"
    else:
        observation = checkpoint.visible
        prefix = prefix_sha256(checkpoint.state)
        prefix_usage = usage_payload(checkpoint.state)
        checkpoint_payload = _checkpoint_payload(checkpoint)
        try:
            captured_value = _captured_integer(checkpoint, spec.tool, spec.field)
        except (TypeError, ValueError):
            stop_reason = "captured_value_mismatch"
        else:
            if captured_value != spec.original_value:
                stop_reason = "captured_value_mismatch"
            elif not _factual_ok(result, observation, spec.tool):
                stop_reason = "factual_invalid"
    if stop_reason is None:
        factual_valid = True
    else:
        factual_valid = False
    stored = StoredBaseline(
        schema_version=XRAY_SCHEMA_VERSION,
        xray_id=spec.xray_id,
        checkpoint=checkpoint_payload,
        final_state=result.state.model_dump(mode="json"),
        decision_parsed=result.decision_parsed,
        parse_ok=result.parse_ok,
        execution_status=result.execution_status,
        flags=list(result.flags),
        error=_baseline_error(result.error, stop_reason),
        observation_original=observation,
        observation_sha256=None if observation is None else sha256_text(observation),
        captured_value=captured_value,
        prefix_sha256=prefix,
        prefix_usage=prefix_usage,
        usage=usage_payload(result.state),
        model_calls=calls,
        provider_response_ids=list(result.state.provider_response_ids),
        factual_valid=factual_valid,
        stop_reason=stop_reason,
    )
    _write_exclusive(output_dir / "baseline.json", stored.model_dump_json(indent=2) + "\n")
    _append_execution_trace(
        output_dir / "traces.jsonl",
        spec,
        result,
        arm="baseline",
        intervention_id=None,
        prefix=prefix,
        observation_original=observation,
        observation_mutated=None,
        oracle=_posthoc_oracle(spec.fixture_id, spec.tool, spec.field, None),
        protocol=_protocol_version(client),
    )


async def _run_remaining(
    spec: XRaySpec,
    output_dir: Path,
    client: Client,
    limiter: "_LimitedAdapter",
    baseline: StoredBaseline,
    executed: Sequence[CandidateRecord],
) -> None:
    done = {row.candidate_value for row in executed}
    checkpoint = _checkpoint_from_payload(baseline.checkpoint or {})
    final_state = AgentState.model_validate(baseline.final_state)
    original_decision = baseline.decision_parsed
    if original_decision is None:
        return
    ordered_done = [row for row in executed if row.candidate_value in done]
    if _should_stop(spec, ordered_done):
        return
    for index, candidate in enumerate(spec.candidates):
        if candidate in done:
            continue
        if _should_stop(spec, _read_candidates(output_dir / "candidates.jsonl")):
            return
        mutated = _edited_visible(checkpoint, spec.tool, spec.field, candidate)
        before_calls = limiter.used
        message_count = len(checkpoint.state.messages)
        branch_result = await _continue(
            client,
            checkpoint,
            final_state,
            limiter,
            mutated,
            spec.fixture_id,
            spec.tool,
        )
        if len(checkpoint.state.messages) != message_count:
            raise RuntimeError("checkpoint was modified")
        applied = _intervention_applied(branch_result, spec.tool, mutated)
        visible = mutated in [message.content for message in branch_result.state.messages if message.role == "tool"]
        status = classify_outcome(
            factual_status=baseline.execution_status,
            factual_parsed=baseline.parse_ok,
            factual_decision=original_decision,
            mutant_status=branch_result.execution_status,
            mutant_parsed=branch_result.parse_ok,
            mutant_decision=branch_result.decision_parsed,
            intervention_applied=applied,
            model_visible=visible,
            factual_flags=baseline.flags,
            mutant_flags=branch_result.flags,
        )
        oracle_before, oracle_after = _posthoc_pair(spec, candidate)
        record = CandidateRecord(
            schema_version=XRAY_SCHEMA_VERSION,
            xray_id=spec.xray_id,
            candidate_value=candidate,
            distance=field_distance(spec.original_value, candidate),
            order_index=index,
            status=status,
            decision_parsed=branch_result.decision_parsed,
            flipped=status == "verified_flip",
            execution_valid=status in {"verified_flip", "no_flip"},
            intervention_applied=applied,
            model_visible=visible,
            factual_decision=original_decision,
            evidence_sha256_original=baseline.observation_sha256,
            evidence_sha256_mutated=sha256_text(mutated),
            checkpoint_id=baseline.prefix_sha256,
            execution_status=branch_result.execution_status,
            flags=list(branch_result.flags),
            error=_public_text(branch_result.error) if branch_result.error else None,
            oracle_before=oracle_before,
            oracle_after=oracle_after,
            usage=usage_payload(branch_result.state),
            model_calls=limiter.used - before_calls,
            provider_response_ids=list(branch_result.state.provider_response_ids),
        )
        _append_record(output_dir / "candidates.jsonl", record)
        _append_execution_trace(
            output_dir / "traces.jsonl",
            spec,
            branch_result,
            arm="mutant",
            intervention_id=f"xray:{spec.field}={candidate}",
            prefix=baseline.prefix_sha256,
            observation_original=baseline.observation_original,
            observation_mutated=mutated,
            oracle=oracle_after,
            protocol=_protocol_version(client),
        )


async def _continue(
    client: Client,
    checkpoint: Checkpoint,
    baseline_state: AgentState,
    adapter: ModelAdapter,
    visible: str,
    incident_id: str,
    tool: str,
) -> AgentResult:
    branch = copy_state(checkpoint.state)
    check_prefix_pair(branch, checkpoint.state)
    restore(branch, checkpoint, content=visible, source="overlay")
    _carry_same_turn_results(branch, baseline_state, checkpoint)
    return await run_agent(
        client,
        branch,
        adapter,
        resolver=OverlayResolver({tool: visible}, incident_id=incident_id),
    )


def _should_stop(spec: XRaySpec, executed: Sequence[CandidateRecord]) -> bool:
    if spec.strategy == "exhaustive":
        return False
    flips = [row for row in executed if row.status == "verified_flip"]
    if spec.target_decision is None:
        return any(row.status == "verified_flip" for row in executed)
    return any(row.decision_parsed == spec.target_decision for row in flips)


def _termination(spec: XRaySpec, executed: Sequence[CandidateRecord], rows: Sequence[SearchRow]) -> str:
    if any(row.status == "not_run" for row in rows) and _should_stop(spec, executed):
        if spec.target_decision is not None and any(
            row.status == "verified_flip" and row.decision_parsed == spec.target_decision for row in executed
        ):
            return "target_reached"
        return "verified_flip"
    if classify_search(rows) == _INCONCLUSIVE_CLAIM:
        return "inconclusive"
    return "candidates_exhausted"


def classify_search(rows: Sequence[SearchRow]) -> str:
    """Classify minimality from recorded outcomes. The oracle is not an input."""

    flips = [row for row in rows if row.status == "verified_flip"]
    if flips:
        best = min(row.distance for row in flips)
        closer = [row for row in rows if row.distance < best]
        if all(row.status == "no_flip" for row in closer):
            return _PROVEN_CLAIM
        return _TESTED_CLAIM
    if any(row.status != "no_flip" for row in rows):
        return _INCONCLUSIVE_CLAIM
    return _NO_FLIP_CLAIM


def _proof_blockers(rows: Sequence[SearchRow]) -> list[int]:
    flips = [row for row in rows if row.status == "verified_flip"]
    if flips:
        best = min(row.distance for row in flips)
        blocked = [row for row in rows if row.distance < best and row.status != "no_flip"]
    else:
        blocked = [row for row in rows if row.status != "no_flip"]
    return [row.candidate_value for row in blocked]


def _minimal_flip(executed: Sequence[CandidateRecord]) -> CandidateRecord | None:
    flips = [row for row in executed if row.status == "verified_flip"]
    if not flips:
        return None
    return min(flips, key=lambda row: (row.distance, row.order_index))


def _search_row(row: CandidateRecord) -> SearchRow:
    return SearchRow(
        candidate_value=row.candidate_value,
        distance=row.distance,
        order_index=row.order_index,
        status=row.status,
        decision=row.decision_parsed,
        flipped=row.flipped,
        execution_valid=row.execution_valid,
        evidence_sha256_original=row.evidence_sha256_original,
        evidence_sha256_mutated=row.evidence_sha256_mutated,
        checkpoint_id=row.checkpoint_id,
        oracle_before=row.oracle_before,
        oracle_after=row.oracle_after,
        error=row.error,
        model_calls=row.model_calls,
    )


def _deduped_usage(baseline: StoredBaseline, executed: Sequence[CandidateRecord]) -> dict[str, int] | None:
    if baseline.usage is None or baseline.prefix_usage is None:
        return None
    total = {
        "input_tokens": baseline.usage["input_tokens"],
        "output_tokens": baseline.usage["output_tokens"],
    }
    for row in executed:
        if row.usage is None:
            return None
        extra_input = row.usage["input_tokens"] - baseline.prefix_usage["input_tokens"]
        extra_output = row.usage["output_tokens"] - baseline.prefix_usage["output_tokens"]
        if extra_input < 0 or extra_output < 0:
            return None
        total["input_tokens"] += extra_input
        total["output_tokens"] += extra_output
    return total


def _factual_ok(result: AgentResult, observation: str, tool: str) -> bool:
    if result.execution_status != "completed" or not result.parse_ok or result.decision_parsed is None:
        return False
    if "provider_error" in result.flags or "tool_error" in result.flags or "engine_error" in result.flags:
        return False
    visible = [message.content for message in result.state.messages if message.role == "tool"]
    if observation not in visible:
        return False
    return any(
        record.name == tool and record.source == "base" and record.model_visible_observation == observation
        for record in result.state.tool_calls
    )


def _intervention_applied(result: AgentResult, tool: str, mutated: str) -> bool:
    return any(
        record.name == tool and record.source == "overlay" and record.model_visible_observation == mutated
        for record in result.state.tool_calls
    )


def edit_evidence(
    *,
    tool_name: str,
    field_path: str,
    structured: Mapping[str, Any],
    visible: str,
    candidate: int,
) -> EditedEvidence:
    """Edit one captured observation. Fixture files and the oracle are untouched."""

    model_type = _searchable_model(tool_name, field_path)
    if not isinstance(structured, dict):
        raise TypeError("captured observation is not an object")
    current = model_type.model_validate(dict(structured))
    reserialized = canonical_json(current.model_dump(mode="json"))
    if reserialized != visible:
        raise ValueError("captured observation did not reserialize")
    original = _stored_integer(current.model_dump(mode="json"), field_path)
    _validate_candidate(model_type, current.model_dump(mode="json"), field_path, original, candidate)
    updated = current.model_copy(update={field_path: candidate})
    validated = model_type.model_validate(updated.model_dump())
    before = current.model_dump(mode="json")
    after = validated.model_dump(mode="json")
    changed = [key for key in before if before[key] != after[key]]
    if changed != [field_path]:
        raise ValueError("edit changed more than the specified field")
    observation = canonical_json(after)
    source_sha256 = sha256_text(visible)
    evidence_sha256 = sha256_text(observation)
    if evidence_sha256 == source_sha256:
        raise ValueError("edit did not change the evidence hash")
    return EditedEvidence(
        tool_name=tool_name,
        field_path=field_path,
        original_value=original,
        candidate_value=candidate,
        observation=observation,
        source_sha256=source_sha256,
        evidence_sha256=evidence_sha256,
    )


def _edited_visible(checkpoint: Checkpoint, tool: str, field: str, candidate: int) -> str:
    structured = checkpoint.raw.get("structured_content")
    if not isinstance(structured, dict):
        raise TypeError("checkpoint has no structured observation")
    return edit_evidence(
        tool_name=tool,
        field_path=field,
        structured=structured,
        visible=checkpoint.visible,
        candidate=candidate,
    ).observation


def _captured_integer(checkpoint: Checkpoint, tool: str, field: str) -> int:
    model_type = TOOL_MODELS[tool]
    structured = checkpoint.raw.get("structured_content")
    if not isinstance(structured, dict):
        raise TypeError("checkpoint has no structured observation")
    current = model_type.model_validate(structured)
    value = getattr(current, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("captured field is not an integer")
    reserialized = canonical_json(current.model_dump(mode="json"))
    if reserialized != checkpoint.visible:
        raise ValueError("captured observation did not reserialize")
    return value


def _posthoc_pair(spec: XRaySpec, candidate: int) -> tuple[str | None, str | None]:
    return (
        _posthoc_oracle(spec.fixture_id, spec.tool, spec.field, None),
        _posthoc_oracle(spec.fixture_id, spec.tool, spec.field, candidate),
    )


def _posthoc_oracle(fixture_id: str, tool: str, field: str, candidate: int | None) -> str | None:
    """Score the edited fixture after the model has answered."""

    from parallax.oracle import decide

    fixture = get_fixture(fixture_id)
    if fixture is None:
        return None
    metrics = fixture.metrics
    logs = fixture.logs
    deployment = fixture.deployment
    if candidate is not None:
        model_type = TOOL_MODELS[tool]
        payload = _tool_payload(fixture, tool)
        payload[field] = candidate
        edited = model_type.model_validate(payload)
        if tool == "get_metrics":
            metrics = edited  # type: ignore[assignment]
        elif tool == "get_logs":
            logs = edited  # type: ignore[assignment]
        elif tool == "get_deployment":
            deployment = edited  # type: ignore[assignment]
    return decide(metrics, logs, deployment).value


def _append_execution_trace(
    path: Path,
    spec: XRaySpec,
    result: AgentResult,
    *,
    arm: Literal["baseline", "mutant"],
    intervention_id: str | None,
    prefix: str | None,
    observation_original: str | None,
    observation_mutated: str | None,
    oracle: str | None,
    protocol: str | None,
) -> None:
    started = utc_now()
    record = TraceRecord(
        run_id=uuid.uuid4().hex,
        pair_id=spec.xray_id,
        arm=arm,
        incident_id=spec.fixture_id,
        fixture_hash=spec.fixture_hash,
        intervention_id=intervention_id,
        model_id=spec.model_id,
        model_params=spec.model_params,
        prompt_hash=spec.prompt_hash,
        sdk_version=spec.sdk_version,
        negotiated_protocol_version=protocol,
        snapshot_id=prefix,
        phase="harness",
        scenario_id=None,
        repeat_index=0,
        tool_calls=list(result.state.tool_calls),
        observation_original=observation_original,
        observation_mutated=observation_mutated,
        model_visible_observations=[message.content for message in result.state.messages if message.role == "tool"],
        usage=usage_payload(result.state),
        decision_raw=result.decision_raw,
        decision_parsed=result.decision_parsed,
        action_observed=result.action_observed,
        parse_ok=result.parse_ok,
        oracle_decision=oracle,
        predicate_m=None,
        predicate_l=None,
        predicate_d=None,
        flags=list(result.flags),
        error=_public_text(result.error) if result.error else None,
        provider_error=result.provider_error,
        started_at=started,
        ended_at=utc_now(),
        prefix_sha256=prefix,
        intervention_source_sha256=spec.search_source_sha256,
        execution_status=result.execution_status,
        provider_response_ids=list(result.state.provider_response_ids),
        experiment_id=spec.xray_id,
    )
    append_trace(path, record)


def _baseline_error(error: str | None, stop_reason: str | None) -> str | None:
    if error:
        return _public_text(error)
    if stop_reason == "captured_value_mismatch":
        return "captured field value does not match the fixture"
    return None


def _checkpoint_payload(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "arguments": checkpoint.arguments,
        "call_id": checkpoint.call_id,
        "latency_ms": checkpoint.latency_ms,
        "raw": checkpoint.raw,
        "state": checkpoint.state.model_dump(mode="json"),
        "tool_name": checkpoint.tool_name,
        "visible": checkpoint.visible,
    }


def _checkpoint_from_payload(payload: Mapping[str, Any]) -> Checkpoint:
    return Checkpoint(
        state=AgentState.model_validate(payload["state"]),
        tool_name=str(payload["tool_name"]),
        arguments=dict(payload["arguments"]),
        raw=dict(payload["raw"]),
        visible=str(payload["visible"]),
        latency_ms=float(payload["latency_ms"]),
        call_id=None if payload.get("call_id") is None else str(payload["call_id"]),
    )


def _searchable_model(tool: str, field: str) -> type[BaseModel]:
    if "." in field or "[" in field or "/" in field:
        raise ValueError(f"unsupported nested path: {field}")
    try:
        model = TOOL_MODELS[tool]
    except KeyError as exc:
        raise ValueError(f"unknown tool: {tool}") from exc
    if field not in model.model_fields:
        raise ValueError(f"unknown field: {field}")
    info = model.model_fields[field]
    if info.annotation is not int:
        raise ValueError(f"{field} is not an integer field")
    if (tool, field) not in SEARCHABLE_FIELDS:
        raise ValueError(f"{field} is not a searchable field")
    return model


def _strategy_name(label: str) -> Strategy:
    if label == "early-stop":
        return "early_stop"
    if label == "exhaustive":
        return "exhaustive"
    raise ValueError("strategy must be early_stop or exhaustive")


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    revision = result.stdout.strip()
    return revision or "unknown"


def _validate_candidate(
    model: type[BaseModel],
    payload: Mapping[str, Any],
    field: str,
    original: int,
    candidate: int,
) -> None:
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise TypeError("candidate must be an integer")
    if candidate == original:
        raise ValueError("candidate equals the original value")
    trial = dict(payload)
    trial[field] = candidate
    try:
        model.model_validate(trial)
    except ValidationError as exc:
        raise ValueError(f"candidate {candidate} is not valid for {field}") from exc


def _reject_duplicate_candidates(candidates: Sequence[int]) -> None:
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate values must be unique")


def _stored_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("stored field is not an integer")
    return value


def _tool_payload(fixture: Any, tool: str) -> dict[str, Any]:
    if tool == "get_metrics":
        return fixture.metrics.model_dump(mode="json")
    if tool == "get_logs":
        return fixture.logs.model_dump(mode="json")
    if tool == "get_deployment":
        return fixture.deployment.model_dump(mode="json")
    raise ValueError(f"unknown tool: {tool}")


def _store_manifest(path: Path, spec: XRaySpec) -> None:
    payload = spec.model_dump_json(indent=2) + "\n"
    if path.is_file():
        existing = XRaySpec.model_validate_json(path.read_text(encoding="utf-8"))
        if _spec_body(existing) != _spec_body(spec):
            raise ValueError("x-ray directory belongs to a different search")
        return
    _write_exclusive(path, payload)


def _spec_body(spec: XRaySpec) -> dict[str, Any]:
    """Search identity used for resume. The checkout revision is provenance."""

    body = spec.model_dump()
    body.pop("git_revision", None)
    return body


def _write_result(path: Path, result: XRayResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    text = result.model_dump_json(indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _append_record(path: Path, record: CandidateRecord) -> None:
    line = record.model_dump_json()
    if "\n" in line or "\r" in line:
        raise ValueError("candidate record must serialize to one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_torn_jsonl_tail(path, CandidateRecord.model_validate_json)
    existing = {row.candidate_value for row in _read_candidates(path)}
    if record.candidate_value in existing:
        return
    _append_line(path, line)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_candidates(path: Path) -> list[CandidateRecord]:
    if not path.is_file():
        return []
    drop_torn_jsonl_tail(path, CandidateRecord.model_validate_json)
    return [
        CandidateRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_baseline(path: Path) -> StoredBaseline:
    return StoredBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def _search_started(baseline_path: Path, candidate_path: Path) -> bool:
    return baseline_path.is_file() or candidate_path.is_file()


def _spent_requests(baseline_path: Path, candidate_path: Path) -> int:
    spent = 0
    if baseline_path.is_file():
        spent += _load_baseline(baseline_path).model_calls
    spent += sum(row.model_calls for row in _read_candidates(candidate_path))
    return spent


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _package_version(name: str, fallback: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback


def _usage_text(usage: dict[str, int] | None) -> str:
    if usage is None:
        return "null"
    return f"input_tokens={usage['input_tokens']} output_tokens={usage['output_tokens']}"


def _public_text(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return "redacted"
    return text


class _LimitedAdapter:
    """Stop before another provider request once the ceiling is reached."""

    def __init__(self, inner: ModelAdapter, limit: int, used: int) -> None:
        self._inner = inner
        self._limit = limit
        self.used = used

    async def complete(
        self,
        messages: Sequence[Any],
        tools: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
    ) -> Any:
        if self.used >= self._limit:
            raise LoopBudget()
        self.used += 1
        return await self._inner.complete(messages, tools, params)


if __name__ == "__main__":
    raise SystemExit(main())
