"""Research controls for Decision X-Ray stochasticity.

One factual baseline acquires a checkpoint. Each planned continuation
copies that checkpoint and returns either the original observation or
one edited value. Null controls replay the captured text and are not
search candidates. The X-Ray editor still rejects an unchanged edit.

Repetitions are fresh continuations from one checkpoint draw. They are
not independent checkpoint samples. Decision changes on the null arm are
variability, not successful evidence interventions. Oracle labels are
recorded after the model answers. Offline ordering simulations reuse
recorded outcomes and do not call a provider.

Rows use ``phase=harness``. This module does not write fixture files or
modify pilot or X-Ray directories.
"""

import argparse
import hashlib
import math
import os
import platform
import random
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp import Client
from pydantic import BaseModel, ConfigDict, Field

from parallax.agent import (
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
from parallax.xray import SEARCHABLE_FIELDS, edit_evidence

CONTROL_SCHEMA_VERSION = 1
DEFAULT_REPETITIONS = 3
MAX_REPETITIONS = 5
MAX_CONDITIONS = 3
DEFAULT_MODEL_CALLS = 4
MAX_MODEL_CALLS = 8
DEFAULT_TOOL_CALLS = 6
MAX_TOOL_CALLS = 12
DEFAULT_INTERVENTION = 499
DEFAULT_NEGATIVE = 501
ROLES = ("null", "intervention", "negative")
SAMPLING_UNIT = "shared_checkpoint_continuation"
SCHEDULE_NAME = "round_robin_repetition"
_WILSON_Z = 1.959963984540054
_SECRET_MARKERS = ("sk-", "api_key", "bearer ", "authorization")
_DIFFERENCE_NOTE = "Descriptive difference. This is not a significance test."
_ORDERING_NOTE = (
    "Offline reordering of recorded outcomes. Not an independent model experiment. "
    "Distance sorting is a baseline, not a claimed novel algorithm."
)
Role = Literal["null", "intervention", "negative"]
ContinuationStatus = Literal[
    "decision_stable",
    "decision_changed",
    "invalid_output",
    "provider_error",
    "mcp_error",
    "budget_exhausted",
    "observation_not_visible",
    "not_run",
]
OrderingMethod = Literal["distance_first", "seeded_random", "exhaustive"]


class ControlCondition(BaseModel):
    """One labeled arm. A null arm is not a search candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    condition_id: str
    field_value: int
    distance: int
    search_candidate: Literal[False] = False


class ScheduleSlot(BaseModel):
    """One planned continuation under the shared checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_index: int
    repetition_index: int
    role: Role
    condition_id: str


class StudySpec(BaseModel):
    """Immutable control-study manifest. The oracle is not an input."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    study_id: str
    fixture_id: str
    tool: str
    field: str
    original_value: int
    conditions: list[ControlCondition]
    schedule: list[ScheduleSlot]
    repetitions: int
    checkpoint_allocation: Literal["shared_block"]
    checkpoint_draws: Literal[1]
    sampling_unit: Literal["shared_checkpoint_continuation"]
    schedule_name: Literal["round_robin_repetition"]
    random_seed: int
    seed_applies_to: Literal["offline_ordering_only"]
    model_id: str
    model_params: dict[str, Any]
    prompt_hash: str
    fixture_hash: str
    source_evidence_sha256: str
    model_call_budget: int
    tool_call_budget: int
    checkpoint_acquisition_requests: int
    null_continuation_requests: int
    intervention_continuation_requests: int
    negative_continuation_requests: int
    worst_case_model_requests: int
    git_revision: str
    source_sha256: str
    editor_sha256: str
    software_version: str
    sdk_version: str
    provider_sdk_version: str
    python_version: str
    trace_schema_version: int
    phase: Literal["harness"] = "harness"
    notes: list[str] = Field(default_factory=list)


class StoredBaseline(BaseModel):
    """Durable factual execution and the single checkpoint draw."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    study_id: str
    checkpoint: dict[str, Any] | None
    checkpoint_draw_index: int
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
    prefix_response_ids: list[str]
    fork_call_id: str | None
    usage: dict[str, int] | None
    model_calls: int
    provider_response_ids: list[str]
    factual_valid: bool
    stop_reason: str | None


class ContinuationRecord(BaseModel):
    """One fresh continuation. A missing line was not durable."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    study_id: str
    condition_id: str
    role: Role
    repetition_index: int
    order_index: int
    checkpoint_id: str | None
    checkpoint_draw_index: int
    sampling_unit: Literal["shared_checkpoint_continuation"]
    independent_checkpoint_sample: Literal[False]
    field_value: int
    distance: int
    search_candidate: Literal[False]
    status: ContinuationStatus
    decision_parsed: str | None
    parse_ok: bool
    execution_status: str | None
    decision_changed: bool
    evidence_intervention: bool
    execution_valid: bool
    observation_sha256: str | None
    fork_call_id: str | None
    flags: list[str]
    error: str | None
    oracle_decision: str | None
    usage: dict[str, int] | None
    model_calls: int
    provider_response_ids: list[str]


class ConditionMetric(BaseModel):
    """Rates for one arm. Flip denominators are valid observations only."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    condition_id: str
    attempt_count: int
    valid_count: int
    flip_count: int
    flip_rate: float | None
    uncertainty: dict[str, float] | None
    oracle_aligned_count: int
    oracle_alignment_rate: float | None
    invalid_output_count: int
    provider_failure_count: int
    mcp_failure_count: int
    completion_count: int


class RecordedOutcome(BaseModel):
    """One already observed candidate. Ordering functions do not rerun it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: int
    distance: int
    verified_flip: bool


class OrderingEvaluation(BaseModel):
    """A planner score over recorded outcomes, not a new model experiment."""

    model_config = ConfigDict(extra="forbid")

    method: OrderingMethod
    seed: int | None
    order: list[int]
    first_verified_flip_position: int | None
    evaluated_count: int
    smallest_discovered_distance: int | None
    coverage: int
    candidate_count: int
    independent_model_experiment: Literal[False] = False
    note: str


class StudyResult(BaseModel):
    """Assembled control study. The journal remains the durable source."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    study_id: str
    original_decision: str | None
    termination_reason: str
    checkpoint_id: str | None
    checkpoint_draws: int
    sampling_unit: str
    continuations: list[ContinuationRecord]
    conditions: list[ConditionMetric]
    intervention_minus_null: dict[str, Any] | None
    unique_provider_response_ids: list[str]
    provider_requests: int
    usage_deduped: dict[str, int] | None
    notes: list[str]


def request_ceiling(model_calls: int, conditions: int, repetitions: int) -> int:
    """Worst-case provider requests for one shared checkpoint block."""

    continuations = conditions * repetitions
    if model_calls < 1 or continuations < 1:
        return 0
    return model_calls + continuations * (model_calls - 1)


def continuation_requests(model_calls: int, repetitions: int) -> int:
    """Worst-case requests for one condition's fresh continuations."""

    if model_calls < 2 or repetitions < 1:
        return 0
    return repetitions * (model_calls - 1)


def comparison_schedule(repetitions: int) -> list[tuple[int, Role]]:
    """Round-robin repetitions. The live order does not use the random seed."""

    return [(repetition, role) for repetition in range(repetitions) for role in ROLES]


def wilson_interval(successes: int, trials: int) -> dict[str, float] | None:
    """95% Wilson interval. An empty denominator has no interval."""

    if trials < 1:
        return None
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("counts must be integers")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie in the trial count")
    z2 = _WILSON_Z * _WILSON_Z
    proportion = successes / trials
    denominator = 1 + z2 / trials
    center = (proportion + z2 / (2 * trials)) / denominator
    margin = _WILSON_Z * math.sqrt(proportion * (1 - proportion) / trials + z2 / (4 * trials * trials))
    margin /= denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def classify_continuation(
    *,
    factual_valid: bool,
    execution_status: str | None,
    parse_ok: bool,
    decision: str | None,
    factual_decision: str | None,
    observation_visible: bool,
    flags: Sequence[str],
) -> ContinuationStatus:
    """Classify one continuation. The oracle and the arm label are not inputs."""

    flags_set = set(flags)
    if "provider_error" in flags_set or execution_status == "model_error":
        return "provider_error"
    if "tool_error" in flags_set or execution_status == "tool_error":
        return "mcp_error"
    if "budget_exhausted" in flags_set or execution_status == "budget_exhausted":
        return "budget_exhausted"
    if not observation_visible:
        return "observation_not_visible"
    if (
        not factual_valid
        or not parse_ok
        or execution_status != "completed"
        or decision is None
        or factual_decision is None
    ):
        return "invalid_output"
    if decision != factual_decision:
        return "decision_changed"
    return "decision_stable"


def prepare_control_observation(
    *,
    role: Role,
    tool_name: str,
    field_path: str,
    structured: Mapping[str, Any],
    visible: str,
    value: int,
) -> str:
    """Return the model-visible text for one arm.

    The null arm returns the captured observation unchanged and does not
    call the candidate editor. Edited arms use that editor, which still
    rejects the original value.
    """

    if role == "null":
        _require_captured_value(structured, tool_name, field_path, visible, value)
        return visible
    if role not in ("intervention", "negative"):
        raise ValueError("unknown control role")
    return edit_evidence(
        tool_name=tool_name,
        field_path=field_path,
        structured=structured,
        visible=visible,
        candidate=value,
    ).observation


def simulate_orderings(outcomes: Sequence[RecordedOutcome], seed: int) -> tuple[OrderingEvaluation, ...]:
    """Compare planners on a fixed recorded set. This does not call a model."""

    rows = _validated_outcomes(outcomes)
    distance_order = sorted(rows, key=lambda item: (item.distance, item.value))
    random_order = list(rows)
    random.Random(seed).shuffle(random_order)
    return (
        _early_stop_report("distance_first", None, distance_order),
        _early_stop_report("seeded_random", seed, random_order),
        _exhaustive_report(distance_order),
    )


def build_study(
    *,
    fixture_id: str,
    tool: str,
    field: str,
    intervention: int,
    negative: int,
    repetitions: int,
    seed: int,
    model_id: str,
    model_params: Mapping[str, Any],
    model_call_budget: int,
    tool_call_budget: int,
) -> StudySpec:
    """Validate a three-arm block. Condition order does not read the oracle."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("repetitions must be an integer")
    if repetitions < 1 or repetitions > MAX_REPETITIONS:
        raise ValueError(f"repetitions must be from 1 to {MAX_REPETITIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an integer of at least 0")
    if model_call_budget < 2 or model_call_budget > MAX_MODEL_CALLS:
        raise ValueError(f"model-call budget must be from 2 to {MAX_MODEL_CALLS}")
    if tool_call_budget < 1 or tool_call_budget > MAX_TOOL_CALLS:
        raise ValueError(f"tool-call budget must be from 1 to {MAX_TOOL_CALLS}")
    if "." in field or "[" in field or "/" in field:
        raise ValueError(f"unsupported nested path: {field}")
    if (tool, field) not in SEARCHABLE_FIELDS:
        raise ValueError(f"{field} is not a searchable field")
    model = model_id.strip()
    if not model:
        raise ValueError("model id is required")
    fixture = get_fixture(fixture_id)
    if fixture is None:
        raise ValueError(f"unknown incident: {fixture_id}")
    payload = _tool_payload(fixture, tool)
    visible = canonical_json(payload)
    original = _stored_integer(payload, field)
    if intervention == original or negative == original:
        raise ValueError("edited controls must differ from the original value")
    if intervention == negative:
        raise ValueError("intervention and negative control must differ")
    prepare_control_observation(
        role="null",
        tool_name=tool,
        field_path=field,
        structured=payload,
        visible=visible,
        value=original,
    )
    edit_evidence(
        tool_name=tool,
        field_path=field,
        structured=payload,
        visible=visible,
        candidate=intervention,
    )
    edit_evidence(
        tool_name=tool,
        field_path=field,
        structured=payload,
        visible=visible,
        candidate=negative,
    )
    values = {"null": original, "intervention": intervention, "negative": negative}
    conditions = [
        ControlCondition(
            role=role,
            condition_id=f"{role}:{field}={values[role]}",
            field_value=values[role],
            distance=abs(values[role] - original),
        )
        for role in ROLES
    ]
    if len(conditions) > MAX_CONDITIONS:
        raise ValueError("condition count exceeds the maximum")
    by_role = {condition.role: condition for condition in conditions}
    schedule = [
        ScheduleSlot(
            order_index=index,
            repetition_index=repetition,
            role=role,
            condition_id=by_role[role].condition_id,
        )
        for index, (repetition, role) in enumerate(comparison_schedule(repetitions))
    ]
    params = dict(model_params)
    source_evidence_sha256 = sha256_text(visible)
    identity = {
        "checkpoint_allocation": "shared_block",
        "checkpoint_draws": 1,
        "conditions": [condition.model_dump() for condition in conditions],
        "editor_sha256": _file_sha256(Path(__file__).with_name("xray.py")),
        "field": field,
        "fixture_hash": fixture_hash(fixture),
        "fixture_id": fixture_id,
        "model_call_budget": model_call_budget,
        "model_id": model,
        "model_params": params,
        "original_value": original,
        "phase": "harness",
        "prompt_hash": prompt_hash(fixture_id),
        "random_seed": seed,
        "repetitions": repetitions,
        "sampling_unit": SAMPLING_UNIT,
        "schedule": [slot.model_dump() for slot in schedule],
        "schema_version": CONTROL_SCHEMA_VERSION,
        "source_evidence_sha256": source_evidence_sha256,
        "source_sha256": _file_sha256(Path(__file__)),
        "tool": tool,
        "tool_call_budget": tool_call_budget,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }
    per_arm = continuation_requests(model_call_budget, repetitions)
    notes = [
        "Repetitions are fresh continuations from one checkpoint draw.",
        "They are not independent checkpoint samples.",
        "Null controls replay the captured observation and are not search candidates.",
        "A null decision change is variability, not an evidence intervention.",
        "The oracle is recorded after the continuation and does not select arms.",
        "Offline ordering simulations are not model experiments.",
        _DIFFERENCE_NOTE,
    ]
    return StudySpec(
        schema_version=CONTROL_SCHEMA_VERSION,
        study_id="ctl-" + sha256_text(canonical_json(identity))[:16],
        fixture_id=fixture_id,
        tool=tool,
        field=field,
        original_value=original,
        conditions=conditions,
        schedule=schedule,
        repetitions=repetitions,
        checkpoint_allocation="shared_block",
        checkpoint_draws=1,
        sampling_unit=SAMPLING_UNIT,
        schedule_name=SCHEDULE_NAME,
        random_seed=seed,
        seed_applies_to="offline_ordering_only",
        model_id=model,
        model_params=params,
        prompt_hash=str(identity["prompt_hash"]),
        fixture_hash=str(identity["fixture_hash"]),
        source_evidence_sha256=source_evidence_sha256,
        model_call_budget=model_call_budget,
        tool_call_budget=tool_call_budget,
        checkpoint_acquisition_requests=model_call_budget,
        null_continuation_requests=per_arm,
        intervention_continuation_requests=per_arm,
        negative_continuation_requests=per_arm,
        worst_case_model_requests=request_ceiling(model_call_budget, len(conditions), repetitions),
        git_revision=_git_revision(),
        source_sha256=str(identity["source_sha256"]),
        editor_sha256=str(identity["editor_sha256"]),
        software_version=_package_version("parallax", "0.1.0"),
        sdk_version=_package_version("mcp", "unknown"),
        provider_sdk_version=_package_version("openai", "absent"),
        python_version=platform.python_version(),
        trace_schema_version=TRACE_SCHEMA_VERSION,
        notes=notes,
    )


def format_dry_run(spec: StudySpec) -> str:
    """Print the plan. No provider client is constructed by this function."""

    lines = [
        "dry-run: no model request will be sent",
        f"study_id={spec.study_id}",
        f"fixture={spec.fixture_id}",
        f"tool={spec.tool}",
        f"field={spec.field}",
        f"original_value={spec.original_value}",
        f"model={spec.model_id}",
        f"phase={spec.phase}",
        f"checkpoint_allocation={spec.checkpoint_allocation}",
        f"checkpoint_draws={spec.checkpoint_draws}",
        f"sampling_unit={spec.sampling_unit}",
        f"schedule={spec.schedule_name}",
        f"repetitions={spec.repetitions}",
        f"random_seed={spec.random_seed}",
        f"seed_applies_to={spec.seed_applies_to}",
        "oracle=post-hoc",
        "null_controls_are_search_candidates=false",
        f"model_call_budget={spec.model_call_budget}",
        f"tool_call_budget={spec.tool_call_budget}",
        f"checkpoint_acquisition_requests={spec.checkpoint_acquisition_requests}",
        f"null_continuation_requests={spec.null_continuation_requests}",
        f"intervention_continuation_requests={spec.intervention_continuation_requests}",
        f"negative_continuation_requests={spec.negative_continuation_requests}",
        f"worst_case_model_requests={spec.worst_case_model_requests}",
        f"prompt_hash={spec.prompt_hash}",
        f"fixture_hash={spec.fixture_hash}",
        f"source_sha256={spec.source_sha256}",
        f"editor_sha256={spec.editor_sha256}",
        f"git_revision={spec.git_revision}",
        "ordering=offline distance_first, seeded_random, exhaustive",
        _ORDERING_NOTE,
    ]
    for condition in spec.conditions:
        lines.append(
            f"condition role={condition.role} value={condition.field_value} "
            f"distance={condition.distance} search_candidate=false"
        )
    return "\n".join(lines) + "\n"


def format_result(result: StudyResult) -> str:
    """Operator summary. Observation bodies are omitted."""

    lines = [
        f"study_id={result.study_id}",
        f"original_decision={result.original_decision}",
        f"termination_reason={result.termination_reason}",
        f"checkpoint_id={result.checkpoint_id}",
        f"checkpoint_draws={result.checkpoint_draws}",
        f"sampling_unit={result.sampling_unit}",
        f"provider_requests={result.provider_requests}",
        f"unique_provider_responses={len(result.unique_provider_response_ids)}",
        "usage_deduped=" + _usage_text(result.usage_deduped),
    ]
    for metric in result.conditions:
        lines.append(
            f"metric role={metric.role} attempts={metric.attempt_count} "
            f"valid={metric.valid_count} flips={metric.flip_count} "
            f"rate={_rate_text(metric.flip_rate)} "
            f"uncertainty={_interval_text(metric.uncertainty)} "
            f"invalid={metric.invalid_output_count} "
            f"provider_failures={metric.provider_failure_count} "
            f"mcp_failures={metric.mcp_failure_count}"
        )
    difference = result.intervention_minus_null
    if difference is None:
        lines.append("intervention_minus_null=none")
    else:
        lines.append(
            "intervention_minus_null="
            f"{difference['estimate']} note={difference['note']}"
        )
    for row in result.continuations:
        lines.append(
            f"continuation role={row.role} repetition={row.repetition_index} "
            f"status={row.status} decision={row.decision_parsed} "
            f"evidence_intervention={str(row.evidence_intervention).lower()}"
        )
    return "\n".join(lines) + "\n"


async def execute_study(
    spec: StudySpec,
    output_dir: Path,
    adapter: ModelAdapter,
    *,
    resume: bool,
) -> StudyResult:
    """Run the remaining continuations. Committed work is not repeated."""

    if spec.worst_case_model_requests < 1:
        raise ValueError("request ceiling is zero")
    _store_manifest(output_dir / "manifest.json", spec)
    baseline_path = output_dir / "baseline.json"
    journal_path = output_dir / "continuations.jsonl"
    if journal_path.is_file() and not baseline_path.is_file():
        raise ValueError("baseline is missing; refusing to repeat checkpoint acquisition")
    if _study_started(baseline_path, journal_path) and not resume:
        raise ValueError("control output exists; pass --resume to continue")
    spent = _spent_requests(baseline_path, journal_path)
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
        done = {(row.condition_id, row.repetition_index) for row in _read_journal(journal_path)}
        if baseline.factual_valid and baseline.checkpoint is not None and baseline.stop_reason is None:
            await _run_remaining(spec, output_dir, client, limiter, baseline, done)
    result = _assemble(spec, _load_baseline(baseline_path), _read_journal(journal_path))
    _write_result(output_dir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Without ``--live`` this only prints the plan."""

    parser = argparse.ArgumentParser(prog="python -m parallax.controls")
    parser.add_argument("--fixture", default="F100")
    parser.add_argument("--tool", default="get_metrics")
    parser.add_argument("--field", default="error_rate_bps")
    parser.add_argument("--intervention", type=int, default=DEFAULT_INTERVENTION)
    parser.add_argument("--negative", type=int, default=DEFAULT_NEGATIVE)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-calls", type=int, default=None)
    parser.add_argument("--tool-calls", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
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
        spec = build_study(
            fixture_id=args.fixture,
            tool=args.tool,
            field=args.field,
            intervention=args.intervention,
            negative=args.negative,
            repetitions=args.repetitions,
            seed=args.seed,
            model_id=model_id or "(unset)",
            model_params=pair_model_params(model_id or "(unset)"),
            model_call_budget=model_calls,
            tool_call_budget=tool_calls,
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
    directory = args.output or Path("runs") / "controls" / spec.study_id
    try:
        _store_manifest(directory / "manifest.json", spec)
        baseline_path = directory / "baseline.json"
        journal_path = directory / "continuations.jsonl"
        if journal_path.is_file() and not baseline_path.is_file():
            raise ValueError("baseline is missing; refusing to repeat checkpoint acquisition")
        if _study_started(baseline_path, journal_path) and not args.resume:
            raise ValueError("control output exists; pass --resume to continue")
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
        result = anyio.run(partial(execute_study, spec, directory, adapter, resume=args.resume))
    except (TypeError, ValueError) as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    sys.stdout.write(format_result(result))
    print(f"output_dir={directory}")
    return 0


def _assemble(spec: StudySpec, baseline: StoredBaseline, executed: Sequence[ContinuationRecord]) -> StudyResult:
    found = {(row.condition_id, row.repetition_index): row for row in executed}
    rows: list[ContinuationRecord] = []
    for slot in spec.schedule:
        key = (slot.condition_id, slot.repetition_index)
        existing = found.get(key)
        if existing is not None:
            rows.append(existing)
            continue
        condition = _condition_by_id(spec, slot.condition_id)
        rows.append(
            ContinuationRecord(
                schema_version=CONTROL_SCHEMA_VERSION,
                study_id=spec.study_id,
                condition_id=slot.condition_id,
                role=slot.role,
                repetition_index=slot.repetition_index,
                order_index=slot.order_index,
                checkpoint_id=baseline.prefix_sha256,
                checkpoint_draw_index=0,
                sampling_unit=SAMPLING_UNIT,
                independent_checkpoint_sample=False,
                field_value=condition.field_value,
                distance=condition.distance,
                search_candidate=False,
                status="not_run",
                decision_parsed=None,
                parse_ok=False,
                execution_status=None,
                decision_changed=False,
                evidence_intervention=False,
                execution_valid=False,
                observation_sha256=None,
                fork_call_id=baseline.fork_call_id,
                flags=[],
                error=None,
                oracle_decision=None,
                usage=None,
                model_calls=0,
                provider_response_ids=[],
            )
        )
    attempted = [row for row in rows if row.status != "not_run"]
    metrics = [_condition_metric(spec, rows, condition) for condition in spec.conditions]
    by_role = {metric.role: metric for metric in metrics}
    return StudyResult(
        schema_version=CONTROL_SCHEMA_VERSION,
        study_id=spec.study_id,
        original_decision=baseline.decision_parsed,
        termination_reason=baseline.stop_reason or "schedule_complete",
        checkpoint_id=baseline.prefix_sha256,
        checkpoint_draws=spec.checkpoint_draws,
        sampling_unit=spec.sampling_unit,
        continuations=rows,
        conditions=metrics,
        intervention_minus_null=_rate_difference(by_role["intervention"], by_role["null"]),
        unique_provider_response_ids=_unique_response_ids(baseline, attempted),
        provider_requests=baseline.model_calls + sum(row.model_calls for row in attempted),
        usage_deduped=_deduped_usage(baseline, attempted),
        notes=list(spec.notes),
    )


def _condition_metric(
    spec: StudySpec,
    rows: Sequence[ContinuationRecord],
    condition: ControlCondition,
) -> ConditionMetric:
    selected = [row for row in rows if row.condition_id == condition.condition_id and row.status != "not_run"]
    valid = [row for row in selected if row.execution_valid]
    flips = [row for row in valid if row.decision_changed]
    aligned = [row for row in valid if row.decision_parsed == row.oracle_decision and row.oracle_decision is not None]
    valid_count = len(valid)
    return ConditionMetric(
        role=condition.role,
        condition_id=condition.condition_id,
        attempt_count=len(selected),
        valid_count=valid_count,
        flip_count=len(flips),
        flip_rate=None if valid_count == 0 else len(flips) / valid_count,
        uncertainty=wilson_interval(len(flips), valid_count),
        oracle_aligned_count=len(aligned),
        oracle_alignment_rate=None if valid_count == 0 else len(aligned) / valid_count,
        invalid_output_count=sum(row.status == "invalid_output" for row in selected),
        provider_failure_count=sum(row.status == "provider_error" for row in selected),
        mcp_failure_count=sum(row.status == "mcp_error" for row in selected),
        completion_count=sum(row.execution_status == "completed" and row.parse_ok for row in selected),
    )


def _rate_difference(intervention: ConditionMetric, null: ConditionMetric) -> dict[str, Any] | None:
    if intervention.flip_rate is None or null.flip_rate is None:
        return None
    return {
        "estimate": intervention.flip_rate - null.flip_rate,
        "intervention_flips": intervention.flip_count,
        "intervention_valid": intervention.valid_count,
        "null_flips": null.flip_count,
        "null_valid": null.valid_count,
        "significance_test": None,
        "note": _DIFFERENCE_NOTE,
    }


async def _run_baseline(
    spec: StudySpec,
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
    checkpoint = recorder.checkpoint
    captured_value = None
    stop_reason = None
    observation = None
    prefix = None
    prefix_usage = None
    prefix_ids: list[str] = []
    fork_call_id = None
    checkpoint_payload = None
    if checkpoint is None:
        stop_reason = "factual_invalid" if _provider_failed(result) else "intervention_unobserved"
    else:
        observation = checkpoint.visible
        prefix = prefix_sha256(checkpoint.state)
        prefix_usage = usage_payload(checkpoint.state)
        prefix_ids = list(checkpoint.state.provider_response_ids)
        fork_call_id = checkpoint.call_id
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
    stored = StoredBaseline(
        schema_version=CONTROL_SCHEMA_VERSION,
        study_id=spec.study_id,
        checkpoint=checkpoint_payload,
        checkpoint_draw_index=0,
        final_state=result.state.model_dump(mode="json"),
        decision_parsed=result.decision_parsed,
        parse_ok=result.parse_ok,
        execution_status=result.execution_status,
        flags=list(result.flags),
        error=_public_text(result.error) if result.error else None,
        observation_original=observation,
        observation_sha256=None if observation is None else sha256_text(observation),
        captured_value=captured_value,
        prefix_sha256=prefix,
        prefix_usage=prefix_usage,
        prefix_response_ids=prefix_ids,
        fork_call_id=fork_call_id,
        usage=usage_payload(result.state),
        model_calls=limiter.used - before,
        provider_response_ids=list(result.state.provider_response_ids),
        factual_valid=stop_reason is None,
        stop_reason=stop_reason,
    )
    _write_exclusive(output_dir / "baseline.json", stored.model_dump_json(indent=2) + "\n")
    _append_execution_trace(
        output_dir / "traces.jsonl",
        spec,
        result,
        arm="baseline",
        condition_id=None,
        repetition_index=0,
        prefix=prefix,
        observation_original=observation,
        observation_mutated=None,
        oracle=_posthoc_oracle(spec.fixture_id, spec.tool, spec.field, None),
        protocol=_protocol_version(client),
    )


async def _run_remaining(
    spec: StudySpec,
    output_dir: Path,
    client: Client,
    limiter: "_LimitedAdapter",
    baseline: StoredBaseline,
    done: set[tuple[str, int]],
) -> None:
    checkpoint = _checkpoint_from_payload(baseline.checkpoint or {})
    final_state = AgentState.model_validate(baseline.final_state)
    journal_path = output_dir / "continuations.jsonl"
    for slot in spec.schedule:
        key = (slot.condition_id, slot.repetition_index)
        if key in done:
            continue
        condition = _condition_by_id(spec, slot.condition_id)
        if limiter.used >= limiter._limit:
            record = _unrun_record(spec, baseline, slot, condition, "budget_exhausted", "request ceiling reached")
            _append_record(journal_path, record)
            done.add(key)
            continue
        structured = checkpoint.raw.get("structured_content")
        if not isinstance(structured, dict):
            record = _unrun_record(
                spec,
                baseline,
                slot,
                condition,
                "observation_not_visible",
                "checkpoint has no structured observation",
            )
            _append_record(journal_path, record)
            done.add(key)
            continue
        try:
            visible = prepare_control_observation(
                role=slot.role,
                tool_name=spec.tool,
                field_path=spec.field,
                structured=structured,
                visible=checkpoint.visible,
                value=condition.field_value,
            )
        except (TypeError, ValueError) as exc:
            record = _unrun_record(spec, baseline, slot, condition, "observation_not_visible", _public_text(str(exc)))
            _append_record(journal_path, record)
            done.add(key)
            continue
        source: Literal["base", "overlay"] = "base" if slot.role == "null" else "overlay"
        before_calls = limiter.used
        message_count = len(checkpoint.state.messages)
        branch_result = await _continue(
            client,
            checkpoint,
            final_state,
            limiter,
            visible,
            spec.fixture_id,
            spec.tool,
            source,
        )
        if len(checkpoint.state.messages) != message_count:
            raise RuntimeError("checkpoint was modified")
        visible_ok = _observation_visible(branch_result, spec.tool, visible, source)
        status = classify_continuation(
            factual_valid=baseline.factual_valid,
            execution_status=branch_result.execution_status,
            parse_ok=branch_result.parse_ok,
            decision=branch_result.decision_parsed,
            factual_decision=baseline.decision_parsed,
            observation_visible=visible_ok,
            flags=branch_result.flags,
        )
        evidence_changed = sha256_text(visible) != baseline.observation_sha256
        evidence_intervention = (
            slot.role == "intervention" and status == "decision_changed" and evidence_changed and visible_ok
        )
        record = ContinuationRecord(
            schema_version=CONTROL_SCHEMA_VERSION,
            study_id=spec.study_id,
            condition_id=slot.condition_id,
            role=slot.role,
            repetition_index=slot.repetition_index,
            order_index=slot.order_index,
            checkpoint_id=baseline.prefix_sha256,
            checkpoint_draw_index=0,
            sampling_unit=SAMPLING_UNIT,
            independent_checkpoint_sample=False,
            field_value=condition.field_value,
            distance=condition.distance,
            search_candidate=False,
            status=status,
            decision_parsed=branch_result.decision_parsed,
            parse_ok=branch_result.parse_ok,
            execution_status=branch_result.execution_status,
            decision_changed=status == "decision_changed",
            evidence_intervention=evidence_intervention,
            execution_valid=status in {"decision_stable", "decision_changed"},
            observation_sha256=sha256_text(visible),
            fork_call_id=checkpoint.call_id,
            flags=list(branch_result.flags),
            error=_public_text(branch_result.error) if branch_result.error else None,
            oracle_decision=_posthoc_oracle(spec.fixture_id, spec.tool, spec.field, condition.field_value),
            usage=usage_payload(branch_result.state),
            model_calls=limiter.used - before_calls,
            provider_response_ids=list(branch_result.state.provider_response_ids),
        )
        _append_record(journal_path, record)
        done.add(key)
        _append_execution_trace(
            output_dir / "traces.jsonl",
            spec,
            branch_result,
            arm="factual_replay" if slot.role == "null" else "mutant",
            condition_id=slot.condition_id,
            repetition_index=slot.repetition_index,
            prefix=baseline.prefix_sha256,
            observation_original=baseline.observation_original,
            observation_mutated=visible,
            oracle=record.oracle_decision,
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
    source: Literal["base", "overlay"],
) -> AgentResult:
    branch = copy_state(checkpoint.state)
    check_prefix_pair(branch, checkpoint.state)
    restore(branch, checkpoint, content=visible, source=source)
    _carry_same_turn_results(branch, baseline_state, checkpoint)
    return await run_agent(
        client,
        branch,
        adapter,
        resolver=OverlayResolver({tool: visible}, incident_id=incident_id),
    )


def _unrun_record(
    spec: StudySpec,
    baseline: StoredBaseline,
    slot: ScheduleSlot,
    condition: ControlCondition,
    status: ContinuationStatus,
    error: str,
) -> ContinuationRecord:
    return ContinuationRecord(
        schema_version=CONTROL_SCHEMA_VERSION,
        study_id=spec.study_id,
        condition_id=slot.condition_id,
        role=slot.role,
        repetition_index=slot.repetition_index,
        order_index=slot.order_index,
        checkpoint_id=baseline.prefix_sha256,
        checkpoint_draw_index=0,
        sampling_unit=SAMPLING_UNIT,
        independent_checkpoint_sample=False,
        field_value=condition.field_value,
        distance=condition.distance,
        search_candidate=False,
        status=status,
        decision_parsed=None,
        parse_ok=False,
        execution_status="budget_exhausted" if status == "budget_exhausted" else None,
        decision_changed=False,
        evidence_intervention=False,
        execution_valid=False,
        observation_sha256=None,
        fork_call_id=baseline.fork_call_id,
        flags=["budget_exhausted"] if status == "budget_exhausted" else [],
        error=error,
        oracle_decision=None,
        usage=None,
        model_calls=0,
        provider_response_ids=[],
    )


def _validated_outcomes(outcomes: Sequence[RecordedOutcome]) -> list[RecordedOutcome]:
    rows = list(outcomes)
    if not rows:
        raise ValueError("at least one recorded outcome is required")
    values = [row.value for row in rows]
    if len(set(values)) != len(values):
        raise ValueError("recorded outcomes must be unique")
    return rows


def _early_stop_report(
    method: OrderingMethod,
    seed: int | None,
    order: Sequence[RecordedOutcome],
) -> OrderingEvaluation:
    evaluated = 0
    position = None
    distance = None
    for index, row in enumerate(order, start=1):
        evaluated = index
        if row.verified_flip:
            position = index
            distance = row.distance
            break
    return OrderingEvaluation(
        method=method,
        seed=seed,
        order=[row.value for row in order],
        first_verified_flip_position=position,
        evaluated_count=evaluated,
        smallest_discovered_distance=distance,
        coverage=evaluated,
        candidate_count=len(order),
        note=_ORDERING_NOTE,
    )


def _exhaustive_report(order: Sequence[RecordedOutcome]) -> OrderingEvaluation:
    flips = [row for row in order if row.verified_flip]
    position = None
    for index, row in enumerate(order, start=1):
        if row.verified_flip:
            position = index
            break
    smallest = None if not flips else min(row.distance for row in flips)
    return OrderingEvaluation(
        method="exhaustive",
        seed=None,
        order=[row.value for row in order],
        first_verified_flip_position=position,
        evaluated_count=len(order),
        smallest_discovered_distance=smallest,
        coverage=len(order),
        candidate_count=len(order),
        note=_ORDERING_NOTE,
    )


def _require_captured_value(
    structured: Mapping[str, Any],
    tool_name: str,
    field_path: str,
    visible: str,
    value: int,
) -> None:
    if not isinstance(structured, dict):
        raise TypeError("captured observation is not an object")
    model_type = TOOL_MODELS[tool_name]
    current = model_type.model_validate(dict(structured))
    reserialized = canonical_json(current.model_dump(mode="json"))
    if reserialized != visible:
        raise ValueError("captured observation did not reserialize")
    captured = _stored_integer(current.model_dump(mode="json"), field_path)
    if captured != value:
        raise ValueError("null control must replay the captured value")


def _observation_visible(result: AgentResult, tool: str, text: str, source: str) -> bool:
    in_messages = text in [message.content for message in result.state.messages if message.role == "tool"]
    in_records = any(
        record.name == tool and record.source == source and record.model_visible_observation == text
        for record in result.state.tool_calls
    )
    return in_messages and in_records


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


def _provider_failed(result: AgentResult) -> bool:
    return "provider_error" in result.flags or result.execution_status == "model_error"


def _captured_integer(checkpoint: Checkpoint, tool: str, field: str) -> int:
    structured = checkpoint.raw.get("structured_content")
    if not isinstance(structured, dict):
        raise TypeError("checkpoint has no structured observation")
    current = TOOL_MODELS[tool].model_validate(structured)
    value = getattr(current, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("captured field is not an integer")
    reserialized = canonical_json(current.model_dump(mode="json"))
    if reserialized != checkpoint.visible:
        raise ValueError("captured observation did not reserialize")
    return value


def _posthoc_oracle(fixture_id: str, tool: str, field: str, value: int | None) -> str | None:
    """Score the arm's evidence after the model has answered."""

    from parallax.oracle import decide

    fixture = get_fixture(fixture_id)
    if fixture is None:
        return None
    metrics = fixture.metrics
    logs = fixture.logs
    deployment = fixture.deployment
    if value is not None:
        payload = _tool_payload(fixture, tool)
        payload[field] = value
        edited = TOOL_MODELS[tool].model_validate(payload)
        if tool == "get_metrics":
            metrics = edited  # type: ignore[assignment]
        elif tool == "get_logs":
            logs = edited  # type: ignore[assignment]
        elif tool == "get_deployment":
            deployment = edited  # type: ignore[assignment]
    return decide(metrics, logs, deployment).value


def _append_execution_trace(
    path: Path,
    spec: StudySpec,
    result: AgentResult,
    *,
    arm: Literal["baseline", "factual_replay", "mutant"],
    condition_id: str | None,
    repetition_index: int,
    prefix: str | None,
    observation_original: str | None,
    observation_mutated: str | None,
    oracle: str | None,
    protocol: str | None,
) -> None:
    started = utc_now()
    record = TraceRecord(
        run_id=uuid.uuid4().hex,
        pair_id=spec.study_id,
        arm=arm,
        incident_id=spec.fixture_id,
        fixture_hash=spec.fixture_hash,
        intervention_id=condition_id,
        model_id=spec.model_id,
        model_params=spec.model_params,
        prompt_hash=spec.prompt_hash,
        sdk_version=spec.sdk_version,
        negotiated_protocol_version=protocol,
        snapshot_id=prefix,
        phase="harness",
        scenario_id=None,
        repeat_index=repetition_index,
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
        intervention_source_sha256=spec.source_sha256,
        execution_status=result.execution_status,
        provider_response_ids=list(result.state.provider_response_ids),
        experiment_id=spec.study_id,
    )
    append_trace(path, record)


def _deduped_usage(baseline: StoredBaseline, rows: Sequence[ContinuationRecord]) -> dict[str, int] | None:
    if baseline.usage is None or baseline.prefix_usage is None:
        return None
    prefix_ids = set(baseline.prefix_response_ids)
    seen_new: set[str] = set()
    total = {
        "input_tokens": baseline.usage["input_tokens"],
        "output_tokens": baseline.usage["output_tokens"],
    }
    for row in rows:
        if row.usage is None:
            return None
        new_ids = [item for item in row.provider_response_ids if item not in prefix_ids]
        if len(new_ids) != len(set(new_ids)) or any(item in seen_new for item in new_ids):
            return None
        seen_new.update(new_ids)
        extra_input = row.usage["input_tokens"] - baseline.prefix_usage["input_tokens"]
        extra_output = row.usage["output_tokens"] - baseline.prefix_usage["output_tokens"]
        if extra_input < 0 or extra_output < 0:
            return None
        total["input_tokens"] += extra_input
        total["output_tokens"] += extra_output
    return total


def _unique_response_ids(baseline: StoredBaseline, rows: Sequence[ContinuationRecord]) -> list[str]:
    found = list(baseline.provider_response_ids)
    seen = set(found)
    for row in rows:
        for item in row.provider_response_ids:
            if item not in seen:
                seen.add(item)
                found.append(item)
    return found


def _condition_by_id(spec: StudySpec, condition_id: str) -> ControlCondition:
    for condition in spec.conditions:
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"unknown condition: {condition_id}")


def _tool_payload(fixture: Any, tool: str) -> dict[str, Any]:
    if tool == "get_metrics":
        return fixture.metrics.model_dump(mode="json")
    if tool == "get_logs":
        return fixture.logs.model_dump(mode="json")
    if tool == "get_deployment":
        return fixture.deployment.model_dump(mode="json")
    raise ValueError(f"unknown tool: {tool}")


def _stored_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("stored field is not an integer")
    return value


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


def _store_manifest(path: Path, spec: StudySpec) -> None:
    payload = spec.model_dump_json(indent=2) + "\n"
    if path.is_file():
        existing = StudySpec.model_validate_json(path.read_text(encoding="utf-8"))
        if _spec_body(existing) != _spec_body(spec):
            raise ValueError("control directory belongs to a different study")
        return
    _write_exclusive(path, payload)


def _spec_body(spec: StudySpec) -> dict[str, Any]:
    body = spec.model_dump()
    body.pop("git_revision", None)
    return body


def _write_result(path: Path, result: StudyResult) -> None:
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
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise ValueError("control record already exists; pass --resume") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _append_record(path: Path, record: ContinuationRecord) -> None:
    line = record.model_dump_json()
    if "\n" in line or "\r" in line:
        raise ValueError("continuation record must serialize to one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_torn_jsonl_tail(path, ContinuationRecord.model_validate_json)
    existing = {(row.condition_id, row.repetition_index) for row in _read_journal(path)}
    if (record.condition_id, record.repetition_index) in existing:
        return
    _append_line(path, line)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_journal(path: Path) -> list[ContinuationRecord]:
    if not path.is_file():
        return []
    drop_torn_jsonl_tail(path, ContinuationRecord.model_validate_json)
    return [
        ContinuationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_baseline(path: Path) -> StoredBaseline:
    return StoredBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def _study_started(baseline_path: Path, journal_path: Path) -> bool:
    return baseline_path.is_file() or journal_path.is_file()


def _spent_requests(baseline_path: Path, journal_path: Path) -> int:
    spent = 0
    if baseline_path.is_file():
        spent += _load_baseline(baseline_path).model_calls
    spent += sum(row.model_calls for row in _read_journal(journal_path))
    return spent


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _package_version(name: str, fallback: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback


def _usage_text(usage: dict[str, int] | None) -> str:
    if usage is None:
        return "null"
    return f"input_tokens={usage['input_tokens']} output_tokens={usage['output_tokens']}"


def _rate_text(rate: float | None) -> str:
    if rate is None:
        return "none"
    return f"{rate:.6f}"


def _interval_text(interval: dict[str, float] | None) -> str:
    if interval is None:
        return "none"
    return f"{interval['low']:.6f},{interval['high']:.6f}"


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
