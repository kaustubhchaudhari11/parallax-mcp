"""Mini-pilot experiment runner.

The default command is a dry run. ``--live`` is required before a
provider client exists. Each slot reuses ``run_pair``. A slot whose
trial id is already in the journal is not run again.

S01, S09, and S17 on F000 are the mini-pilot. The protocol pilot is
three fixtures at R=5. Scripted adapters can exercise the scheduler.
Their rows stay out of the empirical rates.
"""

import argparse
import platform
import sys
from collections.abc import Sequence
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import anyio
from pydantic import BaseModel, ConfigDict, Field

from parallax.agent import ModelAdapter
from parallax.capture import sha256_text
from parallax.fixtures import get_fixture
from parallax.live import configured_budget, os_environ
from parallax.live_pair import max_model_requests, pair_model_params
from parallax.metrics import (
    ExperimentSummary,
    PairAccounting,
    TrialRecord,
    account_pair,
    arm_snapshot,
    format_summary,
    summarize,
)
from parallax.mutate import intervention_source_sha256
from parallax.observe import canonical_json
from parallax.prompt import prompt_hash
from parallax.replay import PairRun, fixture_hash, run_pair
from parallax.scenarios import SCENARIOS, Scenario
from parallax.trace import (
    RESEARCH_PHASES,
    TRACE_SCHEMA_VERSION,
    TraceRecord,
    append_trace,
)

MANIFEST_SCHEMA_VERSION = 1
TRIAL_SCHEMA_VERSION = 1
_DEFAULT_MODEL_CALLS = 4
_DEFAULT_TOOL_CALLS = 6
_DEFAULT_REPEATS = 1
_FAMILIES = ("H1", "H2", "H3")
_MINI_PILOT = ("S01", "S09", "S17")
ScheduleName = Literal["mini-pilot", "custom"]
_SECRET_MARKERS = ("sk-", "api_key", "bearer ", "authorization")
_LABEL_TOKENS = (
    "oracle_decision",
    "oracle_before",
    "oracle_after",
    "expected_decision",
    "declared_oracle",
)
PhaseName = Literal["pilot", "exploratory"]


class ExperimentManifest(BaseModel):
    """Versioned configuration for one research run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    experiment_id: str
    scenario_ids: list[str]
    fixture_ids: list[str]
    hypotheses: list[str]
    provider: Literal["openai"]
    model_id: str
    model_params: dict[str, Any]
    prompt_hashes: dict[str, str]
    fixture_hashes: dict[str, str]
    intervention_source_sha256: str
    scenario_bindings: list[dict[str, str]]
    schedule: ScheduleName
    provider_sdk_version: str
    repeats: int
    model_call_budget: int
    tool_call_budget: int
    pair_model_call_budget: int
    enforced_pair_model_calls: int
    worst_case_model_requests: int
    token_budget: int | None
    phase: PhaseName
    software_version: str
    sdk_version: str
    python_version: str
    trace_schema_version: int
    notes: list[str] = Field(default_factory=list)


class TrialSlot:
    """One scenario repeat. The id is stable across a resume."""

    def __init__(self, scenario: Scenario, repeat_index: int) -> None:
        self.scenario = scenario
        self.repeat_index = repeat_index

    @property
    def trial_id(self) -> str:
        return f"{self.scenario.scenario_id}#{self.repeat_index}"


def pilot_scenarios() -> tuple[Scenario, Scenario, Scenario]:
    """First declared scenario of each hypothesis family."""

    found: dict[str, Scenario] = {}
    for scenario in SCENARIOS:
        if scenario.family in _FAMILIES and scenario.family not in found:
            found[scenario.family] = scenario
    missing = [family for family in _FAMILIES if family not in found]
    if missing:
        raise ValueError("pilot family is missing: " + ",".join(missing))
    return (found["H1"], found["H2"], found["H3"])


def scenario_by_id(scenario_id: str) -> Scenario:
    """Return one declared scenario."""

    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise ValueError(f"unknown scenario: {scenario_id}")


def build_manifest(
    *,
    scenario_ids: Sequence[str],
    repeats: int,
    model_id: str,
    model_call_budget: int,
    tool_call_budget: int,
    pair_model_call_budget: int,
    phase: str = "pilot",
) -> ExperimentManifest:
    """Validate a research configuration and fingerprint its inputs."""

    if phase not in RESEARCH_PHASES:
        raise ValueError("research phase must be pilot or exploratory")
    if repeats < 1:
        raise ValueError("repeats must be an integer of at least 1")
    if model_call_budget < 1 or tool_call_budget < 1 or pair_model_call_budget < 1:
        raise ValueError("budgets must be integers of at least 1")
    if not scenario_ids:
        raise ValueError("at least one scenario is required")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario ids must be unique")
    selected = [scenario_by_id(scenario_id) for scenario_id in scenario_ids]
    model = model_id.strip()
    if not model:
        raise ValueError("model id is required")
    fixture_ids: list[str] = []
    for scenario in selected:
        if scenario.fixture_id not in fixture_ids:
            fixture_ids.append(scenario.fixture_id)
    prompt_hashes = {fixture_id: prompt_hash(fixture_id) for fixture_id in fixture_ids}
    fixture_hashes: dict[str, str] = {}
    for fixture_id in fixture_ids:
        fixture = get_fixture(fixture_id)
        if fixture is None:
            raise ValueError(f"unknown incident: {fixture_id}")
        fixture_hashes[fixture_id] = fixture_hash(fixture)
    hypotheses = [family for family in _FAMILIES if any(row.family == family for row in selected)]
    enforced = max_model_requests(model_call_budget, pair_model_call_budget)
    slots = len(selected) * repeats
    schedule_name = _schedule_name(scenario_ids)
    bindings = [_binding(scenario) for scenario in selected]
    identity = {
        "fixture_hashes": fixture_hashes,
        "hypotheses": hypotheses,
        "intervention_source_sha256": intervention_source_sha256(),
        "model_call_budget": model_call_budget,
        "model_id": model,
        "model_params": pair_model_params(model),
        "pair_model_call_budget": pair_model_call_budget,
        "phase": phase,
        "prompt_hashes": prompt_hashes,
        "provider_sdk_version": _package_version("openai", "absent"),
        "python_version": platform.python_version(),
        "repeats": repeats,
        "scenario_bindings": bindings,
        "scenario_ids": list(scenario_ids),
        "schedule": schedule_name,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sdk_version": _package_version("mcp", "unknown"),
        "software_version": _package_version("parallax", "0.1.0"),
        "token_budget": None,
        "tool_call_budget": tool_call_budget,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }
    notes = [
        "Baselines are the unmodified arm of each pair.",
        "This runner does not share one checkpoint across scenarios.",
        "Each pair is capped at the shared-prefix request bound.",
    ]
    if schedule_name == "mini-pilot":
        notes.append(
            "Schedule label: mini-pilot. Scenarios S01, S09, and S17 on fixture F000. "
            "The protocol pilot uses three fixtures and R=5."
        )
    return ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        experiment_id="exp-3a-" + sha256_text(canonical_json(identity))[:16],
        scenario_ids=list(scenario_ids),
        fixture_ids=fixture_ids,
        hypotheses=hypotheses,
        provider="openai",
        model_id=model,
        model_params=pair_model_params(model),
        prompt_hashes=prompt_hashes,
        fixture_hashes=fixture_hashes,
        intervention_source_sha256=identity["intervention_source_sha256"],
        scenario_bindings=bindings,
        schedule=schedule_name,
        provider_sdk_version=str(identity["provider_sdk_version"]),
        repeats=repeats,
        model_call_budget=model_call_budget,
        tool_call_budget=tool_call_budget,
        pair_model_call_budget=pair_model_call_budget,
        enforced_pair_model_calls=enforced,
        worst_case_model_requests=enforced * slots,
        token_budget=None,
        phase=cast(PhaseName, phase),
        software_version=str(identity["software_version"]),
        sdk_version=str(identity["sdk_version"]),
        python_version=str(identity["python_version"]),
        trace_schema_version=TRACE_SCHEMA_VERSION,
        notes=notes,
    )


def pilot_manifest(
    *,
    repeats: int,
    model_id: str,
    model_call_budget: int,
    tool_call_budget: int,
    pair_model_call_budget: int,
) -> ExperimentManifest:
    """Manifest for the one-scenario-per-family pilot."""

    scenarios = pilot_scenarios()
    return build_manifest(
        scenario_ids=[scenario.scenario_id for scenario in scenarios],
        repeats=repeats,
        model_id=model_id,
        model_call_budget=model_call_budget,
        tool_call_budget=tool_call_budget,
        pair_model_call_budget=pair_model_call_budget,
        phase="pilot",
    )


def schedule(manifest: ExperimentManifest) -> list[TrialSlot]:
    """Expand scenarios and repeats. Order is scenario, then repeat."""

    slots: list[TrialSlot] = []
    for scenario_id in manifest.scenario_ids:
        scenario = scenario_by_id(scenario_id)
        for repeat_index in range(manifest.repeats):
            slots.append(TrialSlot(scenario, repeat_index))
    return slots


def format_manifest(manifest: ExperimentManifest) -> str:
    """Dry-run plan. The key value is not included."""

    lines = [
        "dry-run: no model request will be sent",
        f"experiment_id={manifest.experiment_id}",
        f"schema_version={manifest.schema_version}",
        f"phase={manifest.phase}",
        f"schedule={manifest.schedule}",
        "research_trial=true",
        "scenarios=" + ",".join(manifest.scenario_ids),
        "fixtures=" + ",".join(manifest.fixture_ids),
        "hypotheses=" + ",".join(manifest.hypotheses),
        f"provider={manifest.provider}",
        f"provider_sdk_version={manifest.provider_sdk_version}",
        f"model={manifest.model_id}",
        f"repeats={manifest.repeats}",
        f"slots={len(manifest.scenario_ids) * manifest.repeats}",
        f"model_call_budget={manifest.model_call_budget}",
        f"tool_call_budget={manifest.tool_call_budget}",
        f"pair_model_call_budget={manifest.pair_model_call_budget}",
        f"enforced_pair_model_calls={manifest.enforced_pair_model_calls}",
        f"worst_case_model_requests={manifest.worst_case_model_requests}",
        f"token_budget={manifest.token_budget}",
        f"software_version={manifest.software_version}",
        f"sdk_version={manifest.sdk_version}",
        f"python_version={manifest.python_version}",
        f"trace_schema_version={manifest.trace_schema_version}",
        "prompt_hashes=" + ",".join(
            f"{fixture_id}:{manifest.prompt_hashes[fixture_id]}" for fixture_id in manifest.fixture_ids
        ),
        "fixture_hashes=" + ",".join(
            f"{fixture_id}:{manifest.fixture_hashes[fixture_id]}" for fixture_id in manifest.fixture_ids
        ),
        f"intervention_source_sha256={manifest.intervention_source_sha256}",
        "scenario_bindings="
        + ",".join(
            f"{row['scenario_id']}:{row['family']}:{row['intervention_id']}:{row['fixture_id']}"
            for row in manifest.scenario_bindings
        ),
    ]
    if manifest.schedule == "mini-pilot":
        lines.append("mini-pilot: S01, S09, S17 on F000. Protocol pilot: three fixtures, R=5.")
    return "\n".join(lines) + "\n"


def experiment_directory(root: Path, experiment_id: str) -> Path:
    """Directory for one experiment. It lives under the gitignored runs tree."""

    return root / "runs" / "experiments" / experiment_id


def read_completed_trial_ids(path: Path) -> set[str]:
    """Trial ids already written. A missing journal is an empty set."""

    if not path.is_file():
        return set()
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            found.add(TrialRecord.model_validate_json(line).trial_id)
    return found


def eligibility_reasons(
    run: PairRun,
    *,
    manifest: ExperimentManifest,
    scenario: Scenario,
    provider_executed: Literal["openai", "scripted"],
) -> list[str]:
    """Reasons a pair cannot support an empirical claim.

    An unobserved intervention is a retained outcome, not one of these
    reasons. Provider and engine failures are retained on the arm.
    """

    reasons: list[str] = []
    if provider_executed != "openai":
        reasons.append("scripted_provider")
    if _scripted_model(manifest.model_id) or _scripted_model(run.baseline.model_id):
        reasons.append("scripted_model")
    if run.baseline.phase not in RESEARCH_PHASES:
        reasons.append("harness_phase")
    if run.baseline.phase != manifest.phase:
        reasons.append("phase_mismatch")
    if run.baseline.model_id != manifest.model_id:
        reasons.append("model_mismatch")
    if run.baseline.model_params != manifest.model_params:
        reasons.append("param_mismatch")
    if run.baseline.scenario_id != scenario.scenario_id:
        reasons.append("scenario_mismatch")
    rows = [run.baseline, run.factual, run.mutant]
    if any(row is None or row.execution_status != "completed" for row in rows):
        reasons.append("arm_failure")
    pair_ids = [run.baseline.pair_id, run.mutant.pair_id]
    if run.factual is not None:
        pair_ids.append(run.factual.pair_id)
    if len(set(pair_ids)) != 1:
        reasons.append("pair_id_mismatch")
    if run.baseline.arm != "baseline" or run.mutant.arm != "mutant":
        reasons.append("arm_mismatch")
    if run.factual is not None and run.factual.arm != "factual_replay":
        reasons.append("arm_mismatch")
    if _label_leak(run):
        reasons.append("oracle_label_in_transcript")
    if run.prefix_sha256 is None or run.factual is None or run.snapshot is None:
        reasons.append("checkpoint_unobserved")
        return _unique(reasons)
    if run.factual.prefix_sha256 != run.prefix_sha256 or run.mutant.prefix_sha256 != run.prefix_sha256:
        reasons.append("prefix_hash_mismatch")
    reasons.extend(_continuation_reasons(run))
    reasons.extend(_observation_reasons(run, scenario))
    return _unique(reasons)


def failure_reasons(run: PairRun) -> list[str]:
    """Explicit arm failures. Secret-looking text is redacted."""

    reasons: list[str] = []
    rows = [run.baseline, run.factual, run.mutant]
    for row in rows:
        if row is None:
            reasons.append("factual_replay:missing")
            continue
        if row.execution_status != "completed":
            reasons.append(f"{row.arm}:{row.execution_status or 'missing_status'}")
        if row.error:
            reasons.append(f"{row.arm}:{_public_text(row.error)}")
        for flag in row.flags:
            if flag not in {"parse_failure"}:
                reasons.append(f"{row.arm}:{flag}")
    return _unique(reasons)


async def execute_experiment(
    manifest: ExperimentManifest,
    output_dir: Path,
    adapter: ModelAdapter,
    *,
    provider_executed: Literal["openai", "scripted"],
    resume: bool,
) -> tuple[list[TrialRecord], ExperimentSummary]:
    """Run the remaining slots. Completed trial ids are not repeated."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    _store_manifest(manifest_path, manifest)
    trial_path = output_dir / "trials.jsonl"
    trace_path = output_dir / "traces.jsonl"
    completed = read_completed_trial_ids(trial_path)
    if completed and not resume:
        raise ValueError("experiment journal exists; pass --resume to continue")
    written = list(_read_trials(trial_path))
    for slot in schedule(manifest):
        if slot.trial_id in completed:
            continue
        record = await _run_slot(
            manifest,
            slot,
            adapter,
            provider_executed=provider_executed,
            trace_path=trace_path,
        )
        append_trial(trial_path, record)
        written.append(record)
        completed.add(record.trial_id)
        _write_summary(output_dir / "summary.json", summarize(manifest.experiment_id, written))
    summary = summarize(manifest.experiment_id, written)
    _write_summary(output_dir / "summary.json", summary)
    return written, summary


def append_trial(path: Path, record: TrialRecord) -> None:
    """Append one trial journal line."""

    line = record.model_dump_json()
    if "\n" in line or "\r" in line:
        raise ValueError("trial record must serialize to one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Without ``--live`` this only prints the manifest."""

    parser = argparse.ArgumentParser(prog="python -m parallax.experiment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenarios", nargs="+", metavar="SCENARIO_ID", default=None)
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument("--model-calls", type=int, default=None)
    parser.add_argument("--tool-calls", type=int, default=None)
    parser.add_argument("--pair-model-calls", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(None if argv is None else list(argv))
    if args.live and args.dry_run:
        print("choose either --dry-run or --live", file=sys.stderr)
        return 2
    from parallax.openai_adapter import load_project_env

    load_project_env()
    environ = dict(os_environ())
    model_id = environ.get("PARALLAX_MODEL", "").strip()
    if args.live and not model_id:
        print("PARALLAX_MODEL is not set", file=sys.stderr)
        return 2
    try:
        model_calls = configured_budget(
            args.model_calls, environ, "PARALLAX_MAX_MODEL_CALLS", _DEFAULT_MODEL_CALLS
        )
        tool_calls = configured_budget(
            args.tool_calls, environ, "PARALLAX_MAX_TOOL_CALLS", _DEFAULT_TOOL_CALLS
        )
        pair_calls = args.pair_model_calls
        if pair_calls is None:
            pair_calls = max_model_requests(model_calls, 2 * model_calls - 1)
        manifest_args = {
            "repeats": args.repeats,
            "model_id": model_id or "(unset)",
            "model_call_budget": model_calls,
            "tool_call_budget": tool_calls,
            "pair_model_call_budget": pair_calls,
        }
        if args.scenarios is None:
            manifest = pilot_manifest(**manifest_args)
        else:
            manifest = build_manifest(
                scenario_ids=args.scenarios,
                phase="pilot",
                **manifest_args,
            )
    except ValueError as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    if not args.live:
        sys.stdout.write(format_manifest(manifest))
        return 0
    directory = args.output or experiment_directory(Path("."), manifest.experiment_id)
    try:
        from parallax.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.from_env()
    except RuntimeError as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    try:
        run_experiment = partial(
            execute_experiment,
            manifest,
            directory,
            adapter,
            provider_executed="openai",
            resume=args.resume,
        )
        _written, summary = anyio.run(run_experiment)
    except ValueError as exc:
        print(_public_text(str(exc)), file=sys.stderr)
        return 2
    sys.stdout.write(format_summary(summary))
    print(f"output_dir={directory}")
    completion = _summary_value(summary, "pair_completion")
    validity = _summary_value(summary, "pair_validity")
    if completion != 1 or validity != 1:
        return 1
    return 0


def _summary_value(summary: ExperimentSummary, name: str) -> float | None:
    for row in summary.proportions:
        if row.name == name:
            return row.value
    return None


async def _run_slot(
    manifest: ExperimentManifest,
    slot: TrialSlot,
    adapter: ModelAdapter,
    *,
    provider_executed: Literal["openai", "scripted"],
    trace_path: Path,
) -> TrialRecord:
    scenario = slot.scenario
    try:
        run = _scrub_run(
            await run_pair(
                incident_id=scenario.fixture_id,
                intervention_id=scenario.intervention_id,
                adapter=adapter,
                model_id=manifest.model_id,
                model_params=manifest.model_params,
                step_budget=manifest.model_call_budget,
                token_budget=manifest.token_budget,
                phase=manifest.phase,
                scenario_id=scenario.scenario_id,
                repeat_index=slot.repeat_index,
                tool_call_budget=manifest.tool_call_budget,
                model_call_limit=manifest.enforced_pair_model_calls,
            )
        )
    except ValueError as exc:
        return _failed_slot(manifest, slot, provider_executed, _public_text(str(exc)))
    _write_traces(trace_path, run, manifest.experiment_id)
    reasons = eligibility_reasons(
        run,
        manifest=manifest,
        scenario=scenario,
        provider_executed=provider_executed,
    )
    accounting = account_pair(run)
    arms = [arm_snapshot(run.baseline)]
    if run.factual is not None:
        arms.append(arm_snapshot(run.factual))
    arms.append(arm_snapshot(run.mutant))
    return TrialRecord(
        schema_version=TRIAL_SCHEMA_VERSION,
        experiment_id=manifest.experiment_id,
        trial_id=slot.trial_id,
        scenario_id=scenario.scenario_id,
        fixture_id=scenario.fixture_id,
        family=scenario.family,
        repeat_index=slot.repeat_index,
        phase=manifest.phase,
        provider_executed=provider_executed,
        pair_id=run.baseline.pair_id,
        empirical_eligible=not reasons,
        ineligibility_reasons=reasons,
        failure_reasons=failure_reasons(run),
        arms=arms,
        accounting=accounting,
    )


def _failed_slot(
    manifest: ExperimentManifest,
    slot: TrialSlot,
    provider_executed: Literal["openai", "scripted"],
    reason: str,
) -> TrialRecord:
    empty = PairAccounting(
        unique_model_calls=0,
        cumulative_model_calls=0,
        shared_prefix_model_calls=0,
        incremental_model_calls=0,
        deduped_usage=None,
        cumulative_usage=None,
        deduped_tool_calls=0,
        cumulative_tool_calls=0,
        deduped_latency_ms=None,
        cumulative_latency_ms=None,
        unique_response_ids=[],
        shared_prefix_response_ids=[],
        incremental_response_ids=[],
    )
    return TrialRecord(
        schema_version=TRIAL_SCHEMA_VERSION,
        experiment_id=manifest.experiment_id,
        trial_id=slot.trial_id,
        scenario_id=slot.scenario.scenario_id,
        fixture_id=slot.scenario.fixture_id,
        family=slot.scenario.family,
        repeat_index=slot.repeat_index,
        phase=manifest.phase,
        provider_executed=provider_executed,
        pair_id=None,
        empirical_eligible=False,
        ineligibility_reasons=["runner_error", "scripted_provider"]
        if provider_executed != "openai"
        else ["runner_error"],
        failure_reasons=[reason],
        arms=[],
        accounting=empty,
    )


def _scrub_run(run: PairRun) -> PairRun:
    run.baseline = _scrub_row(run.baseline)
    if run.factual is not None:
        run.factual = _scrub_row(run.factual)
    run.mutant = _scrub_row(run.mutant)
    return run


def _scrub_row(row: TraceRecord) -> TraceRecord:
    if row.error is None:
        return row
    cleaned = _public_text(row.error)
    if cleaned == row.error:
        return row
    return row.model_copy(update={"error": cleaned})


def _write_traces(path: Path, run: PairRun, experiment_id: str) -> None:
    rows: list[TraceRecord] = [run.baseline]
    if run.factual is not None:
        rows.append(run.factual)
    rows.append(run.mutant)
    for row in rows:
        append_trace(path, row.model_copy(update={"experiment_id": experiment_id}))


def _continuation_reasons(run: PairRun) -> list[str]:
    assert run.snapshot is not None
    assert run.factual is not None
    prefix = set(run.snapshot.state.provider_response_ids)
    reasons: list[str] = []
    if not prefix:
        reasons.append("missing_provider_response_ids")
        return reasons
    factual_ids = set(run.factual.provider_response_ids)
    mutant_ids = set(run.mutant.provider_response_ids)
    baseline_ids = set(run.baseline.provider_response_ids)
    if not prefix <= factual_ids or not prefix <= mutant_ids or not prefix <= baseline_ids:
        reasons.append("fork_response_not_reused")
    factual_post = factual_ids - prefix
    mutant_post = mutant_ids - prefix
    baseline_post = baseline_ids - prefix
    if factual_post & mutant_ids or mutant_post & factual_ids:
        reasons.append("shared_post_fork_response")
    if factual_post & baseline_ids or mutant_post & baseline_ids:
        reasons.append("shared_post_fork_response")
    if baseline_post & factual_ids or baseline_post & mutant_ids:
        reasons.append("shared_post_fork_response")
    return reasons


def _observation_reasons(run: PairRun, scenario: Scenario) -> list[str]:
    assert run.factual is not None
    factual = run.factual
    mutant = run.mutant
    baseline = run.baseline
    reasons: list[str] = []
    if factual.observation_original != baseline.observation_original:
        reasons.append("original_observation_diverged")
    if mutant.observation_original != baseline.observation_original:
        reasons.append("original_observation_diverged")
    if factual.observation_mutated is not None:
        reasons.append("factual_arm_mutated")
    if mutant.observation_mutated is None or mutant.observation_mutated == mutant.observation_original:
        reasons.append("mutant_observation_missing")
    if scenario.family in {"H1", "H3"} and factual.oracle_decision != mutant.oracle_decision:
        reasons.append("oracle_shift")
    if scenario.family == "H2" and factual.oracle_decision == mutant.oracle_decision:
        reasons.append("oracle_unchanged")
    if mutant.observation_mutated and mutant.observation_mutated not in mutant.model_visible_observations:
        reasons.append("mutant_not_visible")
    if factual.observation_original and factual.observation_original not in factual.model_visible_observations:
        reasons.append("factual_not_visible")
    return reasons


def _label_leak(run: PairRun) -> bool:
    texts: list[str] = []
    groups = (run.baseline_messages, run.factual_messages, run.mutant_messages)
    for messages in groups:
        if messages is None:
            continue
        texts.extend(message.content for message in messages)
    blob = "\n".join(texts)
    return any(token in blob for token in _LABEL_TOKENS)


def _store_manifest(path: Path, manifest: ExperimentManifest) -> None:
    payload = manifest.model_dump_json(indent=2) + "\n"
    if path.is_file():
        existing = ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if existing.model_dump() != manifest.model_dump():
            raise ValueError("experiment directory belongs to a different manifest")
        return
    path.write_text(payload, encoding="utf-8", newline="\n")


def _write_summary(path: Path, summary: ExperimentSummary) -> None:
    path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")


def _read_trials(path: Path) -> list[TrialRecord]:
    if not path.is_file():
        return []
    return [
        TrialRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _schedule_name(scenario_ids: Sequence[str]) -> ScheduleName:
    if tuple(scenario_ids) == _MINI_PILOT:
        return "mini-pilot"
    return "custom"


def _binding(scenario: Scenario) -> dict[str, str]:
    return {
        "family": scenario.family,
        "fixture_id": scenario.fixture_id,
        "intervention_id": scenario.intervention_id,
        "scenario_id": scenario.scenario_id,
    }


def _scripted_model(model_id: str) -> bool:
    name = model_id.strip().lower()
    return name in {"fake", "scripted"} or name.startswith("fake")


def _public_text(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return "redacted"
    return text


def _unique(items: Sequence[str]) -> list[str]:
    found: list[str] = []
    for item in items:
        if item not in found:
            found.append(item)
    return found


def _package_version(name: str, fallback: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback


if __name__ == "__main__":
    raise SystemExit(main())
