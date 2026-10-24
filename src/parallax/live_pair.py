"""One opt-in counterfactual pair.

The default is a dry run. ``--live`` is required before a provider
client exists. The pair reuses ``run_pair`` and stays ``phase=harness``.
It is not a research trial and it is not the scenario scheduler.

Branches share the copied pre-fork transcript. The OpenAI adapter reads
``previous_response_id`` from that transcript and does not keep a cursor.
Post-fork response ids stay on the arm that received them.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

from parallax.capture import sha256_text
from parallax.live import configured_budget, os_environ, smoke_model_params
from parallax.openai_adapter import OpenAIAdapter, load_project_env
from parallax.replay import PairRun, run_pair
from parallax.scenarios import SCENARIOS, Scenario
from parallax.trace import TraceRecord, research_eligible

_DEFAULT_MODEL_CALLS = 4
_DEFAULT_TOOL_CALLS = 6
_CONTINUATION = "previous_response_id"


def first_h2_scenario() -> Scenario:
    """Return the first declared H2 row. The id is not invented here."""

    for scenario in SCENARIOS:
        if scenario.family == "H2":
            return scenario
    raise ValueError("no H2 scenario is declared")


def resolve_h2_scenario(scenario_id: str | None) -> Scenario:
    """Select one declared H2 scenario. Other families are rejected."""

    if scenario_id is None or not scenario_id.strip():
        return first_h2_scenario()
    selected = scenario_id.strip()
    for scenario in SCENARIOS:
        if scenario.scenario_id == selected:
            if scenario.family != "H2":
                raise ValueError(f"scenario is not H2: {selected}")
            return scenario
    raise ValueError(f"unknown scenario: {selected}")


def full_pair_request_ceiling(per_branch: int) -> int:
    """Most model calls one pair can make.

    The baseline may use the whole per-branch budget. The checkpoint
    copy keeps ``steps_used``, so after a fork on the first call each
    replay has ``per_branch - 1`` calls left. A later fork leaves less.
    """

    if per_branch < 1:
        return 0
    return per_branch + 2 * (per_branch - 1)


def max_model_requests(per_branch: int, total: int) -> int:
    """Smaller of the requested total and the earliest-fork ceiling.

    The total cap can only make this smaller. It does not raise the
    per-branch ceiling.
    """

    if per_branch < 1 or total < 1:
        return 0
    return min(total, full_pair_request_ceiling(per_branch))


def pair_model_params(model: str) -> dict[str, Any]:
    """Decoding parameters plus the continuation record.

    ``provider_continuation`` is logged and omitted from the API request.
    """

    params = smoke_model_params(model)
    params["provider_continuation"] = _CONTINUATION
    return params


def format_pair_dry_run(
    *,
    scenario: Scenario,
    environ: Mapping[str, str],
    model_calls: int,
    tool_calls: int,
    pair_model_calls: int,
) -> str:
    """Describe one pair. The key value is not included."""

    key_state = "set" if environ.get("OPENAI_API_KEY", "").strip() else "missing"
    model = environ.get("PARALLAX_MODEL", "").strip() or "(unset)"
    lines = [
        "dry-run: no model request will be sent",
        "phase=harness",
        "research_trial=false",
        f"scenario_id={scenario.scenario_id}",
        f"family={scenario.family}",
        f"intervention_id={scenario.intervention_id}",
        f"incident_id={scenario.fixture_id}",
        f"declared_oracle_after={scenario.oracle_after}",
        f"model={model}",
        f"OPENAI_API_KEY={key_state}",
        f"model_call_budget={model_calls}",
        f"tool_call_budget={tool_calls}",
        f"pair_model_call_budget={pair_model_calls}",
        "max_model_requests=" + str(max_model_requests(model_calls, pair_model_calls)),
        f"provider_continuation={_CONTINUATION}",
        "tools=get_metrics,get_logs,get_deployment",
    ]
    return "\n".join(lines) + "\n"


def format_pair_summary(run: PairRun) -> str:
    """Operator ledger. Observation bodies are omitted."""

    baseline = run.baseline
    lines = [
        f"phase={baseline.phase}",
        "research_trial=" + str(research_eligible(baseline)).lower(),
        f"scenario_id={baseline.scenario_id}",
        f"incident_id={baseline.incident_id}",
        f"intervention_id={run.mutant.intervention_id}",
        f"model_id={baseline.model_id}",
        "model_params=" + _params_text(baseline.model_params),
        f"pair_id={baseline.pair_id}",
        f"prefix_sha256={run.prefix_sha256}",
        "provider_continuation=" + str(baseline.model_params.get("provider_continuation", "none")),
    ]
    for row in (run.baseline, run.factual, run.mutant):
        if row is not None:
            lines.extend(_arm_lines(row))
    return "\n".join(lines) + "\n"


async def run_live_pair(
    scenario: Scenario,
    model_calls: int,
    tool_calls: int,
    pair_model_calls: int,
    trace_path: Path | None,
) -> int:
    """Run one harness pair. The caller has already required ``--live``."""

    try:
        adapter = OpenAIAdapter.from_env()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        run = await run_pair(
            incident_id=scenario.fixture_id,
            intervention_id=scenario.intervention_id,
            adapter=adapter,
            model_id=adapter.model,
            model_params=pair_model_params(adapter.model),
            step_budget=model_calls,
            token_budget=None,
            phase="harness",
            scenario_id=scenario.scenario_id,
            repeat_index=0,
            trace_path=trace_path,
            tool_call_budget=tool_calls,
            model_call_limit=pair_model_calls,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(format_pair_summary(run))
    if trace_path is not None:
        print(f"trace_path={trace_path}")
    rows = [run.baseline, run.factual, run.mutant]
    if any(row is None or row.execution_status != "completed" for row in rows):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Without ``--live`` this only prints the plan."""

    parser = argparse.ArgumentParser(prog="python -m parallax.live_pair")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model-calls", type=int, default=None)
    parser.add_argument("--tool-calls", type=int, default=None)
    parser.add_argument("--pair-model-calls", type=int, default=None)
    parser.add_argument("--trace", type=Path, default=None)
    args = parser.parse_args(None if argv is None else list(argv))
    if args.live and args.dry_run:
        print("choose either --dry-run or --live", file=sys.stderr)
        return 2
    try:
        scenario = resolve_h2_scenario(args.scenario)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.trace is not None and args.trace.suffix.lower() != ".jsonl":
        print("trace path must end in .jsonl", file=sys.stderr)
        return 2
    load_project_env()
    environ = dict(os_environ())
    try:
        model_calls = configured_budget(
            args.model_calls, environ, "PARALLAX_MAX_MODEL_CALLS", _DEFAULT_MODEL_CALLS
        )
        tool_calls = configured_budget(
            args.tool_calls, environ, "PARALLAX_MAX_TOOL_CALLS", _DEFAULT_TOOL_CALLS
        )
        pair_calls = _pair_budget(args.pair_model_calls, model_calls)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.live:
        sys.stdout.write(
            format_pair_dry_run(
                scenario=scenario,
                environ=environ,
                model_calls=model_calls,
                tool_calls=tool_calls,
                pair_model_calls=pair_calls,
            )
        )
        return 0
    return anyio.run(
        run_live_pair,
        scenario,
        model_calls,
        tool_calls,
        pair_calls,
        args.trace,
    )


def _pair_budget(explicit: int | None, per_branch: int) -> int:
    if explicit is None:
        return max_model_requests(per_branch, 2 * per_branch - 1)
    if explicit < 1:
        raise ValueError("pair model-call budget must be an integer of at least 1")
    return explicit


def _params_text(params: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(params):
        value = params[key]
        if not isinstance(key, str) or not key.isidentifier():
            continue
        text_ok = isinstance(value, str) and value.isidentifier()
        number_ok = isinstance(value, int) and not isinstance(value, bool)
        if text_ok or number_ok:
            parts.append(f"{key}={value}")
    return ",".join(parts) if parts else "(none)"


def _arm_lines(row: TraceRecord) -> list[str]:
    arm = row.arm
    usage = "null" if row.usage is None else str(row.usage)
    names = ",".join(f"{call.name}:{call.source}" for call in row.tool_calls) or "(none)"
    errors = [call.name for call in row.tool_calls if call.is_error]
    latency = sum(call.latency_ms or 0 for call in row.tool_calls)
    lines = [
        f"arm={arm}",
        f"{arm}_run_id={row.run_id}",
        f"{arm}_execution_status={row.execution_status}",
        f"{arm}_decision={row.decision_parsed}",
        f"{arm}_oracle_decision={row.oracle_decision}",
        f"{arm}_provider_response_ids=" + ",".join(row.provider_response_ids),
        f"{arm}_usage={usage}",
        f"{arm}_tool_calls={names}",
        f"{arm}_tool_errors=" + (",".join(errors) if errors else "(none)"),
        f"{arm}_tool_latency_ms={latency}",
        f"{arm}_started_at={row.started_at}",
        f"{arm}_ended_at={row.ended_at}",
    ]
    if row.observation_original is not None:
        lines.append(f"{arm}_observation_original_sha256={sha256_text(row.observation_original)}")
    if row.observation_mutated is not None:
        lines.append(f"{arm}_observation_mutated_sha256={sha256_text(row.observation_mutated)}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
