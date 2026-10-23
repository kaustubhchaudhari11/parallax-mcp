"""Offline checks for the research runner. No provider call is made."""

import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from parallax.agent import ModelCompletion, ToolRequest
from parallax.experiment import (
    ExperimentManifest,
    build_manifest,
    eligibility_reasons,
    execute_experiment,
    experiment_directory,
    format_manifest,
    main,
    pilot_manifest,
    pilot_scenarios,
    scenario_by_id,
    schedule,
)
from parallax.metrics import (
    ArmSnapshot,
    ExperimentSummary,
    PairAccounting,
    TrialRecord,
    account_pair,
    format_summary,
    proportion,
    summarize,
    wilson_interval,
)
from parallax.openai_adapter import OpenAIAdapter, ProviderCallError
from parallax.replay import run_pair
from tests.harness import SCRIPTED_DECISION, ScriptedAdapter

PARAMS_MODEL = "gpt-6-luna"


def _drive(function: object) -> object:
    return anyio.run(function)  # type: ignore[arg-type]


def test_pilot_selects_the_first_scenario_of_each_family() -> None:
    first, second, third = pilot_scenarios()
    assert (first.scenario_id, first.family, first.fixture_id) == ("S01", "H1", "F000")
    assert (second.scenario_id, second.family, second.fixture_id) == ("S09", "H2", "F000")
    assert (third.scenario_id, third.family, third.fixture_id) == ("S17", "H3", "F000")
    manifest = pilot_manifest(
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )
    assert manifest.scenario_ids == ["S01", "S09", "S17"]
    assert manifest.hypotheses == ["H1", "H2", "H3"]
    assert manifest.fixture_ids == ["F000"]
    assert manifest.phase == "pilot"
    assert manifest.schedule == "mini-pilot"
    assert manifest.scenario_bindings == [
        {"family": "H1", "fixture_id": "F000", "intervention_id": "h1-latency", "scenario_id": "S01"},
        {
            "family": "H2",
            "fixture_id": "F000",
            "intervention_id": "h2-cross-error-rate",
            "scenario_id": "S09",
        },
        {
            "family": "H3",
            "fixture_id": "F000",
            "intervention_id": "h3-ignore-policy",
            "scenario_id": "S17",
        },
    ]
    assert any("mini-pilot" in note and "three fixtures" in note and "R=5" in note for note in manifest.notes)
    assert manifest.provider == "openai"
    assert manifest.enforced_pair_model_calls == 7
    assert manifest.worst_case_model_requests == 21
    assert manifest.token_budget is None
    assert manifest.prompt_hashes["F000"]
    assert manifest.fixture_hashes["F000"]
    again = pilot_manifest(
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )
    assert again.experiment_id == manifest.experiment_id
    assert experiment_directory(Path("."), manifest.experiment_id) == Path("runs/experiments") / manifest.experiment_id


def test_manifest_validation_rejects_a_bad_research_configuration() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        build_manifest(
            scenario_ids=["S99"],
            repeats=1,
            model_id=PARAMS_MODEL,
            model_call_budget=4,
            tool_call_budget=6,
            pair_model_call_budget=7,
        )
    with pytest.raises(ValueError, match="unique"):
        build_manifest(
            scenario_ids=["S01", "S01"],
            repeats=1,
            model_id=PARAMS_MODEL,
            model_call_budget=4,
            tool_call_budget=6,
            pair_model_call_budget=7,
        )
    with pytest.raises(ValueError, match="repeats"):
        build_manifest(
            scenario_ids=["S09"],
            repeats=0,
            model_id=PARAMS_MODEL,
            model_call_budget=4,
            tool_call_budget=6,
            pair_model_call_budget=7,
        )
    with pytest.raises(ValueError, match="phase"):
        build_manifest(
            scenario_ids=["S09"],
            repeats=1,
            model_id=PARAMS_MODEL,
            model_call_budget=4,
            tool_call_budget=6,
            pair_model_call_budget=7,
            phase="harness",
        )


def test_schedule_repeats_keep_trial_ids_stable() -> None:
    manifest = build_manifest(
        scenario_ids=["S01", "S09", "S17"],
        repeats=2,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )
    assert [slot.trial_id for slot in schedule(manifest)] == [
        "S01#0",
        "S01#1",
        "S09#0",
        "S09#1",
        "S17#0",
        "S17#1",
    ]
    assert manifest.worst_case_model_requests == 42


def test_dry_run_prints_the_plan_and_hides_the_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("parallax.openai_adapter.load_project_env", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-test-value")
    monkeypatch.setenv("PARALLAX_MODEL", PARAMS_MODEL)
    monkeypatch.setenv("PARALLAX_MAX_MODEL_CALLS", "4")
    monkeypatch.setenv("PARALLAX_MAX_TOOL_CALLS", "6")

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.openai_adapter.OpenAIAdapter.from_env", boom)
    code = main(["--dry-run", "--repeats", "1"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "dry-run: no model request will be sent" in printed
    assert "scenarios=S01,S09,S17" in printed
    assert "phase=pilot" in printed
    assert "schedule=mini-pilot" in printed
    assert "Protocol pilot: three fixtures, R=5." in printed
    assert "worst_case_model_requests=21" in printed
    assert "enforced_pair_model_calls=7" in printed
    assert "sk-live-test-value" not in printed
    assert "sk-" not in printed
    assert "oracle_decision" not in printed


def test_h3_only_dry_run_has_a_distinct_reproducible_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("parallax.openai_adapter.load_project_env", lambda: False)
    monkeypatch.setenv("PARALLAX_MODEL", PARAMS_MODEL)
    monkeypatch.setenv("PARALLAX_MAX_MODEL_CALLS", "4")
    monkeypatch.setenv("PARALLAX_MAX_TOOL_CALLS", "6")

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.openai_adapter.OpenAIAdapter.from_env", boom)
    code = main(["--dry-run", "--scenarios", "S17", "--repeats", "1"])
    printed = capsys.readouterr().out
    h3_manifest = build_manifest(
        scenario_ids=["S17"],
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )
    pilot = pilot_manifest(
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )

    assert code == 0
    assert f"experiment_id={h3_manifest.experiment_id}" in printed
    assert h3_manifest.experiment_id != pilot.experiment_id
    assert "schedule=custom" in printed
    assert "scenarios=S17" in printed
    assert "hypotheses=H3" in printed
    assert "slots=1" in printed
    assert "worst_case_model_requests=7" in printed


def test_live_cli_forwards_coroutine_keywords_offline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("parallax.openai_adapter.load_project_env", lambda: False)
    monkeypatch.setenv("PARALLAX_MODEL", PARAMS_MODEL)
    monkeypatch.setenv("PARALLAX_MAX_MODEL_CALLS", "4")
    monkeypatch.setenv("PARALLAX_MAX_TOOL_CALLS", "6")
    adapter = object()
    monkeypatch.setattr("parallax.openai_adapter.OpenAIAdapter.from_env", lambda: adapter)
    received: dict[str, object] = {}

    async def fake_execute(
        manifest: ExperimentManifest,
        output_dir: Path,
        selected_adapter: object,
        *,
        provider_executed: str,
        resume: bool,
    ) -> tuple[list[TrialRecord], ExperimentSummary]:
        received.update(
            output_dir=output_dir,
            adapter=selected_adapter,
            provider_executed=provider_executed,
            resume=resume,
        )
        trial = _trial(
            "H2",
            "S09",
            decision="INVESTIGATE",
            oracle="INVESTIGATE",
        ).model_copy(update={"experiment_id": manifest.experiment_id})
        return [trial], summarize(manifest.experiment_id, [trial])

    monkeypatch.setattr("parallax.experiment.execute_experiment", fake_execute)
    code = main(["--live", "--resume", "--repeats", "1", "--output", str(tmp_path)])

    assert code == 0
    assert received == {
        "output_dir": tmp_path,
        "adapter": adapter,
        "provider_executed": "openai",
        "resume": True,
    }
    assert f"output_dir={tmp_path}" in capsys.readouterr().out


def test_h3_provider_errors_are_structured_and_not_scored(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(
        scenario_ids=["S17"],
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )

    class _H3Failure:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools, params
            self.calls += 1
            if self.calls == 1:
                return ModelCompletion(
                    tool_calls=(
                        ToolRequest(
                            name="get_metrics",
                            arguments={"incident_id": "F000"},
                            call_id="call_metrics",
                        ),
                        ToolRequest(
                            name="get_logs",
                            arguments={"incident_id": "F000"},
                            call_id="call_logs",
                        ),
                        ToolRequest(
                            name="get_deployment",
                            arguments={"incident_id": "F000"},
                            call_id="call_deploy",
                        ),
                    ),
                    response_id="resp_pre",
                )
            if self.calls == 2:
                return ModelCompletion(content=SCRIPTED_DECISION, response_id="resp_baseline")
            raise ProviderCallError(
                stage="request",
                exception_type="BadRequestError",
                status_code=400,
                error_code="invalid_function_call_output",
                parameter="input",
                request_id=f"req_h3_{self.calls}",
            )

    adapter = _H3Failure()

    async def run() -> object:
        return await execute_experiment(
            manifest,
            tmp_path,
            adapter,
            provider_executed="openai",
            resume=False,
        )

    trials, summary = _drive(run)
    trial = trials[0]  # type: ignore[index]
    arms = {arm.arm: arm for arm in trial.arms}
    assert adapter.calls == 4
    assert arms["baseline"].execution_status == "completed"
    assert arms["factual_replay"].execution_status == "model_error"
    assert arms["mutant"].execution_status == "model_error"
    assert trial.empirical_eligible is False
    assert "arm_failure" in trial.ineligibility_reasons
    for name in ("factual_replay", "mutant"):
        metadata = arms[name].provider_error
        assert metadata is not None
        assert metadata.stage == "request"
        assert metadata.exception_class == "BadRequestError"
        assert metadata.http_status == 400
        assert metadata.error_code == "invalid_function_call_output"
        assert metadata.parameter == "input"
        assert metadata.request_id is not None
    rates = {row.name: (row.numerator, row.denominator) for row in summary.proportions}  # type: ignore[attr-defined]
    assert rates["h3_policy_violation"] == (0, 0)
    assert rates["h3_instruction_rejection"] == (0, 0)
    assert rates["pair_completion"] == (0, 1)
    assert rates["pair_validity"] == (0, 1)
    assert rates["parse_failure"] == (0, 3)
    assert "h3_policy_violation 0/0 NOT EVALUABLE" in format_summary(summary)  # type: ignore[arg-type]
    assert trial.accounting.deduped_tool_calls == 3
    assert trial.accounting.deduped_latency_ms is not None
    journal = (tmp_path / "trials.jsonl").read_text(encoding="utf-8")
    assert '"provider_error":{"stage":"request"' in journal
    assert "invalid_function_call_output" in journal
    assert "sk-" not in journal


def test_scripted_runner_is_resumable_and_not_empirical(tmp_path: Path) -> None:
    manifest = build_manifest(
        scenario_ids=["S09"],
        repeats=2,
        model_id="gpt-test",
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )
    seen = {"calls": 0}

    class _Count:
        def __init__(self) -> None:
            self.inner = ScriptedAdapter("F000", ("get_metrics",))

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            seen["calls"] += 1
            return await self.inner.complete(messages, tools, params)

    async def first() -> object:
        return await execute_experiment(
            manifest,
            tmp_path,
            _Count(),
            provider_executed="scripted",
            resume=False,
        )

    trials, summary = _drive(first)
    assert seen["calls"] == 8
    assert [trial.trial_id for trial in trials] == ["S09#0", "S09#1"]  # type: ignore[attr-defined]
    assert all(trial.empirical_eligible is False for trial in trials)  # type: ignore[attr-defined]
    assert "scripted_provider" in trials[0].ineligibility_reasons  # type: ignore[attr-defined]
    rates = {row.name: (row.numerator, row.denominator) for row in summary.proportions}  # type: ignore[attr-defined]
    assert rates["h2_adaptation"] == (0, 0)
    assert rates["pair_completion"] == (2, 2)
    assert rates["pair_validity"] == (0, 2)
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "trials.jsonl").is_file()
    assert (tmp_path / "traces.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    stored = (tmp_path / "traces.jsonl").read_text(encoding="utf-8")
    assert f'"experiment_id":"{manifest.experiment_id}"' in stored

    async def second() -> object:
        return await execute_experiment(
            manifest,
            tmp_path,
            _Count(),
            provider_executed="scripted",
            resume=True,
        )

    resumed, _summary = _drive(second)
    assert seen["calls"] == 8
    assert [trial.trial_id for trial in resumed] == ["S09#0", "S09#1"]  # type: ignore[attr-defined]

    async def blocked() -> object:
        return await execute_experiment(
            manifest,
            tmp_path,
            _Count(),
            provider_executed="scripted",
            resume=False,
        )

    with pytest.raises(ValueError, match="resume"):
        _drive(blocked)
    assert seen["calls"] == 8


def test_pair_budget_stops_before_another_provider_request(tmp_path: Path) -> None:
    manifest = build_manifest(
        scenario_ids=["S09"],
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=1,
    )
    assert manifest.enforced_pair_model_calls == 1
    assert manifest.worst_case_model_requests == 1
    client = _Client(_Hungry())
    adapter = OpenAIAdapter(client, model=PARAMS_MODEL)

    async def main() -> object:
        return await execute_experiment(
            manifest,
            tmp_path,
            adapter,
            provider_executed="scripted",
            resume=False,
        )

    trials, _summary = _drive(main)
    assert client.responses.calls == 1
    assert trials[0].failure_reasons  # type: ignore[attr-defined]
    assert trials[0].accounting.unique_model_calls == 1  # type: ignore[attr-defined]
    assert "budget_exhausted" in " ".join(trials[0].failure_reasons)  # type: ignore[attr-defined]


def test_provider_calls_are_deduplicated_and_missing_usage_stays_null() -> None:
    billed = _Client(_Responses(usage=True))
    billed_adapter = OpenAIAdapter(billed, model=PARAMS_MODEL)
    manifest = _s09_manifest()

    async def billed_run() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=billed_adapter,
            model_id=PARAMS_MODEL,
            model_params=manifest.model_params,
            step_budget=4,
            token_budget=None,
            phase="pilot",
            scenario_id="S09",
            repeat_index=0,
            tool_call_budget=6,
            model_call_limit=7,
        )

    run = _drive(billed_run)
    accounting = account_pair(run)  # type: ignore[arg-type]
    assert accounting.unique_model_calls == 4
    assert accounting.cumulative_model_calls == 6
    assert accounting.shared_prefix_model_calls == 1
    assert accounting.incremental_model_calls == 3
    assert accounting.deduped_usage == {"input_tokens": 12, "output_tokens": 8}
    assert accounting.cumulative_usage == {"input_tokens": 18, "output_tokens": 12}
    assert accounting.deduped_tool_calls == 1
    assert accounting.cumulative_tool_calls == 3
    assert accounting.deduped_latency_ms is not None
    assert accounting.cumulative_latency_ms is not None
    assert accounting.cumulative_latency_ms > accounting.deduped_latency_ms
    reasons = eligibility_reasons(
        run,  # type: ignore[arg-type]
        manifest=manifest,
        scenario=pilot_scenarios()[1],
        provider_executed="scripted",
    )
    assert "shared_post_fork_response" not in reasons
    assert "fork_response_not_reused" not in reasons
    assert "oracle_label_in_transcript" not in reasons
    assert "scripted_provider" in reasons
    history = billed.responses.history
    assert history[2]["previous_response_id"] == "resp_pre"
    assert history[3]["previous_response_id"] == "resp_pre"
    assert "resp_post_2" not in str(history[3]["input"])
    assert "resp_post_4" not in str(history[2]["input"])

    gap = _Client(_Responses(usage=False))
    gap_adapter = OpenAIAdapter(gap, model=PARAMS_MODEL)

    async def gap_run() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=gap_adapter,
            model_id=PARAMS_MODEL,
            model_params=manifest.model_params,
            step_budget=4,
            token_budget=None,
            phase="pilot",
            scenario_id="S09",
            repeat_index=0,
            model_call_limit=7,
        )

    gap_accounting = account_pair(_drive(gap_run))  # type: ignore[arg-type]
    assert gap_accounting.unique_model_calls == 4
    assert gap_accounting.deduped_usage is None
    assert gap_accounting.cumulative_usage is None


def test_invalid_trial_and_secret_text_are_retained_without_the_key(tmp_path: Path) -> None:
    manifest = _s09_manifest()

    class _Rejected:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools, params
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="remediate_now", arguments={"incident_id": "F000"}),),
            )

    async def rejected() -> object:
        return await execute_experiment(
            manifest,
            tmp_path / "invalid",
            _Rejected(),
            provider_executed="scripted",
            resume=False,
        )

    trials, summary = _drive(rejected)
    assert trials[0].empirical_eligible is False  # type: ignore[attr-defined]
    text = " ".join(trials[0].failure_reasons)  # type: ignore[attr-defined]
    assert "invalid_tool" in text
    assert "intervention_unobserved" in text
    assert summary.failure_breakdown["invalid_tool"] == 1  # type: ignore[attr-defined]

    class _Secret:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools, params
            raise RuntimeError("bad sk-live-test-value body")

    async def secret() -> object:
        return await execute_experiment(
            manifest,
            tmp_path / "secret",
            _Secret(),
            provider_executed="scripted",
            resume=False,
        )

    secret_trials, secret_summary = _drive(secret)
    journal = (tmp_path / "secret" / "trials.jsonl").read_text(encoding="utf-8")
    traces = (tmp_path / "secret" / "traces.jsonl").read_text(encoding="utf-8")
    printed = format_manifest(manifest) + secret_summary.model_dump_json()  # type: ignore[attr-defined]
    assert "sk-live-test-value" not in journal
    assert "sk-live-test-value" not in traces
    assert "sk-" not in journal
    assert "sk-" not in traces
    assert "sk-" not in printed
    assert "redacted" in journal
    assert secret_trials[0].failure_reasons  # type: ignore[attr-defined]


def test_metric_denominators_keep_parse_failures_and_drop_scripted_rows() -> None:
    empty = wilson_interval(0, 0)
    assert empty is None
    assert proportion("empty", 0, 0).value is None
    h1_match = _trial("H1", "S01", decision="MONITOR", oracle="MONITOR")
    h1_miss = _trial("H1", "S01", decision="ESCALATE", oracle="MONITOR", repeat=1)
    h1_unparsed = _trial("H1", "S01", decision=None, oracle="MONITOR", repeat=2, parse_ok=False)
    h2_match = _trial("H2", "S09", decision="INVESTIGATE", oracle="INVESTIGATE")
    h3_clean = _trial("H3", "S17", decision="MONITOR", oracle="MONITOR")
    h3_dirty = _trial(
        "H3",
        "S17",
        decision="ESCALATE",
        oracle="MONITOR",
        repeat=1,
        action="remediate",
        rejected=True,
    )
    provider_failure = _trial(
        "H2",
        "S09",
        decision=None,
        oracle="INVESTIGATE",
        repeat=3,
        parse_ok=False,
        status="model_error",
        flags=["provider_error"],
    )
    harness = _trial("H1", "S01", decision="MONITOR", oracle="MONITOR", repeat=4, phase="harness")
    scripted = _trial(
        "H2",
        "S09",
        decision="INVESTIGATE",
        oracle="INVESTIGATE",
        repeat=4,
        provider="scripted",
    )
    summary = summarize(
        "exp-fixture",
        [h1_match, h1_miss, h1_unparsed, h2_match, h3_clean, h3_dirty, provider_failure, harness, scripted],
    )
    rates = {row.name: (row.numerator, row.denominator) for row in summary.proportions}
    assert rates["h1_invariance"] == (1, 3)
    assert rates["h1_invariance_parsed_observed"] == (1, 2)
    assert rates["h2_adaptation"] == (1, 1)
    assert rates["h3_instruction_rejection"] == (1, 2)
    assert rates["h3_policy_violation"] == (1, 2)
    assert rates["h3_decision_contamination"] == (1, 2)
    assert rates["h3_instruction_adoption"] == (1, 2)
    assert rates["baseline_oracle_match"] == (3, 6)
    assert rates["baseline_stability"] == (3, 5)
    assert summary.discordant_pairs["S01"]["factual_match_mutant_miss"] == 1
    assert summary.usage_deduped == {"input_tokens": 7, "output_tokens": 7}
    assert summary.model_calls_unique == 7
    assert summary.model_calls_cumulative == 14
    assert summary.model_calls_shared_prefix == 7
    printed = str(summary.model_dump())
    assert "sk-" not in printed
    assert "oracle_decision" not in printed


def test_identity_tracks_the_scenario_binding_and_provider_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = {
        "scenario_ids": ["S01", "S09", "S17"],
        "repeats": 1,
        "model_id": PARAMS_MODEL,
        "model_call_budget": 4,
        "tool_call_budget": 6,
        "pair_model_call_budget": 7,
    }
    base = build_manifest(**kwargs)
    assert base.schedule == "mini-pilot"
    repeated = build_manifest(**{**kwargs, "repeats": 2})
    assert repeated.experiment_id != base.experiment_id
    single = build_manifest(**{**kwargs, "scenario_ids": ["S09"]})
    assert single.schedule == "custom"
    assert "mini-pilot" not in " ".join(single.notes)

    def swapped(scenario_id: str) -> object:
        row = scenario_by_id(scenario_id)
        if scenario_id == "S01":
            return row._replace(intervention_id="h1-rollback-flag")
        return row

    monkeypatch.setattr("parallax.experiment.scenario_by_id", swapped)
    rebound = build_manifest(**kwargs)
    assert rebound.experiment_id != base.experiment_id
    assert rebound.scenario_bindings[0]["intervention_id"] == "h1-rollback-flag"

    def other_sdk(name: str, fallback: str) -> str:
        if name == "openai":
            return "9.9.9-audit"
        return fallback

    monkeypatch.setattr("parallax.experiment._package_version", other_sdk)
    shifted = build_manifest(**kwargs)
    assert shifted.provider_sdk_version == "9.9.9-audit"
    assert shifted.experiment_id != rebound.experiment_id


def test_forced_eligibility_flag_cannot_admit_a_scripted_row() -> None:
    scripted = _trial(
        "H2",
        "S09",
        decision="INVESTIGATE",
        oracle="INVESTIGATE",
        provider="scripted",
    ).model_copy(update={"empirical_eligible": True, "ineligibility_reasons": []})
    fake = _trial("H1", "S01", decision="MONITOR", oracle="MONITOR").model_copy(
        update={
            "empirical_eligible": True,
            "arms": [arm.model_copy(update={"model_id": "fake-lab"}) for arm in scripted.arms],
        }
    )
    summary = summarize("exp-fixture", [scripted, fake])
    rates = {row.name: (row.numerator, row.denominator) for row in summary.proportions}
    assert rates["h1_invariance"] == (0, 0)
    assert rates["h2_adaptation"] == (0, 0)
    assert rates["pair_validity"] == (0, 2)
    assert rates["pair_completion"] == (2, 2)
    assert summary.model_calls_unique == 0
    assert summary.usage_deduped is None
    assert summary.latency_ms_deduped is None


def test_mini_pilot_edits_are_visible_hashed_and_keep_the_policy() -> None:
    from parallax.capture import sha256_text
    from parallax.mutate import H3_IGNORE_POLICY, INTERVENTIONS
    from parallax.prompt import POLICY

    runs = {scenario.scenario_id: _drive_pair(scenario) for scenario in pilot_scenarios()}
    h1 = runs["S01"]
    h2 = runs["S09"]
    h3 = runs["S17"]
    assert h1.factual is not None and h1.mutant.observation_mutated is not None
    assert h1.factual.observation_original != h1.mutant.observation_mutated
    assert h1.factual.oracle_decision == h1.mutant.oracle_decision == "MONITOR"
    assert (h1.factual.predicate_m, h1.factual.predicate_l, h1.factual.predicate_d) == (
        h1.mutant.predicate_m,
        h1.mutant.predicate_l,
        h1.mutant.predicate_d,
    )
    original_metrics = json.loads(h1.factual.observation_original or "")
    mutated_metrics = json.loads(h1.mutant.observation_mutated)
    assert original_metrics["latency_p99_ms"] != mutated_metrics["latency_p99_ms"]
    assert original_metrics["error_rate_bps"] == mutated_metrics["error_rate_bps"]

    assert h2.factual is not None and h2.mutant.observation_mutated is not None
    assert h2.factual.predicate_m is False and h2.mutant.predicate_m is True
    assert h2.factual.predicate_l == h2.mutant.predicate_l
    assert h2.factual.predicate_d == h2.mutant.predicate_d
    assert h2.factual.oracle_decision == "MONITOR"
    assert h2.mutant.oracle_decision == "INVESTIGATE"
    before = json.loads(h2.factual.observation_original or "")
    after = json.loads(h2.mutant.observation_mutated)
    assert before["error_rate_bps"] == 499
    assert after["error_rate_bps"] == 500

    assert h3.factual is not None and h3.mutant.observation_mutated is not None
    assert H3_IGNORE_POLICY not in (h3.factual.observation_original or "")
    assert H3_IGNORE_POLICY in h3.mutant.observation_mutated
    assert h3.factual.oracle_decision == h3.mutant.oracle_decision == "MONITOR"
    assert h3.mutant.observation_mutated in h3.mutant.model_visible_observations
    for run in (h1, h2, h3):
        assert run.factual is not None and run.factual_messages is not None
        assert run.mutant_messages is not None
        assert run.baseline_messages[0].content == POLICY
        assert run.factual_messages[0].content == POLICY
        assert run.mutant_messages[0].content == POLICY
        assert H3_IGNORE_POLICY not in POLICY
        factual_hash = sha256_text(run.factual.observation_original or "")
        mutant_hash = sha256_text(run.mutant.observation_mutated or "")
        assert factual_hash != mutant_hash
        assert factual_hash == sha256_text(run.mutant.observation_original or "")
        assert factual_hash in {call.model_visible_sha256 for call in run.factual.tool_calls}
        assert mutant_hash in {call.model_visible_sha256 for call in run.mutant.tool_calls}
        visible = "\n".join(message.content for message in run.mutant_messages)
        assert "oracle_decision" not in visible
        assert "expected_decision" not in visible
        assert "declared_oracle" not in visible
        tool = INTERVENTIONS[run.mutant.intervention_id or ""].tool_name
        assert any(call.name == tool and call.source == "overlay" for call in run.mutant.tool_calls)
        assert any(call.name == tool and call.source == "base" for call in run.factual.tool_calls)


def _drive_pair(scenario: object) -> object:
    from parallax.live_pair import pair_model_params
    from parallax.mutate import INTERVENTIONS

    tool = INTERVENTIONS[scenario.intervention_id].tool_name  # type: ignore[attr-defined]

    async def run() -> object:
        return await run_pair(
            incident_id=scenario.fixture_id,  # type: ignore[attr-defined]
            intervention_id=scenario.intervention_id,  # type: ignore[attr-defined]
            adapter=ScriptedAdapter(scenario.fixture_id, (tool,)),  # type: ignore[attr-defined]
            model_id="gpt-test",
            model_params=pair_model_params("gpt-test"),
            step_budget=4,
            token_budget=None,
            phase="pilot",
            scenario_id=scenario.scenario_id,  # type: ignore[attr-defined]
            repeat_index=0,
            tool_call_budget=6,
            model_call_limit=7,
        )

    return _drive(run)


def _s09_manifest() -> object:
    return build_manifest(
        scenario_ids=["S09"],
        repeats=1,
        model_id=PARAMS_MODEL,
        model_call_budget=4,
        tool_call_budget=6,
        pair_model_call_budget=7,
    )


def _trial(
    family: str,
    scenario_id: str,
    *,
    decision: str | None,
    oracle: str,
    repeat: int = 0,
    parse_ok: bool = True,
    status: str = "completed",
    flags: list[str] | None = None,
    action: str = "none",
    rejected: bool = False,
    phase: str = "pilot",
    provider: str = "openai",
) -> TrialRecord:
    """Ledger fixture for the summary function. It is not a model measurement."""

    def arm(name: str, parsed: str | None, label: str) -> ArmSnapshot:
        return ArmSnapshot(
            run_id=f"{scenario_id}-{name}-{repeat}",
            arm=name,  # type: ignore[arg-type]
            execution_status=status,  # type: ignore[arg-type]
            decision_parsed=parsed,
            action_observed=action,
            parse_ok=parse_ok,
            oracle_decision=label,
            flags=list(flags or []),
            error=None,
            usage={"input_tokens": 2, "output_tokens": 2},
            provider_response_ids=[f"{scenario_id}-{repeat}-pre", f"{scenario_id}-{repeat}-{name}"],
            tool_call_count=1,
            rejected_tool=rejected,
            prefix_sha256="prefix",
            model_id=PARAMS_MODEL,
            phase=phase,  # type: ignore[arg-type]
            incident_id="F000",
        )

    factual_decision = "MONITOR" if family == "H1" and decision == "ESCALATE" else decision
    accounting = PairAccounting(
        unique_model_calls=1,
        cumulative_model_calls=2,
        shared_prefix_model_calls=1,
        incremental_model_calls=0,
        deduped_usage={"input_tokens": 1, "output_tokens": 1},
        cumulative_usage={"input_tokens": 2, "output_tokens": 2},
        deduped_tool_calls=1,
        cumulative_tool_calls=2,
        deduped_latency_ms=1.0,
        cumulative_latency_ms=2.0,
        unique_response_ids=[f"{scenario_id}-{repeat}-pre"],
        shared_prefix_response_ids=[f"{scenario_id}-{repeat}-pre"],
        incremental_response_ids=[],
    )
    return TrialRecord(
        schema_version=1,
        experiment_id="exp-fixture",
        trial_id=f"{scenario_id}#{repeat}",
        scenario_id=scenario_id,
        fixture_id="F000",
        family=family,
        repeat_index=repeat,
        phase=phase,  # type: ignore[arg-type]
        provider_executed=provider,  # type: ignore[arg-type]
        pair_id=f"pair-{scenario_id}-{repeat}",
        empirical_eligible=provider == "openai" and phase == "pilot" and status == "completed",
        ineligibility_reasons=[] if provider == "openai" and phase == "pilot" else ["scripted_or_harness"],
        failure_reasons=[] if status == "completed" else [status],
        arms=[
            arm("baseline", factual_decision if factual_decision else None, "MONITOR"),
            arm("factual_replay", factual_decision if factual_decision else None, "MONITOR"),
            arm("mutant", decision, oracle),
        ],
        accounting=accounting,
    )


class _Responses:
    def __init__(self, *, usage: bool) -> None:
        self.history: list[dict[str, object]] = []
        self.calls = 0
        self.usage = usage

    async def create(self, **kwargs: object) -> object:
        self.history.append(kwargs)
        self.calls += 1
        if self.calls == 1:
            return _tool_response("resp_pre", "call_metrics")
        reported: object
        if self.usage or self.calls != 3:
            reported = SimpleNamespace(input_tokens=3, output_tokens=2)
        else:
            reported = None
        return _decision_response(f"resp_post_{self.calls}", reported)


class _Hungry:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        return _tool_response(f"resp_{self.calls}", f"call_{self.calls}")


class _Client:
    def __init__(self, responses: object) -> None:
        self.responses = responses


def _tool_response(response_id: str, call_id: str) -> SimpleNamespace:
    call = SimpleNamespace(
        type="function_call",
        name="get_metrics",
        arguments='{"incident_id":"F000"}',
        call_id=call_id,
        id=f"fc_{call_id}",
        status="completed",
    )
    return SimpleNamespace(
        id=response_id,
        output=[call],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        error=None,
        status="completed",
    )


def _decision_response(response_id: str, usage: object) -> SimpleNamespace:
    message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=SCRIPTED_DECISION)],
    )
    return SimpleNamespace(
        id=response_id,
        output=[message],
        usage=usage,
        error=None,
        status="completed",
    )
