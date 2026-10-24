"""Offline Decision X-Ray checks. No provider call is made."""

import json
from pathlib import Path

import anyio
import pytest

from parallax.agent import ModelCompletion, ModelUsage, ToolRequest
from parallax.capture import sha256_text
from parallax.fixtures import get_fixture
from parallax.observe import canonical_json
from parallax.schemas import ServiceMetrics
from parallax.xray import (
    DEFAULT_CANDIDATES,
    SearchRow,
    XRayResult,
    build_spec,
    classify_outcome,
    classify_search,
    edit_evidence,
    execute_xray,
    field_distance,
    format_dry_run,
    main,
    order_candidates,
    plan_search,
    request_ceiling,
)
from tests.harness import SCRIPTED_DECISION, EvidenceAdapter

PARAMS = {"temperature": 0}


def _drive(function: object) -> object:
    return anyio.run(function)  # type: ignore[arg-type]


def _spec(candidates: list[int] | None = None, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "fixture_id": "F100",
        "tool": "get_metrics",
        "field": "error_rate_bps",
        "candidates": candidates,
        "target_decision": None,
        "model_id": "gpt-test",
        "model_params": PARAMS,
        "model_call_budget": 4,
        "tool_call_budget": 6,
    }
    arguments.update(overrides)
    return build_spec(**arguments)  # type: ignore[arg-type]


def test_candidates_are_ordered_by_distance_without_the_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("oracle called")

    monkeypatch.setattr("parallax.oracle.decide", boom)
    assert field_distance(500, 495) == 5
    assert order_candidates(500, [505, 499, 498, 501]) == [499, 501, 498, 505]
    assert order_candidates(500, [502, 498]) == [498, 502]
    assert classify_outcome(
        factual_status="completed",
        factual_parsed=True,
        factual_decision="INVESTIGATE",
        mutant_status="completed",
        mutant_parsed=True,
        mutant_decision="MONITOR",
        intervention_applied=True,
        model_visible=True,
        factual_flags=(),
        mutant_flags=(),
    ) == "verified_flip"
    assert classify_outcome(
        factual_status="completed",
        factual_parsed=True,
        factual_decision="INVESTIGATE",
        mutant_status="completed",
        mutant_parsed=True,
        mutant_decision="INVESTIGATE",
        intervention_applied=True,
        model_visible=True,
        factual_flags=(),
        mutant_flags=(),
    ) == "no_flip"
    assert classify_outcome(
        factual_status="completed",
        factual_parsed=True,
        factual_decision="INVESTIGATE",
        mutant_status="invalid_final",
        mutant_parsed=False,
        mutant_decision=None,
        intervention_applied=True,
        model_visible=True,
        factual_flags=(),
        mutant_flags=("parse_failure",),
    ) == "invalid_output"
    assert classify_outcome(
        factual_status="completed",
        factual_parsed=True,
        factual_decision="INVESTIGATE",
        mutant_status="model_error",
        mutant_parsed=False,
        mutant_decision=None,
        intervention_applied=False,
        model_visible=False,
        factual_flags=(),
        mutant_flags=("provider_error",),
    ) == "provider_error"
    assert classify_outcome(
        factual_status="completed",
        factual_parsed=True,
        factual_decision="INVESTIGATE",
        mutant_status="budget_exhausted",
        mutant_parsed=False,
        mutant_decision=None,
        intervention_applied=True,
        model_visible=True,
        factual_flags=(),
        mutant_flags=("budget_exhausted",),
    ) == "budget_exhausted"
    assert classify_outcome(
        factual_status="completed",
        factual_parsed=True,
        factual_decision="INVESTIGATE",
        mutant_status="completed",
        mutant_parsed=True,
        mutant_decision="MONITOR",
        intervention_applied=False,
        model_visible=False,
        factual_flags=(),
        mutant_flags=(),
    ) == "intervention_missing"
    spec = _spec()
    assert spec.original_value == 500  # type: ignore[attr-defined]
    assert spec.candidates == list(order_candidates(500, DEFAULT_CANDIDATES))  # type: ignore[attr-defined]
    assert spec.worst_case_model_requests == request_ceiling(4, 6) == 22  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="integer field"):
        _spec(field="window")
    with pytest.raises(ValueError, match="not valid"):
        _spec([-1])
    with pytest.raises(ValueError, match="original value"):
        _spec([500])
    with pytest.raises(ValueError, match="unique"):
        _spec([499, 499])
    with pytest.raises(ValueError, match="maximum"):
        _spec([1, 2, 3, 4, 5, 6, 7], max_candidates=6)
    with pytest.raises(ValueError, match="unknown incident"):
        _spec(fixture_id="missing")
    with pytest.raises(ValueError, match="nested"):
        _spec(field="window.start")
    with pytest.raises(ValueError, match="searchable"):
        _spec(field="request_count")


def test_dry_run_prints_the_plan_and_does_not_construct_a_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("parallax.openai_adapter.load_project_env", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-test-value")
    monkeypatch.setenv("PARALLAX_MODEL", "gpt-test")

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.openai_adapter.OpenAIAdapter.from_env", boom)
    assert main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "dry-run: no model request will be sent" in printed
    assert "fixture=F100" in printed
    assert "tool=get_metrics" in printed
    assert "field=error_rate_bps" in printed
    assert "original_value=500" in printed
    assert "candidate value=499 distance=1" in printed
    assert "candidate value=501 distance=1" in printed
    assert "worst_case_model_requests=22" in printed
    assert "strategy=early_stop" in printed
    assert "git_revision=" in printed
    assert "shared_checkpoint=true" in printed
    assert "oracle=post-hoc" in printed
    assert "sk-live-test-value" not in printed
    assert "sk-" not in printed
    assert not (tmp_path / "runs").exists()
    assert main(["--dry-run", "--live"]) == 2
    assert main(["--field", "window"]) == 2
    monkeypatch.delenv("PARALLAX_MODEL")
    assert main(["--live"]) == 2


def test_live_cli_forwards_resume_through_anyio(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The real --live entry must call execute_xray. anyio.run does not accept resume=."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("parallax.openai_adapter.load_project_env", lambda: False)
    monkeypatch.setenv("PARALLAX_MODEL", "gpt-test")
    sentinel = object()
    seen: list[bool] = []

    def from_env() -> object:
        return sentinel

    async def execute(spec: object, output_dir: Path, adapter: object, *, resume: bool) -> XRayResult:
        del output_dir
        assert adapter is sentinel
        seen.append(resume)
        return XRayResult(
            schema_version=spec.schema_version,  # type: ignore[attr-defined]
            xray_id=spec.xray_id,  # type: ignore[attr-defined]
            original_decision=None,
            termination_reason="mocked",
            minimality_definition=spec.minimality_definition,  # type: ignore[attr-defined]
            minimality_claim="no verified flip",
            minimal_edit=None,
            proof_blockers=[],
            shared_checkpoint=False,
            checkpoint_id=None,
            candidates=[],
            provider_requests=0,
            usage_deduped=None,
            target_decision=None,
            target_reached=False,
            notes=[],
        )

    monkeypatch.setattr("parallax.openai_adapter.OpenAIAdapter.from_env", from_env)
    monkeypatch.setattr("parallax.xray.execute_xray", execute)
    output = tmp_path / "xray-live"
    argv = [
        "--live",
        "--fixture",
        "F100",
        "--tool",
        "get_metrics",
        "--field",
        "error_rate_bps",
        "--candidates",
        "499",
        "501",
        "--strategy",
        "exhaustive",
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    assert main([*argv, "--resume"]) == 0
    assert seen == [False, True]
    printed = capsys.readouterr().out
    assert "termination_reason=mocked" in printed
    assert "sk-" not in printed
    assert not (tmp_path / "runs").exists()


def test_verified_flip_stops_at_the_nearest_tested_candidate(tmp_path: Path) -> None:
    before = get_fixture("F100")
    assert before is not None
    frozen = before.metrics.model_dump(mode="json")
    spec = _spec([501, 499])
    historical = tmp_path / "runs" / "experiments" / "exp-3a-da9fc3022a77b55b"
    historical.mkdir(parents=True)
    sentinel = historical / "manifest.json"
    sentinel.write_bytes(b'{"frozen":true}\n')

    async def run() -> object:
        return await execute_xray(spec, tmp_path / "xray", EvidenceAdapter("F100"), resume=False)

    result = _drive(run)
    assert isinstance(result, XRayResult)
    assert result.original_decision == "INVESTIGATE"
    assert result.termination_reason == "verified_flip"
    assert result.minimality_claim == "minimum edit distance proven over the enumerated candidate space"
    assert result.minimal_edit == {"decision": "MONITOR", "distance": 1, "value": 499}
    assert result.shared_checkpoint is True
    assert result.checkpoint_id
    rows = {row.candidate_value: row for row in result.candidates}
    assert rows[499].status == "verified_flip"
    assert rows[499].flipped is True
    assert rows[499].execution_valid is True
    assert rows[499].decision == "MONITOR"
    assert rows[499].oracle_before == "INVESTIGATE"
    assert rows[499].oracle_after == "MONITOR"
    assert rows[501].status == "not_run"
    assert rows[501].flipped is False
    assert rows[501].model_calls == 0
    assert result.provider_requests == 7
    mutated = rows[499].evidence_sha256_mutated
    assert mutated
    assert mutated != rows[499].evidence_sha256_original
    trace = (tmp_path / "xray" / "traces.jsonl").read_text(encoding="utf-8")
    assert mutated in trace
    assert "oracle_decision" not in trace.split('"model_visible_observations"')[0]
    assert sentinel.read_bytes() == b'{"frozen":true}\n'
    assert get_fixture("F100").metrics.model_dump(mode="json") == frozen  # type: ignore[union-attr]
    checkpoint = json.loads((tmp_path / "xray" / "baseline.json").read_text(encoding="utf-8"))
    assert all(message["role"] != "tool" for message in checkpoint["checkpoint"]["state"]["messages"])
    assert "499" not in checkpoint["checkpoint"]["visible"]


def test_oracle_change_does_not_select_or_define_a_flip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = _spec([498, 502, 499, 501])
    assert spec.candidates == [499, 501, 498, 502]  # type: ignore[attr-defined]

    class _Opposite:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del tools, params
            visible: dict[str, str] = {}
            for message in messages:  # type: ignore[attr-defined]
                if message.role == "tool" and message.tool_name:
                    visible[message.tool_name] = message.content
            for name in ("get_metrics", "get_logs", "get_deployment"):
                if name not in visible:
                    return ModelCompletion(
                        content="",
                        tool_calls=(ToolRequest(name=name, arguments={"incident_id": "F100"}),),
                    )
            rate = json.loads(visible["get_metrics"])["error_rate_bps"]
            decision = "MONITOR" if rate == 501 else "INVESTIGATE"
            body = json.dumps(
                {"action": "none", "decision": decision, "evidence": ["metrics", "logs", "deployment"]}
            )
            return ModelCompletion(content=body)

    def constant(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["n"] += 1
        return type("Label", (), {"value": "ESCALATE"})()

    calls = {"n": 0}
    monkeypatch.setattr("parallax.oracle.decide", constant)

    async def run() -> object:
        return await execute_xray(spec, tmp_path / "xray", _Opposite(), resume=False)

    result = _drive(run)
    rows = {row.candidate_value: row for row in result.candidates}  # type: ignore[attr-defined]
    assert rows[499].flipped is False
    assert rows[499].oracle_before == "ESCALATE"
    assert rows[499].oracle_after == "ESCALATE"
    assert rows[501].flipped is True
    assert rows[501].decision == "MONITOR"
    assert rows[501].oracle_after == "ESCALATE"
    assert rows[498].status == "not_run"
    assert rows[502].status == "not_run"
    assert calls["n"] > 0


def test_no_flip_runs_the_declared_set(tmp_path: Path) -> None:
    spec = _spec([505, 501])

    async def run() -> object:
        return await execute_xray(spec, tmp_path / "xray", EvidenceAdapter("F100"), resume=False)

    result = _drive(run)
    assert result.termination_reason == "candidates_exhausted"  # type: ignore[attr-defined]
    assert result.minimality_claim == "no verified flip"  # type: ignore[attr-defined]
    assert result.minimal_edit is None  # type: ignore[attr-defined]
    assert result.usage_deduped is None  # type: ignore[attr-defined]
    rows = result.candidates  # type: ignore[attr-defined]
    assert [row.status for row in rows] == ["no_flip", "no_flip"]
    assert {row.checkpoint_id for row in rows} == {result.checkpoint_id}  # type: ignore[attr-defined]
    assert all(row.flipped is False and row.execution_valid is True for row in rows)


def test_invalid_output_and_provider_error_are_not_flips(tmp_path: Path) -> None:
    class _Garbage:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            result = await self.inner.complete(messages, tools, params)
            if self.calls > 4 and not result.tool_calls:
                return ModelCompletion(content="NOPE")
            return result

    class _Boom:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            if self.calls > 4:
                raise RuntimeError("provider stopped")
            return await self.inner.complete(messages, tools, params)

    garbage = _Garbage()

    async def invalid() -> object:
        return await execute_xray(_spec([501]), tmp_path / "invalid", garbage, resume=False)

    invalid_result = _drive(invalid)
    assert invalid_result.candidates[0].status == "invalid_output"  # type: ignore[attr-defined]
    assert invalid_result.candidates[0].flipped is False  # type: ignore[attr-defined]
    assert invalid_result.minimality_claim == "inconclusive due to invalid candidates"  # type: ignore[attr-defined]
    boom = _Boom()

    async def failed() -> object:
        return await execute_xray(_spec([501]), tmp_path / "provider", boom, resume=False)

    failed_result = _drive(failed)
    assert failed_result.candidates[0].status == "provider_error"  # type: ignore[attr-defined]
    assert failed_result.candidates[0].flipped is False  # type: ignore[attr-defined]
    assert boom.calls == 5


def test_missing_intervention_does_not_run_candidates(tmp_path: Path) -> None:
    class _LogsOnly:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del tools, params
            seen = [message.tool_name for message in messages if message.role == "tool"]  # type: ignore[attr-defined]
            if "get_logs" not in seen:
                return ModelCompletion(
                    content="",
                    tool_calls=(ToolRequest(name="get_logs", arguments={"incident_id": "F100"}),),
                )
            return ModelCompletion(content=SCRIPTED_DECISION)

    async def run() -> object:
        return await execute_xray(_spec([499]), tmp_path / "xray", _LogsOnly(), resume=False)

    result = _drive(run)
    assert result.termination_reason == "intervention_unobserved"  # type: ignore[attr-defined]
    assert result.candidates[0].status == "not_run"  # type: ignore[attr-defined]
    assert result.minimal_edit is None  # type: ignore[attr-defined]
    assert result.minimality_claim == "inconclusive due to invalid candidates"  # type: ignore[attr-defined]
    assert not (tmp_path / "xray" / "candidates.jsonl").exists()


def test_same_turn_tool_calls_stay_on_the_branch_and_off_the_checkpoint(tmp_path: Path) -> None:
    class _Batch:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del tools, params
            visible: dict[str, str] = {}
            for message in messages:  # type: ignore[attr-defined]
                if message.role == "tool" and message.tool_name:
                    visible[message.tool_name] = message.content
            if "get_metrics" not in visible:
                return ModelCompletion(
                    content="",
                    tool_calls=(
                        ToolRequest(name="get_metrics", arguments={"incident_id": "F100"}, call_id="metrics"),
                        ToolRequest(name="get_logs", arguments={"incident_id": "F100"}, call_id="logs"),
                    ),
                )
            if "get_deployment" not in visible:
                return ModelCompletion(
                    content="",
                    tool_calls=(
                        ToolRequest(name="get_deployment", arguments={"incident_id": "F100"}, call_id="deploy"),
                    ),
                )
            rate = json.loads(visible["get_metrics"])["error_rate_bps"]
            decision = "MONITOR" if rate < 500 else "INVESTIGATE"
            body = json.dumps(
                {"action": "none", "decision": decision, "evidence": ["metrics", "logs", "deployment"]}
            )
            return ModelCompletion(content=body)

    async def run() -> object:
        return await execute_xray(_spec([499]), tmp_path / "xray", _Batch(), resume=False)

    result = _drive(run)
    assert result.candidates[0].flipped is True  # type: ignore[attr-defined]
    traces = (tmp_path / "xray" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    mutant = json.loads(traces[-1])
    sources = {(row["name"], row["source"]) for row in mutant["tool_calls"]}
    assert ("get_metrics", "overlay") in sources
    assert ("get_logs", "base") in sources
    assert "checkout latency above the page budget" in "\n".join(mutant["model_visible_observations"])
    overlay = next(row for row in mutant["tool_calls"] if row["source"] == "overlay")
    assert overlay["model_visible_sha256"] == sha256_text(overlay["model_visible_observation"])
    assert result.candidates[0].evidence_sha256_mutated == overlay["model_visible_sha256"]  # type: ignore[attr-defined]
    stored = json.loads((tmp_path / "xray" / "baseline.json").read_text(encoding="utf-8"))
    assert all(message["role"] != "tool" for message in stored["checkpoint"]["state"]["messages"])


def test_request_ceiling_counts_each_completion_once(tmp_path: Path) -> None:
    class _Hungry:
        def __init__(self) -> None:
            self.calls = 0
            self.inner = EvidenceAdapter("F100")

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            visible = [message for message in messages if message.role == "tool"]  # type: ignore[attr-defined]
            if self.calls > 4 and len(visible) >= 3:
                return ModelCompletion(
                    content="",
                    tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F100"}),),
                    response_id=f"resp-{self.calls}",
                    usage=ModelUsage(input_tokens=2, output_tokens=3),
                )
            result = await self.inner.complete(messages, tools, params)
            return result.model_copy(
                update={
                    "response_id": f"resp-{self.calls}",
                    "usage": ModelUsage(input_tokens=2, output_tokens=3),
                }
            )

    adapter = _Hungry()
    spec = _spec([499, 501])

    async def run() -> object:
        return await execute_xray(spec, tmp_path / "xray", adapter, resume=False)

    result = _drive(run)
    assert adapter.calls == request_ceiling(4, 2) == 10
    assert result.provider_requests == 10  # type: ignore[attr-defined]
    assert [row.status for row in result.candidates] == ["budget_exhausted", "budget_exhausted"]  # type: ignore[attr-defined]
    assert all(row.flipped is False for row in result.candidates)  # type: ignore[attr-defined]
    assert result.usage_deduped == {"input_tokens": 20, "output_tokens": 30}  # type: ignore[attr-defined]


def test_resume_skips_committed_candidates_and_keeps_history(tmp_path: Path) -> None:
    spec = _spec([501, 505])
    output = tmp_path / "xray"
    historical = tmp_path / "runs" / "experiments" / "exp-3a-da9fc3022a77b55b"
    historical.mkdir(parents=True)
    sentinel = historical / "summary.json"
    sentinel.write_bytes(b'{"historical":true}\n')
    other = tmp_path / "older-xray"
    other.mkdir()
    other_manifest = other / "manifest.json"
    other_manifest.write_bytes(b'{"older":true}\n')

    async def first() -> object:
        return await execute_xray(spec, output, EvidenceAdapter("F100"), resume=False)

    _drive(first)
    baseline_bytes = (output / "baseline.json").read_bytes()
    manifest_bytes = (output / "manifest.json").read_bytes()
    journal = output / "candidates.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    journal.write_text(lines[0] + "\n", encoding="utf-8", newline="\n")
    seen = {"calls": 0}

    class _Count:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            seen["calls"] += 1
            return await self.inner.complete(messages, tools, params)

    async def refused() -> object:
        return await execute_xray(spec, output, _Count(), resume=False)

    with pytest.raises(ValueError, match="resume"):
        _drive(refused)
    assert seen["calls"] == 0

    async def resumed() -> object:
        return await execute_xray(spec, output, _Count(), resume=True)

    result = _drive(resumed)
    assert seen["calls"] == 3
    assert (output / "baseline.json").read_bytes() == baseline_bytes
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    stored = journal.read_text(encoding="utf-8").splitlines()
    assert stored[0] == lines[0]
    assert len(stored) == 2
    assert [row.status for row in result.candidates] == ["no_flip", "no_flip"]  # type: ignore[attr-defined]
    assert result.provider_requests == 10  # type: ignore[attr-defined]
    assert sentinel.read_bytes() == b'{"historical":true}\n'
    assert other_manifest.read_bytes() == b'{"older":true}\n'
    different = _spec([499])

    async def mismatched() -> object:
        return await execute_xray(different, output, _Count(), resume=True)

    with pytest.raises(ValueError, match="different search"):
        _drive(mismatched)
    assert seen["calls"] == 3
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["git_revision"] = "0" * 40
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    async def after_revision_change() -> object:
        return await execute_xray(spec, output, _Count(), resume=True)

    _drive(after_revision_change)
    assert seen["calls"] == 3
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["git_revision"] == "0" * 40


def test_format_names_the_minimality_limit() -> None:
    printed = format_dry_run(_spec())  # type: ignore[arg-type]
    assert "not a minimum over other fields" in printed
    assert "strategy=early_stop" in printed


def _row(value: int, distance: int, status: str) -> SearchRow:
    decision = None
    if status == "verified_flip":
        decision = "MONITOR"
    elif status == "no_flip":
        decision = "INVESTIGATE"
    return SearchRow(
        candidate_value=value,
        distance=distance,
        order_index=value,
        status=status,  # type: ignore[arg-type]
        decision=decision,
        flipped=status == "verified_flip",
        execution_valid=status in {"verified_flip", "no_flip"},
        evidence_sha256_original="a" * 64,
        evidence_sha256_mutated="b" * 64,
        checkpoint_id="c" * 64,
        oracle_before=None,
        oracle_after=None,
        error=None,
        model_calls=0,
    )


def test_search_plan_and_identity_do_not_use_the_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("oracle called")

    monkeypatch.setattr("parallax.oracle.decide", boom)
    plan = plan_search(500, [502, 498, 499], "exhaustive")
    assert plan.ordered_values == (499, 498, 502)
    assert plan.distances == (1, 2, 2)
    assert plan.distance_metric == "absolute_change_in_field_units"
    left = _spec()
    right = _spec()
    assert left.xray_id == right.xray_id  # type: ignore[attr-defined]
    assert left.xray_id != _spec([499]).xray_id  # type: ignore[attr-defined]
    assert left.xray_id != _spec(strategy="exhaustive").xray_id  # type: ignore[attr-defined]
    assert classify_search([_row(499, 1, "verified_flip"), _row(501, 1, "not_run")]) == (
        "minimum edit distance proven over the enumerated candidate space"
    )
    assert classify_search([_row(499, 1, "provider_error"), _row(495, 5, "verified_flip")]) == (
        "smallest verified flip among tested edits"
    )
    assert classify_search([_row(501, 1, "no_flip")]) == "no verified flip"
    assert classify_search([_row(501, 1, "invalid_output")]) == "inconclusive due to invalid candidates"


def test_edit_evidence_changes_only_the_allowlisted_field() -> None:
    fixture = get_fixture("F100")
    assert fixture is not None
    frozen = fixture.metrics.model_dump(mode="json")
    visible = canonical_json(frozen)
    edited = edit_evidence(
        tool_name="get_metrics",
        field_path="error_rate_bps",
        structured=frozen,
        visible=visible,
        candidate=499,
    )
    parsed = ServiceMetrics.model_validate_json(edited.observation)
    assert parsed.error_rate_bps == 499
    assert edited.evidence_sha256 == sha256_text(edited.observation)
    assert edited.evidence_sha256 != edited.source_sha256
    assert edited.source_sha256 == sha256_text(visible)
    assert get_fixture("F100").metrics.model_dump(mode="json") == frozen  # type: ignore[union-attr]


def test_exhaustive_search_runs_the_declared_set_after_a_flip(tmp_path: Path) -> None:
    spec = _spec([499, 501], strategy="exhaustive")

    async def run() -> object:
        return await execute_xray(spec, tmp_path / "xray", EvidenceAdapter("F100"), resume=False)

    result = _drive(run)
    assert [row.status for row in result.candidates] == ["verified_flip", "no_flip"]  # type: ignore[attr-defined]
    assert result.termination_reason == "candidates_exhausted"  # type: ignore[attr-defined]
    assert result.minimality_claim == "minimum edit distance proven over the enumerated candidate space"  # type: ignore[attr-defined]
    assert result.proof_blockers == []  # type: ignore[attr-defined]
    assert result.provider_requests == 10  # type: ignore[attr-defined]


def test_an_invalid_closer_candidate_is_not_a_proven_minimum(tmp_path: Path) -> None:
    class _CloserFails:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            visible = {
                message.tool_name: message.content
                for message in messages  # type: ignore[attr-defined]
                if message.role == "tool" and message.tool_name
            }
            rate = json.loads(visible["get_metrics"])["error_rate_bps"] if "get_metrics" in visible else None
            if rate == 499:
                raise RuntimeError("provider stopped")
            return await self.inner.complete(messages, tools, params)

    async def run() -> object:
        return await execute_xray(_spec([499, 495]), tmp_path / "xray", _CloserFails(), resume=False)

    result = _drive(run)
    rows = {row.candidate_value: row for row in result.candidates}  # type: ignore[attr-defined]
    assert rows[499].status == "provider_error"
    assert rows[495].status == "verified_flip"
    assert rows[495].decision == "MONITOR"
    assert result.minimality_claim == "smallest verified flip among tested edits"  # type: ignore[attr-defined]
    assert result.proof_blockers == [499]  # type: ignore[attr-defined]
    assert result.minimal_edit["value"] == 495  # type: ignore[attr-defined]


def test_mcp_error_does_not_count_as_a_flip(tmp_path: Path) -> None:
    class _BrokenTool:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            tool_messages = [message for message in messages if message.role == "tool"]  # type: ignore[attr-defined]
            if self.calls > 4 and len(tool_messages) == 1:
                return ModelCompletion(
                    content="",
                    tool_calls=(ToolRequest(name="get_logs", arguments={"incident_id": "missing"}, call_id="logs"),),
                )
            if self.calls > 4:
                return ModelCompletion(content="")
            return await self.inner.complete(messages, tools, params)

    async def run() -> object:
        return await execute_xray(_spec([499]), tmp_path / "xray", _BrokenTool(), resume=False)

    result = _drive(run)
    assert result.candidates[0].status == "mcp_error"  # type: ignore[attr-defined]
    assert result.candidates[0].flipped is False  # type: ignore[attr-defined]
    assert result.minimality_claim == "inconclusive due to invalid candidates"  # type: ignore[attr-defined]
