"""Offline research-control checks. Scripted adapters are not empirical findings."""

import json
from pathlib import Path

import anyio
import pytest

from parallax.agent import ModelCompletion, ModelUsage, ToolRequest
from parallax.capture import sha256_text
from parallax.controls import (
    RecordedOutcome,
    StudyResult,
    StudySpec,
    build_study,
    classify_continuation,
    comparison_schedule,
    edit_evidence,
    execute_study,
    main,
    prepare_control_observation,
    request_ceiling,
    simulate_orderings,
    wilson_interval,
)
from parallax.fixtures import get_fixture
from parallax.observe import canonical_json
from tests.harness import EvidenceAdapter

PARAMS = {"temperature": 0}


def _drive(function: object) -> object:
    return anyio.run(function)  # type: ignore[arg-type]


def _study(repetitions: int = 1, **overrides: object) -> StudySpec:
    arguments: dict[str, object] = {
        "fixture_id": "F100",
        "tool": "get_metrics",
        "field": "error_rate_bps",
        "intervention": 499,
        "negative": 501,
        "repetitions": repetitions,
        "seed": 0,
        "model_id": "gpt-test",
        "model_params": PARAMS,
        "model_call_budget": 4,
        "tool_call_budget": 6,
    }
    arguments.update(overrides)
    return build_study(**arguments)  # type: ignore[arg-type]


def test_null_control_replays_the_captured_text_and_the_editor_still_rejects_it() -> None:
    fixture = get_fixture("F100")
    assert fixture is not None
    payload = fixture.metrics.model_dump(mode="json")
    visible = canonical_json(payload)
    replayed = prepare_control_observation(
        role="null",
        tool_name="get_metrics",
        field_path="error_rate_bps",
        structured=payload,
        visible=visible,
        value=500,
    )
    assert replayed == visible
    assert sha256_text(replayed) == sha256_text(visible)
    with pytest.raises(ValueError, match="original value"):
        edit_evidence(
            tool_name="get_metrics",
            field_path="error_rate_bps",
            structured=payload,
            visible=visible,
            candidate=500,
        )
    edited = prepare_control_observation(
        role="intervention",
        tool_name="get_metrics",
        field_path="error_rate_bps",
        structured=payload,
        visible=visible,
        value=499,
    )
    assert sha256_text(edited) != sha256_text(visible)
    with pytest.raises(ValueError, match="null control"):
        prepare_control_observation(
            role="null",
            tool_name="get_metrics",
            field_path="error_rate_bps",
            structured=payload,
            visible=visible,
            value=499,
        )


def test_schedule_is_round_robin_and_null_is_not_a_candidate() -> None:
    assert comparison_schedule(2) == [
        (0, "null"),
        (0, "intervention"),
        (0, "negative"),
        (1, "null"),
        (1, "intervention"),
        (1, "negative"),
    ]
    spec = _study(2)
    assert [slot.role for slot in spec.schedule] == [role for _, role in comparison_schedule(2)]
    assert [slot.repetition_index for slot in spec.schedule] == [0, 0, 0, 1, 1, 1]
    assert all(condition.search_candidate is False for condition in spec.conditions)
    assert spec.checkpoint_draws == 1
    assert spec.sampling_unit == "shared_checkpoint_continuation"
    assert spec.worst_case_model_requests == request_ceiling(4, 3, 2) == 22
    assert spec.null_continuation_requests == 6
    with pytest.raises(ValueError, match="original value"):
        _study(intervention=500)
    with pytest.raises(ValueError, match="1 to 5"):
        _study(repetitions=6)


def test_orderings_are_reproducible_simulations() -> None:
    outcomes = [
        RecordedOutcome(value=505, distance=5, verified_flip=False),
        RecordedOutcome(value=499, distance=1, verified_flip=True),
        RecordedOutcome(value=501, distance=1, verified_flip=False),
        RecordedOutcome(value=495, distance=5, verified_flip=True),
    ]
    first = simulate_orderings(outcomes, 7)
    second = simulate_orderings(outcomes, 7)
    assert first == second
    distance, seeded, exhaustive = first
    assert distance.method == "distance_first"
    assert distance.order == [499, 501, 495, 505]
    assert distance.first_verified_flip_position == 1
    assert distance.evaluated_count == 1
    assert distance.smallest_discovered_distance == 1
    assert distance.coverage == 1
    assert exhaustive.evaluated_count == 4
    assert exhaustive.coverage == 4
    assert exhaustive.smallest_discovered_distance == 1
    assert seeded.seed == 7
    assert seeded.evaluated_count >= 1
    assert all(row.independent_model_experiment is False for row in first)
    assert "Not an independent model experiment" in distance.note
    other = simulate_orderings(outcomes, 8)[1]
    assert other.order == simulate_orderings(outcomes, 8)[1].order


def test_metrics_keep_errors_out_of_the_flip_denominator() -> None:
    assert wilson_interval(0, 0) is None
    interval = wilson_interval(1, 2)
    assert interval is not None
    assert 0.0 < interval["low"] < 0.5 < interval["high"] < 1.0
    assert classify_continuation(
        factual_valid=True,
        execution_status="completed",
        parse_ok=True,
        decision="MONITOR",
        factual_decision="INVESTIGATE",
        observation_visible=True,
        flags=(),
    ) == "decision_changed"
    assert classify_continuation(
        factual_valid=True,
        execution_status="model_error",
        parse_ok=False,
        decision=None,
        factual_decision="INVESTIGATE",
        observation_visible=False,
        flags=("provider_error",),
    ) == "provider_error"
    assert classify_continuation(
        factual_valid=True,
        execution_status="tool_error",
        parse_ok=False,
        decision=None,
        factual_decision="INVESTIGATE",
        observation_visible=True,
        flags=("tool_error",),
    ) == "mcp_error"


def test_shared_checkpoint_block_separates_null_from_evidence(tmp_path: Path) -> None:
    decisions: list[int] = []

    class _Vary:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            result = await self.inner.complete(messages, tools, params)
            visible = {
                message.tool_name: message.content
                for message in messages  # type: ignore[attr-defined]
                if message.role == "tool" and message.tool_name
            }
            rate = json.loads(visible["get_metrics"])["error_rate_bps"] if "get_metrics" in visible else None
            content = result.content
            if rate == 500 and not result.tool_calls:
                seen = sum(item == 500 for item in decisions)
                decisions.append(500)
                if seen == 2:
                    content = json.dumps(
                        {"action": "none", "decision": "MONITOR", "evidence": ["metrics", "logs", "deployment"]}
                    )
            elif rate is not None and not result.tool_calls:
                decisions.append(rate)
            return result.model_copy(
                update={
                    "content": content,
                    "response_id": f"resp-{self.calls}",
                    "usage": ModelUsage(input_tokens=2, output_tokens=3),
                }
            )

    spec = _study(2)
    frozen = get_fixture("F100")
    assert frozen is not None
    frozen_metrics = frozen.metrics.model_dump(mode="json")
    roots = [
        Path("runs/experiments/exp-3a-da9fc3022a77b55b"),
        Path("runs/xray/xray-547963e428ed516b"),
    ]
    before = {str(path): _snapshot(path) for path in roots if path.exists()}

    async def run() -> StudyResult:
        return await execute_study(spec, tmp_path / "controls", _Vary(), resume=False)

    result = _drive(run)
    assert isinstance(result, StudyResult)
    assert result.original_decision == "INVESTIGATE"
    assert result.checkpoint_draws == 1
    assert result.termination_reason == "schedule_complete"
    rows = result.continuations
    assert [(row.role, row.repetition_index) for row in rows] == [
        ("null", 0),
        ("intervention", 0),
        ("negative", 0),
        ("null", 1),
        ("intervention", 1),
        ("negative", 1),
    ]
    assert len({row.checkpoint_id for row in rows}) == 1
    assert all(row.independent_checkpoint_sample is False for row in rows)
    assert all(row.search_candidate is False for row in rows)
    nulls = [row for row in rows if row.role == "null"]
    edits = [row for row in rows if row.role == "intervention"]
    negatives = [row for row in rows if row.role == "negative"]
    assert [row.decision_parsed for row in nulls] == ["INVESTIGATE", "MONITOR"]
    assert all(row.evidence_intervention is False for row in nulls)
    assert nulls[1].decision_changed is True
    assert nulls[1].status == "decision_changed"
    assert len({row.observation_sha256 for row in nulls}) == 1
    assert all(row.evidence_intervention is True for row in edits)
    assert [row.decision_parsed for row in edits] == ["MONITOR", "MONITOR"]
    assert all(row.observation_sha256 != nulls[0].observation_sha256 for row in edits)
    assert all(row.decision_changed is False and row.evidence_intervention is False for row in negatives)
    by_role = {metric.role: metric for metric in result.conditions}
    assert by_role["null"].attempt_count == 2
    assert by_role["null"].valid_count == 2
    assert by_role["null"].flip_count == 1
    assert by_role["null"].flip_rate == 0.5
    assert by_role["intervention"].flip_count == 2
    assert by_role["intervention"].flip_rate == 1.0
    assert by_role["negative"].flip_count == 0
    assert by_role["negative"].oracle_aligned_count == 2
    assert result.intervention_minus_null is not None
    assert result.intervention_minus_null["estimate"] == 0.5
    assert result.intervention_minus_null["significance_test"] is None
    assert "not a significance test" in result.intervention_minus_null["note"]
    assert result.usage_deduped == {"input_tokens": 44, "output_tokens": 66}
    assert len(result.unique_provider_response_ids) == 22
    assert len(set(result.unique_provider_response_ids)) == 22
    checkpoint = json.loads((tmp_path / "controls" / "baseline.json").read_text(encoding="utf-8"))
    assert all(message["role"] != "tool" for message in checkpoint["checkpoint"]["state"]["messages"])
    assert "499" not in checkpoint["checkpoint"]["visible"]
    assert get_fixture("F100").metrics.model_dump(mode="json") == frozen_metrics  # type: ignore[union-attr]
    after = {str(path): _snapshot(path) for path in roots if path.exists()}
    assert after == before


def test_provider_mcp_and_invalid_outputs_are_attempts_not_flips(tmp_path: Path) -> None:
    class _Faults:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            visible = {
                message.tool_name: message.content
                for message in messages  # type: ignore[attr-defined]
                if message.role == "tool" and message.tool_name
            }
            rate = json.loads(visible["get_metrics"])["error_rate_bps"] if "get_metrics" in visible else None
            tool_messages = [message for message in messages if message.role == "tool"]  # type: ignore[attr-defined]
            if rate == 499:
                raise RuntimeError("provider stopped")
            if rate == 501 and len(tool_messages) == 1:
                return ModelCompletion(
                    content="",
                    tool_calls=(ToolRequest(name="get_logs", arguments={"incident_id": "missing"}, call_id="logs"),),
                )
            if rate == 501:
                return ModelCompletion(content="NOPE")
            return await self.inner.complete(messages, tools, params)

    async def run() -> StudyResult:
        return await execute_study(_study(1), tmp_path / "controls", _Faults(), resume=False)

    result = _drive(run)
    by_role = {row.role: row for row in result.continuations}
    assert by_role["null"].status == "decision_stable"
    assert by_role["null"].execution_valid is True
    assert by_role["intervention"].status == "provider_error"
    assert by_role["intervention"].evidence_intervention is False
    assert by_role["negative"].status == "mcp_error"
    metrics = {metric.role: metric for metric in result.conditions}
    assert metrics["intervention"].attempt_count == 1
    assert metrics["intervention"].valid_count == 0
    assert metrics["intervention"].flip_rate is None
    assert metrics["intervention"].provider_failure_count == 1
    assert metrics["negative"].mcp_failure_count == 1
    assert metrics["negative"].flip_count == 0
    assert result.intervention_minus_null is None

    class _Invalid:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            result = await self.inner.complete(messages, tools, params)
            visible = {
                message.tool_name: message.content
                for message in messages  # type: ignore[attr-defined]
                if message.role == "tool" and message.tool_name
            }
            rate = json.loads(visible["get_metrics"])["error_rate_bps"] if "get_metrics" in visible else None
            if rate == 499 and not result.tool_calls:
                return ModelCompletion(content="NOPE")
            return result

    async def invalid() -> StudyResult:
        return await execute_study(_study(1), tmp_path / "invalid", _Invalid(), resume=False)

    invalid_result = _drive(invalid)
    edited = next(row for row in invalid_result.continuations if row.role == "intervention")
    assert edited.status == "invalid_output"
    assert edited.evidence_intervention is False
    metric = next(item for item in invalid_result.conditions if item.role == "intervention")
    assert metric.invalid_output_count == 1
    assert metric.valid_count == 0


def test_duplicate_new_response_ids_null_the_usage(tmp_path: Path) -> None:
    class _Reuse:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")
            self.calls = 0

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            self.calls += 1
            result = await self.inner.complete(messages, tools, params)
            identifier = "resp-shared" if self.calls > 4 else f"resp-{self.calls}"
            return result.model_copy(
                update={
                    "response_id": identifier,
                    "usage": ModelUsage(input_tokens=2, output_tokens=3),
                }
            )

    async def run() -> StudyResult:
        return await execute_study(_study(1), tmp_path / "controls", _Reuse(), resume=False)

    result = _drive(run)
    assert result.usage_deduped is None
    assert len(result.unique_provider_response_ids) < sum(
        len(row.provider_response_ids) for row in result.continuations
    )


def test_resume_skips_committed_continuations_and_rejects_a_missing_baseline(tmp_path: Path) -> None:
    spec = _study(1)
    output = tmp_path / "controls"
    historical = tmp_path / "runs" / "experiments" / "exp-3a-da9fc3022a77b55b"
    historical.mkdir(parents=True)
    sentinel = historical / "manifest.json"
    sentinel.write_bytes(b'{"frozen":true}\n')
    xray = tmp_path / "runs" / "xray" / "xray-547963e428ed516b"
    xray.mkdir(parents=True)
    xray_manifest = xray / "manifest.json"
    xray_manifest.write_bytes(b'{"xray":true}\n')

    async def first() -> StudyResult:
        return await execute_study(spec, output, EvidenceAdapter("F100"), resume=False)

    _drive(first)
    journal = output / "continuations.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    baseline_bytes = (output / "baseline.json").read_bytes()
    manifest_bytes = (output / "manifest.json").read_bytes()
    journal.write_text(lines[0] + "\n", encoding="utf-8", newline="\n")
    seen = {"calls": 0}

    class _Count:
        def __init__(self) -> None:
            self.inner = EvidenceAdapter("F100")

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            seen["calls"] += 1
            return await self.inner.complete(messages, tools, params)

    async def refused() -> StudyResult:
        return await execute_study(spec, output, _Count(), resume=False)

    with pytest.raises(ValueError, match="resume"):
        _drive(refused)
    assert seen["calls"] == 0

    async def resumed() -> StudyResult:
        return await execute_study(spec, output, _Count(), resume=True)

    result = _drive(resumed)
    assert seen["calls"] == 6
    assert (output / "baseline.json").read_bytes() == baseline_bytes
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    stored = journal.read_text(encoding="utf-8").splitlines()
    assert stored[0] == lines[0]
    assert len(stored) == 3
    assert result.provider_requests == 13
    assert sentinel.read_bytes() == b'{"frozen":true}\n'
    assert xray_manifest.read_bytes() == b'{"xray":true}\n'
    (output / "baseline.json").unlink()

    async def unsafe() -> StudyResult:
        return await execute_study(spec, output, _Count(), resume=True)

    with pytest.raises(ValueError, match="baseline is missing"):
        _drive(unsafe)
    assert seen["calls"] == 6


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

    async def run() -> StudyResult:
        return await execute_study(_study(1), tmp_path / "controls", adapter, resume=False)

    result = _drive(run)
    assert adapter.calls == request_ceiling(4, 3, 1) == 13
    assert [row.status for row in result.continuations] == ["budget_exhausted"] * 3
    assert all(row.evidence_intervention is False for row in result.continuations)
    assert result.usage_deduped == {"input_tokens": 26, "output_tokens": 39}


def test_same_turn_sibling_stays_on_the_null_branch(tmp_path: Path) -> None:
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
                    tool_calls=(ToolRequest(name="get_deployment", arguments={"incident_id": "F100"}, call_id="deploy"),),
                )
            rate = json.loads(visible["get_metrics"])["error_rate_bps"]
            decision = "MONITOR" if rate == 499 else "INVESTIGATE"
            return ModelCompletion(
                content=json.dumps(
                    {"action": "none", "decision": decision, "evidence": ["metrics", "logs", "deployment"]}
                )
            )

    async def run() -> StudyResult:
        return await execute_study(_study(1), tmp_path / "controls", _Batch(), resume=False)

    result = _drive(run)
    null = next(row for row in result.continuations if row.role == "null")
    assert null.fork_call_id == "metrics"
    assert null.observation_sha256 == sha256_text(
        json.loads((tmp_path / "controls" / "baseline.json").read_text(encoding="utf-8"))["checkpoint"]["visible"]
    )
    trace = (tmp_path / "controls" / "traces.jsonl").read_text(encoding="utf-8")
    assert "checkout latency above the page budget" in trace
    assert null.evidence_intervention is False


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
    assert main(["--dry-run", "--repetitions", "3"]) == 0
    printed = capsys.readouterr().out
    assert "dry-run: no model request will be sent" in printed
    assert "condition role=null value=500 distance=0 search_candidate=false" in printed
    assert "condition role=intervention value=499 distance=1 search_candidate=false" in printed
    assert "condition role=negative value=501 distance=1 search_candidate=false" in printed
    assert "worst_case_model_requests=31" in printed
    assert "checkpoint_acquisition_requests=4" in printed
    assert "null_continuation_requests=9" in printed
    assert "sampling_unit=shared_checkpoint_continuation" in printed
    assert "seed_applies_to=offline_ordering_only" in printed
    assert "sk-" not in printed
    assert not (tmp_path / "runs").exists()
    assert main(["--dry-run", "--live"]) == 2
    monkeypatch.delenv("PARALLAX_MODEL")
    assert main(["--live"]) == 2


def test_live_cli_forwards_resume_through_anyio(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("parallax.openai_adapter.load_project_env", lambda: False)
    monkeypatch.setenv("PARALLAX_MODEL", "gpt-test")
    sentinel = object()
    seen: list[bool] = []

    def from_env() -> object:
        return sentinel

    async def execute(spec: StudySpec, output_dir: Path, adapter: object, *, resume: bool) -> StudyResult:
        del output_dir
        assert adapter is sentinel
        seen.append(resume)
        return StudyResult(
            schema_version=spec.schema_version,
            study_id=spec.study_id,
            original_decision="INVESTIGATE",
            termination_reason="mocked",
            checkpoint_id=None,
            checkpoint_draws=1,
            sampling_unit=spec.sampling_unit,
            continuations=[],
            conditions=[],
            intervention_minus_null=None,
            unique_provider_response_ids=[],
            provider_requests=0,
            usage_deduped=None,
            notes=[],
        )

    monkeypatch.setattr("parallax.openai_adapter.OpenAIAdapter.from_env", from_env)
    monkeypatch.setattr("parallax.controls.execute_study", execute)
    output = tmp_path / "controls-live"
    argv = ["--live", "--repetitions", "1", "--output", str(output)]
    assert main(argv) == 0
    assert main([*argv, "--resume"]) == 0
    assert seen == [False, True]
    printed = capsys.readouterr().out
    assert "termination_reason=mocked" in printed
    assert "sk-" not in printed


def _snapshot(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.rglob("*")) if item.is_file()}
