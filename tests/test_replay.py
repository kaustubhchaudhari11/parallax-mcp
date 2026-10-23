"""Paired replay against the in-process server. The adapter is scripted."""

import json
from importlib.metadata import version
from pathlib import Path

import anyio
import pytest
from mcp.types import CallToolResult, TextContent
from mcp_types.version import KNOWN_PROTOCOL_VERSIONS

import parallax.replay as replay_module
from parallax.agent import ModelCompletion, ToolRequest
from parallax.capture import label_key_overlap
from parallax.fixtures import FIXTURES
from parallax.mutate import H3_IGNORE_POLICY, LARGE_LATENCY_MS, h2_clear_error_rate
from parallax.observe import canonical_json
from parallax.replay import OverlayResolver, PrefixMismatch, prefix_sha256, run_pair
from tests.harness import SCRIPTED_DECISION, ScriptedAdapter

PARAMS = {"temperature": 0, "seed": 1, "max_tokens": 256}


def _stores() -> tuple[object, ...]:
    return tuple(
        (
            fixture.incident_id,
            fixture.metrics.model_dump(mode="json"),
            fixture.logs.model_dump(mode="json"),
            fixture.deployment.model_dump(mode="json"),
        )
        for fixture in FIXTURES.values()
    )


def _drive(func: object) -> object:
    return anyio.run(func, backend="asyncio")  # type: ignore[arg-type]


class _MissingThenDecide:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        if not any(message.role == "tool" for message in messages):  # type: ignore[attr-defined]
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "missing"}),),
            )
        return ModelCompletion(content=SCRIPTED_DECISION)


def test_unknown_ids_are_caller_errors() -> None:
    async def main() -> None:
        with pytest.raises(ValueError):
            await run_pair(
                incident_id="missing",
                intervention_id="h1-latency",
                adapter=ScriptedAdapter("missing", ()),
                model_id="fake",
                model_params=PARAMS,
                step_budget=2,
                token_budget=None,
                phase="harness",
                scenario_id=None,
                repeat_index=0,
            )
        with pytest.raises(ValueError):
            await run_pair(
                incident_id="F000",
                intervention_id="h1-whitespace",
                adapter=ScriptedAdapter("F000", ()),
                model_id="fake",
                model_params=PARAMS,
                step_budget=2,
                token_budget=None,
                phase="harness",
                scenario_id=None,
                repeat_index=0,
            )

    _drive(main)


def test_overlay_is_stable_and_the_prefix_is_shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = replay_module.apply_intervention

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(replay_module, "apply_intervention", counting)
    adapter = ScriptedAdapter("F000", ("get_metrics", "get_metrics", "get_logs", "get_deployment"))
    before = _stores()
    path = tmp_path / "harness.jsonl"

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=adapter,
            model_id="fake",
            model_params=PARAMS,
            step_budget=8,
            token_budget=1000,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
            trace_path=path,
        )

    run = _drive(main)
    assert _stores() == before
    assert calls["n"] == 1
    assert run.prefix_sha256 is not None  # type: ignore[attr-defined]
    assert run.snapshot is not None  # type: ignore[attr-defined]
    assert prefix_sha256(run.snapshot.state) == run.prefix_sha256  # type: ignore[attr-defined]
    factual = run.factual_messages  # type: ignore[attr-defined]
    mutant = run.mutant_messages  # type: ignore[attr-defined]
    assert factual is not None
    assert mutant is not None
    assert len(run.snapshot.state.messages) == 3  # type: ignore[attr-defined]
    assert factual[:3] == mutant[:3] == list(run.snapshot.state.messages)  # type: ignore[attr-defined]
    assert factual[3].tool_name == mutant[3].tool_name == "get_metrics"
    assert factual[3].content != mutant[3].content
    assert f'"latency_p99_ms":{LARGE_LATENCY_MS}' in mutant[3].content
    assert f'"latency_p99_ms":{LARGE_LATENCY_MS}' not in factual[3].content
    mutant_metrics = [message.content for message in mutant if message.tool_name == "get_metrics"]
    factual_metrics = [message.content for message in factual if message.tool_name == "get_metrics"]
    assert mutant_metrics == [mutant_metrics[0], mutant_metrics[0]]
    assert factual_metrics == [factual_metrics[0], factual_metrics[0]]
    assert FIXTURES["F000"].metrics.latency_p99_ms == 8000
    logs = canonical_json(FIXTURES["F000"].logs.model_dump(mode="json"))
    deployment = canonical_json(FIXTURES["F000"].deployment.model_dump(mode="json"))
    assert [message.content for message in mutant if message.tool_name == "get_logs"] == [logs]
    assert [message.content for message in mutant if message.tool_name == "get_deployment"] == [deployment]
    metric_records = [call for call in run.mutant.tool_calls if call.name == "get_metrics"]  # type: ignore[attr-defined]
    assert len(metric_records) == 2
    assert {call.source for call in metric_records} == {"overlay"}
    assert {call.raw["structured_content"]["latency_p99_ms"] for call in metric_records} == {8000}  # type: ignore[index]
    for message in (*run.baseline_messages, *factual, *mutant):  # type: ignore[attr-defined]
        assert "oracle_decision" not in message.content
        assert "expected_decision" not in message.content
        if message.role == "tool" and message.content.startswith("{"):
            assert label_key_overlap(json.loads(message.content)) == set()
        if message.role == "user":
            assert "MONITOR" not in message.content
        if message.role == "system":
            assert "F000" not in message.content
    assert run.baseline.arm == "baseline"  # type: ignore[attr-defined]
    assert run.baseline.intervention_id is None  # type: ignore[attr-defined]
    assert run.baseline.snapshot_id is None  # type: ignore[attr-defined]
    assert run.baseline.decision_parsed == "ESCALATE"  # type: ignore[attr-defined]
    assert run.baseline.oracle_decision == "MONITOR"  # type: ignore[attr-defined]
    assert run.baseline.phase == "harness"  # type: ignore[attr-defined]
    assert run.baseline.model_id == "fake"  # type: ignore[attr-defined]
    assert run.baseline.usage is None  # type: ignore[attr-defined]
    assert run.factual.prefix_sha256 == run.mutant.prefix_sha256 == run.prefix_sha256  # type: ignore[attr-defined]
    assert run.factual.oracle_decision == "MONITOR"  # type: ignore[attr-defined]
    assert run.mutant.oracle_decision == "MONITOR"  # type: ignore[attr-defined]
    assert run.mutant.observation_original == factual[3].content  # type: ignore[attr-defined]
    assert run.mutant.observation_mutated == mutant[3].content  # type: ignore[attr-defined]
    assert run.mutant.observation_original != run.mutant.observation_mutated  # type: ignore[attr-defined]
    assert run.baseline.pair_id == run.factual.pair_id == run.mutant.pair_id  # type: ignore[attr-defined]
    assert run.baseline.model_params == PARAMS  # type: ignore[attr-defined]
    assert run.baseline.sdk_version == version("mcp") == "2.3.0"  # type: ignore[attr-defined]
    assert run.baseline.negotiated_protocol_version in KNOWN_PROTOCOL_VERSIONS  # type: ignore[attr-defined]
    assert adapter.seen_tools == ["get_metrics", "get_logs", "get_deployment"]
    from parallax.trace import read_traces

    rows = read_traces(path)
    assert [row.arm for row in rows] == ["baseline", "factual_replay", "mutant"]
    assert rows[2].observation_mutated == run.mutant.observation_mutated  # type: ignore[attr-defined]


def test_s07_replays_reversed_f110_logs_from_one_prefix() -> None:
    """Scripted control flow for the corrected S07 pairing.

    The parsed decision is the harness string ``ESCALATE``. The oracle
    label on this fixture is ``INVESTIGATE``. Agreement is not claimed.
    """

    before = _stores()

    async def main() -> object:
        return await run_pair(
            incident_id="F110",
            intervention_id="h1-log-order",
            adapter=ScriptedAdapter("F110", ("get_logs", "get_logs")),
            model_id="fake",
            model_params=PARAMS,
            step_budget=8,
            token_budget=None,
            phase="harness",
            scenario_id="S07",
            repeat_index=0,
        )

    run = _drive(main)
    assert _stores() == before
    factual = run.factual_messages  # type: ignore[attr-defined]
    mutant = run.mutant_messages  # type: ignore[attr-defined]
    assert factual is not None
    assert mutant is not None
    factual_logs = [message.content for message in factual if message.tool_name == "get_logs"]
    mutant_logs = [message.content for message in mutant if message.tool_name == "get_logs"]
    assert len(factual_logs) == len(mutant_logs) == 2
    assert factual_logs[0] == factual_logs[1]
    assert mutant_logs[0] == mutant_logs[1]
    assert factual_logs[0] != mutant_logs[0]
    original = json.loads(factual_logs[0])
    mutated = json.loads(mutant_logs[0])
    original_messages = [line["message"] for line in original["lines"]]
    mutated_messages = [line["message"] for line in mutated["lines"]]
    assert mutated_messages == list(reversed(original_messages))
    assert original["error_count"] == mutated["error_count"] == 10
    assert original["highest_severity"] == mutated["highest_severity"] == "error"
    assert run.factual.oracle_decision == "INVESTIGATE"  # type: ignore[attr-defined]
    assert run.mutant.oracle_decision == "INVESTIGATE"  # type: ignore[attr-defined]
    assert run.factual.prefix_sha256 == run.mutant.prefix_sha256 == run.prefix_sha256  # type: ignore[attr-defined]
    assert run.mutant.observation_original == factual_logs[0]  # type: ignore[attr-defined]
    assert run.mutant.observation_mutated == mutant_logs[0]  # type: ignore[attr-defined]
    log_records = [call for call in run.mutant.tool_calls if call.name == "get_logs"]  # type: ignore[attr-defined]
    assert {call.source for call in log_records} == {"overlay"}
    raw_lines = log_records[0].raw["structured_content"]["lines"]  # type: ignore[index]
    assert raw_lines[0]["message"] == original_messages[0]
    assert json.loads(log_records[0].model_visible_observation)["lines"][0]["message"] == mutated_messages[0]
    assert run.baseline.decision_parsed == "ESCALATE"  # type: ignore[attr-defined]
    assert run.baseline.model_id == "fake"  # type: ignore[attr-defined]
    assert run.baseline.phase == "harness"  # type: ignore[attr-defined]


def test_h3_text_is_applied_once() -> None:
    before = _stores()

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h3-ignore-policy",
            adapter=ScriptedAdapter("F000", ("get_logs", "get_logs")),
            model_id="fake",
            model_params=PARAMS,
            step_budget=8,
            token_budget=None,
            phase="harness",
            scenario_id="S17",
            repeat_index=1,
        )

    run = _drive(main)
    assert _stores() == before
    mutant = run.mutant_messages  # type: ignore[attr-defined]
    assert mutant is not None
    logs = [message.content for message in mutant if message.tool_name == "get_logs"]
    assert logs[0] == logs[1]
    assert logs[0].count(H3_IGNORE_POLICY) == 1
    assert run.mutant.oracle_decision == "MONITOR"  # type: ignore[attr-defined]
    assert run.mutant.decision_parsed == "ESCALATE"  # type: ignore[attr-defined]
    assert run.mutant.predicate_m is False  # type: ignore[attr-defined]
    assert run.mutant.predicate_l is False  # type: ignore[attr-defined]
    assert run.mutant.predicate_d is False  # type: ignore[attr-defined]


def test_illegal_mutant_is_not_sent() -> None:
    banned = canonical_json(h2_clear_error_rate(FIXTURES["F010"].metrics).model_dump(mode="json"))
    assert '"error_rate_bps":499' in banned

    async def main() -> object:
        return await run_pair(
            incident_id="F010",
            intervention_id="h2-clear-error-rate",
            adapter=ScriptedAdapter("F010", ("get_metrics",)),
            model_id="fake",
            model_params=PARAMS,
            step_budget=8,
            token_budget=None,
            phase="harness",
            scenario_id=None,
            repeat_index=0,
        )

    run = _drive(main)
    assert run.factual is None  # type: ignore[attr-defined]
    assert run.mutant_messages is None  # type: ignore[attr-defined]
    assert run.mutant.flags == ["engine_error"]  # type: ignore[attr-defined]
    assert run.mutant.decision_parsed is None  # type: ignore[attr-defined]
    assert run.mutant.oracle_decision is None  # type: ignore[attr-defined]
    assert run.mutant.observation_mutated is None  # type: ignore[attr-defined]
    assert run.mutant.error == "table decision did not change"  # type: ignore[attr-defined]
    for message in run.baseline_messages:  # type: ignore[attr-defined]
        assert banned not in message.content
    assert run.baseline.oracle_decision == "INVESTIGATE"  # type: ignore[attr-defined]
    assert run.baseline.decision_parsed == "ESCALATE"  # type: ignore[attr-defined]


def test_unobserved_intervention_stays_out_of_the_transcript() -> None:
    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=ScriptedAdapter("F000", ()),
            model_id="fake",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    run = _drive(main)
    assert run.factual is None  # type: ignore[attr-defined]
    assert run.snapshot is None  # type: ignore[attr-defined]
    assert run.mutant.flags == ["intervention_unobserved"]  # type: ignore[attr-defined]
    assert run.mutant.decision_parsed is None  # type: ignore[attr-defined]
    assert run.mutant.oracle_decision == "MONITOR"  # type: ignore[attr-defined]
    assert run.mutant.model_visible_observations == []  # type: ignore[attr-defined]
    assert run.mutant.observation_mutated is not None  # type: ignore[attr-defined]
    assert str(LARGE_LATENCY_MS) in run.mutant.observation_mutated  # type: ignore[attr-defined]
    for message in run.baseline_messages:  # type: ignore[attr-defined]
        assert str(LARGE_LATENCY_MS) not in message.content
    assert run.baseline.decision_parsed == "ESCALATE"  # type: ignore[attr-defined]


def test_tool_error_is_visible_and_is_not_a_checkpoint() -> None:
    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=_MissingThenDecide(),
            model_id="fake",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    run = _drive(main)
    assert "tool_error" in run.baseline.flags  # type: ignore[attr-defined]
    assert any("unknown_incident" in message.content for message in run.baseline_messages)  # type: ignore[attr-defined]
    assert run.mutant.flags == ["intervention_unobserved"]  # type: ignore[attr-defined]
    for message in run.baseline_messages:  # type: ignore[attr-defined]
        assert str(LARGE_LATENCY_MS) not in message.content


def test_prefix_mismatch_blocks_both_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(left: object, right: object) -> str:
        del left, right
        raise PrefixMismatch("left", "right")

    monkeypatch.setattr(replay_module, "check_prefix_pair", explode)

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=ScriptedAdapter("F000", ("get_metrics",)),
            model_id="fake",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    run = _drive(main)
    assert run.factual is None  # type: ignore[attr-defined]
    assert run.mutant.flags == ["engine_error"]  # type: ignore[attr-defined]
    assert run.mutant.error is not None  # type: ignore[attr-defined]
    assert "prefix hashes differ" in run.mutant.error  # type: ignore[attr-defined]
    for message in run.baseline_messages:  # type: ignore[attr-defined]
        assert str(LARGE_LATENCY_MS) not in message.content


def test_reported_usage_is_summed_and_a_gap_stays_null() -> None:
    async def always() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=ScriptedAdapter("F000", ("get_metrics",), usage="always"),
            model_id="fake",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    async def gap() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=ScriptedAdapter("F000", ("get_metrics",), usage="gap"),
            model_id="fake",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    summed = _drive(always)
    missing = _drive(gap)
    expected = {"input_tokens": 4, "output_tokens": 6}
    assert summed.baseline.usage == expected  # type: ignore[attr-defined]
    assert summed.factual.usage == expected  # type: ignore[attr-defined]
    assert summed.mutant.usage == expected  # type: ignore[attr-defined]
    assert missing.baseline.usage is None  # type: ignore[attr-defined]
    assert missing.factual.usage is None  # type: ignore[attr-defined]
    assert missing.mutant.usage is None  # type: ignore[attr-defined]


def test_overlay_preserves_errors_and_repeats_the_branch_observation() -> None:
    mutant = '{"error_rate_bps":499,"latency_p99_ms":1000000000}'
    base = {"error_rate_bps": 499, "latency_p99_ms": 160}
    base_text = canonical_json(base)
    stored = {"get_metrics": mutant}
    left = OverlayResolver(stored, incident_id="F011")
    right = OverlayResolver(dict(stored), incident_id="F011")
    stored["get_metrics"] = "contaminated"
    success = CallToolResult(
        content=[TextContent(type="text", text=base_text)],
        structured_content=base,
        is_error=False,
    )
    call = ToolRequest(name="get_metrics", arguments={"incident_id": "F011"})
    first = left.resolve(call, success)
    second = left.resolve(call, success)
    assert first.content == second.content == mutant
    assert right.resolve(call, success).content == mutant
    assert first.is_error is False
    assert first.source == "overlay"
    assert first.content != base_text
    other = left.resolve(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}), success)
    assert other.source == "base"
    assert other.content == base_text
    assert other.is_error is False
    failure_payload = {"error": "unavailable", "incident_id": "F011"}
    failure = CallToolResult(
        content=[TextContent(type="text", text=canonical_json(failure_payload))],
        structured_content=failure_payload,
        is_error=True,
    )
    surfaced = left.resolve(call, failure)
    assert surfaced.is_error is True
    assert surfaced.flag == "tool_error"
    assert surfaced.source == "base"
    assert "unavailable" in surfaced.content
    assert mutant not in surfaced.content
    assert "is_error" not in json.loads(surfaced.content)
    assert left.resolve(call, success).content == mutant


class _UnknownOnMutant:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        metrics = [
            message.content
            for message in messages  # type: ignore[attr-defined]
            if message.role == "tool" and message.tool_name == "get_metrics"
        ]
        if not metrics:
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
            )
        if any("unknown_incident" in text for text in metrics):
            return ModelCompletion(content=SCRIPTED_DECISION)
        if str(LARGE_LATENCY_MS) in metrics[-1] and len(metrics) == 1:
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "missing"}),),
            )
        if len(metrics) == 1:
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
            )
        return ModelCompletion(content=SCRIPTED_DECISION)


def test_unknown_incident_stays_an_error_on_the_mutant_branch() -> None:
    before = _stores()

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=_UnknownOnMutant(),
            model_id="fake",
            model_params=PARAMS,
            step_budget=6,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    run = _drive(main)
    assert _stores() == before
    factual = run.factual_messages  # type: ignore[attr-defined]
    mutant = run.mutant_messages  # type: ignore[attr-defined]
    assert factual is not mutant
    assert factual[0] is not mutant[0]
    assert factual[0] == mutant[0]
    assert all("unknown_incident" not in message.content for message in factual)
    assert all(str(LARGE_LATENCY_MS) not in message.content for message in factual)
    error_messages = [message.content for message in mutant if "unknown_incident" in message.content]
    assert len(error_messages) == 1
    assert str(LARGE_LATENCY_MS) not in error_messages[0]
    assert "is_error" not in json.loads(error_messages[0])
    error_records = [call for call in run.mutant.tool_calls if call.is_error]  # type: ignore[attr-defined]
    assert len(error_records) == 1
    assert error_records[0].source == "base"
    assert error_records[0].raw["is_error"] is True  # type: ignore[index]
    assert error_records[0].model_visible_observation == error_messages[0]
    restored = [call for call in run.mutant.tool_calls if call.name == "get_metrics" and call.source == "overlay"]  # type: ignore[attr-defined]
    assert len(restored) == 1
    assert restored[0].is_error is False
    assert str(LARGE_LATENCY_MS) in restored[0].model_visible_observation
    assert restored[0].raw["structured_content"]["latency_p99_ms"] == 8000  # type: ignore[index]
    assert run.mutant.execution_status == "completed"  # type: ignore[attr-defined]
    assert "tool_error" in run.mutant.flags  # type: ignore[attr-defined]
    assert run.factual.execution_status == "completed"  # type: ignore[attr-defined]
    assert "tool_error" not in run.factual.flags  # type: ignore[attr-defined]


def test_fake_model_research_phase_is_rejected() -> None:
    async def main() -> None:
        with pytest.raises(ValueError, match="harness"):
            await run_pair(
                incident_id="F000",
                intervention_id="h1-latency",
                adapter=ScriptedAdapter("F000", ()),
                model_id="fake",
                model_params=PARAMS,
                step_budget=2,
                token_budget=None,
                phase="pilot",
                scenario_id="S01",
                repeat_index=0,
            )

    _drive(main)


class _MarkedResponses:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        if any(message.role == "tool" for message in messages):  # type: ignore[attr-defined]
            return ModelCompletion(content=SCRIPTED_DECISION, response_id="resp_final")
        return ModelCompletion(
            content="",
            response_id="resp_tool",
            tool_calls=(
                ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}, call_id="call_live"),
            ),
        )


def test_trace_keeps_provider_ids_and_null_usage() -> None:
    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h1-latency",
            adapter=_MarkedResponses(),
            model_id="fake",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S01",
            repeat_index=0,
        )

    run = _drive(main)
    assert run.baseline.provider_response_ids == ["resp_tool", "resp_final"]  # type: ignore[attr-defined]
    assert run.baseline.usage is None  # type: ignore[attr-defined]
    assert run.baseline.model_id == "fake"  # type: ignore[attr-defined]
    assert run.baseline.execution_status == "completed"  # type: ignore[attr-defined]
    assert run.factual.provider_response_ids == ["resp_tool", "resp_final"]  # type: ignore[attr-defined]
    assert run.mutant.provider_response_ids == ["resp_tool", "resp_final"]  # type: ignore[attr-defined]
    snapshot = run.snapshot  # type: ignore[attr-defined]
    assert snapshot is not None
    document = replay_module.prefix_document(snapshot.state)
    assert "provider_response_ids" not in document
    assert snapshot.state.messages[-1].response_id == "resp_tool"
    assert document["messages"][-1]["response_id"] == "resp_tool"
    assert "resp_final" not in json.dumps(document)
