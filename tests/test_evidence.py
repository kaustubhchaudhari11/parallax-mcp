"""Evidence-aware scripted control. These rows are harness checks, not model results."""

import ast
import json
from pathlib import Path

import anyio
import pytest

from parallax.agent import ChatMessage
from parallax.capture import label_key_overlap, sha256_text
from parallax.fixtures import FIXTURES
from parallax.mutate import LARGE_LATENCY_MS
from parallax.replay import first_action_divergence, first_difference, run_pair
from parallax.trace import research_eligible
from tests.harness import (
    SCRIPTED_DECISION,
    EvidenceAdapter,
    ScriptedAdapter,
    decide_from_visible,
)

PARAMS = {"temperature": 0, "seed": 1, "max_tokens": 256}


def _pack(rate: object, latency: int, severity: str, count: int, deployed: bool) -> tuple[str, str, str]:
    metrics = json.dumps({"error_rate_bps": rate, "latency_p99_ms": latency, "request_count": 3})
    logs = json.dumps({"error_count": count, "highest_severity": severity, "lines": []})
    deployment = json.dumps(
        {"change_summary": "note", "deployed_within_window": deployed, "rollback_available": False}
    )
    return metrics, logs, deployment


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


def _contents(messages: list[ChatMessage], tool_name: str) -> list[str]:
    return [message.content for message in messages if message.tool_name == tool_name]


def _run(intervention_id: str, scenario_id: str) -> object:
    async def main() -> object:
        return await run_pair(
            incident_id="F011",
            intervention_id=intervention_id,
            adapter=EvidenceAdapter("F011"),
            model_id="fake",
            model_params=PARAMS,
            step_budget=6,
            token_budget=None,
            phase="harness",
            scenario_id=scenario_id,
            repeat_index=0,
        )

    return anyio.run(main, backend="asyncio")


def test_harness_does_not_import_the_oracle_or_the_fixtures() -> None:
    path = Path(__file__).resolve().parent / "harness.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "parallax.oracle" not in modules
    assert "parallax.fixtures" not in modules
    assert "parallax.scenarios" not in modules
    source = path.read_text(encoding="utf-8")
    decision_source = source.split("def decide_from_visible", 1)[1].split("\nclass ", 1)[0]
    assert "F0" not in decision_source


def test_visible_policy_reads_evidence_and_ignores_latency() -> None:
    original = _pack(499, 160, "error", 1, True)
    crossed = _pack(500, 160, "error", 1, True)
    slower = _pack(499, LARGE_LATENCY_MS, "error", 1, True)
    assert decide_from_visible(*original) == "INVESTIGATE"
    assert decide_from_visible(*crossed) == "ESCALATE"
    assert decide_from_visible(*slower) == "INVESTIGATE"
    with pytest.raises(TypeError, match="error_rate_bps"):
        decide_from_visible(*_pack(True, 1, "info", 0, False))


def test_fixed_decision_control_ignores_the_observations() -> None:
    adapter = ScriptedAdapter("F011", ())

    async def main() -> object:
        return await adapter.complete(
            [ChatMessage(role="tool", content=_pack(0, 1, "info", 0, False)[0], tool_name="get_metrics")],
            [{"name": "get_metrics"}],
            {},
        )

    completion = anyio.run(main, backend="asyncio")
    assert completion.content == SCRIPTED_DECISION  # type: ignore[attr-defined]
    assert completion.usage is None  # type: ignore[attr-defined]


def test_evidence_adapter_adapts_when_the_error_rate_crosses_500() -> None:
    before = _stores()
    run = _run("h2-cross-error-rate", "S16")
    assert _stores() == before
    factual = run.factual_messages  # type: ignore[attr-defined]
    mutant = run.mutant_messages  # type: ignore[attr-defined]
    assert factual is not mutant
    assert factual[0] is not mutant[0]
    prefix_len = len(run.snapshot.state.messages)  # type: ignore[attr-defined]
    assert first_difference(factual, mutant) == prefix_len
    assert factual[:prefix_len] == mutant[:prefix_len]
    factual_metrics = _contents(factual, "get_metrics")
    mutant_metrics = _contents(mutant, "get_metrics")
    assert factual_metrics[0] != mutant_metrics[0]
    assert '"error_rate_bps":499' in factual_metrics[0]
    assert '"error_rate_bps":500' in mutant_metrics[0]
    assert _contents(factual, "get_logs") == _contents(mutant, "get_logs")
    assert _contents(factual, "get_deployment") == _contents(mutant, "get_deployment")
    assert run.factual.prefix_sha256 == run.mutant.prefix_sha256 == run.prefix_sha256  # type: ignore[attr-defined]
    action = first_action_divergence(factual, mutant)
    assert action is not None
    assert factual[action].role == mutant[action].role == "assistant"
    assert '"decision":"INVESTIGATE"' in factual[action].content
    assert '"decision":"ESCALATE"' in mutant[action].content
    assert run.factual.decision_parsed == "INVESTIGATE"  # type: ignore[attr-defined]
    assert run.mutant.decision_parsed == "ESCALATE"  # type: ignore[attr-defined]
    assert run.factual.oracle_decision == "INVESTIGATE"  # type: ignore[attr-defined]
    assert run.mutant.oracle_decision == "ESCALATE"  # type: ignore[attr-defined]
    assert run.factual.execution_status == run.mutant.execution_status == "completed"  # type: ignore[attr-defined]
    _assert_harness_pair(run, factual_metrics[0], mutant_metrics[0])


def test_evidence_adapter_holds_the_decision_when_only_latency_changes() -> None:
    before = _stores()
    run = _run("h1-latency", "S03")
    assert _stores() == before
    factual = run.factual_messages  # type: ignore[attr-defined]
    mutant = run.mutant_messages  # type: ignore[attr-defined]
    prefix_len = len(run.snapshot.state.messages)  # type: ignore[attr-defined]
    assert first_difference(factual, mutant) == prefix_len
    assert factual[:prefix_len] == mutant[:prefix_len]
    factual_metrics = _contents(factual, "get_metrics")
    mutant_metrics = _contents(mutant, "get_metrics")
    assert factual_metrics[0] != mutant_metrics[0]
    assert f'"latency_p99_ms":{LARGE_LATENCY_MS}' in mutant_metrics[0]
    assert f'"latency_p99_ms":{LARGE_LATENCY_MS}' not in factual_metrics[0]
    assert '"error_rate_bps":499' in factual_metrics[0]
    assert '"error_rate_bps":499' in mutant_metrics[0]
    assert first_action_divergence(factual, mutant) is None
    assert run.factual.decision_parsed == run.mutant.decision_parsed == "INVESTIGATE"  # type: ignore[attr-defined]
    assert run.factual.oracle_decision == run.mutant.oracle_decision == "INVESTIGATE"  # type: ignore[attr-defined]
    metric_records = [call for call in run.mutant.tool_calls if call.name == "get_metrics"]  # type: ignore[attr-defined]
    assert metric_records[0].source == "overlay"
    assert metric_records[0].raw["structured_content"]["latency_p99_ms"] == 160  # type: ignore[index]
    assert str(LARGE_LATENCY_MS) in metric_records[0].model_visible_observation
    _assert_harness_pair(run, factual_metrics[0], mutant_metrics[0])


def _assert_harness_pair(run: object, original: str, mutated: str) -> None:
    assert run.baseline.phase == run.factual.phase == run.mutant.phase == "harness"  # type: ignore[attr-defined]
    assert run.baseline.model_id == "fake"  # type: ignore[attr-defined]
    assert run.baseline.usage is None  # type: ignore[attr-defined]
    assert run.factual.usage is None  # type: ignore[attr-defined]
    assert run.mutant.usage is None  # type: ignore[attr-defined]
    assert research_eligible(run.baseline) is False  # type: ignore[attr-defined]
    assert research_eligible(run.factual) is False  # type: ignore[attr-defined]
    assert research_eligible(run.mutant) is False  # type: ignore[attr-defined]
    assert run.baseline.pair_id == run.factual.pair_id == run.mutant.pair_id  # type: ignore[attr-defined]
    assert len({run.baseline.run_id, run.factual.run_id, run.mutant.run_id}) == 3  # type: ignore[attr-defined]
    assert run.baseline.intervention_id is None  # type: ignore[attr-defined]
    assert run.factual.intervention_id == run.mutant.intervention_id  # type: ignore[attr-defined]
    assert run.mutant.observation_original == original  # type: ignore[attr-defined]
    assert run.mutant.observation_mutated == mutated  # type: ignore[attr-defined]
    assert sha256_text(original) != sha256_text(mutated)
    for row in (run.baseline, run.factual, run.mutant):  # type: ignore[attr-defined]
        for text in (row.observation_original, row.observation_mutated, *row.model_visible_observations):
            if isinstance(text, str) and text.startswith("{"):
                assert label_key_overlap(json.loads(text)) == set()
