"""Mutation legality. These tests do not call a model or the MCP server."""

import json

from parallax.fixtures import FIXTURES, IncidentFixture
from parallax.mutate import (
    H3_FAKE_TOOL,
    H3_IGNORE_POLICY,
    INTERVENTIONS,
    LARGE_LATENCY_MS,
    MutationResult,
    apply_declared_edit,
    apply_intervention,
    canonical_observation_changed,
    h2_clear_error_rate,
)
from parallax.observe import canonical_json
from parallax.oracle import decide
from parallax.scenarios import SCENARIOS
from parallax.schemas import IncidentLogs

_EXPECTED_IDS = (
    "h1-log-order",
    "h1-rollback-flag",
    "h1-latency",
    "h2-cross-error-rate",
    "h2-add-deploy",
    "h2-clear-deploy",
    "h2-clear-error-rate",
    "h2-raise-errors",
    "h3-ignore-policy",
    "h3-fake-tool",
)


def _dump(fixture: IncidentFixture) -> tuple[object, object, object]:
    return (
        fixture.metrics.model_dump(mode="json"),
        fixture.logs.model_dump(mode="json"),
        fixture.deployment.model_dump(mode="json"),
    )


def test_registry_matches_the_protocol_ids() -> None:
    assert tuple(INTERVENTIONS) == _EXPECTED_IDS
    assert "h1-key-order" not in INTERVENTIONS
    assert "h1-whitespace" not in INTERVENTIONS
    assert "propose_action" not in INTERVENTIONS


def test_latency_constant_is_outside_every_fixture() -> None:
    assert all(fixture.metrics.latency_p99_ms != LARGE_LATENCY_MS for fixture in FIXTURES.values())


def test_day1_f010_log_order_is_not_model_visible() -> None:
    logs = FIXTURES["F010"].logs
    assert len(logs.lines) == 1
    assert logs.lines[0].message == "inventory read failed once"
    assert logs.error_count == 1
    result = apply_intervention(FIXTURES["F010"], "h1-log-order")
    assert result.ok is False
    assert result.mutated_visible is None
    assert result.error == "canonical observation did not change"


def test_s07_reorders_existing_f110_lines() -> None:
    scenario = next(row for row in SCENARIOS if row.scenario_id == "S07")
    assert (scenario.family, scenario.intervention_id, scenario.fixture_id) == ("H1", "h1-log-order", "F110")
    logs = FIXTURES["F110"].logs
    assert len(logs.lines) == 10
    result = apply_intervention(FIXTURES["F110"], "h1-log-order")
    assert result.ok
    assert result.predicates_before == result.predicates_after == (True, True, False)
    assert result.oracle_before == result.oracle_after == "INVESTIGATE"
    assert result.original_visible != result.mutated_visible
    assert result.mutated is not None
    original_messages = [line.message for line in logs.lines]
    mutated_messages = [line["message"] for line in result.mutated["lines"]]
    assert mutated_messages == list(reversed(original_messages))
    assert mutated_messages != original_messages
    assert result.mutated["error_count"] == 10
    assert result.mutated["highest_severity"] == "error"


def test_every_scenario_is_a_legal_edit() -> None:
    assert tuple(scenario.scenario_id for scenario in SCENARIOS) == tuple(f"S{index:02d}" for index in range(1, 25))
    for scenario in SCENARIOS:
        intervention = INTERVENTIONS[scenario.intervention_id]
        assert intervention.family == scenario.family
        result = apply_intervention(FIXTURES[scenario.fixture_id], scenario.intervention_id)
        assert result.ok, (scenario.scenario_id, result.error)
        assert result.oracle_after == scenario.oracle_after
        assert result.mutated_visible is not None
        assert result.original_visible != result.mutated_visible
        if scenario.family in {"H1", "H3"}:
            assert result.predicates_before == result.predicates_after
        if scenario.family == "H2":
            assert result.oracle_before != result.oracle_after
        if scenario.intervention_id == "h3-ignore-policy":
            assert result.mutated_visible is not None
            assert result.mutated_visible.count(H3_IGNORE_POLICY) == 1
        if scenario.intervention_id == "h3-fake-tool":
            assert result.mutated_visible is not None
            assert result.mutated_visible.count(H3_FAKE_TOOL) == 1


def test_edits_are_pure_and_illegal_outputs_are_withheld() -> None:
    for fixture in FIXTURES.values():
        before = _dump(fixture)
        original_decision = decide(fixture.metrics, fixture.logs, fixture.deployment)
        for intervention in INTERVENTIONS.values():
            result = apply_intervention(fixture, intervention.intervention_id)
            assert _dump(fixture) == before
            assert decide(fixture.metrics, fixture.logs, fixture.deployment) == original_decision
            _assert_family_rules(result, intervention.family, intervention.required_text)


def test_key_order_and_whitespace_do_not_change_canonical_json() -> None:
    payload = FIXTURES["F000"].metrics.model_dump(mode="json")
    reordered = {key: payload[key] for key in reversed(payload)}
    assert list(reordered) != list(payload)
    assert not canonical_observation_changed(payload, reordered)
    pretty = json.dumps(payload, indent=4)
    assert pretty != canonical_json(payload)
    assert not canonical_observation_changed(payload, json.loads(pretty))
    identity = apply_declared_edit(
        FIXTURES["F000"],
        family="H1",
        tool_name="get_metrics",
        edit=lambda model: model,
        intervention_id="h1-key-order",
    )
    assert identity.ok is False
    assert identity.mutated_visible is None
    assert identity.error is not None
    assert "canonical" in identity.error


def test_schema_invalid_edit_is_an_engine_error() -> None:
    broken = apply_intervention(
        FIXTURES["F000"],
        "h1-latency",
        captured={"error_rate_bps": "nope"},
    )
    assert broken.ok is False
    assert broken.mutated_visible is None
    assert broken.error is not None
    assert "schema validation failed" in broken.error

    def smash(logs: IncidentLogs) -> IncidentLogs:
        return IncidentLogs.model_construct(
            highest_severity=logs.highest_severity,
            error_count=99,
            lines=logs.lines,
        )

    smashed = apply_declared_edit(
        FIXTURES["F000"],
        family="H1",
        tool_name="get_logs",
        edit=smash,
    )
    assert smashed.ok is False
    assert smashed.mutated is None
    assert smashed.mutated_visible is None


def test_h2_that_does_not_move_the_table_is_withheld() -> None:
    result = apply_intervention(FIXTURES["F010"], "h2-clear-error-rate")
    assert result.ok is False
    assert result.mutated_visible is None
    assert result.error == "table decision did not change"
    edited = h2_clear_error_rate(FIXTURES["F010"].metrics)
    assert edited.error_rate_bps == 499
    assert '"error_rate_bps":499' in canonical_json(edited.model_dump(mode="json"))


def test_one_line_reversal_is_not_an_intervention() -> None:
    result = apply_intervention(FIXTURES["F000"], "h1-log-order")
    assert len(FIXTURES["F000"].logs.lines) == 1
    assert result.ok is False
    assert result.mutated_visible is None
    assert result.error == "canonical observation did not change"


def test_unknown_intervention_is_rejected() -> None:
    result = apply_intervention(FIXTURES["F000"], "h1-whitespace")
    assert result.ok is False
    assert result.mutated_visible is None


def _assert_family_rules(result: MutationResult, family: str, required_text: str | None) -> None:
    if not result.ok:
        assert result.mutated is None
        assert result.mutated_visible is None
        assert result.oracle_after is None
        return
    assert result.original_visible != result.mutated_visible
    assert result.predicates_after is not None
    if family in {"H1", "H3"}:
        assert result.predicates_before == result.predicates_after
        assert result.oracle_before == result.oracle_after
    if family == "H2":
        assert result.oracle_before != result.oracle_after
    if required_text is not None:
        assert result.mutated_visible is not None
        assert result.mutated_visible.count(required_text) == 1