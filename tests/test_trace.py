"""JSONL ledger records. No model is called."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from parallax.mutate import H3_FAKE_TOOL
from parallax.observe import canonical_json
from parallax.trace import TraceRecord, append_trace, read_traces, research_eligible


def _row(**overrides: object) -> TraceRecord:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "pair_id": "pair-1",
        "arm": "baseline",
        "incident_id": "F000",
        "fixture_hash": "abc",
        "intervention_id": None,
        "model_id": "fake",
        "model_params": {"temperature": 0},
        "prompt_hash": "def",
        "sdk_version": "2.3.0",
        "negotiated_protocol_version": "2026-07-28",
        "snapshot_id": None,
        "phase": "harness",
        "scenario_id": None,
        "repeat_index": 0,
        "tool_calls": [],
        "observation_original": None,
        "observation_mutated": None,
        "model_visible_observations": [],
        "usage": None,
        "decision_raw": None,
        "decision_parsed": None,
        "action_observed": None,
        "parse_ok": False,
        "oracle_decision": "MONITOR",
        "predicate_m": False,
        "predicate_l": False,
        "predicate_d": False,
        "flags": [],
        "error": None,
        "started_at": "2026-10-22T00:00:00+00:00",
        "ended_at": "2026-10-22T00:00:01+00:00",
        "prefix_sha256": None,
        "intervention_source_sha256": "abc",
    }
    payload.update(overrides)
    return TraceRecord.model_validate(payload)


def test_append_writes_one_line_and_reads_it_back(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "harness.jsonl"
    instruction = canonical_json(
        {
            "change_summary": f"refresh catalog copy {H3_FAKE_TOOL}",
            "deployed_within_window": False,
            "rollback_available": True,
            "version": "1.2.0",
        }
    )
    first = _row(observation_original=instruction)
    second = _row(run_id="run-2", arm="mutant", usage={"input_tokens": 2, "output_tokens": 3})
    append_trace(path, first)
    append_trace(path, second)
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 2
    assert "\n\n" not in text
    rows = read_traces(path)
    assert [row.run_id for row in rows] == ["run-1", "run-2"]
    assert rows[0].usage is None
    assert rows[1].usage == {"input_tokens": 2, "output_tokens": 3}
    assert H3_FAKE_TOOL in rows[0].observation_original  # type: ignore[operator]
    assert json.loads(text.splitlines()[0])["model_id"] == "fake"


def test_torn_trace_tail_is_dropped_on_read(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    append_trace(path, _row())
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"torn":')
    rows = read_traces(path)
    assert [row.run_id for row in rows] == ["run-1"]
    assert '{"torn":' not in path.read_text(encoding="utf-8")
    append_trace(path, _row(run_id="run-2"))
    assert [row.run_id for row in read_traces(path)] == ["run-1", "run-2"]


def test_unparsed_row_cannot_carry_a_decision() -> None:
    with pytest.raises(ValidationError):
        _row(parse_ok=False, decision_parsed="MONITOR")


def test_partial_usage_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(usage={"input_tokens": 1})


def test_labeled_observation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(observation_original='{"oracle":"MONITOR"}')
    with pytest.raises(ValidationError):
        _row(model_visible_observations=['{"decision":"MONITOR"}'])


def test_unknown_flag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(flags=["hypothesis_success"])


def test_harness_rows_stay_out_of_research_eligibility() -> None:
    harness = _row(
        phase="harness",
        model_id="fake",
        execution_status="completed",
        parse_ok=True,
        decision_parsed="ESCALATE",
        oracle_decision="MONITOR",
    )
    pilot = _row(phase="pilot", execution_status="completed", parse_ok=True, decision_parsed="MONITOR")
    exploratory = _row(
        phase="exploratory",
        execution_status="completed",
        parse_ok=True,
        decision_parsed="ESCALATE",
        oracle_decision="MONITOR",
    )
    assert research_eligible(harness) is False
    assert research_eligible(pilot) is True
    assert research_eligible(exploratory) is True
    assert exploratory.decision_parsed != exploratory.oracle_decision


def test_unknown_execution_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(execution_status="oracle_mismatch")
