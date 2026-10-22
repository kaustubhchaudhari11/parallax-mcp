"""Fixture store validation. Payloads stay immutable and label-free."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from parallax.fixtures import FIXTURE_IDS, FIXTURES, IncidentFixture, get_fixture
from parallax.schemas import (
    DeploymentInfo,
    IncidentLogs,
    LogLine,
    MetricWindow,
    ServiceMetrics,
    Severity,
    UnknownIncidentError,
)

FORBIDDEN_KEYS = frozenset(
    {"decision", "label", "oracle", "gold", "expected_decision", "oracle_decision"}
)
DECISION_TOKENS = ("MONITOR", "INVESTIGATE", "ESCALATE")


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def _assert_unlabeled(value: object) -> None:
    assert _keys(value).isdisjoint(FORBIDDEN_KEYS)
    text = json.dumps(value)
    for token in DECISION_TOKENS:
        assert token not in text
    assert "oracle" not in text


def test_eight_ids_and_lookup() -> None:
    assert FIXTURE_IDS == ("F000", "F100", "F010", "F001", "F110", "F101", "F011", "F111")
    assert tuple(FIXTURES) == FIXTURE_IDS
    assert get_fixture("F000") is FIXTURES["F000"]
    assert get_fixture("f000") is None
    assert get_fixture("F000 ") is None
    assert get_fixture("missing") is None


def test_store_and_models_are_immutable() -> None:
    with pytest.raises(TypeError):
        FIXTURES["F999"] = FIXTURES["F000"]  # type: ignore[index]
    fixture = FIXTURES["F000"]
    with pytest.raises(ValidationError):
        fixture.metrics.error_rate_bps = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        fixture.logs.lines.append(fixture.logs.lines[0])  # type: ignore[attr-defined]
    broken = fixture.logs.model_dump(mode="json")
    broken["error_count"] = 4
    with pytest.raises(ValidationError):
        IncidentLogs.model_validate(broken)
    assert fixture.logs.error_count == 0


def test_record_has_no_decision_field() -> None:
    assert set(IncidentFixture.__dataclass_fields__) == {
        "incident_id",
        "metrics",
        "logs",
        "deployment",
    }


def test_boundary_values_are_in_the_fixtures() -> None:
    assert FIXTURES["F000"].metrics.error_rate_bps == 499
    assert FIXTURES["F100"].metrics.error_rate_bps == 500
    assert FIXTURES["F110"].metrics.error_rate_bps == 500
    assert FIXTURES["F000"].logs.error_count == 0
    assert FIXTURES["F000"].logs.highest_severity == Severity.INFO
    assert FIXTURES["F100"].logs.highest_severity == Severity.WARN
    assert FIXTURES["F100"].logs.error_count == 0
    assert FIXTURES["F010"].logs.error_count == 1
    assert FIXTURES["F010"].logs.highest_severity == Severity.ERROR
    assert FIXTURES["F110"].logs.error_count == 10
    assert len(FIXTURES["F110"].logs.lines) == 10
    assert FIXTURES["F111"].logs.error_count == 10
    assert len(FIXTURES["F111"].logs.lines) == 11
    assert FIXTURES["F000"].deployment.deployed_within_window is False
    assert FIXTURES["F001"].deployment.deployed_within_window is True
    assert FIXTURES["F000"].metrics.latency_p99_ms == 8000


def test_counters_match_lines_and_timestamps_sit_in_the_window() -> None:
    for fixture in FIXTURES.values():
        logs = fixture.logs
        error_lines = [line for line in logs.lines if line.severity == Severity.ERROR]
        assert logs.error_count == len(error_lines)
        highest = max(logs.lines, key=lambda line: list(Severity).index(line.severity)).severity
        assert logs.highest_severity == highest
        window = fixture.metrics.window
        for line in logs.lines:
            assert window.start <= line.timestamp < window.end
        for model in (fixture.metrics, fixture.logs, fixture.deployment):
            dumped = model.model_dump(mode="json")
            restored = type(model).model_validate(dumped)
            assert restored == model
            _assert_unlabeled(dumped)


def test_documented_field_sets() -> None:
    metrics = FIXTURES["F111"].metrics.model_dump(mode="json")
    logs = FIXTURES["F111"].logs.model_dump(mode="json")
    deployment = FIXTURES["F111"].deployment.model_dump(mode="json")
    assert set(metrics) == {"window", "error_rate_bps", "request_count", "latency_p99_ms"}
    assert set(metrics["window"]) == {"start", "end"}
    assert set(logs) == {"highest_severity", "error_count", "lines"}
    assert set(logs["lines"][0]) == {"timestamp", "severity", "message"}
    assert set(deployment) == {
        "version",
        "deployed_within_window",
        "rollback_available",
        "change_summary",
    }
    assert isinstance(metrics["error_rate_bps"], int)
    assert isinstance(metrics["latency_p99_ms"], int)
    assert isinstance(deployment["deployed_within_window"], bool)


def test_schema_rejects_invalid_observations() -> None:
    window = FIXTURES["F000"].metrics.window
    with pytest.raises(ValidationError):
        ServiceMetrics(
            window=window,
            error_rate_bps=500.0,  # type: ignore[arg-type]
            request_count=1,
            latency_p99_ms=1,
        )
    with pytest.raises(ValidationError):
        ServiceMetrics(
            window=window,
            error_rate_bps=True,  # type: ignore[arg-type]
            request_count=1,
            latency_p99_ms=1,
        )
    with pytest.raises(ValidationError):
        ServiceMetrics(
            window=window,
            error_rate_bps=1,
            request_count=1,
            latency_p99_ms=1,
            decision="MONITOR",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        MetricWindow(
            start=datetime(2026, 10, 21, 12, 0),
            end=datetime(2026, 10, 21, 12, 15, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        MetricWindow(
            start=datetime(2026, 10, 21, 12, 15, tzinfo=UTC),
            end=datetime(2026, 10, 21, 12, 0, tzinfo=UTC),
        )
    line = FIXTURES["F000"].logs.lines[0]
    with pytest.raises(ValidationError):
        IncidentLogs(highest_severity=Severity.WARN, error_count=0, lines=())
    with pytest.raises(ValidationError):
        IncidentLogs(highest_severity=Severity.WARN, error_count=10, lines=(line,))
    with pytest.raises(ValidationError):
        IncidentLogs(highest_severity=Severity.ERROR, error_count=0, lines=(line,))
    with pytest.raises(ValidationError):
        DeploymentInfo(
            version="",
            deployed_within_window=False,
            rollback_available=False,
            change_summary="x",
        )
    with pytest.raises(ValidationError):
        LogLine(timestamp=line.timestamp, severity=Severity.INFO, message="")
    error = UnknownIncidentError(error="unknown_incident", incident_id="missing")
    assert error.model_dump(mode="json") == {"error": "unknown_incident", "incident_id": "missing"}
    with pytest.raises(ValidationError):
        UnknownIncidentError(error="other", incident_id="missing")  # type: ignore[arg-type]
