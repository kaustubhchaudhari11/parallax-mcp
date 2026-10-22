"""Decision-table tests. This module imports the oracle and not the server."""

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from parallax.fixtures import FIXTURES
from parallax.oracle import Decision, decide, predicate_d, predicate_l, predicate_m
from parallax.schemas import (
    DeploymentInfo,
    IncidentLogs,
    LogLine,
    MetricWindow,
    ServiceMetrics,
    Severity,
)

ROOT = Path(__file__).resolve().parents[1]
_WINDOW = MetricWindow(
    start=datetime(2026, 10, 21, 12, 0, tzinfo=UTC),
    end=datetime(2026, 10, 21, 12, 15, tzinfo=UTC),
)

# Independent copy of the protocol table. Production code is checked against
# these literals, not against a table it exports.
_TABLE: dict[tuple[bool, bool, bool], str] = {
    (False, False, False): "MONITOR",
    (True, False, False): "INVESTIGATE",
    (False, True, False): "INVESTIGATE",
    (False, False, True): "INVESTIGATE",
    (True, True, False): "INVESTIGATE",
    (True, False, True): "INVESTIGATE",
    (False, True, True): "INVESTIGATE",
    (True, True, True): "ESCALATE",
}


def _metrics(rate: int, *, latency: int = 20, requests: int = 100) -> ServiceMetrics:
    return ServiceMetrics(
        window=_WINDOW,
        error_rate_bps=rate,
        request_count=requests,
        latency_p99_ms=latency,
    )


def _line(second: int, severity: Severity, message: str) -> LogLine:
    return LogLine(
        timestamp=datetime(2026, 10, 21, 12, 1, second, tzinfo=UTC),
        severity=severity,
        message=message,
    )


def _consistent_logs(error_count: int, highest: Severity) -> IncidentLogs:
    if highest == Severity.ERROR:
        lines = tuple(_line(index, Severity.ERROR, f"failed {index}") for index in range(error_count))
    else:
        lines = (_line(0, highest, "note"),)
    return IncidentLogs(highest_severity=highest, error_count=error_count, lines=lines)


def _deployment(flag: bool, **overrides: object) -> DeploymentInfo:
    payload: dict[str, object] = {
        "version": "1.0.0",
        "deployed_within_window": flag,
        "rollback_available": False,
        "change_summary": "ship the image",
    }
    payload.update(overrides)
    return DeploymentInfo.model_validate(payload)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize(("bits", "expected"), list(_TABLE.items()))
def test_decision_table(bits: tuple[bool, bool, bool], expected: str) -> None:
    metrics_breach, logs_breach, deploy_breach = bits
    metrics = _metrics(500 if metrics_breach else 499)
    logs = _consistent_logs(10 if logs_breach else 0, Severity.ERROR if logs_breach else Severity.INFO)
    deployment = _deployment(deploy_breach)
    decision = decide(metrics, logs, deployment)
    assert isinstance(decision, Decision)
    assert decision == expected
    assert (predicate_m(metrics), predicate_l(logs), predicate_d(deployment)) == bits


def test_fixtures_cover_every_cell() -> None:
    seen: set[tuple[bool, bool, bool]] = set()
    for fixture in FIXTURES.values():
        bits = (
            predicate_m(fixture.metrics),
            predicate_l(fixture.logs),
            predicate_d(fixture.deployment),
        )
        seen.add(bits)
        assert decide(fixture.metrics, fixture.logs, fixture.deployment) == _TABLE[bits]
    assert seen == set(_TABLE)


def test_rate_boundary_belongs_to_the_breach() -> None:
    quiet = _consistent_logs(0, Severity.INFO)
    deployment = _deployment(False)
    assert predicate_m(_metrics(499)) is False
    assert predicate_m(_metrics(500)) is True
    assert predicate_m(_metrics(501)) is True
    assert decide(_metrics(499), quiet, deployment) == "MONITOR"
    assert decide(_metrics(500), quiet, deployment) == "INVESTIGATE"


def test_log_boundaries() -> None:
    metrics = _metrics(0)
    deployment = _deployment(False)
    warn = _consistent_logs(0, Severity.WARN)
    nine_errors = _consistent_logs(9, Severity.ERROR)
    ten_errors = _consistent_logs(10, Severity.ERROR)
    assert predicate_l(warn) is False
    assert predicate_l(nine_errors) is True
    assert predicate_l(ten_errors) is True
    assert decide(metrics, warn, deployment) == "MONITOR"
    assert decide(metrics, nine_errors, deployment) == "INVESTIGATE"
    assert decide(metrics, ten_errors, deployment) == "INVESTIGATE"

    count_without_severity = IncidentLogs.model_construct(
        highest_severity=Severity.WARN,
        error_count=10,
        lines=(_line(0, Severity.WARN, "note"),),
    )
    below_count_without_severity = IncidentLogs.model_construct(
        highest_severity=Severity.WARN,
        error_count=9,
        lines=(_line(0, Severity.WARN, "note"),),
    )
    assert predicate_l(count_without_severity) is True
    assert predicate_l(below_count_without_severity) is False


def test_deploy_flag_must_be_boolean_true() -> None:
    metrics = _metrics(0)
    logs = _consistent_logs(0, Severity.INFO)
    assert predicate_d(_deployment(False)) is False
    assert predicate_d(_deployment(True)) is True
    assert decide(metrics, logs, _deployment(False)) == "MONITOR"
    assert decide(metrics, logs, _deployment(True)) == "INVESTIGATE"
    numeric_flag = DeploymentInfo.model_construct(
        version="1.0.0",
        deployed_within_window=1,
        rollback_available=False,
        change_summary="ship the image",
    )
    assert predicate_d(numeric_flag) is False


def test_ignored_fields_do_not_change_the_decision() -> None:
    metrics = _metrics(499, latency=8000, requests=1)
    logs = _consistent_logs(0, Severity.WARN)
    deployment = _deployment(False, rollback_available=True, change_summary="first", version="9.9.9")
    baseline = decide(metrics, logs, deployment)
    assert baseline == "MONITOR"
    louder = metrics.model_copy(update={"latency_p99_ms": 10**9, "request_count": 10**9})
    shifted = metrics.model_copy(
        update={
            "window": MetricWindow(
                start=datetime(2020, 1, 1, tzinfo=UTC),
                end=datetime(2020, 1, 2, tzinfo=UTC),
            )
        }
    )
    reordered = logs.model_copy(update={"lines": tuple(reversed(logs.lines))})
    rewritten = logs.model_copy(
        update={"lines": (_line(0, Severity.WARN, "a completely different sentence"),)}
    )
    other_deploy = deployment.model_copy(
        update={
            "version": "0.0.1",
            "rollback_available": False,
            "change_summary": "unrelated text",
        }
    )
    assert decide(louder, logs, deployment) == baseline
    assert decide(shifted, logs, deployment) == baseline
    assert decide(metrics, reordered, deployment) == baseline
    assert decide(metrics, rewritten, deployment) == baseline
    assert decide(metrics, logs, other_deploy) == baseline


def test_oracle_source_does_not_import_mcp_or_the_server() -> None:
    modules = _imported_modules(ROOT / "src" / "parallax" / "oracle.py")
    assert modules == {"enum", "parallax.schemas"}
    test_modules = _imported_modules(Path(__file__))
    assert "parallax.server" not in test_modules
    assert "parallax.agent" not in test_modules
