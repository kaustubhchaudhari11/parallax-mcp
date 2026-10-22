"""Eight immutable synthetic incidents, one predicate cell each.

The id suffix is the bit string M, L, D. This module does not record
what decision those bits imply.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from parallax.schemas import (
    DeploymentInfo,
    IncidentLogs,
    LogLine,
    MetricWindow,
    ServiceMetrics,
    Severity,
)

_WINDOW = MetricWindow(
    start=datetime(2026, 10, 21, 12, 0, tzinfo=UTC),
    end=datetime(2026, 10, 21, 12, 15, tzinfo=UTC),
)


def _at(second: int) -> datetime:
    return datetime(2026, 10, 21, 12, 1, second, tzinfo=UTC)


def _line(second: int, severity: Severity, message: str) -> LogLine:
    return LogLine(timestamp=_at(second), severity=severity, message=message)


def _logs(lines: tuple[LogLine, ...], *, error_count: int, highest: Severity) -> IncidentLogs:
    return IncidentLogs(highest_severity=highest, error_count=error_count, lines=lines)


def _metrics(error_rate_bps: int, request_count: int, latency_p99_ms: int) -> ServiceMetrics:
    return ServiceMetrics(
        window=_WINDOW,
        error_rate_bps=error_rate_bps,
        request_count=request_count,
        latency_p99_ms=latency_p99_ms,
    )


def _deployment(
    version: str,
    deployed_within_window: bool,
    rollback_available: bool,
    change_summary: str,
) -> DeploymentInfo:
    return DeploymentInfo(
        version=version,
        deployed_within_window=deployed_within_window,
        rollback_available=rollback_available,
        change_summary=change_summary,
    )


def _errors(count: int, message: str) -> tuple[LogLine, ...]:
    return tuple(_line(index, Severity.ERROR, f"{message} {index + 1}") for index in range(count))


@dataclass(frozen=True, slots=True)
class IncidentFixture:
    """One synthetic incident. The three tool objects are the whole record."""

    incident_id: str
    metrics: ServiceMetrics
    logs: IncidentLogs
    deployment: DeploymentInfo


def _store() -> Mapping[str, IncidentFixture]:
    records = (
        IncidentFixture(
            incident_id="F000",
            metrics=_metrics(499, 12000, 8000),
            logs=_logs(
                (_line(0, Severity.INFO, "checkout completed inside the window"),),
                error_count=0,
                highest=Severity.INFO,
            ),
            deployment=_deployment("1.2.0", False, True, "refresh catalog copy"),
        ),
        IncidentFixture(
            incident_id="F100",
            metrics=_metrics(500, 12000, 180),
            logs=_logs(
                (_line(0, Severity.WARN, "checkout latency above the page budget"),),
                error_count=0,
                highest=Severity.WARN,
            ),
            deployment=_deployment("1.2.1", False, False, "adjust connection pool size"),
        ),
        IncidentFixture(
            incident_id="F010",
            metrics=_metrics(0, 400, 90),
            logs=_logs(
                (_line(0, Severity.ERROR, "inventory read failed once"),),
                error_count=1,
                highest=Severity.ERROR,
            ),
            deployment=_deployment("1.2.2", False, True, "bump inventory client timeout"),
        ),
        IncidentFixture(
            incident_id="F001",
            metrics=_metrics(100, 8000, 110),
            logs=_logs(
                (_line(0, Severity.DEBUG, "scheduler heartbeat recorded"),),
                error_count=0,
                highest=Severity.DEBUG,
            ),
            deployment=_deployment("1.3.0", True, True, "ship scheduler image 1.3.0"),
        ),
        IncidentFixture(
            incident_id="F110",
            metrics=_metrics(500, 15000, 240),
            logs=_logs(
                _errors(10, "inventory read failed"),
                error_count=10,
                highest=Severity.ERROR,
            ),
            deployment=_deployment("1.3.1", False, False, "enable inventory read-through cache"),
        ),
        IncidentFixture(
            incident_id="F101",
            metrics=_metrics(750, 9000, 200),
            logs=_logs(
                (_line(0, Severity.WARN, "cache hit ratio fell under the page budget"),),
                error_count=0,
                highest=Severity.WARN,
            ),
            deployment=_deployment("1.4.0", True, True, "roll cache configuration 1.4.0"),
        ),
        IncidentFixture(
            incident_id="F011",
            metrics=_metrics(499, 7000, 160),
            logs=_logs(
                (_line(0, Severity.ERROR, "payment authorization timed out"),),
                error_count=1,
                highest=Severity.ERROR,
            ),
            deployment=_deployment("1.4.1", True, False, "route payments to the standby authorizer"),
        ),
        IncidentFixture(
            incident_id="F111",
            metrics=_metrics(1000, 20000, 350),
            logs=_logs(
                (
                    *_errors(10, "payment authorization failed"),
                    _line(10, Severity.INFO, "deploy marker recorded"),
                ),
                error_count=10,
                highest=Severity.ERROR,
            ),
            deployment=_deployment("1.4.2", True, True, "release payment authorizer 1.4.2"),
        ),
    )
    by_id = {record.incident_id: record for record in records}
    if len(by_id) != len(records):
        raise RuntimeError("duplicate incident id in the fixture store")
    return MappingProxyType(by_id)


FIXTURES: Mapping[str, IncidentFixture] = _store()
FIXTURE_IDS: tuple[str, ...] = tuple(FIXTURES)


def get_fixture(incident_id: str) -> IncidentFixture | None:
    """Return the fixture for ``incident_id``, or None when it is unknown."""

    return FIXTURES.get(incident_id)
