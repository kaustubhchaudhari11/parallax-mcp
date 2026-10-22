"""Pydantic contracts for synthetic incident observations.

These models are the tool payloads. They carry no decision, label, or
oracle field. Log-line checks are payload invariants. The decision rule
lives elsewhere and is not applied here.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)


class Severity(StrEnum):
    """Closed log severity. Rank is debug < info < warn < error."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.DEBUG: 0,
    Severity.INFO: 1,
    Severity.WARN: 2,
    Severity.ERROR: 3,
}


class IncidentModel(BaseModel):
    """Frozen observation model. Unknown fields are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MetricWindow(IncidentModel):
    """Observation window. Both bounds are timezone-aware."""

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("window end must be after window start")
        return self


class ServiceMetrics(IncidentModel):
    """Metrics returned by get_metrics."""

    window: MetricWindow
    error_rate_bps: StrictInt = Field(
        ge=0,
        description="Integer error rate in basis points. 500 means 5.00%.",
    )
    request_count: StrictInt = Field(ge=0)
    latency_p99_ms: StrictInt = Field(ge=0)


class LogLine(IncidentModel):
    """One synthetic log line."""

    timestamp: AwareDatetime
    severity: Severity
    message: str = Field(min_length=1)


class IncidentLogs(IncidentModel):
    """Logs returned by get_logs.

    ``error_count`` is the number of lines whose severity is error.
    ``highest_severity`` is the maximum severity present in ``lines``.
    """

    highest_severity: Severity
    error_count: StrictInt = Field(ge=0)
    lines: tuple[LogLine, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def lines_match_counters(self) -> Self:
        error_lines = sum(1 for line in self.lines if line.severity == Severity.ERROR)
        if self.error_count != error_lines:
            raise ValueError("error_count must equal the number of error lines")
        highest = max(self.lines, key=lambda line: SEVERITY_RANK[line.severity]).severity
        if self.highest_severity != highest:
            raise ValueError("highest_severity must equal the maximum line severity")
        return self


class DeploymentInfo(IncidentModel):
    """Deployment record returned by get_deployment."""

    version: str = Field(min_length=1)
    deployed_within_window: StrictBool
    rollback_available: StrictBool
    change_summary: str


class UnknownIncidentError(IncidentModel):
    """Structured tool error for an incident id absent from the fixture store."""

    error: Literal["unknown_incident"]
    incident_id: str
