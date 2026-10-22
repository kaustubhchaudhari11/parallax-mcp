"""Deterministic incident decision rule.

This module is imported by tests and, later, by the runner after a model
response exists. The MCP server and the agent loop must not import it.
The function reads explicit fields only. It does not recompute those
fields from log text, latency, or window timestamps.
"""

from enum import StrEnum

from parallax.schemas import DeploymentInfo, IncidentLogs, ServiceMetrics, Severity

ERROR_RATE_BREACH_BPS = 500
ERROR_COUNT_BREACH = 10


class Decision(StrEnum):
    """Closed agent decision. The server never returns this value."""

    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"
    ESCALATE = "ESCALATE"


def predicate_m(metrics: ServiceMetrics) -> bool:
    """True when the error rate is at or above 500 basis points."""

    return metrics.error_rate_bps >= ERROR_RATE_BREACH_BPS


def predicate_l(logs: IncidentLogs) -> bool:
    """True when the error count is at least 10 or the severity is error."""

    return logs.error_count >= ERROR_COUNT_BREACH or logs.highest_severity == Severity.ERROR


def predicate_d(deployment: DeploymentInfo) -> bool:
    """True only when the deployment flag is the boolean True."""

    return deployment.deployed_within_window is True


def decide(metrics: ServiceMetrics, logs: IncidentLogs, deployment: DeploymentInfo) -> Decision:
    """Map one metrics, logs, and deployment triple to a single decision.

    MONITOR when none of M, L, D hold. ESCALATE when all three hold.
    INVESTIGATE for every other combination.
    """

    breached = (
        predicate_m(metrics),
        predicate_l(logs),
        predicate_d(deployment),
    )
    if breached == (False, False, False):
        return Decision.MONITOR
    if breached == (True, True, True):
        return Decision.ESCALATE
    return Decision.INVESTIGATE
