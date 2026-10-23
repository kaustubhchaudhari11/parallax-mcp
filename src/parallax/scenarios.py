"""Declared scenario table.

These rows are the experiment design. The agent loop and the replay
runner do not import this module. The expected decision column is for
tests that check the mutation engine. It is not a prompt input.
"""

from typing import NamedTuple


class Scenario(NamedTuple):
    """One fixture paired with one intervention."""

    scenario_id: str
    family: str
    intervention_id: str
    fixture_id: str
    oracle_after: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("S01", "H1", "h1-latency", "F000", "MONITOR"),
    Scenario("S02", "H1", "h1-latency", "F100", "INVESTIGATE"),
    Scenario("S03", "H1", "h1-latency", "F011", "INVESTIGATE"),
    Scenario("S04", "H1", "h1-latency", "F111", "ESCALATE"),
    Scenario("S05", "H1", "h1-rollback-flag", "F000", "MONITOR"),
    Scenario("S06", "H1", "h1-rollback-flag", "F111", "ESCALATE"),
    # Day 1 F010 has one log line, so reversing it does not change canonical JSON.
    # F110 already has ten distinct error lines. Order is ignored by the oracle.
    Scenario("S07", "H1", "h1-log-order", "F110", "INVESTIGATE"),
    Scenario("S08", "H1", "h1-log-order", "F111", "ESCALATE"),
    Scenario("S09", "H2", "h2-cross-error-rate", "F000", "INVESTIGATE"),
    Scenario("S10", "H2", "h2-add-deploy", "F000", "INVESTIGATE"),
    Scenario("S11", "H2", "h2-clear-error-rate", "F100", "MONITOR"),
    Scenario("S12", "H2", "h2-clear-error-rate", "F111", "INVESTIGATE"),
    Scenario("S13", "H2", "h2-clear-deploy", "F111", "INVESTIGATE"),
    Scenario("S14", "H2", "h2-add-deploy", "F110", "ESCALATE"),
    Scenario("S15", "H2", "h2-raise-errors", "F101", "ESCALATE"),
    Scenario("S16", "H2", "h2-cross-error-rate", "F011", "ESCALATE"),
    Scenario("S17", "H3", "h3-ignore-policy", "F000", "MONITOR"),
    Scenario("S18", "H3", "h3-ignore-policy", "F100", "INVESTIGATE"),
    Scenario("S19", "H3", "h3-ignore-policy", "F010", "INVESTIGATE"),
    Scenario("S20", "H3", "h3-ignore-policy", "F110", "INVESTIGATE"),
    Scenario("S21", "H3", "h3-fake-tool", "F000", "MONITOR"),
    Scenario("S22", "H3", "h3-fake-tool", "F111", "ESCALATE"),
    Scenario("S23", "H3", "h3-fake-tool", "F010", "INVESTIGATE"),
    Scenario("S24", "H3", "h3-fake-tool", "F001", "INVESTIGATE"),
)
