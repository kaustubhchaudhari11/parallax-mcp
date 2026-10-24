"""Append-only JSONL trial records.

A writer flushes one complete line or no line. Token usage stays null
when the adapter does not report it. This module does not score a
hypothesis rate.
"""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from parallax.capture import label_key_overlap

Arm = Literal["baseline", "factual_replay", "mutant"]
Phase = Literal["pilot", "exploratory", "harness"]
ExecutionStatus = Literal[
    "completed",
    "invalid_final",
    "budget_exhausted",
    "invalid_tool",
    "tool_error",
    "model_error",
]
TRACE_SCHEMA_VERSION = 1
EXECUTION_STATUSES = frozenset(
    {
        "completed",
        "invalid_final",
        "budget_exhausted",
        "invalid_tool",
        "tool_error",
        "model_error",
    }
)
RESEARCH_PHASES = frozenset({"pilot", "exploratory"})
FLAG_NAMES = frozenset(
    {
        "parse_failure",
        "budget_exhausted",
        "engine_error",
        "provider_error",
        "intervention_unobserved",
        "tool_error",
    }
)


def utc_now() -> str:
    """Timezone-aware timestamp for a trace row."""

    return datetime.now(UTC).isoformat()


class ToolCallRecord(BaseModel):
    """One tool request recorded on a trial."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]
    latency_ms: float | None = None
    is_error: bool | None = None
    raw: dict[str, Any] | None = None
    model_visible_observation: str | None = None
    model_visible_sha256: str | None = None
    dispatched: bool
    source: Literal["base", "overlay", "rejected"]


class ProviderErrorMetadata(BaseModel):
    """Sanitized provider failure fields. Request and response bodies are excluded."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["request", "response_parsing", "continuation", "agent_controller"]
    exception_class: str
    http_status: int | None = None
    error_code: str | None = None
    parameter: str | None = None
    request_id: str | None = None


class TraceRecord(BaseModel):
    """One ledger row. Harness rows are control-flow fixtures, not results."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pair_id: str
    arm: Arm
    incident_id: str
    fixture_hash: str
    intervention_id: str | None
    model_id: str
    model_params: dict[str, Any]
    prompt_hash: str
    sdk_version: str
    negotiated_protocol_version: str | None
    snapshot_id: str | None
    phase: Phase
    scenario_id: str | None
    repeat_index: int
    tool_calls: list[ToolCallRecord]
    observation_original: str | None
    observation_mutated: str | None
    model_visible_observations: list[str]
    usage: dict[str, int] | None
    decision_raw: str | None
    decision_parsed: str | None
    action_observed: str | None
    parse_ok: bool
    oracle_decision: str | None
    predicate_m: bool | None
    predicate_l: bool | None
    predicate_d: bool | None
    flags: list[str] = Field(default_factory=list)
    error: str | None
    provider_error: ProviderErrorMetadata | None = None
    started_at: str
    ended_at: str
    prefix_sha256: str | None
    intervention_source_sha256: str
    execution_status: ExecutionStatus | None = None
    provider_response_ids: list[str] = Field(default_factory=list)
    experiment_id: str | None = None

    @model_validator(mode="after")
    def row_is_internally_consistent(self) -> Self:
        unknown = [flag for flag in self.flags if flag not in FLAG_NAMES]
        if unknown:
            raise ValueError(f"unknown flags: {unknown}")
        if self.execution_status is not None and self.execution_status not in EXECUTION_STATUSES:
            raise ValueError(f"unknown execution status: {self.execution_status}")
        if self.parse_ok and self.decision_parsed is None:
            raise ValueError("a parsed decision is missing")
        if not self.parse_ok and self.decision_parsed is not None:
            raise ValueError("an unparsed row must not carry a decision")
        if self.usage is not None:
            allowed = {"input_tokens", "output_tokens"}
            if set(self.usage) != allowed or any(not isinstance(value, int) for value in self.usage.values()):
                raise ValueError("usage must report integer input_tokens and output_tokens")
        for text in (
            self.observation_original,
            self.observation_mutated,
            *self.model_visible_observations,
        ):
            _reject_labeled_observation(text)
        return self


def drop_torn_jsonl_tail(path: Path, parse_line: Callable[[str], object]) -> None:
    """Repair a crash that stopped before the closing newline.

    A file that already ends with a newline is not rewritten. A final
    line that parses gains that newline. A final line that does not
    parse is removed. Earlier lines stay as they were.
    """

    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if not text or text.endswith("\n"):
        return
    lines = text.splitlines()
    if not lines:
        return
    try:
        parse_line(lines[-1])
    except (ValidationError, ValueError):
        body = "".join(f"{line}\n" for line in lines[:-1])
    else:
        body = text + "\n"
    path.write_text(body, encoding="utf-8", newline="\n")


def append_trace(path: Path, record: TraceRecord) -> None:
    """Append one JSON object and flush it."""

    line = record.model_dump_json()
    if "\n" in line or "\r" in line:
        raise ValueError("trace record must serialize to one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_torn_jsonl_tail(path, TraceRecord.model_validate_json)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def research_eligible(record: TraceRecord) -> bool:
    """Return whether a row may enter an empirical hypothesis rate.

    Harness rows are scripted controls. This guard does not score a
    decision and does not compute a rate.
    """

    return record.phase in RESEARCH_PHASES


def read_traces(path: Path) -> list[TraceRecord]:
    """Read every complete line in a ledger file."""

    if not path.exists():
        return []
    drop_torn_jsonl_tail(path, TraceRecord.model_validate_json)
    rows: list[TraceRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(TraceRecord.model_validate_json(line))
    return rows


def _reject_labeled_observation(text: str | None) -> None:
    if text is None or not text.startswith("{"):
        return
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return
    overlap = label_key_overlap(parsed)
    if overlap:
        raise ValueError(f"observation contains label fields: {sorted(overlap)}")
