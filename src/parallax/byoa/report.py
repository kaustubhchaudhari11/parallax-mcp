"""Versioned BYOA report. It is not a PARALLAX pilot or X-Ray result."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from parallax.byoa.contract import ComparisonMode, FrameworkName, Verdict
from parallax.capture import sha256_text
from parallax.observe import canonical_json

MAX_REPORT_BYTES = 256_000
SECRET_MARKERS = ("sk-", "api_key", "bearer ", "authorization")
POPULATION = "byoa"


class ByoaReportError(ValueError):
    """A report file cannot be shown."""


class ByoaReport(BaseModel):
    """One local evaluation. Populations stay separate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    population: Literal["byoa"] = POPULATION
    run_id: str
    created_at: str
    task_id: str
    agent_framework: FrameworkName
    framework_version: str
    model_id: str | None
    factory_fingerprint: str
    contract_hash: str
    evidence_source: str
    modified_field: str
    original_evidence_sha256: str
    modified_evidence_sha256: str
    factual_decision: str | None
    counterfactual_decision: str | None
    expected_behavior: str
    verdict: Verdict
    execution_valid: bool
    comparison_mode: ComparisonMode
    shared_checkpoint: bool
    checkpoint_id: str | None
    model_requests: int = Field(ge=0)
    tool_requests: int = Field(ge=0)
    token_usage: dict[str, int] | None
    errors: list[str]
    limitations: list[str]


def contract_hash(payload: dict[str, object]) -> str:
    """Hash the spec body. The output directory is not part of the identity."""

    body = dict(payload)
    body.pop("output_dir", None)
    return sha256_text(canonical_json(body))


def evidence_hash(evidence: dict[str, object]) -> str:
    """Hash one evidence object."""

    return sha256_text(canonical_json(evidence))


def utc_now() -> str:
    """UTC timestamp for a report. It is not part of the run id."""

    return datetime.now(UTC).isoformat()


def public_error(exc: BaseException) -> str:
    """Short error text with credential-shaped strings removed."""

    text = f"{type(exc).__name__}"
    detail = str(exc).strip()
    if detail and not _secret(detail):
        text = f"{text}: {detail}"
    if len(text) > 300:
        return text[:300]
    return text


def reject_secret_text(text: str) -> None:
    """Refuse a report body that carries a credential marker."""

    if _secret(text):
        raise ByoaReportError("report contains a secret marker")


def read_report_bytes(payload: bytes) -> ByoaReport:
    """Validate one uploaded report. The bytes are not executed."""

    if len(payload) > MAX_REPORT_BYTES:
        raise ByoaReportError("report exceeds the size limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ByoaReportError("report is not utf-8") from exc
    reject_secret_text(text)
    try:
        report = ByoaReport.model_validate_json(text)
    except ValidationError as exc:
        raise ByoaReportError("report failed validation") from exc
    reject_secret_text(report.model_dump_json())
    return report


def write_report(report: ByoaReport, directory: str) -> str:
    """Write one report and return its path. The directory is created."""

    text = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    reject_secret_text(text)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{report.run_id}.json"
    path.write_text(text, encoding="utf-8", newline="\n")
    return str(path)


def _secret(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)
