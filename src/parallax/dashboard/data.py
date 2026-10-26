"""Load sanitized PARALLAX results for the read-only dashboard.

The bundled file is a release snapshot. A local ``runs/`` directory can be
published in memory through the same checks. This module does not call a
model, does not read ``.env``, and does not open raw traces.
"""

import json
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from parallax.capture import sha256_text
from parallax.controls import StudyResult, StudySpec
from parallax.experiment import ExperimentManifest
from parallax.fixtures import get_fixture
from parallax.metrics import ExperimentSummary, Proportion
from parallax.observe import canonical_json
from parallax.oracle import decide, predicate_d, predicate_l, predicate_m
from parallax.xray import XRayResult, XRaySpec

BUNDLE_PATH = Path(__file__).resolve().parent / "catalog" / "release.json"
MAX_SUMMARY_BYTES = 1_000_000
SECRET_MARKERS = ("sk-", "api_key", "bearer ", "authorization")
ALLOWED_DIRS = frozenset({"experiments", "xray", "controls"})
ALLOWED_NAMES = {
    "experiments": frozenset({"manifest.json", "summary.json"}),
    "xray": frozenset({"manifest.json", "result.json"}),
    "controls": frozenset({"manifest.json", "result.json"}),
}
SUMMARY_KEYS_THAT_MAY_BE_ABSENT = (
    "latency_ms_deduped",
    "latency_ms_cumulative",
    "fixture_rates",
    "scenario_rates",
    "phase_ledgers",
)
_ALLOWED_STRICT_OMISSIONS = frozenset({"result summary is missing", "no summary files"})
PROVENANCE = (
    "Sanitized summaries of recorded runs. Provider response identifiers, "
    "fork call identifiers, checkpoints, prompts, and raw traces are omitted."
)
ARCHITECTURE_PATH = (
    ("Synthetic fixtures", "Eight immutable incidents. The fixture store is not edited."),
    ("MCP tools", "get_metrics, get_logs, and get_deployment. The server does not know the arm."),
    ("Agent execution", "One control loop chooses tools and a final decision."),
    ("Shared checkpoint", "The prefix is snapshotted before the edited observation is shown."),
    ("Factual / counterfactual continuations", "The captured observation, or one declared edit, is restored."),
    ("Post-hoc scoring", "The oracle scores the facts after the decision exists."),
    ("Research dashboard", "This view reads summaries. It does not call a model."),
)
_RECORDED_ROLES = {
    "factual_baseline": ("factual_baseline", "Factual baseline"),
    "verified_flip": ("verified_intervention", "Verified intervention"),
    "no_flip": ("negative_candidate", "Negative candidate"),
}
NOT_IMPLEMENTED = (
    "Exploratory evaluation of the 24 declared scenarios",
    "A second model or a multi-model comparison",
    "Provider-backed reliability measurements for LangGraph or CrewAI agents",
    "Headroom on the measurement path",
    "Starting a model run from the dashboard",
    "A paid or hosted execution endpoint",
)
_SCHEDULE_ORDER = {"protocol-pilot": 0, "mini-pilot": 1, "custom": 2}


class DashboardLoadError(ValueError):
    """A result file cannot be shown."""


class SecretMarker(DashboardLoadError):
    """A file contains a credential marker and is not published."""


class ExperimentRelease(BaseModel):
    """One experiment manifest and the summary file it identifies."""

    model_config = ConfigDict(extra="forbid")

    record_status: Literal["measured"] = "measured"
    manifest: ExperimentManifest
    summary: ExperimentSummary
    fields_absent_from_file: list[str] = Field(default_factory=list)


class XRayRelease(BaseModel):
    """One completed Decision X-Ray search."""

    model_config = ConfigDict(extra="forbid")

    record_status: Literal["measured"] = "measured"
    manifest: XRaySpec
    result: XRayResult
    checkpoint_withheld: bool = False


class ControlRelease(BaseModel):
    """One control study with provider identifiers removed."""

    model_config = ConfigDict(extra="forbid")

    record_status: Literal["measured"] = "measured"
    manifest: StudySpec
    result: StudyResult
    unique_provider_response_count: int
    fork_call_ids_removed: int
    redactions: list[str]
    checkpoint_withheld: bool = False


class OmittedRun(BaseModel):
    """A directory that is not a measured result in this bundle."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["experiment", "xray", "control"]
    run_id: str
    reason: str


class ReleaseBundle(BaseModel):
    """The dashboard's published result snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    product: Literal["PARALLAX"] = "PARALLAX"
    package_version: str
    bundle_kind: Literal["sanitized_result_summaries"] = "sanitized_result_summaries"
    provenance: str
    experiments: list[ExperimentRelease]
    xray: list[XRayRelease]
    controls: list[ControlRelease]
    omitted: list[OmittedRun]
    not_implemented: list[str]


class PublishResult(NamedTuple):
    """A bundle plus warnings that did not enter the bundle."""

    bundle: ReleaseBundle
    warnings: tuple[str, ...]


class BoundaryRow(BaseModel):
    """Oracle illustration joined to one recorded X-Ray outcome."""

    model_config = ConfigDict(extra="forbid")

    value: int
    distance: int
    predicate_m: bool
    predicate_l: bool
    predicate_d: bool
    oracle_decision: str
    oracle_kind: Literal["oracle_illustration"] = "oracle_illustration"
    canonical_json: str
    sha256: str
    recorded_sha256: str | None
    hash_matches_record: bool
    measured_decision: str | None
    measured_status: str | None
    measured_kind: Literal["measured", "not_recorded"]


class Disclosure(NamedTuple):
    """What a page may show for one result source."""

    mode: Literal["public", "local"]
    show_checkpoint_id: bool
    show_omitted_runs: bool
    show_loader_warnings: bool


class RecordedKpi(BaseModel):
    """One homepage figure taken from a single recorded study."""

    model_config = ConfigDict(extra="forbid")

    key: Literal["protocol_pairs", "xray_distance", "null_flips"]
    label: str
    value: str
    scope: str
    kind: Literal["measured", "unavailable"]


class RecordedEvidence(BaseModel):
    """One evidence value whose decision was recorded. Untested values are absent."""

    model_config = ConfigDict(extra="forbid")

    value: int
    distance: int
    role: Literal["factual_baseline", "verified_intervention", "negative_candidate"]
    role_label: str
    decision: str
    status: str
    flipped: bool | None
    study_id: str
    model_id: str
    fixture_id: str
    tool: str
    field: str
    original_value: int
    phase: str
    minimality_claim: str
    shared_checkpoint: bool
    canonical_json: str | None = None
    evidence_sha256: str | None = None


class ControlArmDecisions(BaseModel):
    """Recorded continuation decisions for one control value."""

    model_config = ConfigDict(extra="forbid")

    value: int
    role: str
    decisions: list[str]
    statuses: list[str]
    record_status: Literal["measured"] = "measured"


def repo_root() -> Path:
    """Repository root that contains this package."""

    return Path(__file__).resolve().parents[3]


def local_runs_dir() -> Path | None:
    """Return ``runs/`` under the repository, when that directory exists."""

    candidate = repo_root() / "runs"
    if candidate.is_dir():
        return candidate
    return None


def package_version() -> str:
    """Installed package version, or the declared 0.1.0 fallback."""

    try:
        return version("parallax")
    except PackageNotFoundError:
        return "0.1.0"


def reject_secret_text(text: str) -> None:
    """Refuse text that carries a credential marker."""

    lowered = text.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        raise SecretMarker("research file contains a secret marker")


def locate_research_file(root: Path, kind: str, study_id: str, name: str) -> Path | None:
    """Resolve one allowlisted summary file inside ``root``.

    The function returns ``None`` when that file is absent. Any other
    path is refused.
    """

    if kind not in ALLOWED_DIRS or name not in ALLOWED_NAMES[kind]:
        raise DashboardLoadError("research file is not a summary")
    if not _valid_study_id(study_id):
        raise DashboardLoadError("research study id is not valid")
    root_resolved = root.resolve()
    candidate = (root_resolved / kind / study_id / name).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise DashboardLoadError("research path escapes the allowed directory")
    if not candidate.is_file():
        return None
    if candidate.stat().st_size > MAX_SUMMARY_BYTES:
        raise DashboardLoadError("research file exceeds the summary size limit")
    return candidate


def read_research_file(root: Path, kind: str, study_id: str, name: str) -> Path:
    """Return one allowlisted file, or raise when it is missing."""

    located = locate_research_file(root, kind, study_id, name)
    if located is None:
        raise DashboardLoadError("research file is missing")
    return located


def load_summary_text(text: str) -> tuple[ExperimentSummary, list[str]]:
    """Validate a summary. Missing optional keys stay marked absent."""

    reject_secret_text(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DashboardLoadError("summary is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise DashboardLoadError("summary is not an object")
    absent = [key for key in SUMMARY_KEYS_THAT_MAY_BE_ABSENT if key not in payload]
    repaired = dict(payload)
    for key in absent:
        repaired[key] = None if key.startswith("latency_") else {}
    try:
        summary = ExperimentSummary.model_validate(repaired)
    except ValidationError as exc:
        raise DashboardLoadError("summary failed validation") from exc
    return summary, absent


def redact_control(result: StudyResult) -> tuple[StudyResult, int, int]:
    """Drop provider identifiers and keep the recorded counts."""

    response_count = len(result.unique_provider_response_ids)
    fork_count = sum(1 for row in result.continuations if row.fork_call_id)
    payload = result.model_dump(mode="json")
    payload["unique_provider_response_ids"] = []
    for row in payload["continuations"]:
        row["provider_response_ids"] = []
        row["fork_call_id"] = None
    cleaned = StudyResult.model_validate(payload)
    _assert_control_redacted(cleaned)
    return cleaned, response_count, fork_count


def publish_runs(root: Path, *, strict: bool = False) -> PublishResult:
    """Build a release bundle from allowlisted files under ``root``."""

    if not root.is_dir():
        raise DashboardLoadError("research directory is missing")
    root = root.resolve()
    warnings: list[str] = []
    omitted: list[OmittedRun] = []
    experiments = _load_experiments(root, strict, warnings, omitted)
    xray = _load_xray(root, strict, warnings, omitted)
    controls = _load_controls(root, strict, warnings, omitted)
    bundle = ReleaseBundle(
        package_version=package_version(),
        provenance=PROVENANCE,
        experiments=experiments,
        xray=xray,
        controls=controls,
        omitted=sorted(omitted, key=lambda item: (item.kind, item.run_id)),
        not_implemented=list(NOT_IMPLEMENTED),
    )
    reject_secret_text(bundle.model_dump_json())
    return PublishResult(bundle, tuple(warnings))


def write_bundled_release(runs_root: Path | None = None, destination: Path | None = None) -> Path:
    """Publish ``runs/`` into the packaged release file."""

    root = runs_root if runs_root is not None else repo_root() / "runs"
    target = destination if destination is not None else BUNDLE_PATH
    result = publish_runs(root, strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = public_bundle(result.bundle).model_dump_json(indent=2) + "\n"
    reject_secret_text(text)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def load_bundled(path: Path | None = None) -> ReleaseBundle:
    """Read and validate the packaged release snapshot."""

    source = path if path is not None else BUNDLE_PATH
    if not source.is_file():
        raise DashboardLoadError("bundled release is missing")
    text = source.read_text(encoding="utf-8")
    reject_secret_text(text)
    try:
        return ReleaseBundle.model_validate_json(text)
    except ValidationError as exc:
        raise DashboardLoadError("bundled release failed validation") from exc


def find_proportion(summary: ExperimentSummary, name: str) -> Proportion | None:
    """Return one recorded rate by its summary name."""

    for row in summary.proportions:
        if row.name == name:
            return row
    return None


def protocol_pilot(bundle: ReleaseBundle) -> ExperimentRelease | None:
    """Return the single protocol-pilot release, when the bundle has one."""

    matches = [item for item in bundle.experiments if item.manifest.schedule == "protocol-pilot"]
    if len(matches) == 1:
        return matches[0]
    return None


def format_proportion(row: Proportion | None) -> str:
    """Lead with the recorded fraction."""

    if row is None:
        return "not recorded"
    if row.denominator == 0 or row.value is None:
        return "not evaluable"
    return f"{row.numerator}/{row.denominator}"


def format_wilson(row: Proportion | None) -> str:
    """Format a recorded 95% Wilson interval."""

    if row is None or row.wilson_low is None or row.wilson_high is None:
        return "no interval"
    return f"{row.wilson_low:.3f}–{row.wilson_high:.3f}"


def boundary_comparison(study: XRayRelease) -> list[BoundaryRow]:
    """Join the fixture oracle to the recorded X-Ray decisions.

    The oracle column is an illustration of the policy. The measured
    column comes from the result file. A hash match says the canonical
    fixture text is the recorded evidence string.
    """

    if study.manifest.tool != "get_metrics" or study.manifest.field != "error_rate_bps":
        raise DashboardLoadError("boundary comparison is implemented for error_rate_bps")
    fixture = get_fixture(study.manifest.fixture_id)
    if fixture is None:
        raise DashboardLoadError("fixture is missing")
    values = sorted({study.manifest.original_value, *study.manifest.candidates})
    rows: list[BoundaryRow] = []
    for value in values:
        metrics = fixture.metrics.model_copy(update={"error_rate_bps": value})
        text = canonical_json(metrics.model_dump(mode="json"))
        digest = sha256_text(text)
        recorded = _recorded_hash(study, value)
        decision = decide(metrics, fixture.logs, fixture.deployment)
        measured_decision, measured_status, measured_kind = _measured_xray(study, value)
        rows.append(
            BoundaryRow(
                value=value,
                distance=abs(value - study.manifest.original_value),
                predicate_m=predicate_m(metrics),
                predicate_l=predicate_l(fixture.logs),
                predicate_d=predicate_d(fixture.deployment),
                oracle_decision=decision.value,
                canonical_json=text,
                sha256=digest,
                recorded_sha256=recorded,
                hash_matches_record=recorded is not None and recorded == digest,
                measured_decision=measured_decision,
                measured_status=measured_status,
                measured_kind=measured_kind,
            )
        )
    return rows


def control_decisions(study: ControlRelease) -> list[ControlArmDecisions]:
    """Group recorded continuation decisions by value and role."""

    grouped: dict[tuple[int, str], list[tuple[int, str, str]]] = {}
    for row in study.result.continuations:
        decision = row.decision_parsed if row.decision_parsed is not None else "unparsed"
        grouped.setdefault((row.field_value, row.role), []).append((row.order_index, decision, row.status))
    arms: list[ControlArmDecisions] = []
    for value, role in sorted(grouped):
        ordered = [item for _, item, _ in sorted(grouped[(value, role)])]
        statuses = [status for _, _, status in sorted(grouped[(value, role)])]
        arms.append(ControlArmDecisions(value=value, role=role, decisions=ordered, statuses=statuses))
    return arms


def disclosure_for(source: str) -> Disclosure:
    """Public pages withhold local diagnostics. Local research mode keeps them."""

    if source == "local":
        return Disclosure(
            mode="local",
            show_checkpoint_id=True,
            show_omitted_runs=True,
            show_loader_warnings=True,
        )
    return Disclosure(
        mode="public",
        show_checkpoint_id=False,
        show_omitted_runs=False,
        show_loader_warnings=False,
    )


def provenance_for(disclosure: Disclosure) -> str:
    """Caption that matches the checkpoint policy for this source."""

    if disclosure.mode == "public":
        return PROVENANCE
    return (
        "Local research summaries. Provider response identifiers, fork call identifiers, "
        "prompts, and raw traces are omitted. Checkpoint identifiers from the local result files are shown."
    )


def checkpoint_text(checkpoint_id: str | None, *, withheld: bool, disclosure: Disclosure) -> str:
    """Render a checkpoint identifier under the active disclosure policy."""

    if disclosure.show_checkpoint_id:
        return checkpoint_id or "not recorded"
    if checkpoint_id or withheld:
        return "withheld in the public release"
    return "not recorded"


def visible_omissions(bundle: ReleaseBundle, disclosure: Disclosure) -> list[OmittedRun]:
    """Incomplete directories stay off the public demo."""

    if not disclosure.show_omitted_runs:
        return []
    return list(bundle.omitted)


def warnings_for_display(warnings: Sequence[str], disclosure: Disclosure) -> tuple[str, ...]:
    """Loader notes are a local diagnostic."""

    if not disclosure.show_loader_warnings:
        return ()
    return tuple(warnings)


def release_kpis(bundle: ReleaseBundle) -> tuple[RecordedKpi, RecordedKpi, RecordedKpi]:
    """Three homepage figures. Each one is a single study, or explicitly absent."""

    return (
        _protocol_pair_kpi(bundle),
        _xray_distance_kpi(bundle),
        _null_flip_kpi(bundle),
    )


def recorded_evidence(study: XRayRelease) -> list[RecordedEvidence]:
    """Recorded evidence values only. A missing or failed candidate is not given a decision."""

    options: list[RecordedEvidence] = []
    original = study.result.original_decision
    if original:
        options.append(_evidence_option(study, study.manifest.original_value, 0, "factual_baseline", original, None))
    for row in study.result.candidates:
        if row.decision is None or row.status not in _RECORDED_ROLES or row.status == "factual_baseline":
            continue
        options.append(_evidence_option(study, row.candidate_value, row.distance, row.status, row.decision, row.flipped))
    options.sort(key=lambda item: item.value)
    return [_with_matching_evidence(study, item) for item in options]


def evidence_at(study: XRayRelease, value: int) -> RecordedEvidence | None:
    """Return one recorded value. Untested values return ``None``."""

    for item in recorded_evidence(study):
        if item.value == value:
            return item
    return None


def xray_provenance_rows(study: XRayRelease, disclosure: Disclosure) -> list[tuple[str, str]]:
    """Study provenance. The checkpoint cell follows the disclosure policy."""

    manifest = study.manifest
    result = study.result
    return [
        ("checkpoint id", checkpoint_text(result.checkpoint_id, withheld=study.checkpoint_withheld, disclosure=disclosure)),
        ("shared checkpoint", str(result.shared_checkpoint).lower()),
        ("git revision", manifest.git_revision),
        ("prompt hash", manifest.prompt_hash),
        ("fixture hash", manifest.fixture_hash),
        ("source evidence sha256", manifest.source_evidence_sha256),
        ("strategy", manifest.strategy),
        ("model", manifest.model_id),
        ("provider requests", str(result.provider_requests)),
        ("deduped usage", _usage_pair(result.usage_deduped)),
    ]


def control_provenance_rows(study: ControlRelease, disclosure: Disclosure) -> list[tuple[str, str]]:
    """Control provenance. The checkpoint cell follows the disclosure policy."""

    manifest = study.manifest
    return [
        (
            "checkpoint id",
            checkpoint_text(study.result.checkpoint_id, withheld=study.checkpoint_withheld, disclosure=disclosure),
        ),
        ("source evidence sha256", manifest.source_evidence_sha256),
        ("prompt hash", manifest.prompt_hash),
        ("fixture hash", manifest.fixture_hash),
        ("git revision", manifest.git_revision),
    ]


def public_bundle(bundle: ReleaseBundle) -> ReleaseBundle:
    """Copy a bundle for the packaged demo.

    Checkpoint identifiers and directories without a result summary are removed.
    Measured decisions, rates, and evidence hashes stay.
    """

    identifiers = _checkpoint_identifiers(bundle)
    released = bundle.model_copy(
        update={
            "xray": [_public_xray(item) for item in bundle.xray],
            "controls": [_public_control(item) for item in bundle.controls],
            "omitted": [],
            "provenance": PROVENANCE,
        }
    )
    text = released.model_dump_json()
    for identifier in identifiers:
        if identifier in text:
            raise DashboardLoadError("checkpoint id remains in the public release")
    return released


def usage_by_role(result: StudyResult) -> dict[str, dict[str, int] | None]:
    """Sum recorded continuation usage by role.

    A missing usage report makes that role null. The sum repeats the
    shared prefix inside every continuation and is not the deduped total.
    """

    buckets: dict[str, dict[str, int] | None] = {}
    for row in result.continuations:
        if row.status == "not_run":
            continue
        if row.role not in buckets:
            buckets[row.role] = {"input_tokens": 0, "output_tokens": 0}
        if row.usage is None:
            buckets[row.role] = None
            continue
        bucket = buckets[row.role]
        if bucket is None:
            continue
        bucket["input_tokens"] += row.usage["input_tokens"]
        bucket["output_tokens"] += row.usage["output_tokens"]
    return buckets


def _protocol_pair_kpi(bundle: ReleaseBundle) -> RecordedKpi:
    pilot = protocol_pilot(bundle)
    if pilot is None:
        return RecordedKpi(
            key="protocol_pairs",
            label="Valid protocol-pilot pairs",
            value="not recorded",
            scope="No protocol-pilot summary is in this source.",
            kind="unavailable",
        )
    row = find_proportion(pilot.summary, "pair_validity")
    manifest = pilot.manifest
    scope = (
        f"{manifest.experiment_id} · {manifest.schedule} · {pilot.summary.trials} trials · "
        f"{manifest.repeats} repeats · {len(manifest.scenario_ids)} scenarios · pair validity"
    )
    if row is None or row.denominator == 0 or row.value is None:
        return RecordedKpi(
            key="protocol_pairs",
            label="Valid protocol-pilot pairs",
            value=format_proportion(row),
            scope=scope,
            kind="unavailable",
        )
    return RecordedKpi(
        key="protocol_pairs",
        label="Valid protocol-pilot pairs",
        value=format_proportion(row),
        scope=scope,
        kind="measured",
    )


def _xray_distance_kpi(bundle: ReleaseBundle) -> RecordedKpi:
    verified = [item for item in bundle.xray if _verified_distance(item) is not None]
    if len(verified) != 1:
        return RecordedKpi(
            key="xray_distance",
            label="Verified X-Ray edit distance",
            value="not recorded",
            scope="This source does not contain one X-Ray search with a verified flip.",
            kind="unavailable",
        )
    study = verified[0]
    distance = _verified_distance(study)
    manifest = study.manifest
    return RecordedKpi(
        key="xray_distance",
        label="Verified X-Ray edit distance",
        value=f"{distance} bps",
        scope=(
            f"{study.result.xray_id} · {manifest.fixture_id} · {manifest.tool}.{manifest.field} · "
            f"{len(study.result.candidates)} enumerated candidates · phase {manifest.phase} · "
            f"{study.result.minimality_claim}"
        ),
        kind="measured",
    )


def _null_flip_kpi(bundle: ReleaseBundle) -> RecordedKpi:
    studies = [item for item in bundle.controls if _null_metric(item) is not None]
    if len(studies) != 1:
        return RecordedKpi(
            key="null_flips",
            label="Null-control decision flips",
            value="not recorded",
            scope="This source does not contain one control study.",
            kind="unavailable",
        )
    study = studies[0]
    metric = _null_metric(study)
    assert metric is not None
    scope = (
        f"{study.result.study_id} · null replays {study.manifest.original_value} · "
        f"{metric.valid_count} valid continuations · {study.result.checkpoint_draws} checkpoint draw"
    )
    if metric.valid_count == 0:
        return RecordedKpi(
            key="null_flips",
            label="Null-control decision flips",
            value="not evaluable",
            scope=scope,
            kind="unavailable",
        )
    return RecordedKpi(
        key="null_flips",
        label="Null-control decision flips",
        value=f"{metric.flip_count}/{metric.valid_count}",
        scope=scope,
        kind="measured",
    )


def _verified_distance(study: XRayRelease) -> int | None:
    edit = study.result.minimal_edit
    if not edit or "distance" not in edit:
        return None
    distance = edit["distance"]
    if isinstance(distance, bool) or not isinstance(distance, int):
        return None
    return distance


def _null_metric(study: ControlRelease):
    for metric in study.result.conditions:
        if metric.role == "null":
            return metric
    return None


def _evidence_option(
    study: XRayRelease,
    value: int,
    distance: int,
    status: str,
    decision: str,
    flipped: bool | None,
) -> RecordedEvidence:
    role, label = _RECORDED_ROLES[status]
    manifest = study.manifest
    return RecordedEvidence(
        value=value,
        distance=distance,
        role=role,
        role_label=label,
        decision=decision,
        status=status,
        flipped=flipped,
        study_id=study.result.xray_id,
        model_id=manifest.model_id,
        fixture_id=manifest.fixture_id,
        tool=manifest.tool,
        field=manifest.field,
        original_value=manifest.original_value,
        phase=manifest.phase,
        minimality_claim=study.result.minimality_claim,
        shared_checkpoint=study.result.shared_checkpoint,
    )


def _with_matching_evidence(study: XRayRelease, item: RecordedEvidence) -> RecordedEvidence:
    try:
        rows = {row.value: row for row in boundary_comparison(study)}
    except DashboardLoadError:
        return item
    row = rows.get(item.value)
    if row is None or not row.hash_matches_record:
        return item
    return item.model_copy(update={"canonical_json": row.canonical_json, "evidence_sha256": row.sha256})


def _usage_pair(usage: dict[str, int] | None) -> str:
    if usage is None:
        return "recorded null"
    return f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out"


def _checkpoint_identifiers(bundle: ReleaseBundle) -> set[str]:
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            identifier = node.get("checkpoint_id")
            if isinstance(identifier, str) and identifier:
                found.add(identifier)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(bundle.model_dump(mode="json"))
    return found


def _null_checkpoint_ids(node: object) -> bool:
    cleared = False
    if isinstance(node, dict):
        identifier = node.get("checkpoint_id")
        if isinstance(identifier, str) and identifier:
            node["checkpoint_id"] = None
            cleared = True
        for value in node.values():
            if _null_checkpoint_ids(value):
                cleared = True
    elif isinstance(node, list):
        for value in node:
            if _null_checkpoint_ids(value):
                cleared = True
    return cleared


def _public_xray(study: XRayRelease) -> XRayRelease:
    payload = study.result.model_dump(mode="json")
    cleared = _null_checkpoint_ids(payload)
    return study.model_copy(
        update={
            "result": XRayResult.model_validate(payload),
            "checkpoint_withheld": cleared or study.checkpoint_withheld,
        }
    )


def _public_control(study: ControlRelease) -> ControlRelease:
    payload = study.result.model_dump(mode="json")
    cleared = _null_checkpoint_ids(payload)
    cleaned = StudyResult.model_validate(payload)
    _assert_control_redacted(cleaned)
    return study.model_copy(
        update={
            "result": cleaned,
            "checkpoint_withheld": cleared or study.checkpoint_withheld,
        }
    )


def _measured_xray(study: XRayRelease, value: int) -> tuple[str | None, str | None, Literal["measured", "not_recorded"]]:
    if value == study.manifest.original_value:
        if study.result.original_decision is None:
            return None, None, "not_recorded"
        return study.result.original_decision, "factual_baseline", "measured"
    matches = [row for row in study.result.candidates if row.candidate_value == value]
    if len(matches) != 1:
        return None, None, "not_recorded"
    return matches[0].decision, matches[0].status, "measured"


def _recorded_hash(study: XRayRelease, value: int) -> str | None:
    if value == study.manifest.original_value:
        hashes = {
            row.evidence_sha256_original
            for row in study.result.candidates
            if row.evidence_sha256_original is not None
        }
        if len(hashes) != 1:
            return None
        return next(iter(hashes))
    matches = [row for row in study.result.candidates if row.candidate_value == value]
    if len(matches) != 1:
        return None
    return matches[0].evidence_sha256_mutated


def _load_experiments(
    root: Path,
    strict: bool,
    warnings: list[str],
    omitted: list[OmittedRun],
) -> list[ExperimentRelease]:
    loaded: list[ExperimentRelease] = []
    for study_id in _study_ids(root, "experiments"):
        manifest_path = locate_research_file(root, "experiments", study_id, "manifest.json")
        summary_path = locate_research_file(root, "experiments", study_id, "summary.json")
        if manifest_path is None and summary_path is None:
            _omit(strict, warnings, omitted, "experiment", study_id, "no summary files")
            continue
        if summary_path is None:
            _omit(strict, warnings, omitted, "experiment", study_id, "result summary is missing")
            continue
        if manifest_path is None:
            _omit(strict, warnings, omitted, "experiment", study_id, "manifest is missing")
            continue
        try:
            manifest = ExperimentManifest.model_validate_json(_read_text(manifest_path))
            summary, absent = load_summary_text(_read_text(summary_path))
        except SecretMarker:
            _omit(strict, warnings, omitted, "experiment", study_id, "secret marker")
            continue
        except (DashboardLoadError, ValidationError, UnicodeError):
            _omit(strict, warnings, omitted, "experiment", study_id, "summary failed validation")
            continue
        if manifest.experiment_id != summary.experiment_id:
            _omit(strict, warnings, omitted, "experiment", study_id, "experiment id does not match the manifest")
            continue
        loaded.append(
            ExperimentRelease(manifest=manifest, summary=summary, fields_absent_from_file=absent)
        )
    return sorted(loaded, key=lambda item: (_SCHEDULE_ORDER.get(item.manifest.schedule, 9), item.summary.experiment_id))


def _load_xray(
    root: Path,
    strict: bool,
    warnings: list[str],
    omitted: list[OmittedRun],
) -> list[XRayRelease]:
    loaded: list[XRayRelease] = []
    for study_id in _study_ids(root, "xray"):
        manifest_path = locate_research_file(root, "xray", study_id, "manifest.json")
        result_path = locate_research_file(root, "xray", study_id, "result.json")
        if manifest_path is None and result_path is None:
            _omit(strict, warnings, omitted, "xray", study_id, "no summary files")
            continue
        if result_path is None:
            run_id = _manifest_id(manifest_path, XRaySpec, "xray_id", study_id)
            _omit(strict, warnings, omitted, "xray", run_id, "result summary is missing")
            continue
        if manifest_path is None:
            _omit(strict, warnings, omitted, "xray", study_id, "manifest is missing")
            continue
        try:
            manifest = XRaySpec.model_validate_json(_read_text(manifest_path))
            result = XRayResult.model_validate_json(_read_text(result_path))
        except SecretMarker:
            _omit(strict, warnings, omitted, "xray", study_id, "secret marker")
            continue
        except (DashboardLoadError, ValidationError, UnicodeError):
            _omit(strict, warnings, omitted, "xray", study_id, "summary failed validation")
            continue
        if manifest.xray_id != result.xray_id:
            _omit(strict, warnings, omitted, "xray", study_id, "experiment id does not match the manifest")
            continue
        loaded.append(XRayRelease(manifest=manifest, result=result))
    return sorted(loaded, key=lambda item: item.result.xray_id)


def _load_controls(
    root: Path,
    strict: bool,
    warnings: list[str],
    omitted: list[OmittedRun],
) -> list[ControlRelease]:
    loaded: list[ControlRelease] = []
    for study_id in _study_ids(root, "controls"):
        manifest_path = locate_research_file(root, "controls", study_id, "manifest.json")
        result_path = locate_research_file(root, "controls", study_id, "result.json")
        if manifest_path is None and result_path is None:
            _omit(strict, warnings, omitted, "control", study_id, "no summary files")
            continue
        if result_path is None:
            run_id = _manifest_id(manifest_path, StudySpec, "study_id", study_id)
            _omit(strict, warnings, omitted, "control", run_id, "result summary is missing")
            continue
        if manifest_path is None:
            _omit(strict, warnings, omitted, "control", study_id, "manifest is missing")
            continue
        try:
            manifest = StudySpec.model_validate_json(_read_text(manifest_path))
            result = StudyResult.model_validate_json(_read_text(result_path))
        except SecretMarker:
            _omit(strict, warnings, omitted, "control", study_id, "secret marker")
            continue
        except (DashboardLoadError, ValidationError, UnicodeError):
            _omit(strict, warnings, omitted, "control", study_id, "summary failed validation")
            continue
        if manifest.study_id != result.study_id:
            _omit(strict, warnings, omitted, "control", study_id, "experiment id does not match the manifest")
            continue
        cleaned, response_count, fork_count = redact_control(result)
        loaded.append(
            ControlRelease(
                manifest=manifest,
                result=cleaned,
                unique_provider_response_count=response_count,
                fork_call_ids_removed=fork_count,
                redactions=["provider_response_ids", "fork_call_id"],
            )
        )
    return sorted(loaded, key=lambda item: item.result.study_id)


def _manifest_id(path: Path | None, model: type[BaseModel], field: str, fallback: str) -> str:
    if path is None:
        return fallback
    try:
        parsed = model.model_validate_json(_read_text(path))
    except (DashboardLoadError, ValidationError, UnicodeError, SecretMarker):
        return fallback
    value = getattr(parsed, field)
    if isinstance(value, str) and value:
        return value
    return fallback


def _omit(
    strict: bool,
    warnings: list[str],
    omitted: list[OmittedRun],
    kind: Literal["experiment", "xray", "control"],
    run_id: str,
    reason: str,
) -> None:
    omitted.append(OmittedRun(kind=kind, run_id=run_id, reason=reason))
    if reason in _ALLOWED_STRICT_OMISSIONS:
        return
    if strict:
        raise DashboardLoadError(f"{kind}/{run_id}: {reason}")
    warnings.append(f"{kind}/{run_id}: {reason}")


def _study_ids(root: Path, kind: str) -> list[str]:
    parent = root / kind
    if not parent.is_dir():
        return []
    names: list[str] = []
    root_resolved = root.resolve()
    for child in sorted(parent.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or not _valid_study_id(child.name):
            continue
        resolved = child.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise DashboardLoadError("research path escapes the allowed directory")
        names.append(child.name)
    return names


def _valid_study_id(study_id: str) -> bool:
    if not study_id or study_id in {".", ".."}:
        return False
    return not any(separator in study_id for separator in ("/", "\\", "\x00"))


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    reject_secret_text(text)
    return text


def _assert_control_redacted(result: StudyResult) -> None:
    if result.unique_provider_response_ids:
        raise DashboardLoadError("provider response ids remain in the release")
    for row in result.continuations:
        if row.provider_response_ids or row.fork_call_id:
            raise DashboardLoadError("provider response ids remain in the release")
