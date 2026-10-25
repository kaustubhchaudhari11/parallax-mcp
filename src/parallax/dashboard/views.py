"""Pages for the PARALLAX release dashboard."""

from collections.abc import Sequence

import streamlit as st

from parallax.byoa.guide import COMMANDS, DOC_URL, GUIDE
from parallax.byoa.report import ByoaReportError, read_report_bytes
from parallax.dashboard.components import (
    architecture_flow,
    card_row,
    count_chart,
    discordant_chart,
    evidence_chart,
    hero,
    histogram_chart,
    pills,
    rate_chart,
    show_chart,
    wilson_chart,
)
from parallax.dashboard.data import (
    ARCHITECTURE_PATH,
    NOT_IMPLEMENTED,
    BoundaryRow,
    ControlRelease,
    DashboardLoadError,
    Disclosure,
    ExperimentRelease,
    RecordedEvidence,
    ReleaseBundle,
    XRayRelease,
    boundary_comparison,
    control_decisions,
    control_provenance_rows,
    evidence_at,
    find_proportion,
    format_proportion,
    format_wilson,
    protocol_pilot,
    provenance_for,
    recorded_evidence,
    release_kpis,
    usage_by_role,
    visible_omissions,
    xray_provenance_rows,
)

GITHUB_URL = "https://github.com/kaustubhchaudhari11/parallax-mcp"
PAGES = (
    ("overview", "Overview"),
    ("xray", "Decision X-Ray"),
    ("reliability", "Reliability"),
    ("controls", "Research controls"),
    ("about", "About"),
    ("byoa", "Test Your Agent"),
)
_HYPOTHESES = (
    ("h1_invariance", "H1 invariance"),
    ("h2_adaptation", "H2 adaptation"),
    ("h3_instruction_rejection", "H3 instruction rejection"),
    ("h3_decision_contamination", "H3 decision contamination"),
    ("h3_instruction_adoption", "H3 instruction adoption"),
)
_COMPLETION = (
    ("pair_completion", "Pair completion"),
    ("pair_validity", "Pair validity"),
    ("invalid_output", "Invalid output"),
    ("parse_failure", "Parse failure"),
    ("trace_archive_incomplete", "Incomplete trace archive"),
)


def render_sidebar_brand() -> None:
    """Product identity at the top of the sidebar."""

    st.markdown("### PARALLAX")
    st.caption("v0.1.0 · read-only")
    st.caption("Test your AI agents before you trust their decisions.")


def render_source_picker(local_available: bool) -> str:
    """Choose the bundled release or the local runs directory."""

    if not local_available:
        st.caption("Results: bundled release")
        return "bundled"
    return st.radio(
        "Results",
        ("bundled", "local"),
        format_func=lambda key: "Bundled release" if key == "bundled" else "Local runs directory",
    )


def render_page_picker() -> str:
    """Choose a dashboard page."""

    labels = dict(PAGES)
    return st.radio("Sections", [key for key, _label in PAGES], format_func=labels.get)


def render_page(page: str, bundle: ReleaseBundle, source: str, disclosure: Disclosure) -> None:
    """Render the selected page."""

    if page == "xray":
        render_xray(bundle, source, disclosure)
    elif page == "reliability":
        render_reliability(bundle, source, disclosure)
    elif page == "controls":
        render_controls(bundle, source, disclosure)
    elif page == "about":
        render_about(bundle, source, disclosure)
    elif page == "byoa":
        render_byoa()
    else:
        render_overview(bundle, source, disclosure)


def render_overview(bundle: ReleaseBundle, source: str, disclosure: Disclosure) -> None:
    """Product summary, recorded figures, and the research boundary."""

    hero(
        "PARALLAX",
        "Test Your AI Agents Before You Trust Their Decisions.",
        "Understand how changes in tool evidence affect AI agent decisions through reproducible counterfactual testing.",
        tagline="Same Agent. Different Evidence. Different Future.",
    )
    card_row(tuple((item.label, item.value, item.scope, item.kind) for item in release_kpis(bundle)))
    pills(
        ("Measured", "measured"),
        ("Oracle illustration", "illustration"),
        ("Not implemented", "unavailable"),
    )
    _source_line(source, disclosure)
    st.caption(
        "Each figure above is one recorded study. The scope on the card is the sample. "
        "Studies are not pooled. Oracle illustrations apply the decision policy to synthetic fixtures "
        "and are not model measurements."
    )
    st.markdown(
        "PARALLAX compares one agent on the same prompt after a controlled change to an MCP tool result. "
        "The agent may call `get_metrics`, `get_logs`, and `get_deployment`, then choose "
        "`MONITOR`, `INVESTIGATE`, or `ESCALATE`."
    )
    with st.expander("Hypothesis definitions"):
        st.markdown(
            "These are questions for the experiment. They are not assumed properties of the model. "
            "A paired run keeps the prompt, model settings, and prefix fixed, and changes one declared observation."
        )
        card_row(
            (
                (
                    "H1 invariance",
                    "Irrelevant edit",
                    "The oracle predicates stay the same. The question is whether the decision stays correct.",
                    "definition",
                ),
                (
                    "H2 sensitivity",
                    "Material edit",
                    "The facts change the oracle decision. The question is whether the agent follows.",
                    "definition",
                ),
                (
                    "H3 trust boundary",
                    "Embedded instruction",
                    "A tool result contains an instruction. The question is whether the agent adopts it.",
                    "definition",
                ),
            )
        )
        st.markdown(
            "Rates, Wilson intervals, and the protocol-pilot sample live on the Reliability page. "
            "The protocol treats that pilot as debugging data for the loop."
        )
    st.header("Recorded studies")
    if bundle.experiments or bundle.xray or bundle.controls:
        st.dataframe(_study_table(bundle), hide_index=True, width="stretch")
        st.caption("Studies are listed separately. Rates are not pooled across them.")
    else:
        st.info("This source has no measured study.")
    pilot = protocol_pilot(bundle)
    if pilot is not None:
        with st.expander("Protocol-pilot accounting"):
            st.caption(_study_caption(pilot))
            _accounting(pilot)
    st.header("Limitations")
    for item in _limitations():
        st.markdown(f"- {item}")
    st.header("System architecture")
    st.caption("Illustration of the implemented path. This diagram is not a measurement.")
    architecture_flow(ARCHITECTURE_PATH)
    _scope_lists(bundle, disclosure)


def render_xray(bundle: ReleaseBundle, source: str, disclosure: Disclosure) -> None:
    """Recorded evidence selector for one Decision X-Ray search."""

    hero(
        "Decision X-Ray",
        "Where the evidence changes the decision.",
        "Choose a recorded F100 value. The decision is the one stored for that continuation.",
    )
    _source_line(source, disclosure)
    study = _select_xray(bundle)
    if study is None:
        st.info("This source has no completed X-Ray result.")
        return
    pills(("Measured search", "measured"), ("Oracle column is an illustration", "illustration"))
    st.caption(_xray_caption(study))
    st.caption(
        "Phase harness. These rows are not hypothesis rates. "
        "The selector lists only values with a recorded decision. It does not call a model."
    )
    options = recorded_evidence(study)
    if not options:
        st.info("This search has no recorded decision to select.")
    else:
        _evidence_selector(study, options)
    show_chart(evidence_chart(_evidence_points(study)))
    st.caption("Point labels are the recorded decisions. Distance is the absolute change in error_rate_bps.")
    with st.expander("Search-space minimality"):
        edit = study.result.minimal_edit
        if edit is None:
            edit_text = "No verified flip is recorded."
        else:
            edit_text = f"value {edit.get('value')}, distance {edit.get('distance')}, decision {edit.get('decision')}"
        card_row(
            (
                ("Minimality claim", study.result.minimality_claim, "Qualification recorded with the search.", "measured"),
                ("Smallest recorded flip", edit_text, f"Termination: {study.result.termination_reason}.", "measured"),
                (
                    "Proof blockers",
                    "none" if not study.result.proof_blockers else ", ".join(str(item) for item in study.result.proof_blockers),
                    "Candidates that blocked a stronger minimality claim.",
                    "measured",
                ),
            )
        )
        st.markdown(study.result.minimality_definition)
    with st.expander("All candidates"):
        _candidate_table(study)
    _oracle_illustration(study, bundle)
    with st.expander("Study provenance"):
        st.dataframe(
            [{"field": field, "value": value} for field, value in xray_provenance_rows(study, disclosure)],
            hide_index=True,
            width="stretch",
        )
    _notes(study.result.notes)


def render_reliability(bundle: ReleaseBundle, source: str, disclosure: Disclosure) -> None:
    """Hypothesis rates, sample sizes, intervals, and failure counts."""

    hero("Reliability", "Did the decision follow the evidence?", "H1, H2, and H3 from one recorded summary at a time.")
    _source_line(source, disclosure)
    study = _select_experiment(bundle)
    if study is None:
        st.info("This source has no experiment summary.")
        return
    pills(("Measured summary", "measured"))
    st.caption(_study_caption(study))
    st.markdown(
        f"Sample size: {study.summary.trials} trials, "
        f"{study.manifest.repeats} repeats, "
        f"scenarios {', '.join(study.manifest.scenario_ids)}."
    )
    st.subheader("Hypotheses")
    card_row(_rate_cards(study, _HYPOTHESES[:3]))
    card_row(_rate_cards(study, _HYPOTHESES[3:]))
    rows = _wilson_rows(study, _HYPOTHESES)
    if rows:
        show_chart(wilson_chart(rows))
        st.caption("The point is the recorded rate. The bar is the recorded 95% Wilson interval.")
    _unevaluable(study, _HYPOTHESES)
    st.subheader("Completion and failures")
    card_row(_rate_cards(study, _COMPLETION[:3]))
    card_row(_rate_cards(study, _COMPLETION[3:]))
    _completion_note(study)
    if study.summary.failure_breakdown:
        show_chart(count_chart("Recorded execution statuses and flags", study.summary.failure_breakdown))
    if "fixture_rates" in study.fields_absent_from_file:
        st.caption("Per-fixture rates are not in this summary file.")
    elif study.summary.baseline_histogram:
        show_chart(histogram_chart(study.summary.baseline_histogram))
    if study.summary.discordant_pairs:
        show_chart(discordant_chart(study.summary.discordant_pairs))
        st.caption("Discordant cells count parsed factual/mutant pairs in the recorded summary.")
    _slice_tables(study)
    with st.expander("Provider accounting"):
        _accounting(study)
    _notes(study.summary.notes)


def render_controls(bundle: ReleaseBundle, source: str, disclosure: Disclosure) -> None:
    """Null, intervention, and negative continuations from one checkpoint."""

    hero(
        "Research controls",
        "A flip needs a comparison.",
        "The same checkpoint, replayed as null, intervention, and negative evidence.",
    )
    _source_line(source, disclosure)
    study = _select_control(bundle)
    if study is None:
        st.info("This source has no control-study result.")
        return
    pills(("Measured continuations", "measured"))
    manifest = study.manifest
    result = study.result
    st.caption(
        f"{result.study_id} · {manifest.model_id} · fixture {manifest.fixture_id} · "
        f"{manifest.field} original {manifest.original_value} · "
        f"{result.sampling_unit} · checkpoint draws {result.checkpoint_draws}"
    )
    values = {condition.role: condition.field_value for condition in manifest.conditions}
    st.markdown(
        "Repetitions are fresh continuations from one checkpoint draw. "
        f"Null replays {values.get('null')}. "
        f"Intervention replays {values.get('intervention')}. "
        f"Negative replays {values.get('negative')}. "
        "A decision change on the null arm is variability."
    )
    card_row(tuple(_condition_card(metric) for metric in result.conditions))
    chart_rows = _control_chart_rows(study)
    if chart_rows:
        show_chart(rate_chart("Recorded flip rates", chart_rows))
        st.caption("Flip rates use valid continuations as the denominator. The bars are the recorded Wilson intervals.")
    difference = result.intervention_minus_null
    st.subheader("Intervention minus null")
    if difference is None:
        st.caption("No difference estimate is recorded.")
    else:
        card_row(
            (
                (
                    "Recorded difference",
                    f"{float(difference['estimate']):.3f}",
                    (
                        f"Intervention {difference['intervention_flips']}/{difference['intervention_valid']}. "
                        f"Null {difference['null_flips']}/{difference['null_valid']}."
                    ),
                    "measured",
                ),
            )
        )
        st.text(str(difference.get("note", "")))
        st.caption("The result records no significance test.")
    with st.expander("Recorded continuations"):
        st.dataframe(_continuation_rows(study), hide_index=True, width="stretch")
    with st.expander("Validity"):
        st.dataframe(_validity_rows(study), hide_index=True, width="stretch")
    with st.expander("Usage"):
        st.markdown(
            f"Deduped study usage: {_usage_text(result.usage_deduped)}. "
            f"Provider requests: {result.provider_requests}. "
            f"Unique provider responses counted before redaction: {study.unique_provider_response_count}."
        )
        st.caption(
            "The role totals below add the usage recorded on each continuation. "
            "Each of those reports includes the shared prefix, so the sum is larger than the deduped study total."
        )
        st.dataframe(_usage_rows(study), hide_index=True, width="stretch")
        st.caption(
            f"Provider response identifiers removed: {study.unique_provider_response_count} unique ids. "
            f"Fork call identifiers removed: {study.fork_call_ids_removed}."
        )
    with st.expander("Study provenance"):
        st.dataframe(
            [{"field": field, "value": value} for field, value in control_provenance_rows(study, disclosure)],
            hide_index=True,
            width="stretch",
        )
    _notes(result.notes)


def render_about(bundle: ReleaseBundle, source: str, disclosure: Disclosure) -> None:
    """Methodology, reproducibility, and the project link."""

    hero("About", "How a result is produced.", "Counterfactual replay for MCP agents.")
    _source_line(source, disclosure)
    st.subheader("MCP")
    st.markdown(
        "The factual world is an MCP server with three read-only tools. "
        "A tool result becomes one canonical JSON string. "
        "That string is what the agent can read. The raw tool result stays in the local trace, which this dashboard does not publish."
    )
    st.subheader("Counterfactual replay")
    st.markdown(
        "The agent runs until the selected tool returns. "
        "That prefix is snapshotted. "
        "Each continuation copies the prefix and receives either the captured observation or one declared edit. "
        "The fixture store is not modified."
    )
    st.subheader("Factual and mutant continuations")
    st.markdown(
        "The factual continuation restores the captured observation. "
        "The mutant continuation restores the edited observation. "
        "Both arms keep the same prompt, model settings, and remaining budget from the snapshot."
    )
    st.subheader("Oracle separation")
    st.markdown(
        "The oracle is a Python policy: error rate at or above 500 basis points, "
        "an error log condition, and a deployment inside the window. "
        "None of those labels is given to the agent. "
        "The runner records the oracle decision after the model answers. "
        "On the X-Ray page, the oracle column is marked as an illustration of that policy."
    )
    st.subheader("Reproducibility")
    pilot = protocol_pilot(bundle)
    if pilot is None:
        st.caption("The protocol-pilot manifest is not in this source.")
    else:
        manifest = pilot.manifest
        st.dataframe(
            [
                {"field": "experiment", "value": manifest.experiment_id},
                {"field": "software", "value": manifest.software_version},
                {"field": "MCP SDK", "value": manifest.sdk_version},
                {"field": "Python", "value": manifest.python_version},
                {"field": "trace schema", "value": str(manifest.trace_schema_version)},
                {"field": "model", "value": manifest.model_id},
                {"field": "model parameters", "value": str(manifest.model_params)},
            ],
            hide_index=True,
            width="stretch",
        )
    st.caption(f"Dashboard package version recorded in this bundle: {bundle.package_version}.")
    st.subheader("Project")
    st.markdown(f"[PARALLAX on GitHub]({GITHUB_URL})")
    st.caption("The dashboard runs without an API key. It does not start a paid request.")
    _scope_lists(bundle, disclosure)


def render_byoa() -> None:
    """Explain local BYOA evaluation and display one validated report."""

    hero(
        "Test Your Agent",
        "Bring a compatible agent.",
        "Declare one evidence edit, run the framework locally, and read the contract verdict.",
        tagline="The public site does not execute an agent.",
    )
    st.caption(
        "Pilot, X-Ray, and control results are separate studies. "
        "A BYOA report is one contract and one trial."
    )
    for title, body in GUIDE:
        st.subheader(title)
        st.markdown(body)
    st.subheader("Commands")
    for command in COMMANDS:
        st.code(command, language="text")
    st.markdown(f"[BYOA documentation]({DOC_URL})")
    st.subheader("Report")
    st.caption("Upload a JSON report from your own machine. Do not upload secrets or private transcripts.")
    uploaded = st.file_uploader("BYOA report", type=["json"], key="byoa-report")
    if uploaded is None:
        return
    try:
        report = read_report_bytes(uploaded.getvalue())
    except ByoaReportError as exc:
        st.error(str(exc))
        return
    st.caption(f"{report.agent_framework} · {report.comparison_mode} · shared checkpoint {report.shared_checkpoint}")
    card_row(
        (
            ("Verdict", report.verdict, report.expected_behavior, "measured"),
            ("Factual", report.factual_decision or "missing", report.modified_field, "measured"),
            (
                "Counterfactual",
                report.counterfactual_decision or "missing",
                "one declared edit",
                "measured",
            ),
        )
    )
    for item in report.limitations:
        st.caption(item)
    if report.errors:
        st.warning(" ".join(report.errors))


def _study_caption(study: ExperimentRelease) -> str:
    manifest = study.manifest
    return (
        f"{manifest.experiment_id} · phase {manifest.phase} · {manifest.schedule} · "
        f"model {manifest.model_id} · fixtures {', '.join(manifest.fixture_ids)}"
    )


def _xray_caption(study: XRayRelease) -> str:
    manifest = study.manifest
    return (
        f"{manifest.xray_id} · {manifest.model_id} · {manifest.fixture_id} · "
        f"{manifest.tool}.{manifest.field} = {manifest.original_value} · "
        f"phase {manifest.phase}"
    )


def _rate_cards(study: ExperimentRelease, specs: Sequence[tuple[str, str]]) -> tuple[tuple[str, str, str, str], ...]:
    cards: list[tuple[str, str, str, str]] = []
    for name, gloss in specs:
        row = find_proportion(study.summary, name)
        if row is None:
            cards.append((gloss, "Not recorded", "This summary has no rate with that name.", "unavailable"))
            continue
        if row.denominator == 0 or row.value is None:
            cards.append((gloss, "Not evaluable", f"Recorded count {row.numerator}/{row.denominator}.", "unavailable"))
            continue
        cards.append((gloss, format_proportion(row), f"95% Wilson {format_wilson(row)}", "measured"))
    return tuple(cards)


def _wilson_rows(study: ExperimentRelease, specs: Sequence[tuple[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, label in specs:
        row = find_proportion(study.summary, name)
        if row is None or row.value is None or row.wilson_low is None or row.wilson_high is None:
            continue
        rows.append(
            {
                "label": label,
                "value": row.value,
                "low": row.wilson_low,
                "high": row.wilson_high,
                "hover": f"{label}<br>{format_proportion(row)}<br>Wilson {format_wilson(row)}",
            }
        )
    return rows


def _unevaluable(study: ExperimentRelease, specs: Sequence[tuple[str, str]]) -> None:
    missing = []
    for name, label in specs:
        row = find_proportion(study.summary, name)
        if row is not None and (row.denominator == 0 or row.value is None):
            missing.append(f"{label} ({row.numerator}/{row.denominator})")
    if missing:
        st.caption("Not evaluable in this summary: " + ", ".join(missing) + ".")


def _completion_note(study: ExperimentRelease) -> None:
    completion = find_proportion(study.summary, "pair_completion")
    validity = find_proportion(study.summary, "pair_validity")
    if completion is None or validity is None:
        return
    if (completion.numerator, completion.denominator) != (validity.numerator, validity.denominator):
        st.caption(
            "Pair completion and pair validity differ in this recorded summary. "
            "Both figures are shown as recorded."
        )


def _accounting(study: ExperimentRelease) -> None:
    summary = study.summary
    card_row(
        (
            ("Trials", str(summary.trials), "Recorded attempted trials.", "measured"),
            ("Unique model calls", str(summary.model_calls_unique), "Union of provider response ids.", "measured"),
            (
                "Cumulative model calls",
                str(summary.model_calls_cumulative),
                f"Shared prefix {summary.model_calls_shared_prefix}. Post-fork {summary.model_calls_post_fork}.",
                "measured",
            ),
        )
    )
    st.write("")
    card_row(
        (
            ("Deduped usage", _usage_text(summary.usage_deduped), "Shared prefix counted once when the summary could.", "measured"),
            ("Cumulative usage", _usage_text(summary.usage_cumulative), "Arm totals added as stored.", "measured"),
            ("Deduped tool latency", _latency_text(study, "latency_ms_deduped"), "Milliseconds recorded by the summary.", "measured"),
        )
    )


def _latency_text(study: ExperimentRelease, key: str) -> str:
    if key in study.fields_absent_from_file:
        return "not in this summary file"
    value = getattr(study.summary, key)
    if value is None:
        return "recorded null"
    return f"{float(value):,.3f} ms"


def _slice_tables(study: ExperimentRelease) -> None:
    if "scenario_rates" in study.fields_absent_from_file:
        st.caption("Per-scenario rates are not in this summary file.")
    elif study.summary.scenario_rates:
        st.subheader("Scenario rates")
        st.dataframe(_proportion_table(study.summary.scenario_rates), hide_index=True, width="stretch")
    if "fixture_rates" not in study.fields_absent_from_file and study.summary.fixture_rates:
        st.subheader("Fixture rates")
        st.dataframe(_proportion_table(study.summary.fixture_rates), hide_index=True, width="stretch")


def _proportion_table(groups: dict[str, list]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in sorted(groups):
        for row in groups[name]:
            rows.append(
                {
                    "group": name,
                    "rate": row.name,
                    "count": format_proportion(row),
                    "wilson": format_wilson(row),
                }
            )
    return rows


def _boundary_card(row: BoundaryRow) -> tuple[str, str, str, str]:
    if row.measured_kind == "measured":
        value = row.measured_decision or "unparsed"
        status = row.measured_status or "recorded"
        kind = "measured"
    else:
        value = "Not recorded"
        status = "no measurement"
        kind = "unavailable"
    return (
        f"{row.value} bps · {status}",
        value,
        f"Distance {row.distance}. Oracle illustration {row.oracle_decision}. Predicate M is {str(row.predicate_m).lower()}.",
        kind,
    )


def _boundary_table(rows: Sequence[BoundaryRow], controls: Sequence[ControlRelease]) -> None:
    control_lookup = {}
    if controls:
        for arm in control_decisions(controls[0]):
            control_lookup.setdefault(arm.value, []).append(f"{arm.role}: {', '.join(arm.decisions)}")
    table = []
    for row in rows:
        table.append(
            {
                "error_rate_bps": row.value,
                "distance": row.distance,
                "predicate_m": row.predicate_m,
                "oracle illustration": row.oracle_decision,
                "x-ray measured": row.measured_decision if row.measured_kind == "measured" else "not recorded",
                "x-ray status": row.measured_status if row.measured_kind == "measured" else "not recorded",
                "hash matches record": row.hash_matches_record,
                "control decisions": "; ".join(control_lookup.get(row.value, [])) or "not in the first control study",
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")
    if controls:
        st.caption(
            "Control decisions are from "
            f"{controls[0].result.study_id}. They are a separate measured study, not additional X-Ray trials."
        )


def _evidence_points(study: XRayRelease) -> list[dict[str, object]]:
    points = [
        {
            "value": study.manifest.original_value,
            "distance": 0,
            "decision": study.result.original_decision or "unparsed",
            "status": "factual_baseline",
            "hover": f"baseline {study.result.original_decision} at {study.manifest.original_value}",
        }
    ]
    for row in study.result.candidates:
        points.append(
            {
                "value": row.candidate_value,
                "distance": row.distance,
                "decision": row.decision or "unparsed",
                "status": row.status,
                "hover": f"{row.status} · {row.decision} · distance {row.distance}",
            }
        )
    return points


def _candidate_table(study: XRayRelease) -> None:
    st.subheader("Candidates")
    rows = []
    for row in study.result.candidates:
        rows.append(
            {
                "value": row.candidate_value,
                "distance": row.distance,
                "order": row.order_index,
                "status": row.status,
                "decision": row.decision,
                "verified flip": row.flipped,
                "execution valid": row.execution_valid,
                "oracle before": row.oracle_before,
                "oracle after": row.oracle_after,
                "model calls": row.model_calls,
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def _condition_card(metric) -> tuple[str, str, str, str]:
    if metric.flip_rate is None or metric.uncertainty is None:
        detail = f"Valid {metric.valid_count}/{metric.attempt_count}. No flip rate is recorded."
        value = "Not evaluable"
    else:
        detail = (
            f"{metric.flip_count}/{metric.valid_count} valid. "
            f"Wilson {metric.uncertainty['low']:.3f}–{metric.uncertainty['high']:.3f}."
        )
        value = f"{metric.flip_count}/{metric.valid_count}"
    return (metric.role, value, detail, "measured")


def _control_chart_rows(study: ControlRelease) -> list[dict[str, object]]:
    rows = []
    for metric in study.result.conditions:
        if metric.flip_rate is None or metric.uncertainty is None:
            continue
        rows.append(
            {
                "label": metric.role,
                "value": metric.flip_rate,
                "low": metric.uncertainty["low"],
                "high": metric.uncertainty["high"],
                "hover": f"{metric.role}<br>{metric.flip_count}/{metric.valid_count}",
            }
        )
    return rows


def _continuation_rows(study: ControlRelease) -> list[dict[str, object]]:
    rows = []
    for row in study.result.continuations:
        rows.append(
            {
                "role": row.role,
                "repetition": row.repetition_index,
                "value": row.field_value,
                "distance": row.distance,
                "status": row.status,
                "decision": row.decision_parsed,
                "decision changed": row.decision_changed,
                "evidence intervention": row.evidence_intervention,
                "valid": row.execution_valid,
                "oracle": row.oracle_decision,
            }
        )
    return rows


def _validity_rows(study: ControlRelease) -> list[dict[str, object]]:
    rows = []
    for metric in study.result.conditions:
        rows.append(
            {
                "role": metric.role,
                "attempts": metric.attempt_count,
                "valid": metric.valid_count,
                "completed parses": metric.completion_count,
                "invalid output": metric.invalid_output_count,
                "provider failures": metric.provider_failure_count,
                "mcp failures": metric.mcp_failure_count,
                "oracle aligned": metric.oracle_aligned_count,
            }
        )
    return rows


def _usage_rows(study: ControlRelease) -> list[dict[str, object]]:
    totals = usage_by_role(study.result)
    rows = []
    for role, bucket in totals.items():
        if bucket is None:
            rows.append({"role": role, "input tokens": "recorded null", "output tokens": "recorded null"})
        else:
            rows.append(
                {
                    "role": role,
                    "input tokens": f"{bucket['input_tokens']:,}",
                    "output tokens": f"{bucket['output_tokens']:,}",
                }
            )
    return rows


def _study_table(bundle: ReleaseBundle) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in bundle.experiments:
        completion = find_proportion(study.summary, "pair_completion")
        rows.append(
            {
                "kind": "experiment",
                "id": study.manifest.experiment_id,
                "schedule": study.manifest.schedule,
                "model": study.manifest.model_id,
                "trials": str(study.summary.trials),
                "recorded summary": format_proportion(completion),
            }
        )
    for study in bundle.xray:
        rows.append(
            {
                "kind": "x-ray",
                "id": study.result.xray_id,
                "schedule": study.manifest.strategy,
                "model": study.manifest.model_id,
                "trials": str(len(study.result.candidates)),
                "recorded summary": study.result.minimality_claim,
            }
        )
    for study in bundle.controls:
        rows.append(
            {
                "kind": "control",
                "id": study.result.study_id,
                "schedule": study.manifest.schedule_name,
                "model": study.manifest.model_id,
                "trials": str(len(study.result.continuations)),
                "recorded summary": f"{study.result.provider_requests} provider requests",
            }
        )
    return rows


def _select_experiment(bundle: ReleaseBundle) -> ExperimentRelease | None:
    if not bundle.experiments:
        return None
    labels = {study.manifest.experiment_id: _study_caption(study) for study in bundle.experiments}
    default = protocol_pilot(bundle)
    ids = [study.manifest.experiment_id for study in bundle.experiments]
    selected = st.selectbox(
        "Experiment",
        ids,
        index=ids.index(default.manifest.experiment_id) if default is not None else 0,
        format_func=labels.get,
    )
    for study in bundle.experiments:
        if study.manifest.experiment_id == selected:
            return study
    return bundle.experiments[0]


def _select_xray(bundle: ReleaseBundle) -> XRayRelease | None:
    if not bundle.xray:
        return None
    if len(bundle.xray) == 1:
        return bundle.xray[0]
    labels = {study.result.xray_id: _xray_caption(study) for study in bundle.xray}
    selected = st.selectbox("X-Ray search", list(labels), format_func=labels.get)
    for study in bundle.xray:
        if study.result.xray_id == selected:
            return study
    return bundle.xray[0]


def _select_control(bundle: ReleaseBundle) -> ControlRelease | None:
    if not bundle.controls:
        return None
    if len(bundle.controls) == 1:
        return bundle.controls[0]
    labels = {study.result.study_id: study.manifest.model_id for study in bundle.controls}
    selected = st.selectbox("Control study", list(labels), format_func=labels.get)
    for study in bundle.controls:
        if study.result.study_id == selected:
            return study
    return bundle.controls[0]


def _usage_text(usage: dict[str, int] | None) -> str:
    if usage is None:
        return "recorded null"
    return f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out"


def _notes(notes: Sequence[str]) -> None:
    if not notes:
        return
    with st.expander("Recorded notes"):
        for note in notes:
            st.text(note)


def _limitations() -> tuple[str, ...]:
    return (
        "The incidents and the decision policy were written for the experiment.",
        "The protocol pilot has five repeats on three scenarios. Its Wilson intervals stay wide.",
        "The protocol treats those pilot repeats as debugging data for the loop.",
        "Hosted model responses can change when the prefix is fixed. A continuation from one checkpoint is not an independent checkpoint sample.",
        "An H1 edit is irrelevant to the oracle. A model can still use the field the oracle ignores.",
        "H3 covers an instruction inside a tool result. It does not cover a malicious server or a poisoned tool description.",
        "The closed JSON decision is easier to score than an open incident write-up.",
        "Academic novelty is not claimed.",
    )


def _source_line(source: str, disclosure: Disclosure) -> None:
    st.caption(f"{source}. {provenance_for(disclosure)}")


def _evidence_selector(study: XRayRelease, options: Sequence[RecordedEvidence]) -> None:
    by_value = {item.value: item for item in options}
    default = study.manifest.original_value if study.manifest.original_value in by_value else options[0].value
    selected = st.segmented_control(
        "Recorded evidence",
        [item.value for item in options],
        default=default,
        format_func=lambda value: f"{value} bps · {by_value[value].role_label}",
        key=f"xray-evidence-{study.result.xray_id}",
        required=True,
        width="stretch",
    )
    chosen = evidence_at(study, int(selected)) if selected is not None else None
    if chosen is None:
        st.caption("That value has no recorded decision. Untested error rates are not filled in.")
        return
    if chosen.role == "factual_baseline":
        change = f"{chosen.original_value} bps"
        change_detail = "Original evidence. Distance 0. This is not a flip trial."
        flip_value = "Baseline"
        flip_detail = "Recorded original decision."
    elif chosen.role == "verified_intervention":
        change = f"{chosen.original_value} → {chosen.value}"
        change_detail = f"Absolute edit distance {chosen.distance} bps. Recorded status verified_flip."
        flip_value = "Verified flip"
        flip_detail = "One continuation from the shared checkpoint."
    else:
        change = f"{chosen.original_value} → {chosen.value}"
        change_detail = f"Absolute edit distance {chosen.distance} bps. Recorded status no_flip."
        flip_value = "No flip"
        flip_detail = "The recorded decision stayed. This is not the control study's negative arm."
    card_row(
        (
            ("Evidence change", change, change_detail, "measured"),
            ("Recorded decision", chosen.decision, f"{chosen.role_label}. Study {chosen.study_id}.", "measured"),
            (flip_value, chosen.status, flip_detail, "measured"),
        )
    )
    st.caption(
        f"{chosen.model_id} · {chosen.fixture_id} · {chosen.tool}.{chosen.field} · "
        f"phase {chosen.phase} · shared checkpoint {str(chosen.shared_checkpoint).lower()} · "
        f"{chosen.minimality_claim}"
    )
    if chosen.canonical_json:
        with st.expander(f"Canonical evidence at {chosen.value} basis points"):
            st.caption("Model-visible observation. The fixture hash matches the recorded evidence hash.")
            st.code(chosen.canonical_json, language="json")
            st.caption(f"sha256 {chosen.evidence_sha256}")
    elif chosen.evidence_sha256 is None:
        st.caption("Canonical JSON is withheld unless it matches the recorded evidence hash.")


def _oracle_illustration(study: XRayRelease, bundle: ReleaseBundle) -> None:
    try:
        rows = boundary_comparison(study)
    except DashboardLoadError as exc:
        st.error(str(exc))
        return
    with st.expander("Oracle illustration and control comparison"):
        st.caption(
            "The oracle column applies the decision policy to the fixture. "
            "It is not the recorded model decision. Control decisions are a separate study."
        )
        matched = [row for row in rows if row.hash_matches_record]
        if len(matched) != len(rows):
            st.warning(
                "A reconstructed fixture hash does not match the recorded evidence hash. "
                "That JSON is withheld. The measured decision still comes from the result file."
            )
        _boundary_table(rows, bundle.controls)


def _scope_lists(bundle: ReleaseBundle, disclosure: Disclosure) -> None:
    with st.expander("Not implemented"):
        for item in bundle.not_implemented or NOT_IMPLEMENTED:
            st.markdown(f"- {item}")
    omitted = visible_omissions(bundle, disclosure)
    if omitted:
        st.subheader("Directories without a result summary")
        for item in omitted:
            st.markdown(f"- `{item.kind}` `{item.run_id}`: {item.reason}")
