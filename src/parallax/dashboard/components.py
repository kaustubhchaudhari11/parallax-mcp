"""Visual components for the PARALLAX dashboard."""

import html
from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

CHART_CONFIG = {"displayModeBar": "hover", "responsive": True}
_DECISION_COLORS = {
    "MONITOR": "#1D4E89",
    "INVESTIGATE": "#A15C07",
    "ESCALATE": "#8F2D3C",
    "unparsed": "#5C6B7A",
}
_STATUS_COLORS = {
    "verified_flip": "#0F6E6B",
    "no_flip": "#1D4E89",
    "factual_baseline": "#142033",
    "decision_changed": "#0F6E6B",
    "decision_stable": "#1D4E89",
}


def inject_css() -> None:
    """Apply the release theme. Content styles are static."""

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.35rem; padding-bottom: 3.2rem; max-width: 1120px; }
          section[data-testid="stSidebar"] { border-right: 1px solid #e4e9ef; }
          .px-kicker {
            letter-spacing: 0.14em; text-transform: uppercase; font-size: 0.74rem;
            font-weight: 700; color: #0f6e6b; margin: 0;
          }
          .px-title {
            font-size: clamp(1.85rem, 3vw, 2.4rem); line-height: 1.12; letter-spacing: -0.035em;
            color: #142033; margin: 0.15rem 0 0.45rem; font-weight: 700;
          }
          .px-position {
            font-size: 1.08rem; line-height: 1.5; color: #243140; margin: 0 0 0.35rem;
            max-width: 42rem;
          }
          .px-tagline {
            font-size: 0.92rem; line-height: 1.4; color: #0c5c59; font-weight: 700;
            margin: 0 0 0.85rem; letter-spacing: 0.01em;
          }
          .px-card {
            background: #ffffff; border: 1px solid #e4e9ef; border-left: 4px solid #0f6e6b;
            border-radius: 12px; padding: 0.9rem 1rem 0.85rem; min-height: 7.4rem; height: 100%;
            box-shadow: 0 1px 0 rgba(20, 32, 51, 0.03); margin-bottom: 0.35rem;
          }
          .px-card.illustration { border-left-color: #a15c07; }
          .px-card.unavailable { border-left-color: #8b98a5; }
          .px-card.definition { border-left-color: #1d4e89; }
          .px-card.alert { border-left-color: #8f2d3c; }
          .px-card .label { color: #526173; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.35rem; }
          .px-card .value { color: #142033; font-size: 1.55rem; letter-spacing: -0.03em; font-weight: 700; }
          .px-card .detail { color: #526173; font-size: 0.86rem; margin-top: 0.35rem; line-height: 1.4; }
          .px-pill {
            display: inline-block; border-radius: 999px; padding: 0.12rem 0.55rem;
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em; margin-right: 0.35rem;
          }
          .px-pill.measured { background: #e5f4f3; color: #0c5c59; }
          .px-pill.illustration { background: #f8efdc; color: #7a4b04; }
          .px-pill.unavailable { background: #eef1f4; color: #3e4c59; }
          .px-arch { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.45rem; }
          .px-step {
            flex: 1 1 9.2rem; background: #ffffff; border: 1px solid #e4e9ef; border-radius: 12px;
            padding: 0.7rem 0.75rem; min-height: 6.6rem;
          }
          .px-step-index {
            display: block; color: #0c5c59; font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.08em; margin-bottom: 0.2rem;
          }
          .px-step strong { display: block; color: #142033; margin-bottom: 0.25rem; }
          .px-step span { color: #243140; font-size: 0.86rem; line-height: 1.4; }
          .px-arrow { align-self: center; color: #0c5c59; font-weight: 700; font-size: 1.15rem; }
          section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.74rem; color: #526173;
          }
          @media (max-width: 800px) {
            .px-arch { flex-direction: column; }
            .px-arrow { align-self: center; transform: rotate(90deg); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, position: str, tagline: str | None = None) -> None:
    """Render the product header."""

    tagline_html = f"<p class='px-tagline'>{html.escape(tagline)}</p>" if tagline else ""
    st.markdown(
        (
            f"<p class='px-kicker'>{html.escape(kicker)}</p>"
            f"<h1 class='px-title'>{html.escape(title)}</h1>"
            f"<p class='px-position'>{html.escape(position)}</p>"
            f"{tagline_html}"
        ),
        unsafe_allow_html=True,
    )


def pills(*kinds: tuple[str, str]) -> None:
    """Render static status pills. Each item is ``(label, kind)``."""

    markup = []
    for label, kind in kinds:
        safe_kind = kind if kind in {"measured", "illustration", "unavailable"} else "unavailable"
        markup.append(f"<span class='px-pill {safe_kind}'>{html.escape(label)}</span>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def card_row(cards: Sequence[tuple[str, str, str, str]]) -> None:
    """Render one row of metric cards.

    Each card is ``(label, value, detail, kind)``.
    """

    if not cards:
        return
    columns = st.columns(len(cards))
    for column, (label, value, detail, kind) in zip(columns, cards, strict=True):
        safe_kind = kind if kind in {"measured", "illustration", "unavailable", "alert", "definition"} else "measured"
        with column:
            st.markdown(
                (
                    f"<div class='px-card {safe_kind}'>"
                    f"<div class='label'>{html.escape(label)}</div>"
                    f"<div class='value'>{html.escape(value)}</div>"
                    f"<div class='detail'>{html.escape(detail)}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def architecture_flow(steps: Sequence[tuple[str, str]]) -> None:
    """Render a wrapping, ordered path. Each step is ``(title, body)``."""

    parts = ["<div class='px-arch' role='list'>"]
    for index, (title, body) in enumerate(steps):
        if index:
            parts.append("<div class='px-arrow' aria-hidden='true'>→</div>")
        parts.append(
            "<div class='px-step' role='listitem'>"
            f"<span class='px-step-index'>{index + 1:02d}</span>"
            f"<strong>{html.escape(title)}</strong>"
            f"<span>{html.escape(body)}</span>"
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def show_chart(figure: go.Figure) -> None:
    """Show one Plotly figure at the content width."""

    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def wilson_chart(rows: Sequence[dict[str, object]]) -> go.Figure:
    """Forest plot of recorded rates and their Wilson intervals."""

    figure = _base("Recorded rates and 95% Wilson intervals")
    labels = [str(row["label"]) for row in rows]
    values = [float(row["value"]) for row in rows]
    high = [max(0.0, float(row["high"]) - float(row["value"])) for row in rows]
    low = [max(0.0, float(row["value"]) - float(row["low"])) for row in rows]
    figure.add_trace(
        go.Scatter(
            x=values,
            y=labels,
            mode="markers",
            marker={"size": 11, "color": "#0F6E6B"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": high,
                "arrayminus": low,
                "color": "#0F6E6B",
                "thickness": 1.5,
                "width": 0,
            },
            hovertext=[str(row["hover"]) for row in rows],
            hoverinfo="text",
            name="Recorded rate",
        )
    )
    figure.update_xaxes(range=[-0.02, 1.02], tickformat=".0%", title="Rate", showgrid=True, gridcolor="#e6ebf1")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return figure


def histogram_chart(histogram: dict[str, dict[str, int]]) -> go.Figure:
    """Grouped baseline decision counts."""

    figure = _base("Baseline decisions by fixture")
    decisions = ("MONITOR", "INVESTIGATE", "ESCALATE", "unparsed")
    fixtures = list(histogram)
    for decision in decisions:
        figure.add_trace(
            go.Bar(
                name=decision,
                x=fixtures,
                y=[histogram[fixture].get(decision, 0) for fixture in fixtures],
                marker_color=_DECISION_COLORS[decision],
            )
        )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title="Baseline trials", dtick=1, showgrid=True, gridcolor="#e6ebf1")
    return figure


def discordant_chart(table: dict[str, dict[str, int]]) -> go.Figure:
    """Discordant-pair cells recorded for each scenario."""

    figure = _base("Discordant pairs")
    cells = (
        "parsed_pairs",
        "both_match",
        "factual_match_mutant_miss",
        "factual_miss_mutant_match",
        "both_miss",
    )
    scenarios = list(table)
    for index, cell in enumerate(cells):
        figure.add_trace(
            go.Bar(
                name=cell,
                x=scenarios,
                y=[table[scenario].get(cell, 0) for scenario in scenarios],
                marker_color=["#142033", "#0F6E6B", "#A15C07", "#1D4E89", "#8F2D3C"][index],
            )
        )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title="Parsed pairs", dtick=1, showgrid=True, gridcolor="#e6ebf1")
    return figure


def count_chart(title: str, counts: dict[str, int]) -> go.Figure:
    """Bar chart of recorded counts."""

    figure = _base(title, height=320)
    keys = list(counts)
    figure.add_trace(go.Bar(x=keys, y=[counts[key] for key in keys], marker_color="#1D4E89"))
    figure.update_yaxes(title="Count", showgrid=True, gridcolor="#e6ebf1")
    return figure


def rate_chart(title: str, rows: Sequence[dict[str, object]]) -> go.Figure:
    """Point estimates with optional Wilson bars."""

    figure = _base(title, height=340)
    labels = [str(row["label"]) for row in rows]
    values = [float(row["value"]) for row in rows]
    high = [max(0.0, float(row["high"]) - float(row["value"])) for row in rows]
    low = [max(0.0, float(row["value"]) - float(row["low"])) for row in rows]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#0F6E6B",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": high,
                "arrayminus": low,
                "color": "#142033",
                "thickness": 1.4,
                "width": 6,
            },
            hovertext=[str(row["hover"]) for row in rows],
            hoverinfo="text",
            name="Recorded rate",
        )
    )
    figure.update_yaxes(range=[0, 1.05], tickformat=".0%", title="Flip rate", showgrid=True, gridcolor="#e6ebf1")
    return figure


def evidence_chart(rows: Sequence[dict[str, object]]) -> go.Figure:
    """Recorded decision against edit distance for one evidence field."""

    figure = _base("Recorded decisions by evidence value", height=360)
    for status in dict.fromkeys(str(row["status"]) for row in rows):
        selected = [row for row in rows if str(row["status"]) == status]
        figure.add_trace(
            go.Scatter(
                x=[int(row["value"]) for row in selected],
                y=[int(row["distance"]) for row in selected],
                mode="markers+text",
                text=[str(row["decision"]) for row in selected],
                textposition="top center",
                marker={"size": 16, "color": _STATUS_COLORS.get(status, "#A15C07")},
                name=status,
                hovertext=[str(row["hover"]) for row in selected],
                hoverinfo="text",
            )
        )
    figure.update_xaxes(title="error_rate_bps", dtick=1)
    figure.update_yaxes(title="Absolute edit distance", dtick=1, showgrid=True, gridcolor="#e6ebf1")
    return figure


def _base(title: str, height: int = 400) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title={"text": title, "x": 0, "font": {"size": 16}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "color": "#142033", "size": 13},
        margin={"l": 56, "r": 16, "t": 56, "b": 48},
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"bgcolor": "#ffffff", "font_size": 13},
    )
    return figure
