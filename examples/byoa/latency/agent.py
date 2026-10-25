"""Latency graph connected only through the public BYOA contract.

The decision labels and the edited field differ from the incident
example. This file does not call the native agent loop or the incident
oracle, and it does not change the adapter.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from parallax.byoa.langgraph_adapter import LangGraphTarget


class LatencyState(TypedDict, total=False):
    """Evidence has no reducer, so the checkpoint edit replaces it."""

    evidence: dict
    decision: str
    model_requests: int
    tool_requests: int


def collect(state: LatencyState) -> dict:
    """Record one read of the latency evidence."""

    if "latency_p99_ms" not in state["evidence"]["metrics"]:
        raise ValueError("latency evidence is missing")
    return {"tool_requests": state.get("tool_requests", 0) + 1}


def decide(state: LatencyState) -> dict:
    """Map p99 latency to KEEP, WATCH, or PAGE."""

    latency = state["evidence"]["metrics"]["latency_p99_ms"]
    if latency >= 1000:
        decision = "PAGE"
    elif latency >= 400:
        decision = "WATCH"
    else:
        decision = "KEEP"
    return {"decision": decision, "model_requests": state.get("model_requests", 0)}


def build_target() -> LangGraphTarget:
    """Compile a graph that pauses before the latency decision."""

    graph = StateGraph(LatencyState)
    graph.add_node("collect", collect)
    graph.add_node("decide", decide)
    graph.add_edge(START, "collect")
    graph.add_edge("collect", "decide")
    graph.add_edge("decide", END)
    compiled = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["decide"])
    return LangGraphTarget(
        graph=compiled,
        evidence_key="evidence",
        decision_key="decision",
        decision_node="decide",
        side_effects="read_only",
        model_id="scripted-latency-policy",
    )
