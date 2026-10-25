"""Reference LangGraph incident agent.

The decision is produced by this graph. It does not call the native
agent loop or the incident oracle.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from parallax.byoa.langgraph_adapter import LangGraphTarget


class IncidentState(TypedDict, total=False):
    """State channels. ``evidence`` has no reducer, so an edit replaces it."""

    evidence: dict
    decision: str
    model_requests: int
    tool_requests: int


def collect(state: IncidentState) -> dict:
    """Record that the graph read the evidence object."""

    if "metrics" not in state["evidence"]:
        raise ValueError("metrics evidence is missing")
    return {"tool_requests": state.get("tool_requests", 0) + 1}


def decide(state: IncidentState) -> dict:
    """Map the error rate in the graph state to a decision label."""

    rate = state["evidence"]["metrics"]["error_rate_bps"]
    if rate >= 500:
        decision = "ESCALATE"
    elif rate >= 200:
        decision = "INVESTIGATE"
    else:
        decision = "MONITOR"
    return {"decision": decision, "model_requests": state.get("model_requests", 0)}


def build_target() -> LangGraphTarget:
    """Compile a real graph that pauses before the decision node."""

    graph = StateGraph(IncidentState)
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
        model_id="scripted-incident-policy",
    )
