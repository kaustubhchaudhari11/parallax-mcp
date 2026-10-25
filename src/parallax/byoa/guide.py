"""Static Bring Your Own Agent copy for the read-only dashboard.

This module does not import LangGraph, CrewAI, or a model provider.
"""

GUIDE = (
    (
        "What this is",
        (
            "Bring Your Own Agent evaluates a compatible local LangGraph graph or CrewAI crew. "
            "You declare one evidence edit and the behavior you expect. "
            "PARALLAX records the decisions the framework actually produced and scores them "
            "with that contract. It does not reuse the incident oracle unless you write that "
            "same condition yourself."
        ),
    ),
    (
        "LangGraph",
        (
            "A supported graph is compiled with a checkpointer and can pause before the node "
            "that reads the evidence. PARALLAX captures that checkpoint, resumes one factual "
            "continuation, and forks a second continuation with update_state. "
            "Channels that use a reducer may accumulate the edit instead of replacing it. "
            "That run is reported as inconclusive."
        ),
    ),
    (
        "CrewAI",
        (
            "A supported crew builds a new Crew for each arm and reads evidence through a tool. "
            "The arms are separate executions. comparison_mode is controlled_fresh_rerun and "
            "shared_checkpoint is false. That is not checkpoint replay. "
            "Delegation and hierarchical managers are rejected."
        ),
    ),
    (
        "Verdicts",
        (
            "PASS means both parsed decisions satisfy the contract. "
            "FAIL means the executions were valid and at least one decision violates the contract. "
            "INCONCLUSIVE means the evidence, execution, or comparison is not enough to judge. "
            "A parse failure is never PASS. A missing observation is never a successful edit. "
            "A decision change is not automatically good or bad."
        ),
    ),
    (
        "Unsupported",
        (
            "The public site cannot run an agent. There is no Run Agent button. "
            "Uploaded Python is never executed. "
            "Workflows that send mail, change databases, move money, deploy infrastructure, "
            "or write to a production service are out of scope. "
            "Arbitrary agents, uncheckpointed graphs, and crews that delegate on their own "
            "are out of scope."
        ),
    ),
)

COMMANDS = (
    (
        "uv run --extra langgraph python -m parallax.byoa --framework langgraph "
        "--agent-factory examples/byoa/langgraph/agent.py:build_target "
        "--spec examples/byoa/langgraph/spec.json --dry-run"
    ),
    (
        "uv run --extra langgraph python -m parallax.byoa --framework langgraph "
        "--agent-factory examples/byoa/langgraph/agent.py:build_target "
        "--spec examples/byoa/langgraph/spec.json --live --output byoa-runs"
    ),
    (
        "uv run --extra langgraph python -m parallax.byoa --framework langgraph "
        "--agent-factory examples/byoa/latency/agent.py:build_target "
        "--spec examples/byoa/latency/spec.json --dry-run"
    ),
    (
        "uv run --extra crewai python -m parallax.byoa --framework crewai "
        "--agent-factory examples/byoa/crewai/crew.py:build_target "
        "--spec examples/byoa/crewai/spec.json --dry-run"
    ),
)

DOC_URL = "https://github.com/kaustubhchaudhari11/parallax-mcp/blob/main/docs/BRING_YOUR_OWN_AGENT.md"
