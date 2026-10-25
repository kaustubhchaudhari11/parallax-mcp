"""Reference CrewAI incident crew.

Two agents run in sequence. The metrics analyst reads one tool. The
incident investigator turns that tool output into a decision. This crew
does not call the native agent loop or the incident oracle.
"""

import json
import re

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from parallax.byoa.crewai_adapter import CrewAITarget

_TOOL = "get_incident_evidence"


class PolicyLLM(BaseLLM):
    """Scripted model. It follows CrewAI's text tool format and does not call a provider."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        text = _message_text(messages)
        role = getattr(from_agent, "role", "") or ""
        if role == "Metrics Analyst" and _rate(text) is None:
            return (
                "Thought: I need the incident evidence.\n"
                f"Action: {_TOOL}\n"
                "Action Input: {}"
            )
        if role == "Metrics Analyst":
            observed = _observation(text)
            body = observed if observed is not None else "{}"
            return "Thought: I now can give a great answer\nFinal Answer: " + body
        rate = _rate(text)
        if rate is None:
            return "Thought: I now can give a great answer\nFinal Answer: {}"
        if rate >= 500:
            decision = "ESCALATE"
        elif rate >= 200:
            decision = "INVESTIGATE"
        else:
            decision = "MONITOR"
        return (
            "Thought: I now can give a great answer\n"
            f'Final Answer: {{"decision": "{decision}"}}'
        )


def build_target() -> CrewAITarget:
    """Return a factory that builds a new crew for each evidence object."""

    return CrewAITarget(
        build_crew=_build_crew,
        tool_name=_TOOL,
        decision_key="decision",
        side_effects="read_only",
        model_id="scripted-incident-policy",
    )


def _build_crew(evidence: dict) -> Crew:
    @tool(_TOOL)
    def get_incident_evidence() -> str:
        """Return the controlled incident evidence as JSON."""

        return json.dumps(evidence)

    llm = PolicyLLM(model="scripted-incident-policy", temperature=0)
    analyst = Agent(
        role="Metrics Analyst",
        goal="Read the incident evidence tool and report its JSON.",
        backstory="You only report the tool result.",
        llm=llm,
        tools=[get_incident_evidence],
        allow_delegation=False,
        allow_code_execution=False,
        max_iter=4,
        max_retry_limit=0,
        verbose=False,
    )
    investigator = Agent(
        role="Incident Investigator",
        goal="Choose MONITOR, INVESTIGATE, or ESCALATE from the evidence.",
        backstory="You return one JSON object with a decision.",
        llm=llm,
        tools=[],
        allow_delegation=False,
        allow_code_execution=False,
        max_iter=4,
        max_retry_limit=0,
        verbose=False,
    )
    reading = Task(
        description="Use get_incident_evidence and return its JSON unchanged.",
        expected_output="The JSON object returned by the tool.",
        agent=analyst,
    )
    decision = Task(
        description="Read the metrics evidence and return JSON with a decision key.",
        expected_output='A JSON object with a decision field.',
        agent=investigator,
        context=[reading],
    )
    return Crew(
        agents=[analyst, investigator],
        tasks=[reading, decision],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
        tracing=False,
        planning=False,
    )


def _message_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for item in messages:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("content"), str):
            parts.append(item["content"])
    return "\n".join(parts)


def _observation(text: str) -> str | None:
    index = text.rfind("Observation:")
    if index < 0:
        return None
    start = text.find("{", index)
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return json.dumps(value)


def _rate(text: str) -> int | None:
    match = re.search(r'error_rate_bps"?\s*:\s*(\d+)', text)
    if match is None:
        return None
    return int(match.group(1))
