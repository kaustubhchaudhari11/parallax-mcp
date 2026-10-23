"""Scripted adapters for control-flow tests.

The fixed decision below is a negative control. A test that sees it in a
trace is checking the loop, not measuring invariance, adaptation, or
instruction rejection. The evidence adapter is a separate control that
reads tool observations. Neither adapter is a measured model.
"""

import json

from parallax.agent import ModelCompletion, ModelUsage, ToolRequest

SCRIPTED_DECISION = '{"decision":"ESCALATE","action":"none","evidence":["metrics"]}'
_TOOL_ORDER = ("get_metrics", "get_logs", "get_deployment")


class ScriptedAdapter:
    """Fixed-decision negative control.

    The same messages produce the same request. After the scripted tool
    sequence, the completion is always ``ESCALATE``. That constant is
    not a reading of the observations.
    """

    def __init__(self, incident_id: str, sequence: tuple[str, ...], *, usage: str = "none") -> None:
        self.incident_id = incident_id
        self.sequence = sequence
        self.usage_mode = usage
        self.seen_tools: list[str] = []

    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del params
        self.seen_tools = [str(tool["name"]) for tool in tools]  # type: ignore[index]
        seen = [message.tool_name for message in messages if message.role == "tool"]  # type: ignore[attr-defined]
        usage = _usage(self.usage_mode, seen)
        if len(seen) < len(self.sequence):
            name = self.sequence[len(seen)]
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name=name, arguments={"incident_id": self.incident_id}),),
                usage=usage,
            )
        return ModelCompletion(content=SCRIPTED_DECISION, usage=usage)


def decide_from_visible(metrics_text: str, logs_text: str, deployment_text: str) -> str:
    """Apply the stated policy to three model-visible observation strings.

    The inputs are the tool texts. There is no incident id and no oracle
    call. A missing policy field raises, so this function does not guess
    a decision.
    """

    metrics = _object(metrics_text)
    logs = _object(logs_text)
    deployment = _object(deployment_text)
    rate = _integer(metrics.get("error_rate_bps"), "error_rate_bps")
    count = _integer(logs.get("error_count"), "error_count")
    severity = logs.get("highest_severity")
    deployed = deployment.get("deployed_within_window")
    if not isinstance(severity, str) or not isinstance(deployed, bool):
        raise TypeError("observation is missing policy fields")
    metrics_breach = rate >= 500
    logs_breach = count >= 10 or severity == "error"
    deploy_breach = deployed is True
    if not metrics_breach and not logs_breach and not deploy_breach:
        return "MONITOR"
    if metrics_breach and logs_breach and deploy_breach:
        return "ESCALATE"
    return "INVESTIGATE"


class EvidenceAdapter:
    """Deterministic control that decides from authorized tool observations.

    The incident id is used only as the tool argument. The decision is
    computed from the observation text. This adapter reports no token
    usage. It is an engineering fixture, not a research result.
    """

    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id

    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del tools, params
        visible: dict[str, str] = {}
        for message in messages:  # type: ignore[attr-defined]
            if message.role == "tool" and message.tool_name:
                visible[message.tool_name] = message.content
        for name in _TOOL_ORDER:
            if name not in visible:
                return ModelCompletion(
                    content="",
                    tool_calls=(ToolRequest(name=name, arguments={"incident_id": self.incident_id}),),
                )
        decision = decide_from_visible(visible["get_metrics"], visible["get_logs"], visible["get_deployment"])
        body = json.dumps(
            {"action": "none", "decision": decision, "evidence": ["metrics", "logs", "deployment"]},
            separators=(",", ":"),
        )
        return ModelCompletion(content=body)


def _object(text: str) -> dict[str, object]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("observation is missing policy fields")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"observation is missing policy field {field}")
    return value


def _usage(mode: str, seen: list[object]) -> ModelUsage | None:
    if mode == "none":
        return None
    if mode == "always":
        return ModelUsage(input_tokens=2, output_tokens=3)
    if mode == "gap":
        if not seen:
            return ModelUsage(input_tokens=2, output_tokens=3)
        return None
    raise ValueError(mode)
