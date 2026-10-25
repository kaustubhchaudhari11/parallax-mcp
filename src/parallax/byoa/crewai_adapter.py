"""Evaluate one CrewAI crew with two fresh executions.

The APIs used here were checked against CrewAI 1.6.1:

- ``crewai.Agent``
- ``crewai.Task``
- ``crewai.Crew``
- ``crewai.Crew.kickoff``
- ``crewai.Process.sequential``
- ``crewai.llms.base_llm.BaseLLM.call``
- ``crewai.tools.tool``

Each arm builds a new crew. This is not checkpoint continuation.
``comparison_mode`` stays ``controlled_fresh_rerun`` and
``shared_checkpoint`` stays false.
"""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from parallax.byoa.contract import EvaluationSpec
from parallax.byoa.policy import judge
from parallax.byoa.report import ByoaReport, public_error, utc_now

_LIMITATIONS = (
    "One trial is not a reliability rate.",
    "This population is not pooled with the PARALLAX pilot, X-Ray, or control studies.",
    "The verdict uses the evaluation contract, not the PARALLAX incident oracle.",
    "A decision change is not automatically correct.",
    "CrewAI arms are independent fresh runs. They do not share a checkpoint.",
    "A fresh rerun can vary when the model or workflow is stochastic.",
    "This comparison does not by itself prove that the evidence edit caused the decision.",
    "Delegation, hierarchical managers, memory, cache, tracing, and planning are unsupported.",
    "The adapter rejects a target that declares side effects. It cannot see effects hidden in tool code.",
    "Request ceilings wrap model and tool callables exposed on the crew.",
)
_MAX_AGENTS = 4
_MAX_TASKS = 4
_MAX_ITER = 6
_MAX_RETRIES = 1


@dataclass(frozen=True)
class CrewAITarget:
    """A local factory that builds one fresh crew for one evidence object."""

    build_crew: Any
    tool_name: str
    decision_key: str
    side_effects: str = "read_only"
    model_id: str | None = None


@dataclass
class _Budget:
    max_model: int
    max_tool: int
    model: int = 0
    tool: int = 0
    exceeded: bool = False


@dataclass
class _Arm:
    executed: bool
    identity: int | None = None
    decision: str | None = None
    tokens: dict[str, int] = field(default_factory=dict)


def evaluate_crewai(
    spec: EvaluationSpec,
    factory: Any,
    original: dict[str, Any],
    mutated: dict[str, Any],
    common: dict[str, Any],
) -> ByoaReport:
    """Run two isolated crews. The second crew is a new execution."""

    errors: list[str] = []
    target = _target(factory, errors)
    if target is None:
        return _report(spec, common, errors)
    if target.side_effects != "read_only":
        errors.append("side-effecting workflows are unsupported")
        return _report(spec, common, errors, model_id=target.model_id)
    if target.tool_name != spec.evidence_source:
        errors.append("tool name does not match the evidence source")
        return _report(spec, common, errors, model_id=target.model_id)
    budget = _Budget(spec.max_model_requests, spec.max_tool_requests)
    factual = _arm(spec, target, original, _lookup(original, spec.selected_intervention.field), budget, errors)
    counter = _Arm(executed=False)
    if factual.executed and not budget.exceeded and not _blocked(errors):
        counter = _arm(
            spec,
            target,
            mutated,
            spec.selected_intervention.value,
            budget,
            errors,
        )
    if factual.executed and counter.executed and factual.identity == counter.identity:
        errors.append("crews were not reinitialized")
    _finish_decisions(spec, factual.decision, counter.decision, errors)
    if budget.exceeded and not any("budget" in item for item in errors):
        errors.append("request budget was exceeded")
    execution_valid = not errors and factual.executed and counter.executed
    verdict = judge(
        factual.decision,
        counter.decision,
        spec.expected_behavior,
        set(spec.allowed_decisions),
        execution_valid=execution_valid,
    )
    return _report(
        spec,
        common,
        errors,
        model_id=target.model_id,
        factual=factual.decision,
        counterfactual=counter.decision if counter.executed else None,
        verdict=verdict,
        execution_valid=execution_valid,
        model_requests=budget.model,
        tool_requests=budget.tool,
        token_usage=_merge_tokens(factual.tokens, counter.tokens),
    )


def _arm(
    spec: EvaluationSpec,
    target: CrewAITarget,
    evidence: dict[str, Any],
    expected: Any,
    budget: _Budget,
    errors: list[str],
) -> _Arm:
    try:
        crew = target.build_crew(copy.deepcopy(evidence))
    except Exception as exc:  # noqa: BLE001 — a crew failure is a report row, not a crash
        errors.append(public_error(exc))
        return _Arm(executed=False)
    problem = _supported(crew, target.tool_name)
    if problem is not None:
        errors.append(problem)
        return _Arm(executed=False)
    tool = _find_tool(crew, target.tool_name)
    if tool is None:
        errors.append("evidence tool is not attached to the crew")
        return _Arm(executed=False)
    boundary = _probe(tool, expected)
    if boundary is not None:
        errors.append(boundary)
        return _Arm(executed=False)
    if budget.tool >= spec.max_tool_requests or budget.model >= spec.max_model_requests:
        budget.exceeded = True
        errors.append("request budget was exceeded")
        return _Arm(executed=False)
    _guard(crew, tool, spec, budget)
    counted_before = budget.tool
    try:
        result = _kickoff(crew, spec.timeout_seconds)
    except TimeoutError as exc:
        errors.append(public_error(exc))
        return _Arm(executed=False)
    except Exception as exc:  # noqa: BLE001 — a crew failure is a report row, not a crash
        errors.append(public_error(exc))
        if budget.exceeded and not any("budget" in item for item in errors):
            errors.append("request budget was exceeded")
        return _Arm(executed=False)
    used = _used_tools(crew)
    counted = budget.tool - counted_before
    if counted < used:
        budget.tool += used - counted
    if used < 1 and counted < 1:
        errors.append("tool was not invoked")
    if budget.tool > spec.max_tool_requests or budget.model > spec.max_model_requests:
        budget.exceeded = True
        errors.append("request budget was exceeded")
    return _Arm(
        executed=True,
        identity=id(crew),
        decision=_decision(result, target.decision_key),
        tokens=_tokens(crew),
    )


def _supported(crew: Any, tool_name: str) -> str | None:
    if not _is_crew(crew):
        return "compiled object is not a CrewAI crew"
    process = getattr(getattr(crew, "process", None), "value", getattr(crew, "process", None))
    if process != "sequential":
        return "crew process must be sequential"
    if getattr(crew, "manager_agent", None) is not None or getattr(crew, "manager_llm", None) is not None:
        return "a manager agent is unsupported"
    if getattr(crew, "memory", False):
        return "crew memory is unsupported"
    if getattr(crew, "cache", False):
        return "crew cache is unsupported"
    if getattr(crew, "tracing", None) is not False:
        return "crew tracing must be disabled"
    if getattr(crew, "planning", False):
        return "crew planning is unsupported"
    agents = list(getattr(crew, "agents", []) or [])
    tasks = list(getattr(crew, "tasks", []) or [])
    if len(agents) < 2:
        return "crew needs at least two agents"
    if len(agents) > _MAX_AGENTS or len(tasks) > _MAX_TASKS:
        return "crew workflow is too large"
    for agent in agents:
        if getattr(agent, "allow_delegation", False):
            return "delegation is unsupported"
        if getattr(agent, "allow_code_execution", False):
            return "code execution is unsupported"
        if int(getattr(agent, "max_iter", 0) or 0) > _MAX_ITER:
            return "agent iteration bound is too high"
        if int(getattr(agent, "max_retry_limit", 0) or 0) > _MAX_RETRIES:
            return "agent retry bound is too high"
    if not tool_name.strip():
        return "evidence tool name is missing"
    return None


def _is_crew(crew: Any) -> bool:
    if not type(crew).__module__.startswith("crewai"):
        return False
    return callable(getattr(crew, "kickoff", None))


def _find_tool(crew: Any, name: str) -> Any | None:
    found: list[Any] = []
    for agent in crew.agents:
        for item in getattr(agent, "tools", []) or []:
            if getattr(item, "name", None) == name:
                found.append(item)
    if len(found) != 1:
        return None
    return found[0]


def _probe(tool: Any, expected: Any) -> str | None:
    runner = getattr(tool, "_run", None)
    if not callable(runner):
        return "tool boundary did not return the evidence"
    try:
        output = runner()
    except Exception:  # noqa: BLE001 — a tool probe failure is a missing observation
        return "tool boundary did not return the evidence"
    rendered = output if isinstance(output, str) else json.dumps(output, default=str)
    if json.dumps(expected) not in rendered:
        return "tool boundary did not return the evidence"
    return None


def _guard(crew: Any, tool: Any, spec: EvaluationSpec, budget: _Budget) -> None:
    seen: set[int] = set()
    for agent in crew.agents:
        llm = getattr(agent, "llm", None)
        if llm is None or id(llm) in seen or not callable(getattr(llm, "call", None)):
            continue
        seen.add(id(llm))
        original = llm.call

        def limited(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            budget.model += 1
            if budget.model > spec.max_model_requests:
                budget.exceeded = True
                raise RuntimeError("model request budget was exceeded")
            return _original(*args, **kwargs)

        llm.call = limited
    original_func = getattr(tool, "func", None)
    if not callable(original_func):
        return

    def limited_tool(*args: Any, **kwargs: Any) -> Any:
        budget.tool += 1
        if budget.tool > spec.max_tool_requests:
            budget.exceeded = True
            raise RuntimeError("tool request budget was exceeded")
        return original_func(*args, **kwargs)

    tool.func = limited_tool


def _kickoff(crew: Any, timeout: float) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(crew.kickoff)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise TimeoutError("crew execution timed out") from exc


def _used_tools(crew: Any) -> int:
    total = 0
    for task in getattr(crew, "tasks", []) or []:
        value = getattr(task, "used_tools", 0) or 0
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def _decision(result: Any, key: str) -> str | None:
    raw = getattr(result, "raw", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    payload = _json_object(raw)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
        return value


def _tokens(crew: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    seen: set[int] = set()
    for agent in crew.agents:
        llm = getattr(agent, "llm", None)
        if llm is None or id(llm) in seen or not hasattr(llm, "get_token_usage_summary"):
            continue
        seen.add(id(llm))
        summary = llm.get_token_usage_summary()
        data = summary.model_dump() if hasattr(summary, "model_dump") else {}
        for key, value in data.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    if not totals or all(value == 0 for value in totals.values()):
        return {}
    return totals


def _merge_tokens(left: dict[str, int], right: dict[str, int]) -> dict[str, int] | None:
    if not left and not right:
        return None
    keys = set(left) | set(right)
    return {key: left.get(key, 0) + right.get(key, 0) for key in sorted(keys)}


def _finish_decisions(
    spec: EvaluationSpec,
    factual: str | None,
    counterfactual: str | None,
    errors: list[str],
) -> None:
    if errors:
        return
    if factual is None or counterfactual is None:
        errors.append("decision is missing")
        return
    allowed = set(spec.allowed_decisions)
    if factual not in allowed or counterfactual not in allowed:
        errors.append("decision is not an allowed label")


def _blocked(errors: list[str]) -> bool:
    return any(
        item.startswith(
            (
                "delegation",
                "crew ",
                "compiled object",
                "a manager",
                "code execution",
                "agent ",
                "evidence tool",
                "tool boundary",
                "tool was not",
                "side-effect",
            )
        )
        for item in errors
    )


def _target(factory: Any, errors: list[str]) -> CrewAITarget | None:
    try:
        value = factory()
    except Exception as exc:  # noqa: BLE001 — a crew failure is a report row, not a crash
        errors.append(public_error(exc))
        return None
    if not isinstance(value, CrewAITarget) or not callable(value.build_crew):
        errors.append("factory did not return a CrewAI target")
        return None
    return value


def _lookup(evidence: Any, path: str) -> Any:
    current = evidence
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _report(
    spec: EvaluationSpec,
    common: dict[str, Any],
    errors: list[str],
    *,
    model_id: str | None = None,
    factual: str | None = None,
    counterfactual: str | None = None,
    verdict: str = "INCONCLUSIVE",
    execution_valid: bool = False,
    model_requests: int = 0,
    tool_requests: int = 0,
    token_usage: dict[str, int] | None = None,
) -> ByoaReport:
    return ByoaReport(
        created_at=utc_now(),
        framework_version=_framework_version(),
        model_id=model_id,
        factual_decision=factual,
        counterfactual_decision=counterfactual,
        verdict=verdict,  # type: ignore[arg-type]
        execution_valid=execution_valid,
        shared_checkpoint=False,
        checkpoint_id=None,
        model_requests=model_requests,
        tool_requests=tool_requests,
        token_usage=token_usage,
        errors=errors,
        limitations=list(_LIMITATIONS),
        **common,
    )


def _framework_version() -> str:
    try:
        return version("crewai")
    except PackageNotFoundError:
        return "absent"
