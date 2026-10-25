"""Evaluate one compiled LangGraph graph from a shared checkpoint.

The APIs used here were checked against LangGraph 1.2.17:

- ``langgraph.graph.StateGraph``
- ``StateGraph.compile(..., interrupt_before=...)``
- ``langgraph.checkpoint.memory.InMemorySaver``
- ``CompiledStateGraph.invoke``
- ``CompiledStateGraph.get_state``
- ``CompiledStateGraph.get_state_history``
- ``CompiledStateGraph.update_state``

``update_state`` creates a new checkpoint. It does not edit the checkpoint
it was given. Channels that declare a reducer accumulate; channels
without a reducer overwrite. A graph that does not expose a checkpoint
before the decision node is reported as unsupported.
"""

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from parallax.byoa.contract import EvaluationSpec
from parallax.byoa.policy import judge
from parallax.byoa.report import ByoaReport, public_error, utc_now

_REQUIRED = ("invoke", "get_state", "get_state_history", "update_state")
_LIMITATIONS = (
    "One trial is not a reliability rate.",
    "This population is not pooled with the PARALLAX pilot, X-Ray, or control studies.",
    "The verdict uses the evaluation contract, not the PARALLAX incident oracle.",
    "A decision change is not automatically correct.",
    "Checkpoint continuation re-executes nodes after the shared checkpoint.",
    "A state channel with a reducer can accumulate an edit instead of replacing it.",
    "A graph with no checkpointer, or no checkpoint before the decision node, is unsupported.",
    "The adapter rejects a target that declares side effects. It cannot see effects hidden in arbitrary node code.",
    "Model and tool counts come from state fields the graph itself updates.",
)


@dataclass(frozen=True)
class LangGraphTarget:
    """A compiled graph and the state fields this adapter is allowed to use."""

    graph: Any
    evidence_key: str
    decision_key: str
    decision_node: str
    side_effects: str = "read_only"
    model_id: str | None = None


def evaluate_langgraph(
    spec: EvaluationSpec,
    factory: Any,
    original: dict[str, Any],
    mutated: dict[str, Any],
    common: dict[str, Any],
) -> ByoaReport:
    """Fork one real graph at the pre-decision checkpoint."""

    errors: list[str] = []
    target = _target(factory, errors)
    if target is None:
        return _report(spec, common, errors, execution_valid=False, shared=False)
    if not _graph_is_langgraph(target.graph):
        errors.append("compiled object is not a LangGraph graph")
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    if target.side_effects != "read_only":
        errors.append("side-effecting workflows are unsupported")
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    thread_id = "byoa-" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        target.evidence_key: copy.deepcopy(original),
        "model_requests": 0,
        "tool_requests": 0,
    }
    try:
        _invoke(target.graph, initial, config, spec.timeout_seconds)
        history = list(target.graph.get_state_history(config))
    except TimeoutError as exc:
        errors.append(public_error(exc))
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    except Exception as exc:  # noqa: BLE001 — a graph failure is a report row, not a crash
        errors.append("checkpoint history is unavailable: " + public_error(exc))
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    pre = next((item for item in history if item.next == (target.decision_node,)), None)
    if pre is None:
        errors.append("no checkpoint exists before the decision node")
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    if target.evidence_key not in pre.values:
        errors.append("captured evidence is missing")
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    field = spec.selected_intervention.field
    if _lookup(pre.values.get(target.evidence_key), field) != _lookup(original, field):
        errors.append("captured evidence does not match the spec")
        return _report(spec, common, errors, execution_valid=False, shared=False, model_id=target.model_id)
    checkpoint_id = pre.config["configurable"]["checkpoint_id"]
    try:
        fork = target.graph.update_state(pre.config, {target.evidence_key: copy.deepcopy(mutated)})
        fork_state = target.graph.get_state(fork)
        parent_state = target.graph.get_state(
            {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        )
    except Exception as exc:  # noqa: BLE001 — a graph failure is a report row, not a crash
        errors.append(public_error(exc))
        return _report(
            spec,
            common,
            errors,
            execution_valid=False,
            shared=False,
            model_id=target.model_id,
            checkpoint_id=checkpoint_id,
        )
    if not _same_evidence(parent_state.values.get(target.evidence_key), original):
        errors.append("the original checkpoint was overwritten")
        return _report(
            spec,
            common,
            errors,
            execution_valid=False,
            shared=False,
            model_id=target.model_id,
            checkpoint_id=checkpoint_id,
        )
    observed = _lookup(fork_state.values.get(target.evidence_key), spec.selected_intervention.field)
    if observed != spec.selected_intervention.value:
        errors.append("state reducer did not replace the evidence field")
        return _report(
            spec,
            common,
            errors,
            execution_valid=False,
            shared=False,
            model_id=target.model_id,
            checkpoint_id=checkpoint_id,
        )
    try:
        factual = _invoke(target.graph, None, pre.config, spec.timeout_seconds)
        counterfactual = _invoke(target.graph, None, fork, spec.timeout_seconds)
    except TimeoutError as exc:
        errors.append(public_error(exc))
        return _report(
            spec,
            common,
            errors,
            execution_valid=False,
            shared=False,
            model_id=target.model_id,
            checkpoint_id=checkpoint_id,
        )
    except Exception as exc:  # noqa: BLE001 — a graph failure is a report row, not a crash
        errors.append(public_error(exc))
        return _report(
            spec,
            common,
            errors,
            execution_valid=False,
            shared=False,
            model_id=target.model_id,
            checkpoint_id=checkpoint_id,
        )
    shared = _shares_parent(fork_state, checkpoint_id)
    if not shared:
        errors.append("branches do not share the pre-decision checkpoint")
    factual_decision = _decision(factual, target.decision_key)
    counter_decision = _decision(counterfactual, target.decision_key)
    model_requests, tool_requests = _usage(pre.values, factual, counterfactual)
    if model_requests > spec.max_model_requests or tool_requests > spec.max_tool_requests:
        errors.append("request budget was exceeded")
    if factual_decision is None or counter_decision is None:
        errors.append("decision is missing")
    elif factual_decision not in spec.allowed_decisions or counter_decision not in spec.allowed_decisions:
        errors.append("decision is not an allowed label")
    execution_valid = not errors
    verdict = judge(
        factual_decision,
        counter_decision,
        spec.expected_behavior,
        set(spec.allowed_decisions),
        execution_valid=execution_valid,
    )
    return _report(
        spec,
        common,
        errors,
        execution_valid=execution_valid,
        shared=shared,
        model_id=target.model_id,
        checkpoint_id=checkpoint_id,
        factual=factual_decision,
        counterfactual=counter_decision,
        verdict=verdict,
        model_requests=model_requests,
        tool_requests=tool_requests,
    )


def _target(factory: Any, errors: list[str]) -> LangGraphTarget | None:
    try:
        value = factory()
    except Exception as exc:  # noqa: BLE001 — a graph failure is a report row, not a crash
        errors.append(public_error(exc))
        return None
    if not isinstance(value, LangGraphTarget):
        errors.append("factory did not return a LangGraph target")
        return None
    return value


def _graph_is_langgraph(graph: Any) -> bool:
    if not type(graph).__module__.startswith("langgraph"):
        return False
    return all(callable(getattr(graph, name, None)) for name in _REQUIRED)


def _invoke(graph: Any, value: Any, config: dict[str, Any], timeout: float) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(graph.invoke, value, config)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise TimeoutError("graph execution timed out") from exc


def _decision(result: Any, key: str) -> str | None:
    if not isinstance(result, dict):
        return None
    value = result.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _usage(prefix: dict[str, Any], factual: Any, counterfactual: Any) -> tuple[int, int]:
    prefix_model = _count(prefix, "model_requests")
    prefix_tool = _count(prefix, "tool_requests")
    model = prefix_model
    tool = prefix_tool
    model += max(0, _count(factual, "model_requests") - prefix_model)
    model += max(0, _count(counterfactual, "model_requests") - prefix_model)
    tool += max(0, _count(factual, "tool_requests") - prefix_tool)
    tool += max(0, _count(counterfactual, "tool_requests") - prefix_tool)
    return model, tool


def _count(value: Any, key: str) -> int:
    if not isinstance(value, dict):
        return 0
    found = value.get(key, 0)
    if isinstance(found, bool) or not isinstance(found, int) or found < 0:
        return 0
    return found


def _shares_parent(fork_state: Any, checkpoint_id: str) -> bool:
    parent = getattr(fork_state, "parent_config", None)
    if not isinstance(parent, dict):
        return False
    configurable = parent.get("configurable")
    if not isinstance(configurable, dict):
        return False
    return configurable.get("checkpoint_id") == checkpoint_id


def _same_evidence(left: Any, right: dict[str, Any]) -> bool:
    return left == right


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
    execution_valid: bool,
    shared: bool,
    model_id: str | None = None,
    checkpoint_id: str | None = None,
    factual: str | None = None,
    counterfactual: str | None = None,
    verdict: str = "INCONCLUSIVE",
    model_requests: int = 0,
    tool_requests: int = 0,
) -> ByoaReport:
    return ByoaReport(
        created_at=utc_now(),
        framework_version=_framework_version(),
        model_id=model_id,
        factual_decision=factual,
        counterfactual_decision=counterfactual,
        verdict=verdict,  # type: ignore[arg-type]
        execution_valid=execution_valid,
        shared_checkpoint=shared,
        checkpoint_id=checkpoint_id,
        model_requests=model_requests,
        tool_requests=tool_requests,
        token_usage=None,
        errors=errors,
        limitations=list(_LIMITATIONS),
        **common,
    )


def _framework_version() -> str:
    try:
        return version("langgraph")
    except PackageNotFoundError:
        return "absent"
