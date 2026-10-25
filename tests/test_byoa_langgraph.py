"""LangGraph checkpoint evaluation. The model is the graph, not PARALLAX."""

import ast
import copy
import time
from pathlib import Path
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from parallax.byoa.contract import DecisionRule, load_spec
from parallax.byoa.langgraph_adapter import LangGraphTarget
from parallax.byoa.report import evidence_hash
from parallax.byoa.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "examples" / "byoa" / "langgraph" / "spec.json"
AGENT = ROOT / "examples" / "byoa" / "langgraph" / "agent.py"


def test_reference_graph_forks_one_checkpoint() -> None:
    spec = load_spec(SPEC_PATH.read_text(encoding="utf-8"))
    before = copy.deepcopy(spec.evidence)
    report = evaluate(spec, ROOT)
    assert spec.evidence == before
    assert type(report).__name__ == "ByoaReport"
    assert report.verdict == "PASS"
    assert report.execution_valid is True
    assert report.shared_checkpoint is True
    assert report.comparison_mode == "checkpoint_continuation"
    assert report.factual_decision == "MONITOR"
    assert report.counterfactual_decision == "ESCALATE"
    assert report.checkpoint_id
    assert report.model_requests == 0
    assert report.tool_requests == 1
    assert report.population == "byoa"
    assert report.original_evidence_sha256 == evidence_hash(before)
    assert report.original_evidence_sha256 != report.modified_evidence_sha256
    assert "sk-" not in report.model_dump_json()
    graph = _compiled_example()
    assert type(graph).__module__.startswith("langgraph")
    assert hasattr(graph, "get_state_history")
    assert hasattr(graph, "update_state")


def test_parent_checkpoint_is_not_reported_as_overwritten() -> None:
    spec = load_spec(SPEC_PATH.read_text(encoding="utf-8"))
    report = evaluate(spec, ROOT)
    assert report.shared_checkpoint is True
    assert report.errors == []


def test_reducer_that_keeps_the_old_value_is_inconclusive() -> None:
    spec = _spec()
    spec = spec.model_copy(update={"agent_factory": "tests/test_byoa_langgraph.py:build_reducer_target"})
    # The runner loads a file. Build the target in-process instead.
    from parallax.byoa.langgraph_adapter import evaluate_langgraph
    from parallax.byoa.report import contract_hash

    identity = contract_hash(spec.model_dump(mode="json"))
    report = evaluate_langgraph(spec, build_reducer_target, spec.evidence, _edited(spec), _common(spec, identity))
    assert report.verdict == "INCONCLUSIVE"
    assert report.execution_valid is False
    assert any("reducer" in item for item in report.errors)


def test_missing_decision_checkpoint_is_unsupported() -> None:
    spec = _spec()
    from parallax.byoa.langgraph_adapter import evaluate_langgraph
    from parallax.byoa.report import contract_hash

    identity = contract_hash(spec.model_dump(mode="json"))
    report = evaluate_langgraph(spec, build_unsupported_target, spec.evidence, _edited(spec), _common(spec, identity))
    assert report.verdict == "INCONCLUSIVE"
    assert any("no checkpoint" in item for item in report.errors)


def test_wiped_evidence_is_not_a_successful_edit() -> None:
    spec = _spec()
    from parallax.byoa.langgraph_adapter import evaluate_langgraph
    from parallax.byoa.report import contract_hash

    identity = contract_hash(spec.model_dump(mode="json"))
    report = evaluate_langgraph(spec, build_wiping_target, spec.evidence, _edited(spec), _common(spec, identity))
    assert report.verdict == "INCONCLUSIVE"
    assert any("captured evidence" in item for item in report.errors)


def test_policy_violation_is_fail_and_a_bad_label_is_inconclusive() -> None:
    from parallax.byoa.langgraph_adapter import evaluate_langgraph
    from parallax.byoa.report import contract_hash

    spec = _spec()
    behavior = spec.expected_behavior.model_copy(
        update={
            "factual": DecisionRule(op="equals", value="ESCALATE"),
            "name": "require-escalation-on-both",
        }
    )
    spec = spec.model_copy(update={"expected_behavior": behavior})
    identity = contract_hash(spec.model_dump(mode="json"))
    report = evaluate_langgraph(spec, _example_factory, spec.evidence, _edited(spec), _common(spec, identity))
    assert report.execution_valid is True
    assert report.verdict == "FAIL"
    assert report.factual_decision == "MONITOR"

    invalid = evaluate_langgraph(
        _spec(),
        build_invalid_decision_target,
        _spec().evidence,
        _edited(_spec()),
        _common(_spec(), "invalid"),
    )
    assert invalid.verdict == "INCONCLUSIVE"
    assert any("allowed label" in item for item in invalid.errors)


def test_provider_error_and_side_effects_do_not_pass() -> None:
    from parallax.byoa.langgraph_adapter import evaluate_langgraph

    spec = _spec()
    failed = evaluate_langgraph(spec, build_failing_target, spec.evidence, _edited(spec), _common(spec, "err"))
    assert failed.verdict == "INCONCLUSIVE"
    assert failed.factual_decision is None
    assert all("sk-" not in item for item in failed.errors)

    sided = evaluate_langgraph(spec, build_side_effect_target, spec.evidence, _edited(spec), _common(spec, "side"))
    assert sided.verdict == "INCONCLUSIVE"
    assert any("side-effect" in item for item in sided.errors)


def test_latency_agent_uses_its_own_field_and_policy() -> None:
    spec = load_spec((ROOT / "examples" / "byoa" / "latency" / "spec.json").read_text(encoding="utf-8"))
    report = evaluate(spec, ROOT)
    assert report.verdict == "PASS"
    assert report.execution_valid is True
    assert report.shared_checkpoint is True
    assert report.comparison_mode == "checkpoint_continuation"
    assert report.modified_field == "metrics.latency_p99_ms"
    assert report.factual_decision == "KEEP"
    assert report.counterfactual_decision == "PAGE"
    assert report.expected_behavior == "page-when-p99-latency-is-raised"
    assert report.model_id == "scripted-latency-policy"
    assert report.model_requests == 0
    assert spec.evidence["metrics"]["error_rate_bps"] == 100
    source = (ROOT / "examples" / "byoa" / "latency" / "agent.py").read_text(encoding="utf-8")
    assert "error_rate_bps" not in source
    assert "parallax.agent" not in _imports(ROOT / "examples" / "byoa" / "latency" / "agent.py")
    assert "parallax.oracle" not in _imports(ROOT / "examples" / "byoa" / "latency" / "agent.py")


def test_reference_source_is_a_langgraph_graph() -> None:
    text = AGENT.read_text(encoding="utf-8")
    modules = _imports(AGENT)
    assert "langgraph" in modules
    assert "parallax.oracle" not in modules
    assert "StateGraph" in text
    assert "InMemorySaver" in text
    assert "interrupt_before" in text


def build_slow_target() -> LangGraphTarget:
    class State(TypedDict, total=False):
        evidence: dict
        decision: str

    def collect(state: State) -> dict:
        time.sleep(1.0)
        return {"tool_requests": 1}

    def decide(state: State) -> dict:
        return {"decision": "MONITOR"}

    return _graph(State, collect, decide)


def test_timeout_is_inconclusive() -> None:
    spec = _spec().model_copy(update={"timeout_seconds": 0.2})
    from parallax.byoa.langgraph_adapter import evaluate_langgraph

    report = evaluate_langgraph(spec, build_slow_target, spec.evidence, _edited(spec), _common(spec, "timeout"))
    assert report.verdict == "INCONCLUSIVE"
    assert any("timed out" in item for item in report.errors)


def test_dry_run_does_not_invoke_the_graph(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from langgraph.graph.state import CompiledStateGraph

    from parallax.byoa.cli import main

    calls: list[str] = []
    original = CompiledStateGraph.invoke

    def _spy(self, *args, **kwargs):
        calls.append("invoke")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CompiledStateGraph, "invoke", _spy)
    code = main(
        [
            "--framework",
            "langgraph",
            "--agent-factory",
            "examples/byoa/langgraph/agent.py:build_target",
            "--spec",
            str(SPEC_PATH),
            "--dry-run",
        ]
    )
    assert code == 0
    assert calls == []
    output = capsys.readouterr().out
    assert "dry-run: no model request will be sent" in output
    assert "live_execution=false" in output
    assert "max_model_requests=0" in output


def build_reducer_target() -> LangGraphTarget:
    """A graph whose evidence channel ignores the forked value."""

    def keep_first(left: dict, right: dict) -> dict:
        if left:
            return left
        return right

    class State(TypedDict, total=False):
        evidence: Annotated[dict, keep_first]
        decision: str

    def collect(state: State) -> dict:
        return {"tool_requests": 1}

    def decide(state: State) -> dict:
        return {"decision": "MONITOR"}

    return _graph(State, collect, decide)


def build_unsupported_target() -> LangGraphTarget:
    """A graph that never checkpoints the declared decision node."""

    class State(TypedDict, total=False):
        evidence: dict
        decision: str

    def only(state: State) -> dict:
        return {"decision": "MONITOR"}

    graph = StateGraph(State)
    graph.add_node("only", only)
    graph.add_edge(START, "only")
    graph.add_edge("only", END)
    compiled = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["only"])
    return LangGraphTarget(
        graph=compiled,
        evidence_key="evidence",
        decision_key="decision",
        decision_node="decide",
        model_id="unsupported",
    )


def build_wiping_target() -> LangGraphTarget:
    """A node that drops the evidence before the decision checkpoint."""

    class State(TypedDict, total=False):
        evidence: dict
        decision: str

    def collect(state: State) -> dict:
        return {"evidence": {}}

    def decide(state: State) -> dict:
        return {"decision": "MONITOR"}

    return _graph(State, collect, decide)


def build_invalid_decision_target() -> LangGraphTarget:
    class State(TypedDict, total=False):
        evidence: dict
        decision: str

    def collect(state: State) -> dict:
        return {}

    def decide(state: State) -> dict:
        return {"decision": "REBOOT"}

    return _graph(State, collect, decide)


def build_failing_target() -> LangGraphTarget:
    class State(TypedDict, total=False):
        evidence: dict
        decision: str

    def collect(state: State) -> dict:
        return {}

    def decide(state: State) -> dict:
        raise RuntimeError("provider unavailable")

    return _graph(State, collect, decide)


def build_side_effect_target() -> LangGraphTarget:
    target = _example_factory()
    return LangGraphTarget(
        graph=target.graph,
        evidence_key=target.evidence_key,
        decision_key=target.decision_key,
        decision_node=target.decision_node,
        side_effects="external",
        model_id=target.model_id,
    )


def _example_factory() -> LangGraphTarget:
    return _load_example()()


def _compiled_example():
    return _load_example()().graph


def _load_example():
    import importlib.util

    spec = importlib.util.spec_from_file_location("byoa_langgraph_example", AGENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_target


def _graph(state, collect, decide) -> LangGraphTarget:
    graph = StateGraph(state)
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
        model_id="scripted",
    )


def _spec():
    return load_spec(SPEC_PATH.read_text(encoding="utf-8"))


def _edited(spec):
    from parallax.byoa.contract import edited_evidence

    return edited_evidence(spec)


def _common(spec, identity: str) -> dict:
    from parallax.byoa.report import evidence_hash as digest

    edited = _edited(spec)
    return {
        "run_id": "byoa-" + identity[:16],
        "task_id": spec.task_id,
        "agent_framework": spec.framework,
        "factory_fingerprint": "test",
        "contract_hash": identity,
        "evidence_source": spec.evidence_source,
        "modified_field": spec.selected_intervention.field,
        "original_evidence_sha256": digest(spec.evidence),
        "modified_evidence_sha256": digest(edited),
        "expected_behavior": spec.expected_behavior.name,
        "comparison_mode": spec.comparison_mode,
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
            modules.add(node.module)
    return modules
