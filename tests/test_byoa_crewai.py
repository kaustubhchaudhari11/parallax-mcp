"""CrewAI fresh-rerun evaluation. The decision comes from a Crew, not PARALLAX."""

import ast
import copy
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("crewai")

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from parallax.byoa.contract import DecisionRule, edited_evidence, load_spec
from parallax.byoa.crewai_adapter import CrewAITarget, evaluate_crewai
from parallax.byoa.report import contract_hash, evidence_hash
from parallax.byoa.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "examples" / "byoa" / "crewai" / "spec.json"
CREW_FILE = ROOT / "examples" / "byoa" / "crewai" / "crew.py"


def test_kickoff_consumes_the_tool_result() -> None:
    """The crew decision follows the tool payload observed during kickoff."""

    example = _example_module()
    in_kickoff = {"active": False, "arm": 0}
    payloads: list[tuple[int, str]] = []
    investigator: list[tuple[int, str]] = []

    class Watching(example.PolicyLLM):
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
            role = getattr(from_agent, "role", "") or ""
            if in_kickoff["active"] and role == "Incident Investigator":
                investigator.append((in_kickoff["arm"], example._message_text(messages)))
            return super().call(
                messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

    def build(evidence: dict) -> Crew:
        crew = example._build_crew(evidence)
        tool = next(
            item for agent in crew.agents for item in agent.tools if item.name == "get_incident_evidence"
        )
        raw = tool.func

        def wrapped():
            result = raw()
            if in_kickoff["active"]:
                payloads.append((in_kickoff["arm"], result))
            return result

        tool.func = wrapped
        for agent in crew.agents:
            agent.llm = Watching(model="scripted-incident-policy", temperature=0)
        return crew

    original = Crew.kickoff

    def _spy(self, *args, **kwargs):
        in_kickoff["arm"] += 1
        in_kickoff["active"] = True
        try:
            return original(self, *args, **kwargs)
        finally:
            in_kickoff["active"] = False

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Crew, "kickoff", _spy)
    try:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        payload["evidence"]["metrics"]["error_rate_bps"] = 250
        payload["interventions"] = {"metrics.error_rate_bps": [900]}
        payload["selected_intervention"] = {"field": "metrics.error_rate_bps", "value": 900}
        payload["expected_behavior"] = {
            "name": "follow-the-tool-observation",
            "factual": {"op": "equals", "value": "INVESTIGATE"},
            "counterfactual": {"op": "equals", "value": "ESCALATE"},
        }
        spec = load_spec(json.dumps(payload))
        report = _run(
            lambda: CrewAITarget(
                build_crew=build,
                tool_name="get_incident_evidence",
                decision_key="decision",
                model_id="scripted-incident-policy",
            ),
            spec=spec,
        )
    finally:
        monkeypatch.undo()

    assert report.verdict == "PASS"
    assert report.factual_decision == "INVESTIGATE"
    assert report.counterfactual_decision == "ESCALATE"
    factual_tools = [body for arm, body in payloads if arm == 1]
    counter_tools = [body for arm, body in payloads if arm == 2]
    assert factual_tools and all('"error_rate_bps": 250' in body for body in factual_tools)
    assert counter_tools and all('"error_rate_bps": 900' in body for body in counter_tools)
    factual_notes = [text for arm, text in investigator if arm == 1]
    counter_notes = [text for arm, text in investigator if arm == 2]
    assert factual_notes and any("250" in text for text in factual_notes)
    assert counter_notes and any("900" in text for text in counter_notes)
    assert all("800" not in text for text in factual_notes + counter_notes)


def test_reference_crew_reruns_with_separate_evidence() -> None:
    spec = load_spec(SPEC_PATH.read_text(encoding="utf-8"))
    before = copy.deepcopy(spec.evidence)
    report = evaluate(spec, ROOT)
    assert spec.evidence == before
    assert report.verdict == "PASS"
    assert report.execution_valid is True
    assert report.shared_checkpoint is False
    assert report.comparison_mode == "controlled_fresh_rerun"
    assert report.checkpoint_id is None
    assert report.factual_decision == "MONITOR"
    assert report.counterfactual_decision == "ESCALATE"
    assert report.population == "byoa"
    assert report.tool_requests == 2
    assert report.model_requests == 6
    assert report.original_evidence_sha256 == evidence_hash(before)
    assert report.original_evidence_sha256 != report.modified_evidence_sha256
    assert report.framework_version != "absent"
    assert "sk-" not in report.model_dump_json()
    crew = _load_example()().build_crew(copy.deepcopy(before))
    assert type(crew).__module__.startswith("crewai")
    assert isinstance(crew, Crew)
    assert len(crew.agents) >= 2
    assert crew.process == Process.sequential


def test_reference_source_is_a_crewai_crew() -> None:
    text = CREW_FILE.read_text(encoding="utf-8")
    modules = _imports(CREW_FILE)
    assert "crewai" in modules
    assert "parallax.agent" not in modules
    assert "parallax.oracle" not in modules
    assert "class PolicyLLM" in text
    assert "BaseLLM" in text
    assert "def kickoff" not in text


def test_delegation_and_hierarchy_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = Crew.kickoff

    def _spy(self, *args, **kwargs):
        calls.append("kickoff")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Crew, "kickoff", _spy)
    delegated = _run(build_delegating_target)
    assert delegated.verdict == "INCONCLUSIVE"
    assert any("delegation" in item for item in delegated.errors)
    hierarchical = _run(build_hierarchical_target)
    assert hierarchical.verdict == "INCONCLUSIVE"
    assert any("sequential" in item or "manager" in item for item in hierarchical.errors)
    assert calls == []


def test_missing_tool_evidence_is_not_a_successful_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(Crew, "kickoff", lambda self, *args, **kwargs: calls.append("kickoff"))
    report = _run(build_empty_tool_target)
    assert report.verdict == "INCONCLUSIVE"
    assert any("tool boundary" in item for item in report.errors)
    assert calls == []


def test_invalid_and_incomplete_outputs_are_inconclusive() -> None:
    invalid = _run(build_invalid_decision_target)
    assert invalid.verdict == "INCONCLUSIVE"
    assert invalid.factual_decision == "REBOOT" or any("allowed label" in item for item in invalid.errors)
    assert any("allowed label" in item or "decision is missing" in item for item in invalid.errors)

    incomplete = _run(build_incomplete_target)
    assert incomplete.verdict == "INCONCLUSIVE"
    assert any("decision is missing" in item for item in incomplete.errors)


def test_provider_error_and_side_effects_do_not_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = _run(build_failing_target)
    assert failed.verdict == "INCONCLUSIVE"
    assert failed.factual_decision is None
    assert all("sk-" not in item for item in failed.errors)
    assert any("provider unavailable" in item for item in failed.errors)

    calls: list[str] = []

    def _refuse(self, evidence):
        calls.append("build")
        raise AssertionError("side-effecting target was built")

    target = CrewAITarget(
        build_crew=_refuse,
        tool_name="get_incident_evidence",
        decision_key="decision",
        side_effects="external",
        model_id="side",
    )
    report = _run(lambda: target)
    assert report.verdict == "INCONCLUSIVE"
    assert any("side-effect" in item for item in report.errors)
    assert calls == []
    monkeypatch.setattr(Crew, "kickoff", lambda self, *args, **kwargs: calls.append("kickoff"))


def test_request_budget_stops_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(Crew, "kickoff", lambda self, *args, **kwargs: calls.append("kickoff"))
    report = _run(build_reference_target, max_model_requests=0)
    assert report.verdict == "INCONCLUSIVE"
    assert any("budget" in item for item in report.errors)
    assert calls == []

    monkeypatch.undo()
    limited = _run(build_reference_target, max_model_requests=1)
    assert limited.verdict == "INCONCLUSIVE"
    assert limited.execution_valid is False
    assert any("budget" in item for item in limited.errors)
    assert limited.model_requests <= 2
    assert limited.counterfactual_decision is None


def test_token_usage_is_recorded_when_the_model_reports_it() -> None:
    report = _run(build_counting_target)
    assert report.token_usage is not None
    assert report.token_usage["prompt_tokens"] > 0
    assert report.token_usage["successful_requests"] > 0


def test_policy_violation_is_fail() -> None:
    spec = _spec()
    behavior = spec.expected_behavior.model_copy(
        update={
            "name": "require-escalation-on-both",
            "factual": DecisionRule(op="equals", value="ESCALATE"),
        }
    )
    spec = spec.model_copy(update={"expected_behavior": behavior})
    report = _run(build_reference_target, spec=spec)
    assert report.execution_valid is True
    assert report.verdict == "FAIL"
    assert report.factual_decision == "MONITOR"
    assert report.counterfactual_decision == "ESCALATE"


def test_timeout_is_inconclusive() -> None:
    report = _run(build_slow_target, timeout_seconds=0.2)
    assert report.verdict == "INCONCLUSIVE"
    assert report.execution_valid is False
    assert any("timed out" in item for item in report.errors)


def test_dry_run_does_not_kickoff(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from parallax.byoa.cli import main

    calls: list[str] = []
    monkeypatch.setattr(Crew, "kickoff", lambda self, *args, **kwargs: calls.append("kickoff"))
    code = main(
        [
            "--framework",
            "crewai",
            "--agent-factory",
            "examples/byoa/crewai/crew.py:build_target",
            "--spec",
            str(SPEC_PATH),
            "--dry-run",
        ]
    )
    assert code == 0
    assert calls == []
    output = capsys.readouterr().out
    assert "dry-run: no model request will be sent" in output
    assert "controlled_fresh_rerun" in output
    assert "live_execution=false" in output


def build_reference_target() -> CrewAITarget:
    return _load_example()()


def build_counting_target() -> CrewAITarget:
    example = _example_module()

    class Counting(example.PolicyLLM):
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
            self._track_token_usage_internal({"prompt_tokens": 2, "completion_tokens": 1})
            return super().call(
                messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

    def build(evidence: dict) -> Crew:
        crew = example._build_crew(evidence)
        for agent in crew.agents:
            agent.llm = Counting(model="scripted-incident-policy", temperature=0)
        return crew

    return CrewAITarget(
        build_crew=build,
        tool_name="get_incident_evidence",
        decision_key="decision",
        model_id="scripted-incident-policy",
    )


def build_delegating_target() -> CrewAITarget:
    return _variant(allow_delegation=True)


def build_hierarchical_target() -> CrewAITarget:
    return _variant(process=Process.hierarchical)


def build_empty_tool_target() -> CrewAITarget:
    return _variant(evidence_body="{}")


def build_invalid_decision_target() -> CrewAITarget:
    return _variant(decision="REBOOT")


def build_incomplete_target() -> CrewAITarget:
    return _variant(decision="")


def build_failing_target() -> CrewAITarget:
    return _variant(explode=True)


def build_slow_target() -> CrewAITarget:
    return _variant(delay=1.0)


def _variant(
    *,
    allow_delegation: bool = False,
    process: Process = Process.sequential,
    evidence_body: str | None = None,
    decision: str | None = None,
    explode: bool = False,
    delay: float = 0,
) -> CrewAITarget:
    def build(evidence: dict) -> Crew:
        @tool("get_incident_evidence")
        def get_incident_evidence() -> str:
            """Return the controlled incident evidence as JSON."""

            if delay:
                time.sleep(delay)
            if evidence_body is not None:
                return evidence_body
            return json.dumps(evidence)

        llm = _Scripted(decision=decision, explode=explode)
        analyst = _agent("Metrics Analyst", llm, [get_incident_evidence], allow_delegation)
        investigator = _agent("Incident Investigator", llm, [], allow_delegation)
        first = Task(
            description="Use get_incident_evidence and return its JSON.",
            expected_output="JSON evidence.",
            agent=analyst,
        )
        second = Task(
            description="Return JSON with a decision key.",
            expected_output="JSON decision.",
            agent=investigator,
            context=[first],
        )
        return Crew(
            agents=[analyst, investigator],
            tasks=[first, second],
            process=process,
            verbose=False,
            memory=False,
            cache=False,
            tracing=False,
            planning=False,
        )

    return CrewAITarget(
        build_crew=build,
        tool_name="get_incident_evidence",
        decision_key="decision",
        model_id="scripted-variant",
    )


class _Scripted(BaseLLM):
    def __init__(self, *, decision: str | None, explode: bool) -> None:
        super().__init__(model="scripted-variant", temperature=0)
        self._decision = decision
        self._explode = explode

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
        if self._explode:
            raise RuntimeError("provider unavailable")
        text = messages if isinstance(messages, str) else json.dumps(messages, default=str)
        role = getattr(from_agent, "role", "") or ""
        if role == "Metrics Analyst" and "error_rate_bps" not in text:
            return (
                "Thought: I need the incident evidence.\n"
                "Action: get_incident_evidence\n"
                "Action Input: {}"
            )
        if role == "Metrics Analyst":
            return 'Thought: I now can give a great answer\nFinal Answer: {"metrics":{"error_rate_bps": 100}}'
        if not self._decision:
            return "Thought: I now can give a great answer\nFinal Answer: not-json"
        return (
            "Thought: I now can give a great answer\n"
            f'Final Answer: {{"decision": "{self._decision}"}}'
        )


def _agent(role: str, llm: BaseLLM, tools: list, delegation: bool) -> Agent:
    return Agent(
        role=role,
        goal="Follow the supplied evidence.",
        backstory="You stay inside the task.",
        llm=llm,
        tools=tools,
        allow_delegation=delegation,
        allow_code_execution=False,
        max_iter=4,
        max_retry_limit=0,
        verbose=False,
    )


def _run(factory, *, max_model_requests: int | None = None, timeout_seconds: float | None = None, spec=None):
    selected = spec or _spec()
    updates = {}
    if max_model_requests is not None:
        updates["max_model_requests"] = max_model_requests
    if timeout_seconds is not None:
        updates["timeout_seconds"] = timeout_seconds
    if updates:
        selected = selected.model_copy(update=updates)
    identity = contract_hash(selected.model_dump(mode="json"))
    return evaluate_crewai(
        selected,
        factory,
        selected.evidence,
        edited_evidence(selected),
        _common(selected, identity),
    )


def _spec():
    return load_spec(SPEC_PATH.read_text(encoding="utf-8"))


def _load_example():
    module = _example_module()
    return module.build_target


def _example_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("byoa_crewai_example", CREW_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common(spec, identity: str) -> dict:
    edited = edited_evidence(spec)
    return {
        "run_id": "byoa-" + identity[:16],
        "task_id": spec.task_id,
        "agent_framework": spec.framework,
        "factory_fingerprint": "test",
        "contract_hash": identity,
        "evidence_source": spec.evidence_source,
        "modified_field": spec.selected_intervention.field,
        "original_evidence_sha256": evidence_hash(spec.evidence),
        "modified_evidence_sha256": evidence_hash(edited),
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
