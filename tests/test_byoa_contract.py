"""Evaluation contract tests. These tests do not call a model."""

import ast
import json
from pathlib import Path

import pytest

from parallax.byoa.contract import ByoaSpecError, EvaluationSpec, load_spec
from parallax.byoa.policy import judge
from parallax.byoa.report import (
    MAX_REPORT_BYTES,
    ByoaReport,
    ByoaReportError,
    read_report_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "examples" / "byoa" / "langgraph" / "spec.json"


def test_example_spec_is_valid() -> None:
    spec = load_spec(SPEC.read_text(encoding="utf-8"))
    assert spec.framework == "langgraph"
    assert spec.comparison_mode == "checkpoint_continuation"
    assert spec.credentials == "none"
    assert spec.side_effects == "read_only"


def test_framework_mode_is_fixed() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    payload["comparison_mode"] = "controlled_fresh_rerun"
    with pytest.raises(ByoaSpecError, match="checkpoint_continuation"):
        load_spec(json.dumps(payload))


def test_unknown_edit_and_noop_are_rejected() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    payload["selected_intervention"] = {"field": "logs.error_count", "value": 1}
    with pytest.raises(ByoaSpecError, match="not editable"):
        load_spec(json.dumps(payload))
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    payload["interventions"] = {"metrics.error_rate_bps": [100, 800]}
    payload["selected_intervention"] = {"field": "metrics.error_rate_bps", "value": 100}
    with pytest.raises(ByoaSpecError, match="does not change"):
        load_spec(json.dumps(payload))
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    payload["selected_intervention"] = {"field": "metrics.error_rate_bps", "value": 50}
    with pytest.raises(ByoaSpecError, match="not an allowed value"):
        load_spec(json.dumps(payload))


def test_factory_reference_stays_inside_the_project() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    payload["agent_factory"] = "../outside.py:build_target"
    with pytest.raises(ByoaSpecError):
        load_spec(json.dumps(payload))
    payload["agent_factory"] = "examples/byoa/langgraph/agent.py:build_target"
    assert EvaluationSpec.model_validate(payload).agent_factory.endswith("build_target")


def test_verdicts_follow_the_contract() -> None:
    spec = load_spec(SPEC.read_text(encoding="utf-8"))
    behavior = spec.expected_behavior
    allowed = set(spec.allowed_decisions)
    assert judge("MONITOR", "ESCALATE", behavior, allowed, execution_valid=True) == "PASS"
    assert judge("MONITOR", "MONITOR", behavior, allowed, execution_valid=True) == "FAIL"
    assert judge(None, "ESCALATE", behavior, allowed, execution_valid=True) == "INCONCLUSIVE"
    assert judge("MONITOR", "ESCALATE", behavior, allowed, execution_valid=False) == "INCONCLUSIVE"
    assert judge("NOPE", "ESCALATE", behavior, allowed, execution_valid=True) == "INCONCLUSIVE"


def test_report_rejects_secrets_and_oversize_files() -> None:
    with pytest.raises(ByoaReportError, match="size"):
        read_report_bytes(b"{" * (MAX_REPORT_BYTES + 1))
    with pytest.raises(ByoaReportError, match="secret"):
        read_report_bytes(b'{"api_key":"sk-test"}')


def test_report_schema_requires_the_byoa_population() -> None:
    payload = {
        "schema_version": 1,
        "population": "pilot",
        "run_id": "byoa-test",
        "created_at": "2026-10-25T00:00:00+00:00",
        "task_id": "incident-error-rate",
        "agent_framework": "langgraph",
        "framework_version": "1.2.17",
        "model_id": None,
        "factory_fingerprint": "abc",
        "contract_hash": "def",
        "evidence_source": "get_metrics",
        "modified_field": "metrics.error_rate_bps",
        "original_evidence_sha256": "1",
        "modified_evidence_sha256": "2",
        "factual_decision": "MONITOR",
        "counterfactual_decision": "ESCALATE",
        "expected_behavior": "example",
        "verdict": "PASS",
        "execution_valid": True,
        "comparison_mode": "checkpoint_continuation",
        "shared_checkpoint": True,
        "checkpoint_id": "checkpoint",
        "model_requests": 0,
        "tool_requests": 1,
        "token_usage": None,
        "errors": [],
        "limitations": ["one trial"],
    }
    with pytest.raises(ValueError):
        ByoaReport.model_validate(payload)


def test_dashboard_and_contract_do_not_import_a_framework() -> None:
    root = ROOT / "src" / "parallax"
    for relative in (
        "byoa/__init__.py",
        "byoa/contract.py",
        "byoa/policy.py",
        "byoa/report.py",
        "byoa/guide.py",
        "dashboard/views.py",
        "dashboard/data.py",
    ):
        modules = _imports(root / relative)
        assert "langgraph" not in modules
        assert "crewai" not in modules
        assert "openai" not in modules
        assert "parallax.agent" not in modules


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
