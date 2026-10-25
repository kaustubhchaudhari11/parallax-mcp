"""Dispatch one validated spec to the selected framework adapter."""

from pathlib import Path

from parallax.byoa.contract import ByoaSpecError, EvaluationSpec, edited_evidence
from parallax.byoa.loader import factory_fingerprint, load_factory
from parallax.byoa.report import ByoaReport, contract_hash, evidence_hash


def evaluate(spec: EvaluationSpec, root: Path) -> ByoaReport:
    """Run the selected adapter. Imports of a framework happen here."""

    factory = load_factory(spec.agent_factory, root)
    fingerprint = factory_fingerprint(spec.agent_factory, root)
    identity = contract_hash(spec.model_dump(mode="json"))
    original = spec.evidence
    mutated = edited_evidence(spec)
    common = {
        "run_id": "byoa-" + identity[:16],
        "task_id": spec.task_id,
        "agent_framework": spec.framework,
        "factory_fingerprint": fingerprint,
        "contract_hash": identity,
        "evidence_source": spec.evidence_source,
        "modified_field": spec.selected_intervention.field,
        "original_evidence_sha256": evidence_hash(original),
        "modified_evidence_sha256": evidence_hash(mutated),
        "expected_behavior": spec.expected_behavior.name,
        "comparison_mode": spec.comparison_mode,
    }
    if spec.framework == "langgraph":
        from parallax.byoa.langgraph_adapter import evaluate_langgraph

        return evaluate_langgraph(spec, factory, original, mutated, common)
    if spec.framework == "crewai":
        from parallax.byoa.crewai_adapter import evaluate_crewai

        return evaluate_crewai(spec, factory, original, mutated, common)
    raise ByoaSpecError("framework is not supported")
