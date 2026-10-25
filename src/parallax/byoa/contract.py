"""Typed contract for one local Bring Your Own Agent evaluation.

The contract is data. It does not load a factory, call a model, or apply
the PARALLAX incident oracle. A customer policy is valid only when it is
written in this spec.
"""

import copy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FrameworkName = Literal["langgraph", "crewai"]
ComparisonMode = Literal["checkpoint_continuation", "controlled_fresh_rerun"]
SideEffectPolicy = Literal["read_only"]
CredentialPolicy = Literal["none", "env"]
EvidenceType = Literal["int", "str", "bool"]
Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]
_FRAMEWORK_MODE = {
    "langgraph": "checkpoint_continuation",
    "crewai": "controlled_fresh_rerun",
}


class ByoaSpecError(ValueError):
    """The evaluation spec cannot be executed."""


class DecisionRule(BaseModel):
    """One user-written condition on a parsed decision label."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["equals", "one_of"]
    value: str | None = None
    values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def rule_matches_its_operator(self) -> "DecisionRule":
        if self.op == "equals":
            if not self.value or self.values:
                raise ValueError("equals requires one value")
        elif not self.values or self.value is not None:
            raise ValueError("one_of requires values")
        return self


class ExpectedBehavior(BaseModel):
    """The customer's condition for this pair. It is not an oracle."""

    model_config = ConfigDict(extra="forbid")

    name: str
    factual: DecisionRule
    counterfactual: DecisionRule

    @field_validator("name")
    @classmethod
    def name_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("expected behavior name is required")
        return value.strip()


class Intervention(BaseModel):
    """The one evidence edit this run applies."""

    model_config = ConfigDict(extra="forbid")

    field: str
    value: int | str | bool


class EvaluationSpec(BaseModel):
    """One validated local evaluation. Execution is a separate step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    framework: FrameworkName
    agent_factory: str
    task_id: str
    evidence_source: str
    evidence: dict[str, Any]
    evidence_schema: dict[str, EvidenceType]
    editable_fields: list[str]
    interventions: dict[str, list[int | str | bool]]
    selected_intervention: Intervention
    allowed_decisions: list[str]
    expected_behavior: ExpectedBehavior
    max_model_requests: int = Field(ge=0)
    max_tool_requests: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0, le=120)
    output_dir: str = "byoa-runs"
    comparison_mode: ComparisonMode
    side_effects: SideEffectPolicy = "read_only"
    credentials: CredentialPolicy = "none"

    @model_validator(mode="after")
    def spec_is_executable(self) -> "EvaluationSpec":
        if not self.task_id.strip():
            raise ValueError("task id is required")
        if not self.evidence_source.strip():
            raise ValueError("evidence source is required")
        if _FRAMEWORK_MODE[self.framework] != self.comparison_mode:
            raise ValueError(
                f"{self.framework} uses comparison mode {_FRAMEWORK_MODE[self.framework]}"
            )
        if len(self.allowed_decisions) != len(set(self.allowed_decisions)):
            raise ValueError("decision labels must be unique")
        if any(not label.strip() for label in self.allowed_decisions):
            raise ValueError("decision labels must be non-empty")
        _require_factory_reference(self.agent_factory)
        _validate_evidence(self.evidence, self.evidence_schema)
        editable = set(self.editable_fields)
        if editable != set(self.interventions):
            raise ValueError("interventions must match the editable fields")
        for field in self.editable_fields:
            if field not in self.evidence_schema:
                raise ValueError(f"editable field is not in the schema: {field}")
            allowed = self.interventions[field]
            if len(allowed) != len({_stable(item) for item in allowed}):
                raise ValueError(f"intervention values must be unique: {field}")
            for item in allowed:
                _check_type(field, item, self.evidence_schema[field])
        selected = self.selected_intervention
        if selected.field not in editable:
            raise ValueError("selected field is not editable")
        if not any(item == selected.value for item in self.interventions[selected.field]):
            raise ValueError("selected intervention is not an allowed value")
        if _lookup(self.evidence, selected.field) == selected.value:
            raise ValueError("selected intervention does not change the evidence")
        for rule in (self.expected_behavior.factual, self.expected_behavior.counterfactual):
            labels = [rule.value] if rule.op == "equals" else rule.values
            for label in labels:
                if label not in self.allowed_decisions:
                    raise ValueError("expected decision is not an allowed label")
        return self


def load_spec(payload: str | bytes) -> EvaluationSpec:
    """Parse a spec and reject anything the model does not declare."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        return EvaluationSpec.model_validate_json(text)
    except ValueError as exc:
        raise ByoaSpecError(str(exc)) from exc


def edited_evidence(spec: EvaluationSpec) -> dict[str, Any]:
    """Return a deep copy of the evidence with the one selected edit applied."""

    return _assign(spec.evidence, spec.selected_intervention.field, spec.selected_intervention.value)


def _require_factory_reference(reference: str) -> None:
    if reference.count(":") != 1 or "\\" in reference or reference.startswith("/"):
        raise ValueError("agent factory must be a relative file and attribute")
    path, attribute = reference.split(":")
    parts = tuple(Path(path).parts)
    if not path.endswith(".py") or ".." in parts or not attribute.isidentifier():
        raise ValueError("agent factory must be a relative file and attribute")


def _validate_evidence(evidence: dict[str, Any], schema: dict[str, EvidenceType]) -> None:
    if not schema:
        raise ValueError("evidence schema is required")
    leaves = dict(_leaves(evidence))
    if set(leaves) != set(schema):
        raise ValueError("evidence leaves must match the schema")
    for path, expected in schema.items():
        _check_type(path, leaves[path], expected)


def _leaves(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        if not value:
            raise ValueError("evidence object cannot be empty")
        for key, item in value.items():
            if not key.isidentifier():
                raise ValueError("evidence field names must be identifiers")
            child = f"{prefix}.{key}" if prefix else key
            yield from _leaves(item, child)
        return
    if prefix:
        yield prefix, value


def _check_type(path: str, value: Any, expected: EvidenceType) -> None:
    if expected == "bool":
        valid = isinstance(value, bool)
    elif expected == "int":
        valid = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid = isinstance(value, str)
    if not valid:
        raise ValueError(f"evidence field has the wrong type: {path}")


def _lookup(evidence: dict[str, Any], path: str) -> Any:
    current: Any = evidence
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"evidence field is missing: {path}")
        current = current[part]
    return current


def _assign(evidence: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    copied = copy.deepcopy(evidence)
    current = copied
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value
    return copied


def _stable(value: int | str | bool) -> str:
    return f"{type(value).__name__}:{value}"
