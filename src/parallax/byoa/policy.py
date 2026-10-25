"""User-defined verdicts. Parse failures are never a pass."""

from parallax.byoa.contract import DecisionRule, ExpectedBehavior, Verdict


def rule_holds(rule: DecisionRule, decision: str) -> bool:
    """Return whether one parsed label meets the customer's rule."""

    if rule.op == "equals":
        return decision == rule.value
    return decision in rule.values


def judge(
    factual: str | None,
    counterfactual: str | None,
    behavior: ExpectedBehavior,
    allowed: set[str],
    *,
    execution_valid: bool,
) -> Verdict:
    """Judge a pair. Invalid execution is inconclusive, never a pass."""

    if not execution_valid:
        return "INCONCLUSIVE"
    if factual is None or counterfactual is None:
        return "INCONCLUSIVE"
    if factual not in allowed or counterfactual not in allowed:
        return "INCONCLUSIVE"
    factual_ok = rule_holds(behavior.factual, factual)
    counterfactual_ok = rule_holds(behavior.counterfactual, counterfactual)
    if factual_ok and counterfactual_ok:
        return "PASS"
    return "FAIL"
