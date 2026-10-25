"""Bring Your Own Agent evaluation.

Importing this package does not import LangGraph, CrewAI, or a model
provider. Framework adapters are loaded by the CLI when that framework
is selected.
"""

from parallax.byoa.contract import EvaluationSpec
from parallax.byoa.report import ByoaReport

__all__ = ["ByoaReport", "EvaluationSpec"]
