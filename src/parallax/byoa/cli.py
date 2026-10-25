"""CLI for one local Bring Your Own Agent evaluation.

``--dry-run`` validates the spec and factory and does not invoke the
agent. ``--live`` is required before a framework run. The public
dashboard does not call this module.
"""

import argparse
import sys
from pathlib import Path

from parallax.byoa.contract import ByoaSpecError, load_spec
from parallax.byoa.loader import load_factory
from parallax.byoa.report import write_report


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and either print a plan or run one evaluation."""

    parser = argparse.ArgumentParser(prog="python -m parallax.byoa")
    parser.add_argument("--framework", required=True, choices=("langgraph", "crewai"))
    parser.add_argument("--agent-factory", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.live and args.dry_run:
        print("choose either --dry-run or --live", file=sys.stderr)
        return 2
    if not args.live and not args.dry_run:
        print("pass --dry-run or --live", file=sys.stderr)
        return 2
    try:
        spec = load_spec(args.spec.read_text(encoding="utf-8"))
    except (OSError, ByoaSpecError) as exc:
        print(_public(str(exc)), file=sys.stderr)
        return 2
    if args.framework != spec.framework or args.agent_factory != spec.agent_factory:
        print("framework and agent factory must match the spec", file=sys.stderr)
        return 2
    root = _project_root()
    if args.output is not None:
        spec = spec.model_copy(update={"output_dir": str(args.output)})
    if args.dry_run:
        return _dry_run(spec, root)
    if spec.credentials == "env":
        from parallax.openai_adapter import load_project_env

        load_project_env()
    from parallax.byoa.runner import evaluate

    try:
        report = evaluate(spec, root)
    except ByoaSpecError as exc:
        print(_public(str(exc)), file=sys.stderr)
        return 2
    path = write_report(report, spec.output_dir)
    print(f"verdict={report.verdict}")
    print(f"comparison_mode={report.comparison_mode}")
    print(f"shared_checkpoint={str(report.shared_checkpoint).lower()}")
    print(f"model_requests={report.model_requests}")
    print(f"tool_requests={report.tool_requests}")
    print(f"report={path}")
    return 0 if report.verdict != "INCONCLUSIVE" else 1


def _dry_run(spec, root: Path) -> int:
    try:
        factory = load_factory(spec.agent_factory, root)
        target = factory()
    except (ByoaSpecError, OSError, TypeError, ValueError) as exc:
        print(_public(str(exc)), file=sys.stderr)
        return 2
    if target is None or getattr(target, "side_effects", None) != "read_only":
        print("factory must return a read-only target", file=sys.stderr)
        return 2
    selected = spec.selected_intervention
    print("dry-run: no model request will be sent")
    print(f"framework={spec.framework}")
    print(f"comparison_mode={spec.comparison_mode}")
    print(f"task_id={spec.task_id}")
    print(f"evidence_source={spec.evidence_source}")
    print(f"modified_field={selected.field}")
    print(f"intervention_value={selected.value}")
    print(f"expected_behavior={spec.expected_behavior.name}")
    print(f"max_model_requests={spec.max_model_requests}")
    print(f"max_tool_requests={spec.max_tool_requests}")
    print(f"timeout_seconds={spec.timeout_seconds}")
    print("live_execution=false")
    print(f"target={type(target).__name__}")
    return 0


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "examples").is_dir():
            return candidate
    raise ByoaSpecError("PARALLAX project root was not found")


def _public(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in ("sk-", "api_key", "bearer ", "authorization")):
        return "evaluation failed"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
