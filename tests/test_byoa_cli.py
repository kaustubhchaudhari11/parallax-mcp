"""CLI guards. Framework execution tests skip when that extra is absent."""

import json
from pathlib import Path

import pytest

from parallax.byoa.cli import main

ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_SPEC = ROOT / "examples" / "byoa" / "langgraph" / "spec.json"
FACTORY = "examples/byoa/langgraph/agent.py:build_target"


def test_cli_requires_one_mode() -> None:
    args = ["--framework", "langgraph", "--agent-factory", FACTORY, "--spec", str(LANGGRAPH_SPEC)]
    assert main(args) == 2
    assert main([*args, "--dry-run", "--live"]) == 2


def test_cli_rejects_a_framework_mismatch() -> None:
    code = main(
        [
            "--framework",
            "crewai",
            "--agent-factory",
            FACTORY,
            "--spec",
            str(LANGGRAPH_SPEC),
            "--dry-run",
        ]
    )
    assert code == 2


def test_live_scripted_graph_writes_a_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph")
    import builtins

    loaded: list[str] = []
    real_import = builtins.__import__

    def _guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "parallax.openai_adapter":
            loaded.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guard)
    code = main(
        [
            "--framework",
            "langgraph",
            "--agent-factory",
            FACTORY,
            "--spec",
            str(LANGGRAPH_SPEC),
            "--live",
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert loaded == []
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["population"] == "byoa"
    assert report["verdict"] == "PASS"
    assert report["shared_checkpoint"] is True
    assert report["comparison_mode"] == "checkpoint_continuation"
    assert report["factual_decision"] == "MONITOR"
    assert report["counterfactual_decision"] == "ESCALATE"
    assert "sk-" not in written[0].read_text(encoding="utf-8")
    assert "api_key" not in written[0].read_text(encoding="utf-8").lower()


def test_live_latency_agent_writes_its_own_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph")
    import builtins

    loaded: list[str] = []
    real_import = builtins.__import__

    def _guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "parallax.openai_adapter" or name == "openai":
            loaded.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guard)
    code = main(
        [
            "--framework",
            "langgraph",
            "--agent-factory",
            "examples/byoa/latency/agent.py:build_target",
            "--spec",
            str(ROOT / "examples" / "byoa" / "latency" / "spec.json"),
            "--live",
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert loaded == []
    report = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["factual_decision"] == "KEEP"
    assert report["counterfactual_decision"] == "PAGE"
    assert report["modified_field"] == "metrics.latency_p99_ms"
    assert report["shared_checkpoint"] is True
    assert report["model_id"] == "scripted-latency-policy"
    assert report["population"] == "byoa"
