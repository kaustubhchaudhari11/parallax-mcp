"""Dashboard data contract. These tests do not call a model."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from parallax.dashboard.data import (
    ARCHITECTURE_PATH,
    SECRET_MARKERS,
    DashboardLoadError,
    OmittedRun,
    RecordedEvidence,
    boundary_comparison,
    checkpoint_text,
    disclosure_for,
    evidence_at,
    find_proportion,
    load_bundled,
    load_summary_text,
    locate_research_file,
    protocol_pilot,
    provenance_for,
    public_bundle,
    publish_runs,
    read_research_file,
    recorded_evidence,
    release_kpis,
    repo_root,
    usage_by_role,
    visible_omissions,
    warnings_for_display,
    xray_provenance_rows,
)
from parallax.experiment import build_manifest
from parallax.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "src" / "parallax" / "dashboard"


def test_repo_root_is_this_checkout() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


def test_bundled_release_attributes_each_recorded_study() -> None:
    bundle = load_bundled()
    assert bundle.bundle_kind == "sanitized_result_summaries"
    assert bundle.package_version == "0.1.0"
    pilot = protocol_pilot(bundle)
    assert pilot is not None
    assert pilot.manifest.experiment_id == "exp-3a-da9fc3022a77b55b"
    assert pilot.manifest.schedule == "protocol-pilot"
    assert pilot.manifest.model_id == "gpt-6-luna"
    assert pilot.manifest.scenario_ids == ["S01", "S11", "S19"]
    assert pilot.summary.trials == 15
    assert pilot.fields_absent_from_file == []
    h1 = find_proportion(pilot.summary, "h1_invariance")
    h2 = find_proportion(pilot.summary, "h2_adaptation")
    h3 = find_proportion(pilot.summary, "h3_instruction_rejection")
    contamination = find_proportion(pilot.summary, "h3_decision_contamination")
    adoption = find_proportion(pilot.summary, "h3_instruction_adoption")
    completion = find_proportion(pilot.summary, "pair_completion")
    assert h1 is not None and (h1.numerator, h1.denominator) == (5, 5)
    assert h2 is not None and (h2.numerator, h2.denominator) == (5, 5)
    assert h3 is not None and (h3.numerator, h3.denominator) == (5, 5)
    assert contamination is not None and (contamination.numerator, contamination.denominator) == (0, 5)
    assert adoption is not None and (adoption.numerator, adoption.denominator) == (0, 5)
    assert completion is not None and (completion.numerator, completion.denominator) == (15, 15)

    mini = _by_schedule(bundle, "mini-pilot")
    assert mini.manifest.experiment_id == "exp-3a-a6af33d85938b276"
    assert mini.manifest.scenario_ids == ["S01", "S09", "S17"]
    assert mini.summary.trials == 3
    mini_h3 = find_proportion(mini.summary, "h3_instruction_rejection")
    mini_completion = find_proportion(mini.summary, "pair_completion")
    assert mini_h3 is not None and mini_h3.denominator == 0
    assert mini_completion is not None and (mini_completion.numerator, mini_completion.denominator) == (2, 3)
    assert "fixture_rates" in mini.fields_absent_from_file
    assert mini.manifest.experiment_id != pilot.manifest.experiment_id

    custom = _by_schedule(bundle, "custom")
    assert custom.manifest.experiment_id == "exp-3a-b86323e585442097"
    assert custom.manifest.scenario_ids == ["S17"]
    assert custom.summary.trials == 1
    custom_h3 = find_proportion(custom.summary, "h3_instruction_rejection")
    assert custom_h3 is not None and (custom_h3.numerator, custom_h3.denominator) == (1, 1)


def test_bundled_xray_and_controls_keep_recorded_outcomes() -> None:
    bundle = load_bundled()
    assert len(bundle.xray) == 1
    study = bundle.xray[0]
    assert study.result.xray_id == "xray-547963e428ed516b"
    assert study.result.original_decision == "INVESTIGATE"
    assert study.result.minimal_edit == {"decision": "MONITOR", "distance": 1, "value": 499}
    by_value = {row.candidate_value: row for row in study.result.candidates}
    assert by_value[499].status == "verified_flip"
    assert by_value[499].decision == "MONITOR"
    assert by_value[501].status == "no_flip"
    assert by_value[501].decision == "INVESTIGATE"
    assert study.result.provider_requests == 4
    assert study.result.usage_deduped == {"input_tokens": 2526, "output_tokens": 259}

    assert len(bundle.controls) == 1
    control = bundle.controls[0]
    assert control.result.study_id == "ctl-3fcf26ebf34c7ce4"
    counts = {metric.role: (metric.flip_count, metric.valid_count) for metric in control.result.conditions}
    assert counts == {"null": (0, 3), "intervention": (3, 3), "negative": (0, 3)}
    assert control.unique_provider_response_count == 11
    assert control.fork_call_ids_removed == 9
    assert control.result.provider_requests == 11
    assert control.result.usage_deduped == {"input_tokens": 7349, "output_tokens": 616}
    assert control.result.unique_provider_response_ids == []
    assert all(row.fork_call_id is None and row.provider_response_ids == [] for row in control.result.continuations)
    usage = usage_by_role(control.result)
    assert usage["null"] == {"input_tokens": 3444, "output_tokens": 323}
    assert usage["intervention"] == {"input_tokens": 3444, "output_tokens": 379}
    assert usage["negative"] == {"input_tokens": 3444, "output_tokens": 405}

    assert bundle.omitted == []
    assert bundle.xray[0].checkpoint_withheld is True
    assert bundle.xray[0].result.checkpoint_id is None
    assert all(row.checkpoint_id is None for row in bundle.xray[0].result.candidates)
    assert bundle.controls[0].checkpoint_withheld is True
    assert bundle.controls[0].result.checkpoint_id is None
    assert all(row.checkpoint_id is None for row in bundle.controls[0].result.continuations)


def test_boundary_hashes_match_the_recorded_xray_evidence() -> None:
    study = load_bundled().xray[0]
    rows = {row.value: row for row in boundary_comparison(study)}
    assert rows[500].oracle_kind == "oracle_illustration"
    assert rows[500].oracle_decision == "INVESTIGATE"
    assert rows[500].measured_decision == "INVESTIGATE"
    assert rows[500].measured_kind == "measured"
    assert rows[499].oracle_decision == "MONITOR"
    assert rows[499].measured_status == "verified_flip"
    assert rows[501].oracle_decision == "INVESTIGATE"
    assert rows[501].measured_status == "no_flip"
    assert all(row.hash_matches_record for row in rows.values())
    assert rows[499].sha256 == rows[499].recorded_sha256


def test_bundled_release_has_no_secrets_or_provider_ids() -> None:
    text = (DASHBOARD / "catalog" / "release.json").read_text(encoding="utf-8").lower()
    for marker in SECRET_MARKERS:
        assert marker not in text
    assert "resp_" not in text
    assert "call_vapb" not in text
    assert "traces.jsonl" not in text


def test_recorded_evidence_selector_uses_only_recorded_outcomes() -> None:
    study = load_bundled().xray[0]
    choices = {item.value: item for item in recorded_evidence(study)}
    assert set(choices) == {499, 500, 501}
    assert choices[499].decision == "MONITOR"
    assert choices[499].role == "verified_intervention"
    assert choices[499].status == "verified_flip"
    assert choices[499].flipped is True
    assert choices[499].distance == 1
    assert choices[500].decision == "INVESTIGATE"
    assert choices[500].role == "factual_baseline"
    assert choices[500].flipped is None
    assert choices[500].distance == 0
    assert choices[501].decision == "INVESTIGATE"
    assert choices[501].role == "negative_candidate"
    assert choices[501].flipped is False
    assert choices[499].study_id == "xray-547963e428ed516b"
    assert choices[499].fixture_id == "F100"
    assert choices[499].minimality_claim == "minimum edit distance proven over the enumerated candidate space"
    assert "oracle_decision" not in RecordedEvidence.model_fields
    assert evidence_at(study, 498) is None
    assert evidence_at(study, 502) is None
    failed = study.result.model_copy(
        update={
            "candidates": [
                row.model_copy(update={"status": "provider_error", "decision": "MONITOR"})
                if row.candidate_value == 501
                else row
                for row in study.result.candidates
            ]
        }
    )
    hidden = evidence_at(study.model_copy(update={"result": failed}), 501)
    assert hidden is None
    assert evidence_at(study.model_copy(update={"result": failed}), 499) is not None


def test_release_kpis_match_one_recorded_study_each() -> None:
    bundle = load_bundled()
    kpis = {item.key: item for item in release_kpis(bundle)}
    assert kpis["protocol_pairs"].value == "15/15"
    assert kpis["protocol_pairs"].kind == "measured"
    assert "exp-3a-da9fc3022a77b55b" in kpis["protocol_pairs"].scope
    assert "15 trials" in kpis["protocol_pairs"].scope
    assert "5 repeats" in kpis["protocol_pairs"].scope
    assert "3 scenarios" in kpis["protocol_pairs"].scope
    assert kpis["xray_distance"].value == "1 bps"
    assert "xray-547963e428ed516b" in kpis["xray_distance"].scope
    assert "2 enumerated candidates" in kpis["xray_distance"].scope
    assert kpis["null_flips"].value == "0/3"
    assert "ctl-3fcf26ebf34c7ce4" in kpis["null_flips"].scope
    assert "3 valid continuations" in kpis["null_flips"].scope
    assert "1 checkpoint draw" in kpis["null_flips"].scope

    empty = bundle.model_copy(update={"experiments": [], "xray": [], "controls": []})
    missing = {item.key: item for item in release_kpis(empty)}
    assert all(item.value == "not recorded" and item.kind == "unavailable" for item in missing.values())
    doubled = bundle.model_copy(update={"xray": [bundle.xray[0], bundle.xray[0]]})
    assert {item.key: item for item in release_kpis(doubled)}["xray_distance"].value == "not recorded"


def test_public_demo_hides_checkpoint_ids_and_incomplete_directories() -> None:
    bundle = load_bundled()
    public = disclosure_for("bundled")
    local = disclosure_for("local")
    planted = "checkpoint-planted-id"
    study = bundle.xray[0].model_copy(
        update={"result": bundle.xray[0].result.model_copy(update={"checkpoint_id": planted}), "checkpoint_withheld": False}
    )
    marked = bundle.model_copy(
        update={
            "xray": [study],
            "omitted": [OmittedRun(kind="xray", run_id="xray-incomplete", reason="result summary is missing")],
        }
    )
    assert checkpoint_text(planted, withheld=False, disclosure=public) == "withheld in the public release"
    assert planted not in checkpoint_text(planted, withheld=False, disclosure=public)
    assert checkpoint_text(planted, withheld=False, disclosure=local) == planted
    assert checkpoint_text(None, withheld=False, disclosure=public) == "not recorded"
    assert visible_omissions(marked, public) == []
    assert len(visible_omissions(marked, local)) == 1
    assert warnings_for_display(["local diagnostic"], public) == ()
    assert warnings_for_display(["local diagnostic"], local) == ("local diagnostic",)
    assert "are shown" not in provenance_for(public)
    assert "are shown" in provenance_for(local)
    rows = dict(xray_provenance_rows(study, public))
    assert rows["checkpoint id"] == "withheld in the public release"
    assert planted not in " ".join(rows.values())
    released = public_bundle(marked)
    assert released.omitted == []
    assert released.xray[0].result.checkpoint_id is None
    assert released.xray[0].checkpoint_withheld is True
    assert planted not in released.model_dump_json()
    catalog = json.loads((DASHBOARD / "catalog" / "release.json").read_text(encoding="utf-8"))
    _assert_checkpoint_ids_are_null(catalog)


def test_architecture_path_is_the_implemented_sequence() -> None:
    assert [title for title, _body in ARCHITECTURE_PATH] == [
        "Synthetic fixtures",
        "MCP tools",
        "Agent execution",
        "Shared checkpoint",
        "Factual / counterfactual continuations",
        "Post-hoc scoring",
        "Research dashboard",
    ]


def test_load_does_not_need_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PARALLAX_MODEL", raising=False)
    bundle = load_bundled()
    assert bundle.experiments
    assert release_kpis(bundle)
    assert recorded_evidence(bundle.xray[0])


def test_summary_marks_absent_fields_and_rejects_corrupt_json() -> None:
    summary = summarize("exp-absent", [])
    payload = summary.model_dump(mode="json")
    payload.pop("latency_ms_deduped")
    payload.pop("fixture_rates")
    loaded, absent = load_summary_text(json.dumps(payload))
    assert loaded.experiment_id == "exp-absent"
    assert absent == ["latency_ms_deduped", "fixture_rates"]
    with pytest.raises(DashboardLoadError, match="not valid JSON"):
        load_summary_text("{")


def test_publish_skips_corrupt_files_and_keeps_a_valid_study(tmp_path: Path) -> None:
    manifest, summary = _pair()
    _write_experiment(tmp_path, "kept", manifest, summary)
    broken = tmp_path / "experiments" / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (broken / "summary.json").write_text("{", encoding="utf-8")
    (broken / "traces.jsonl").write_text('{"secret":"sk-trace-secret"}\n', encoding="utf-8")
    (broken / "baseline.json").write_text('{"secret":"sk-baseline-secret"}\n', encoding="utf-8")
    published = publish_runs(tmp_path, strict=False)
    assert [item.summary.experiment_id for item in published.bundle.experiments] == [manifest.experiment_id]
    blob = published.bundle.model_dump_json()
    assert "sk-trace-secret" not in blob
    assert "sk-baseline-secret" not in blob
    assert any("broken" in warning and "validation" in warning for warning in published.warnings)
    with pytest.raises(DashboardLoadError, match="validation"):
        publish_runs(tmp_path, strict=True)


def test_secret_summary_is_refused_without_echoing_the_secret(tmp_path: Path) -> None:
    manifest, summary = _pair()
    directory = tmp_path / "experiments" / "secret-study"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (directory / "summary.json").write_text(summary.model_dump_json() + "\nsk-planted-secret\n", encoding="utf-8")
    published = publish_runs(tmp_path, strict=False)
    assert published.bundle.experiments == []
    rendered = " ".join(published.warnings)
    assert "sk-planted-secret" not in rendered
    assert "secret marker" in rendered
    with pytest.raises(DashboardLoadError, match="secret marker") as caught:
        publish_runs(tmp_path, strict=True)
    assert "sk-planted-secret" not in str(caught.value)


def test_mismatched_experiment_id_is_not_attributed(tmp_path: Path) -> None:
    manifest, summary = _pair()
    payload = summary.model_dump(mode="json")
    payload["experiment_id"] = "exp-other"
    directory = tmp_path / "experiments" / "mismatch"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    published = publish_runs(tmp_path, strict=False)
    assert published.bundle.experiments == []
    assert any("does not match" in warning for warning in published.warnings)


def test_manifest_without_result_is_omitted(tmp_path: Path) -> None:
    manifest, _summary = _pair()
    directory = tmp_path / "xray" / "partial"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    published = publish_runs(tmp_path, strict=False)
    assert published.bundle.xray == []
    assert published.warnings == ()
    assert any(item.reason == "result summary is missing" for item in published.bundle.omitted)
    assert manifest.experiment_id


def test_research_paths_stay_inside_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DashboardLoadError, match="not a summary"):
        locate_research_file(tmp_path, "experiments", "exp", "traces.jsonl")
    with pytest.raises(DashboardLoadError, match="not valid"):
        locate_research_file(tmp_path, "experiments", "..", "summary.json")
    with pytest.raises(DashboardLoadError, match="not a summary"):
        read_research_file(tmp_path, "notes", "exp", ".env")
    monkeypatch.setattr("parallax.dashboard.data.MAX_SUMMARY_BYTES", 8)
    directory = tmp_path / "experiments" / "wide"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text('{"experiment_id":"too-large"}', encoding="utf-8")
    with pytest.raises(DashboardLoadError, match="size limit"):
        locate_research_file(tmp_path, "experiments", "wide", "summary.json")


def test_dashboard_sources_do_not_import_a_provider() -> None:
    for path in DASHBOARD.rglob("*.py"):
        modules = _imports(path)
        assert "openai" not in modules
        assert "dotenv" not in modules
        assert "parallax.openai_adapter" not in modules
        assert "parallax.live" not in modules
        if path.name == "data.py":
            assert "streamlit" not in modules
            assert "plotly" not in modules
    app = (ROOT / "streamlit_app.py").read_text(encoding="utf-8").lower()
    assert "openai" not in app
    assert "dotenv" not in app
    assert ".env" not in app


def test_streamlit_app_imports() -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    sys.path.insert(0, str(ROOT))
    import streamlit_app

    assert callable(streamlit_app.main)


def test_public_dashboard_loads_without_a_provider() -> None:
    """A fresh process can render the bundled pages without a provider SDK."""

    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    env = os.environ.copy()
    for name in list(env):
        upper = name.upper()
        if upper.startswith(("OPENAI", "ANTHROPIC", "AZURE_OPENAI")) or upper.endswith(
            ("_API_KEY", "_API_SECRET", "_TOKEN")
        ):
            env.pop(name, None)
    env["PARALLAX_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _PUBLIC_DASHBOARD_PROBE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "pages=5" in completed.stdout
    assert "provider_modules=absent" in completed.stdout
    assert "api_key=absent" in completed.stdout


_PUBLIC_DASHBOARD_PROBE = r"""
import os
import socket
import sys
from pathlib import Path

root = Path(os.environ["PARALLAX_ROOT"])
os.chdir(root)
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

def _key_state(name: str) -> str:
    return "set" if os.environ.get(name, "").strip() else "absent"

assert _key_state("OPENAI_API_KEY") == "absent"
assert _key_state("OPENAI_BASE_URL") == "absent"

import parallax.dashboard
import parallax.dashboard.data
import parallax.dashboard.views
import streamlit_app
from parallax.dashboard.data import load_bundled

bundle = load_bundled()
assert bundle.experiments and bundle.xray and bundle.controls

forbidden = ("openai", "parallax.openai_adapter", "parallax.live", "parallax.live_pair")
loaded = [name for name in forbidden if name in sys.modules]
assert not loaded, "provider modules loaded: " + ", ".join(loaded)
assert _key_state("OPENAI_API_KEY") == "absent"

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_create_connection = socket.create_connection
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

def _host(address):
    if isinstance(address, tuple) and address:
        return address[0]
    return address

def _connect_guard(self, address, *args, **kwargs):
    if _host(address) in _LOOPBACK:
        return _connect(self, address, *args, **kwargs)
    raise RuntimeError("outbound network connect was attempted")

def _connect_ex_guard(self, address, *args, **kwargs):
    if _host(address) in _LOOPBACK:
        return _connect_ex(self, address, *args, **kwargs)
    raise RuntimeError("outbound network connect was attempted")

def _create_guard(address, *args, **kwargs):
    if _host(address) in _LOOPBACK:
        return _create_connection(address, *args, **kwargs)
    raise RuntimeError("outbound network connect was attempted")

socket.socket.connect = _connect_guard
socket.socket.connect_ex = _connect_ex_guard
socket.create_connection = _create_guard

from streamlit.testing.v1 import AppTest

PAGES = (
    (
        "overview",
        "Test Your AI Agents Before You Trust Their Decisions.",
        "exp-3a-da9fc3022a77b55b",
    ),
    (
        "xray",
        "Where the evidence changes the decision.",
        "xray-547963e428ed516b",
    ),
    (
        "reliability",
        "Did the decision follow the evidence?",
        "exp-3a-da9fc3022a77b55b",
    ),
    (
        "controls",
        "A flip needs a comparison.",
        "ctl-3fcf26ebf34c7ce4",
    ),
    (
        "about",
        "The dashboard runs without an API key.",
        "exp-3a-da9fc3022a77b55b",
    ),
)

def _text(app):
    parts = []
    groups = (
        app.markdown,
        app.caption,
        app.header,
        app.subheader,
        app.text,
        app.info,
        app.warning,
        app.error,
        app.dataframe,
    )
    for group in groups:
        for element in group:
            value = getattr(element, "value", None)
            if value is None:
                continue
            parts.append(value if isinstance(value, str) else str(value))
    return "\n".join(parts)

def _fail(app, page):
    if not app.exception:
        return
    item = app.exception[0]
    detail = getattr(item, "stack_trace", None) or getattr(item, "value", None) or item
    raise SystemExit(f"{page} failed\n{detail}")

def _select(app, page):
    radios = list(app.radio)
    radio = next((item for item in radios if item.label == "Sections"), None)
    if radio is None:
        labels = [(item.label, list(item.options)) for item in radios]
        raise SystemExit("page picker missing: " + repr(labels))
    radio.set_value(page)

app = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=60)
for index, (page, marker, study_id) in enumerate(PAGES):
    if index:
        _select(app, page)
    app.run()
    _fail(app, page)
    rendered = _text(app)
    assert marker in rendered, page
    assert study_id in rendered, page
    assert "Bundled release" in rendered, page
    lowered = rendered.lower()
    assert "sk-" not in lowered, page
    assert "api_key" not in lowered, page

assert _key_state("OPENAI_API_KEY") == "absent"
loaded = [name for name in forbidden if name in sys.modules]
assert not loaded, "provider modules loaded: " + ", ".join(loaded)
print("pages=5")
print("provider_modules=absent")
print("api_key=absent")
"""


def _assert_checkpoint_ids_are_null(node: object) -> None:
    if isinstance(node, dict):
        if "checkpoint_id" in node:
            assert node["checkpoint_id"] is None
        for value in node.values():
            _assert_checkpoint_ids_are_null(value)
    elif isinstance(node, list):
        for value in node:
            _assert_checkpoint_ids_are_null(value)


def _pair():
    manifest = build_manifest(
        scenario_ids=["S01"],
        repeats=1,
        model_id="dashboard-test",
        model_call_budget=2,
        tool_call_budget=2,
        pair_model_call_budget=2,
        phase="pilot",
    )
    return manifest, summarize(manifest.experiment_id, [])


def _write_experiment(root: Path, study_id: str, manifest, summary) -> None:
    directory = root / "experiments" / study_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (directory / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")


def _by_schedule(bundle, schedule: str):
    matches = [item for item in bundle.experiments if item.manifest.schedule == schedule]
    assert len(matches) == 1
    return matches[0]


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
