"""Prompt and snapshot isolation. Gold labels stay out of the loop input."""

import ast
import tomllib
from pathlib import Path

from parallax.agent import build_initial_state
from parallax.capture import FORBIDDEN_LABEL_KEYS, label_key_overlap
from parallax.fixtures import FIXTURE_IDS
from parallax.prompt import POLICY, prompt_hash, user_message
from parallax.replay import Checkpoint, copy_state, prefix_document, restore

ROOT = Path(__file__).resolve().parents[1]
DECISION_WORDS = ("MONITOR", "INVESTIGATE", "ESCALATE")
NETWORK_MODULES = {"openai", "anthropic", "httpx", "requests", "headroom", "streamlit", "urllib"}
PROVIDER_PACKAGES = {"openai", "anthropic", "headroom", "streamlit"}
CORE_MODULES = ("agent.py", "prompt.py", "replay.py", "mutate.py", "trace.py", "capture.py")


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


def _requirement_name(requirement: str) -> str:
    name = requirement.split(";", 1)[0].strip().split("[", 1)[0]
    for separator in ("=", "<", ">", "!", "~", " "):
        name = name.split(separator, 1)[0]
    return name.strip().lower()


def _requirement_names(requirements: list[str]) -> set[str]:
    return {_requirement_name(item) for item in requirements}


def test_policy_states_the_thresholds_and_names_no_fixture() -> None:
    assert "500" in POLICY
    assert "10" in POLICY
    assert "5.00%" in POLICY
    for word in DECISION_WORDS:
        assert word in POLICY
    for incident_id in FIXTURE_IDS:
        assert incident_id not in POLICY
    for banned in ("oracle_decision", "expected_decision", "gold"):
        assert banned not in POLICY


def test_user_message_names_the_incident_and_no_decision() -> None:
    for incident_id in FIXTURE_IDS:
        text = user_message(incident_id)
        assert incident_id in text
        for word in DECISION_WORDS:
            assert word not in text
        assert prompt_hash(incident_id) == prompt_hash(incident_id)
        assert prompt_hash(incident_id) != prompt_hash("other")


def test_initial_state_has_no_label_field() -> None:
    for incident_id in FIXTURE_IDS:
        state = build_initial_state(
            incident_id=incident_id,
            model_id="fake",
            model_params={"temperature": 0},
            step_budget=4,
            token_budget=None,
            fixture_hash="abc",
            tool_schemas=[{"name": "get_metrics"}],
        )
        assert label_key_overlap(state.model_dump(mode="json")) == set()
        assert label_key_overlap(prefix_document(state)) == set()
        system, user = state.messages
        assert system.role == "system"
        assert incident_id not in system.content
        assert user.role == "user"
        assert incident_id in user.content
        for word in DECISION_WORDS:
            assert word not in user.content


def test_restore_appends_only_the_selected_observation() -> None:
    state = build_initial_state(
        incident_id="F000",
        model_id="fake",
        model_params={},
        step_budget=2,
        token_budget=None,
        fixture_hash="abc",
        tool_schemas=[],
    )
    visible = '{"latency_p99_ms":8000}'
    checkpoint = Checkpoint(
        state=copy_state(state),
        tool_name="get_metrics",
        arguments={"incident_id": "F000"},
        raw={"is_error": False, "structured_content": {"latency_p99_ms": 8000}},
        visible=visible,
        latency_ms=0.1,
    )
    before = len(state.messages)
    restore(state, checkpoint, content=visible, source="base")
    assert len(state.messages) == before + 1
    assert state.messages[-1].role == "tool"
    assert state.messages[-1].content == visible
    dumped = state.model_dump(mode="json")
    assert label_key_overlap(dumped).isdisjoint(FORBIDDEN_LABEL_KEYS)
    assert "oracle_decision" not in state.model_dump_json()


def test_loop_modules_do_not_import_labels_or_a_provider() -> None:
    source = ROOT / "src" / "parallax"
    agent = _imports(source / "agent.py")
    prompt = _imports(source / "prompt.py")
    replay = _imports(source / "replay.py")
    mutate = _imports(source / "mutate.py")
    assert "parallax.oracle" not in agent
    assert "parallax.scenarios" not in agent
    assert "parallax.oracle" not in prompt
    assert "parallax.scenarios" not in replay
    assert "parallax.oracle" in replay
    assert "parallax.oracle" in mutate
    for name in CORE_MODULES:
        modules = _imports(source / name)
        assert modules.isdisjoint(NETWORK_MODULES)
        assert "parallax.openai_adapter" not in modules
        assert "parallax.dashboard" not in modules
        text = (source / name).read_text(encoding="utf-8").lower()
        assert "headroom" not in text
        assert "streamlit" not in text
    project_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "headroom" not in project_text
    with (ROOT / "pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    project = document["project"]
    base = _requirement_names(project["dependencies"])
    dev = _requirement_names(document["dependency-groups"]["dev"])
    assert base.isdisjoint(PROVIDER_PACKAGES)
    assert dev.isdisjoint(PROVIDER_PACKAGES)
    assert "openai" not in base
    extras = project["optional-dependencies"]
    assert "dashboard" in extras
    for extra_name, requirements in extras.items():
        names = _requirement_names(requirements)
        if extra_name == "live":
            assert "openai" in names
            assert "python-dotenv" in names
            assert names.isdisjoint(PROVIDER_PACKAGES - {"openai"})
            continue
        if extra_name == "dashboard":
            assert "streamlit" in names
            assert "plotly" in names
            assert "openai" not in names
            assert names.isdisjoint({"anthropic", "headroom"})
            continue
        assert "openai" not in names
        assert "streamlit" not in names
