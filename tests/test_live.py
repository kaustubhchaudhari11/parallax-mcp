"""Offline checks for the smoke command. No provider client is constructed."""

import ast
import json
import os
from pathlib import Path

import anyio
import pytest

from parallax.agent import (
    FactualResolver,
    ModelCompletion,
    ModelUsage,
    ToolRequest,
    run_agent,
)
from parallax.capture import label_key_overlap, sha256_text
from parallax.fixtures import get_fixture
from parallax.live import (
    _print_result,
    configured_budget,
    format_debug_report,
    format_dry_run,
    format_smoke_summary,
    main,
    smoke_model_params,
    smoke_record,
)
from parallax.observe import canonical_json
from parallax.openai_adapter import (
    OpenAIAdapter,
    ProviderCallError,
    load_project_env,
    project_root,
)
from parallax.trace import read_traces, research_eligible
from tests.harness import SCRIPTED_DECISION, ScriptedAdapter
from tests.test_agent import _state, _StubClient

ROOT = Path(__file__).resolve().parents[1]


class _Decide:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        return ModelCompletion(content=SCRIPTED_DECISION, response_id="resp_final")


def test_dry_run_names_the_plan_and_hides_the_key() -> None:
    text = format_dry_run(
        incident_id="F000",
        environ={
            "OPENAI_API_KEY": "test-key",
            "PARALLAX_MODEL": "gpt-test",
            "PARALLAX_MAX_MODEL_CALLS": "4",
            "PARALLAX_MAX_TOOL_CALLS": "6",
        },
    )
    assert "incident_id=F000" in text
    assert "model=gpt-test" in text
    assert "OPENAI_API_KEY=set" in text
    assert "model_call_budget=4" in text
    assert "tool_call_budget=6" in text
    assert "get_metrics,get_logs,get_deployment" in text
    assert "test-key" not in text
    assert "sk-" not in text


def test_dry_run_command_does_not_construct_a_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("parallax.live.load_project_env", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PARALLAX_MODEL", "gpt-test")

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("client constructed")

    monkeypatch.setattr("parallax.openai_adapter.AsyncOpenAI", boom)
    monkeypatch.setattr("parallax.live.OpenAIAdapter.from_env", boom)
    code = main(["--incident", "F000", "--dry-run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "gpt-test" in captured.out
    assert "test-key" not in captured.out
    assert "sk-" not in captured.out


def test_unknown_incident_stops_before_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.live.OpenAIAdapter.from_env", boom)
    code = main(["--incident", "NOPE"])
    assert code == 2


def test_project_root_env_is_loaded_and_a_nested_file_is_not(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PARALLAX_MODEL", raising=False)
    nested = tmp_path / "src" / "parallax"
    nested.mkdir(parents=True)
    (nested / ".env").write_text("OPENAI_API_KEY=nested-key\nPARALLAX_MODEL=nested-model\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-key\nPARALLAX_MODEL=gpt-test\n", encoding="utf-8")
    monkeypatch.setattr("parallax.openai_adapter.project_root", lambda: tmp_path)
    seen: dict[str, str] = {}

    def factory(api_key: str) -> object:
        seen["key"] = api_key
        return object()

    adapter = OpenAIAdapter.from_env(client_factory=factory)
    if seen.get("key") != "test-key":
        raise AssertionError("project-root .env did not supply the test key")
    assert adapter.model == "gpt-test"
    assert os.environ["PARALLAX_MODEL"] == "gpt-test"
    assert load_project_env(tmp_path) is True


def test_project_root_is_the_repository() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "parallax" / "live.py").is_file()


def test_smoke_params_for_luna_use_low_effort_without_sampling() -> None:
    luna = smoke_model_params("gpt-6-luna")
    assert luna == {"reasoning_effort": "low"}
    for name in ("temperature", "top_p", "top_logprobs", "seed"):
        assert name not in luna
    assert smoke_model_params("gpt-test") == {"temperature": 0}


def test_smoke_command_passes_luna_effort_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[dict[str, object]] = []

    class _Adapter:
        model = "gpt-6-luna"

        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools
            seen.append(dict(params))  # type: ignore[arg-type]
            return ModelCompletion(content=SCRIPTED_DECISION, response_id="resp_smoke")

    monkeypatch.setattr("parallax.live.OpenAIAdapter.from_env", lambda: _Adapter())
    code = main(["--incident", "F000", "--model-calls", "1", "--tool-calls", "1"])
    printed = capsys.readouterr().out
    assert code == 0
    assert seen == [{"reasoning_effort": "low"}]
    assert "sk-" not in printed
    assert "temperature" not in printed
    assert "tool_names=" not in printed
    assert "phase=harness" not in printed


def test_budget_errors_do_not_echo_the_raw_value() -> None:
    try:
        configured_budget(None, {"PARALLAX_MAX_MODEL_CALLS": "test-key"}, "PARALLAX_MAX_MODEL_CALLS", 4)
    except ValueError as exc:
        assert "test-key" not in str(exc)
    else:
        raise AssertionError("a non-integer budget was accepted")


def test_smoke_record_keeps_provider_ids_and_leaves_the_oracle_empty() -> None:
    async def run() -> object:
        return await run_agent(_StubClient(), _state(model_id="gpt-test"), _Decide(), resolver=FactualResolver())

    result = anyio.run(run, backend="asyncio")
    record = smoke_record(result, sdk_version="test", protocol="2025-06-18", started_at="t0", ended_at="t1")  # type: ignore[arg-type]
    assert record.provider_response_ids == ["resp_final"]
    assert record.usage is None
    assert record.model_id == "gpt-test"
    assert record.execution_status == "completed"
    assert record.oracle_decision is None
    assert record.predicate_m is None
    assert record.phase == "harness"
    assert research_eligible(record) is False
    assert all("oracle" not in text for text in record.model_visible_observations)


def test_live_module_does_not_import_the_oracle() -> None:
    path = ROOT / "src" / "parallax" / "live.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "parallax.oracle" not in modules
    text = path.read_text(encoding="utf-8")
    assert "api_key" not in text
    assert "{environ" not in text


class _CodedFailure:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        raise ProviderCallError(
            stage="request",
            exception_type="BadRequestError",
            status_code=400,
            error_code="unsupported_parameter",
            request_id="req_abc",
            parameter="temperature",
        )


class _LeakingFailure:
    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del messages, tools, params
        raise RuntimeError("sk-secret prompt text")


def test_debug_errors_print_the_safe_classification(capsys: pytest.CaptureFixture[str]) -> None:
    async def run() -> object:
        return await run_agent(_StubClient(), _state(), _CodedFailure(), resolver=FactualResolver())

    result = anyio.run(run, backend="asyncio")
    assert result.execution_status == "model_error"  # type: ignore[attr-defined]
    assert result.state.provider_response_ids == []  # type: ignore[attr-defined]
    _print_result(result, debug_errors=True)  # type: ignore[arg-type]
    printed = capsys.readouterr().out
    assert "execution_status=model_error" in printed
    assert "stage=request" in printed
    assert "http_status=400" in printed
    assert "error_code=unsupported_parameter" in printed
    assert "parameter=temperature" in printed
    assert "request_id=req_abc" in printed


def test_debug_errors_drop_unstructured_exception_text(capsys: pytest.CaptureFixture[str]) -> None:
    report = format_debug_report("ProviderCallError: stage=request parameter=sk-secret")
    assert "stage=request" in report
    assert "sk-" not in report

    async def run() -> object:
        return await run_agent(_StubClient(), _state(), _LeakingFailure(), resolver=FactualResolver())

    result = anyio.run(run, backend="asyncio")
    _print_result(result, debug_errors=False)  # type: ignore[arg-type]
    quiet = capsys.readouterr().out
    assert "sk-" not in quiet
    assert "stage=" not in quiet
    _print_result(result, debug_errors=True)  # type: ignore[arg-type]
    printed = capsys.readouterr().out
    assert "stage=agent_controller" in printed
    assert "exception=RuntimeError" in printed
    assert "sk-" not in printed
    assert "prompt text" not in printed


class _StateAdapter:
    """Scripted tool sequence with one response id per model call."""

    model = "gpt-test"

    def __init__(self, incident_id: str, sequence: tuple[str, ...]) -> None:
        self.inner = ScriptedAdapter(incident_id, sequence, usage="always")
        self.calls = 0

    async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
        del params
        self.calls += 1
        completion = await self.inner.complete(messages, tools, {})
        return ModelCompletion(
            content=completion.content,
            tool_calls=completion.tool_calls,
            usage=completion.usage,
            response_id=f"resp_{self.calls}",
        )


def test_trace_records_each_server_tool_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _StateAdapter("F000", ("get_metrics", "get_logs", "get_deployment"))
    monkeypatch.setattr("parallax.live.OpenAIAdapter.from_env", lambda: adapter)
    path = tmp_path / "f000.jsonl"
    code = main(
        [
            "--incident",
            "F000",
            "--model-calls",
            "6",
            "--tool-calls",
            "6",
            "--trace",
            str(path),
            "--trace-summary",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    fixture = get_fixture("F000")
    assert fixture is not None
    expected = {
        "get_metrics": canonical_json(fixture.metrics.model_dump(mode="json")),
        "get_logs": canonical_json(fixture.logs.model_dump(mode="json")),
        "get_deployment": canonical_json(fixture.deployment.model_dump(mode="json")),
    }
    rows = read_traces(path)
    assert len(rows) == 1
    row = rows[0]
    names = [call.name for call in row.tool_calls]
    assert names == ["get_metrics", "get_logs", "get_deployment"]
    assert row.phase == "harness"
    assert research_eligible(row) is False
    assert row.execution_status == "completed"
    assert row.decision_parsed == "ESCALATE"
    assert row.oracle_decision is None
    assert row.predicate_m is None
    assert row.predicate_l is None
    assert row.predicate_d is None
    assert row.provider_response_ids == ["resp_1", "resp_2", "resp_3", "resp_4"]
    assert row.usage == {"input_tokens": 8, "output_tokens": 12}
    assert row.model_visible_observations == [expected[name] for name in names]
    for call in row.tool_calls:
        assert call.dispatched is True
        assert call.is_error is False
        assert call.source == "base"
        visible = call.model_visible_observation
        assert visible == expected[call.name]
        assert call.model_visible_sha256 == sha256_text(visible or "")
        assert label_key_overlap(json.loads(visible or "")) == set()
    assert len(set(row.model_visible_observations)) == 3
    stored = path.read_text(encoding="utf-8")
    assert "sk-" not in stored
    assert "OPENAI_API_KEY" not in stored
    assert "Bearer " not in stored
    assert "phase=harness" in printed
    assert "model_calls=4" in printed
    assert "mcp_tool_calls=3" in printed
    assert "tool_names=get_metrics,get_logs,get_deployment" in printed
    assert "tool.0 name=get_metrics dispatched=true is_error=false source=base" in printed
    assert "tool.1 name=get_logs dispatched=true is_error=false source=base" in printed
    assert "tool.2 name=get_deployment dispatched=true is_error=false source=base" in printed
    assert "tool_errors=(none)" in printed
    assert f"trace_path={path}" in printed
    assert "execution_status=completed" in printed
    assert "error_rate_bps" not in printed
    assert "checkout completed" not in printed
    assert "sk-" not in printed


def test_trace_records_a_tool_error_separately_from_the_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _StateAdapter("missing", ("get_metrics",))
    monkeypatch.setattr("parallax.live.OpenAIAdapter.from_env", lambda: adapter)
    path = tmp_path / "missing.jsonl"
    code = main(
        [
            "--incident",
            "F000",
            "--model-calls",
            "4",
            "--tool-calls",
            "4",
            "--trace",
            str(path),
            "--trace-summary",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    row = read_traces(path)[0]
    assert row.execution_status == "completed"
    assert row.decision_parsed == "ESCALATE"
    assert row.provider_response_ids == ["resp_1", "resp_2"]
    assert len(row.tool_calls) == 1
    call = row.tool_calls[0]
    assert call.name == "get_metrics"
    assert call.dispatched is True
    assert call.is_error is True
    assert call.source == "base"
    visible = call.model_visible_observation or ""
    assert "unknown_incident" in visible
    assert "error_rate_bps" not in visible
    assert visible in row.model_visible_observations
    assert "tool_errors=get_metrics" in printed
    assert "mcp_tool_calls=1" in printed
    assert "model_calls=2" in printed
    assert "is_error=true" in printed
    assert "unknown_incident" not in printed
    assert "sk-" not in printed
    assert "sk-" not in path.read_text(encoding="utf-8")


def test_summary_omits_observation_text(capsys: pytest.CaptureFixture[str]) -> None:
    from mcp.types import CallToolResult, TextContent

    class _Client:
        async def call_tool(self, name: str, arguments: dict[str, object]) -> CallToolResult:
            del name, arguments
            payload = {"note": "sk-secret-observation"}
            return CallToolResult(
                content=[TextContent(type="text", text=canonical_json(payload))],
                structured_content=payload,
                is_error=False,
            )

    class _TwoStep:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del tools, params
            seen = [message for message in messages if message.role == "tool"]  # type: ignore[attr-defined]
            if not seen:
                return ModelCompletion(
                    content="",
                    tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
                    usage=ModelUsage(input_tokens=5, output_tokens=1),
                    response_id="resp_a",
                )
            return ModelCompletion(
                content=SCRIPTED_DECISION,
                usage=ModelUsage(input_tokens=7, output_tokens=2),
                response_id="resp_b",
            )

    async def run() -> object:
        return await run_agent(_Client(), _state(model_id="gpt-test"), _TwoStep(), resolver=FactualResolver())

    result = anyio.run(run, backend="asyncio")
    text = format_smoke_summary(result)  # type: ignore[arg-type]
    assert "phase=harness" in text
    assert "model_calls=2" in text
    assert "mcp_tool_calls=1" in text
    assert "tool.0 name=get_metrics dispatched=true is_error=false source=base" in text
    assert "sk-secret-observation" not in text
    assert "sk-" not in text
    record = smoke_record(result, sdk_version="test", protocol=None, started_at="t0", ended_at="t1")  # type: ignore[arg-type]
    assert record.phase == "harness"
    assert record.execution_status == "completed"
    assert record.model_visible_observations == [canonical_json({"note": "sk-secret-observation"})]
    assert record.oracle_decision is None
    _print_result(result, debug_errors=False)  # type: ignore[arg-type]
    printed = capsys.readouterr().out
    assert "sk-" not in printed


def test_rejected_tool_is_not_counted_as_an_mcp_call() -> None:
    class _Rejected:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools, params
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="remediate_now", arguments={"incident_id": "F000"}),),
                response_id="resp_bad",
            )

    async def run() -> object:
        return await run_agent(_StubClient(), _state(), _Rejected(), resolver=FactualResolver())

    result = anyio.run(run, backend="asyncio")
    assert result.execution_status == "invalid_tool"  # type: ignore[attr-defined]
    text = format_smoke_summary(result)  # type: ignore[arg-type]
    assert "model_calls=1" in text
    assert "mcp_tool_calls=0" in text
    assert "tool_names=remediate_now" in text
    assert "tool.0 name=remediate_now dispatched=false is_error=null source=rejected" in text
    assert "tool_errors=(none)" in text


def test_non_jsonl_trace_path_stops_before_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.live.OpenAIAdapter.from_env", boom)
    code = main(["--incident", "F000", "--trace", "runs/live/f000.txt"])
    assert code == 2


def test_trace_output_paths_are_gitignored() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "runs/" in text.splitlines()
    assert "*.jsonl" in text.splitlines()
