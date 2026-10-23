"""Offline checks for one live counterfactual pair. No provider call is made."""

import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from parallax.agent import ModelCompletion, ToolRequest
from parallax.fixtures import get_fixture
from parallax.live_pair import (
    first_h2_scenario,
    format_pair_summary,
    main,
    max_model_requests,
    pair_model_params,
)
from parallax.openai_adapter import OpenAIAdapter
from parallax.oracle import decide
from parallax.replay import prefix_sha256, run_pair
from parallax.trace import read_traces, research_eligible
from tests.harness import SCRIPTED_DECISION, ScriptedAdapter

PARAMS = pair_model_params("gpt-6-luna")


class _Responses:
    def __init__(self) -> None:
        self.history: list[dict[str, object]] = []
        self.calls = 0

    async def create(self, **kwargs: object) -> object:
        self.history.append(kwargs)
        self.calls += 1
        if self.calls == 1:
            return _tool_response("resp_pre", "call_metrics")
        if self.calls == 3:
            return _decision_response(f"resp_post_{self.calls}", SCRIPTED_DECISION, usage=None)
        return _decision_response(
            f"resp_post_{self.calls}",
            SCRIPTED_DECISION,
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
        )


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


def _tool_response(response_id: str, call_id: str) -> SimpleNamespace:
    call = SimpleNamespace(
        type="function_call",
        name="get_metrics",
        arguments='{"incident_id":"F000"}',
        call_id=call_id,
        id=f"fc_{call_id}",
        status="completed",
    )
    return SimpleNamespace(
        id=response_id,
        output=[call],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        error=None,
        status="completed",
    )


def _decision_response(response_id: str, content: str, usage: object) -> SimpleNamespace:
    message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=content)],
    )
    return SimpleNamespace(
        id=response_id,
        output=[message],
        usage=usage,
        error=None,
        status="completed",
    )


def _drive(function: object) -> object:
    return anyio.run(function, backend="asyncio")  # type: ignore[arg-type]


def test_first_h2_scenario_is_the_declared_f000_threshold() -> None:
    scenario = first_h2_scenario()
    assert scenario.scenario_id == "S09"
    assert scenario.family == "H2"
    assert scenario.intervention_id == "h2-cross-error-rate"
    assert scenario.fixture_id == "F000"
    assert scenario.oracle_after == "INVESTIGATE"
    assert max_model_requests(4, 7) == 7
    params = pair_model_params("gpt-6-luna")
    assert params["reasoning_effort"] == "low"
    assert params["provider_continuation"] == "previous_response_id"
    assert "temperature" not in params


def test_dry_run_hides_the_key_and_does_not_call_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("parallax.live_pair.load_project_env", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-test-value")
    monkeypatch.setenv("PARALLAX_MODEL", "gpt-6-luna")

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.live_pair.OpenAIAdapter.from_env", boom)
    code = main(["--scenario", "S09", "--dry-run"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "dry-run: no model request will be sent" in printed
    assert "scenario_id=S09" in printed
    assert "intervention_id=h2-cross-error-rate" in printed
    assert "incident_id=F000" in printed
    assert "phase=harness" in printed
    assert "research_trial=false" in printed
    assert "declared_oracle_after=INVESTIGATE" in printed
    assert "max_model_requests=7" in printed
    assert "provider_continuation=previous_response_id" in printed
    assert "sk-live-test-value" not in printed
    assert "sk-" not in printed


def test_live_pair_keeps_post_fork_response_ids_on_their_own_arm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client()
    adapter = OpenAIAdapter(client, model="gpt-6-luna")
    monkeypatch.setattr("parallax.live_pair.load_project_env", lambda: False)
    monkeypatch.setattr("parallax.live_pair.OpenAIAdapter.from_env", lambda: adapter)
    path = tmp_path / "s09.jsonl"
    code = main(
        [
            "--live",
            "--scenario",
            "S09",
            "--model-calls",
            "4",
            "--tool-calls",
            "6",
            "--pair-model-calls",
            "8",
            "--trace",
            str(path),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    rows = read_traces(path)
    assert [row.arm for row in rows] == ["baseline", "factual_replay", "mutant"]
    assert all(row.phase == "harness" for row in rows)
    assert all(research_eligible(row) is False for row in rows)
    assert all(row.execution_status == "completed" for row in rows)
    fixture = get_fixture("F000")
    assert fixture is not None
    factual_oracle = decide(fixture.metrics, fixture.logs, fixture.deployment).value
    assert rows[1].oracle_decision == factual_oracle
    assert rows[2].oracle_decision == "INVESTIGATE"
    assert rows[1].prefix_sha256 == rows[2].prefix_sha256
    assert rows[1].prefix_sha256 is not None
    assert rows[1].observation_original == rows[0].observation_original
    assert rows[2].observation_original != rows[2].observation_mutated
    assert '"error_rate_bps":499' in (rows[1].observation_original or "")
    assert '"error_rate_bps":500' in (rows[2].observation_mutated or "")
    assert rows[0].provider_response_ids == ["resp_pre", "resp_post_2"]
    assert rows[1].provider_response_ids == ["resp_pre", "resp_post_3"]
    assert rows[2].provider_response_ids == ["resp_pre", "resp_post_4"]
    assert rows[1].usage is None
    assert rows[0].usage == {"input_tokens": 7, "output_tokens": 4}
    assert rows[2].usage == {"input_tokens": 7, "output_tokens": 4}
    factual_tools = rows[1].tool_calls
    mutant_tools = rows[2].tool_calls
    assert [call.name for call in factual_tools] == ["get_metrics"]
    assert factual_tools[0].source == "base"
    assert mutant_tools[0].source == "overlay"
    assert factual_tools[0].model_visible_sha256 != mutant_tools[0].model_visible_sha256
    history = client.responses.history
    assert len(history) == 4
    assert "previous_response_id" not in history[0]
    assert history[2]["previous_response_id"] == "resp_pre"
    assert history[3]["previous_response_id"] == "resp_pre"
    factual_input = history[2]["input"]
    mutant_input = history[3]["input"]
    assert isinstance(factual_input, list)
    assert isinstance(mutant_input, list)
    assert factual_input[0]["call_id"] == "call_metrics"
    assert mutant_input[0]["call_id"] == "call_metrics"
    assert '"error_rate_bps":499' in str(factual_input[0]["output"])
    assert '"error_rate_bps":500' in str(mutant_input[0]["output"])
    assert "resp_post_2" not in str(mutant_input)
    assert "resp_post_3" not in str(mutant_input)
    assert "resp_post_4" not in str(factual_input)
    assert "oracle_decision" not in str(factual_input)
    assert "oracle_decision" not in str(mutant_input)
    for sent in history:
        assert sent["model"] == "gpt-6-luna"
        assert "temperature" not in sent
        assert "provider_continuation" not in sent
        assert sent["reasoning"] == {"effort": "low"}
        assert "sk-" not in json.dumps(sent)
    stored = path.read_text(encoding="utf-8")
    assert "sk-" not in stored
    assert "OPENAI_API_KEY" not in stored
    assert "Bearer " not in stored
    assert "checkout completed" not in printed
    assert "error_rate_bps" not in printed
    assert "phase=harness" in printed
    assert "prefix_sha256=" in printed
    assert "provider_continuation=previous_response_id" in printed
    assert "factual_replay_provider_response_ids=resp_pre,resp_post_3" in printed
    assert "mutant_provider_response_ids=resp_pre,resp_post_4" in printed
    assert "mutant_oracle_decision=INVESTIGATE" in printed
    assert "factual_replay_usage=null" in printed
    assert adapter.omitted_params == ("provider_continuation",)


def test_checkpoint_hash_is_shared_and_the_snapshot_stays_put() -> None:
    client = _Client()
    adapter = OpenAIAdapter(client, model="gpt-6-luna")

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=adapter,
            model_id="gpt-6-luna",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S09",
            repeat_index=0,
            tool_call_budget=6,
            model_call_limit=8,
        )

    run = _drive(main)
    snapshot = run.snapshot  # type: ignore[attr-defined]
    assert snapshot is not None
    assert snapshot.state.messages[-1].role == "assistant"
    assert snapshot.state.provider_response_ids == ["resp_pre"]
    assert run.prefix_sha256 == prefix_sha256(snapshot.state)  # type: ignore[attr-defined]
    assert run.factual.prefix_sha256 == run.mutant.prefix_sha256  # type: ignore[attr-defined]
    factual_blob = json.dumps([message.model_dump(mode="json") for message in run.factual_messages])  # type: ignore[attr-defined]
    mutant_blob = json.dumps([message.model_dump(mode="json") for message in run.mutant_messages])  # type: ignore[attr-defined]
    assert "oracle_decision" not in factual_blob
    assert "oracle_decision" not in mutant_blob
    assert "resp_post_4" not in factual_blob
    assert "resp_post_3" not in mutant_blob
    text = format_pair_summary(run)  # type: ignore[arg-type]
    assert "sk-" not in text
    assert "checkout completed" not in text


def test_same_turn_siblings_stay_factual_on_both_arms() -> None:
    class _BatchResponses:
        def __init__(self) -> None:
            self.history: list[dict[str, object]] = []
            self.calls = 0

        async def create(self, **kwargs: object) -> object:
            self.history.append(kwargs)
            self.calls += 1
            if self.calls == 1:
                calls = [
                    _function_call("get_metrics", "call_metrics"),
                    _function_call("get_logs", "call_logs"),
                    _function_call("get_deployment", "call_deploy"),
                ]
                return SimpleNamespace(
                    id="resp_pre",
                    output=calls,
                    usage=SimpleNamespace(input_tokens=3, output_tokens=2),
                    error=None,
                    status="completed",
                )
            return _decision_response(
                f"resp_post_{self.calls}",
                SCRIPTED_DECISION,
                SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    responses = _BatchResponses()
    adapter = OpenAIAdapter(SimpleNamespace(responses=responses), model="gpt-6-luna")

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=adapter,
            model_id="gpt-6-luna",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S09",
            repeat_index=0,
            model_call_limit=8,
        )

    run = _drive(main)
    assert run.snapshot.state.messages[-1].role == "assistant"  # type: ignore[attr-defined]
    factual_input = responses.history[2]["input"]
    mutant_input = responses.history[3]["input"]
    assert isinstance(factual_input, list)
    assert isinstance(mutant_input, list)
    assert [item["call_id"] for item in factual_input] == ["call_metrics", "call_logs", "call_deploy"]
    assert [item["call_id"] for item in mutant_input] == ["call_metrics", "call_logs", "call_deploy"]
    assert factual_input[1]["output"] == mutant_input[1]["output"]
    assert factual_input[2]["output"] == mutant_input[2]["output"]
    assert factual_input[0]["output"] != mutant_input[0]["output"]
    assert responses.history[2]["previous_response_id"] == "resp_pre"
    assert responses.history[3]["previous_response_id"] == "resp_pre"
    assert "resp_post_2" not in str(mutant_input)
    names = [call.name for call in run.mutant.tool_calls]  # type: ignore[attr-defined]
    assert names == ["get_metrics", "get_logs", "get_deployment"]
    assert run.mutant.tool_calls[0].source == "overlay"  # type: ignore[attr-defined]
    assert run.mutant.tool_calls[1].source == "base"  # type: ignore[attr-defined]
    assert run.factual.tool_calls[0].source == "base"  # type: ignore[attr-defined]


def _function_call(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments='{"incident_id":"F000"}',
        call_id=call_id,
        id=f"fc_{call_id}",
        status="completed",
    )


def test_pair_model_limit_stops_before_another_provider_request() -> None:
    seen = {"calls": 0}

    class _Count:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools, params
            seen["calls"] += 1
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
            )

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=_Count(),
            model_id="fake",
            model_params={"temperature": 0},
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S09",
            repeat_index=0,
            model_call_limit=1,
        )

    run = _drive(main)
    assert seen["calls"] == 1
    assert run.baseline.execution_status == "budget_exhausted"  # type: ignore[attr-defined]
    assert run.factual.execution_status == "budget_exhausted"  # type: ignore[attr-defined]
    assert run.mutant.execution_status == "budget_exhausted"  # type: ignore[attr-defined]
    assert "provider_error" not in run.factual.flags  # type: ignore[attr-defined]
    assert run.factual.usage is None  # type: ignore[attr-defined]


def test_step_budget_is_inherited_by_both_arms() -> None:
    counter = ScriptedAdapter("F000", ("get_metrics",))
    seen = {"calls": 0}

    class _Count:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            seen["calls"] += 1
            return await counter.complete(messages, tools, params)

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=_Count(),
            model_id="fake",
            model_params={"temperature": 0},
            step_budget=1,
            token_budget=None,
            phase="harness",
            scenario_id="S09",
            repeat_index=0,
        )

    run = _drive(main)
    assert seen["calls"] == 1
    assert run.baseline.execution_status == "budget_exhausted"  # type: ignore[attr-defined]
    assert run.factual.execution_status == "budget_exhausted"  # type: ignore[attr-defined]
    assert run.mutant.execution_status == "budget_exhausted"  # type: ignore[attr-defined]
    assert run.prefix_sha256 is not None  # type: ignore[attr-defined]


def test_invalid_tool_does_not_invent_a_branch() -> None:
    class _Rejected:
        async def complete(self, messages: object, tools: object, params: object) -> ModelCompletion:
            del messages, tools, params
            return ModelCompletion(
                content="",
                tool_calls=(ToolRequest(name="remediate_now", arguments={"incident_id": "F000"}),),
            )

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=_Rejected(),
            model_id="fake",
            model_params={"temperature": 0},
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S09",
            repeat_index=0,
        )

    run = _drive(main)
    assert run.baseline.execution_status == "invalid_tool"  # type: ignore[attr-defined]
    assert run.factual is None  # type: ignore[attr-defined]
    assert run.mutant.flags == ["intervention_unobserved"]  # type: ignore[attr-defined]
    assert run.snapshot is None  # type: ignore[attr-defined]


def test_provider_error_trace_drops_the_secret_body(tmp_path: Path) -> None:
    class _Status(Exception):
        def __init__(self) -> None:
            super().__init__("bad sk-secret body")
            self.status_code = 400
            self.request_id = "req_pair"
            self.body = {"error": {"message": "sk-secret", "code": "invalid_request", "param": "input"}}

    class _Responses:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            raise _Status()

    adapter = OpenAIAdapter(SimpleNamespace(responses=_Responses()), model="gpt-6-luna")

    async def main() -> object:
        return await run_pair(
            incident_id="F000",
            intervention_id="h2-cross-error-rate",
            adapter=adapter,
            model_id="gpt-6-luna",
            model_params=PARAMS,
            step_budget=4,
            token_budget=None,
            phase="harness",
            scenario_id="S09",
            repeat_index=0,
            trace_path=tmp_path / "error.jsonl",
        )

    run = _drive(main)
    assert run.baseline.execution_status == "model_error"  # type: ignore[attr-defined]
    assert run.factual is None  # type: ignore[attr-defined]
    stored = (tmp_path / "error.jsonl").read_text(encoding="utf-8")
    assert "sk-" not in stored
    assert "stage=request" in stored
    assert "http_status=400" in stored


def test_non_h2_scenario_and_bad_trace_stop_before_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("provider client constructed")

    monkeypatch.setattr("parallax.live_pair.OpenAIAdapter.from_env", boom)
    assert main(["--scenario", "S01", "--live"]) == 2
    assert main(["--live", "--trace", "runs/live/s09.txt"]) == 2
    assert main(["--live", "--dry-run"]) == 2
