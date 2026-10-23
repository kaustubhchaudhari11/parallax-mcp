"""Offline checks for the optional OpenAI adapter. No network client is constructed."""

import ast
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from parallax.agent import ChatMessage, ToolRequest
from parallax.openai_adapter import OpenAIAdapter, ProviderCallError

ROOT = Path(__file__).resolve().parents[1]


class _Responses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None
        self.history: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        self.history.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _Client:
    def __init__(self, response: object) -> None:
        self.responses = _Responses(response)


def _response(
    content: str | None,
    tool_calls: object = None,
    usage: object = None,
    response_id: str | None = None,
    *,
    error: object = None,
    status: str = "completed",
    prefix: object = None,
) -> SimpleNamespace:
    output: list[object] = []
    if prefix:
        output.extend(prefix)  # type: ignore[arg-type]
    if tool_calls:
        output.extend(tool_calls)  # type: ignore[arg-type]
    if content is not None:
        output.append(
            SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=content)])
        )
    return SimpleNamespace(id=response_id, output=output, usage=usage, error=error, status=status)


def _function_call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments,
        call_id=call_id,
        id=f"fc_{call_id}",
        status="completed",
    )


def test_completion_maps_a_decision_and_reported_usage() -> None:
    usage = SimpleNamespace(input_tokens=11, output_tokens=4)
    client = _Client(_response('{"decision":"INVESTIGATE","action":"none","evidence":["logs"]}', usage=usage))
    adapter = OpenAIAdapter(client, model="gpt-test")
    messages = [
        ChatMessage(role="system", content="policy"),
        ChatMessage(role="user", content="Investigate incident F000."),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}),),
        ),
        ChatMessage(role="tool", content='{"error_rate_bps":499}', tool_name="get_metrics"),
    ]

    async def main() -> object:
        return await adapter.complete(
            messages,
            [{"name": "get_metrics", "description": "metrics", "input_schema": {"type": "object"}}],
            {"temperature": 0, "seed": 1, "max_tokens": 256, "mystery": 9},
        )

    completion = anyio.run(main, backend="asyncio")
    assert completion.content.startswith("{")  # type: ignore[attr-defined]
    assert completion.tool_calls == ()  # type: ignore[attr-defined]
    assert completion.usage is not None  # type: ignore[attr-defined]
    assert completion.usage.input_tokens == 11  # type: ignore[attr-defined]
    assert completion.usage.output_tokens == 4  # type: ignore[attr-defined]
    sent = client.responses.kwargs
    assert sent is not None
    assert sent["model"] == "gpt-test"
    assert sent["temperature"] == 0
    assert sent["max_output_tokens"] == 256
    assert "max_tokens" not in sent
    assert "seed" not in sent
    assert "previous_response_id" not in sent
    assert "mystery" not in sent
    assert "api_key" not in sent
    assert adapter.omitted_params == ("seed", "mystery")
    provider_input = sent["input"]
    assert isinstance(provider_input, list)
    assert provider_input[2]["type"] == "function_call"
    assert provider_input[2]["call_id"] == "call_2_0"
    assert provider_input[2]["name"] == "get_metrics"
    assert provider_input[3]["type"] == "function_call_output"
    assert provider_input[3]["call_id"] == "call_2_0"
    assert provider_input[3]["output"] == '{"error_rate_bps":499}'
    tools = sent["tools"]
    assert isinstance(tools, list)
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "get_metrics"


def test_gpt6_luna_low_effort_omits_sampling_fields() -> None:
    usage = SimpleNamespace(input_tokens=8, output_tokens=3)
    call = _function_call("get_metrics", '{"incident_id":"F000"}', "c1")
    client = _Client(_response(None, tool_calls=[call], usage=usage, response_id="resp_luna"))
    adapter = OpenAIAdapter(client, model="gpt-6-luna")
    tools = [{"name": "get_metrics", "description": "metrics", "input_schema": {"type": "object"}}]
    messages = [ChatMessage(role="user", content="Investigate incident F000.")]

    async def main() -> object:
        return await adapter.complete(
            messages,
            tools,
            {
                "temperature": 0,
                "top_p": 1,
                "top_logprobs": 5,
                "reasoning_effort": "low",
                "max_tokens": 256,
                "seed": 1,
                "api_key": "sk-should-not-send",
            },
        )

    completion = anyio.run(main, backend="asyncio")
    assert completion.response_id == "resp_luna"  # type: ignore[attr-defined]
    sent = client.responses.kwargs
    assert sent is not None
    assert sent["model"] == "gpt-6-luna"
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["max_output_tokens"] == 256
    for name in ("temperature", "top_p", "top_logprobs", "seed", "api_key"):
        assert name not in sent
    assert all(value is not None for value in sent.values())
    provider_input = sent["input"]
    assert isinstance(provider_input, list)
    assert provider_input[0] == {"role": "user", "content": "Investigate incident F000."}
    sent_tools = sent["tools"]
    assert isinstance(sent_tools, list)
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["name"] == "get_metrics"
    assert "sk-" not in str(sent)
    assert adapter.omitted_params == ("temperature", "top_p", "top_logprobs", "seed", "api_key")


def test_gpt6_luna_continuation_keeps_the_provider_call_id() -> None:
    call = _function_call("get_metrics", '{"incident_id":"F000"}', "c1")
    client = _Client(_response(None, tool_calls=[call], response_id="resp_123"))
    adapter = OpenAIAdapter(client, model="gpt-6-luna")
    params = {
        "temperature": 0,
        "top_p": 1,
        "top_logprobs": 2,
        "reasoning_effort": "low",
        "max_tokens": 128,
    }
    tools = [{"name": "get_metrics", "description": "metrics", "input_schema": {"type": "object"}}]

    async def main() -> object:
        first = await adapter.complete(
            [ChatMessage(role="user", content="Investigate incident F000.")],
            tools,
            params,
        )
        continued = [
            ChatMessage(role="user", content="Investigate incident F000."),
            ChatMessage(
                role="assistant",
                content=first.content,  # type: ignore[attr-defined]
                tool_calls=first.tool_calls,  # type: ignore[attr-defined]
                response_id="resp_123",
                continuation=first.continuation,  # type: ignore[attr-defined]
            ),
            ChatMessage(role="tool", content="{}", tool_name="get_metrics", call_id="c1"),
        ]
        return await adapter.complete(continued, tools, params)

    anyio.run(main, backend="asyncio")
    follow = client.responses.history[1]
    assert follow["model"] == "gpt-6-luna"
    assert follow["previous_response_id"] == "resp_123"
    assert follow["reasoning"] == {"effort": "low"}
    assert follow["max_output_tokens"] == 128
    assert follow["input"] == [{"type": "function_call_output", "call_id": "c1", "output": "{}"}]
    for name in ("temperature", "top_p", "top_logprobs", "seed"):
        assert name not in follow
    assert "api_key" not in follow
    assert "Investigate incident F000." not in str(follow["input"])


def test_unset_luna_effort_omits_sampling_without_inventing_effort() -> None:
    client = _Client(_response("ok"))
    adapter = OpenAIAdapter(client, model="gpt-6-luna")

    async def main() -> None:
        await adapter.complete(
            [ChatMessage(role="user", content="task")],
            [],
            {"temperature": 0, "top_p": 1, "top_logprobs": 0, "max_tokens": 32},
        )

    anyio.run(main, backend="asyncio")
    sent = client.responses.kwargs
    assert sent is not None
    assert sent["model"] == "gpt-6-luna"
    assert sent["max_output_tokens"] == 32
    assert "reasoning" not in sent
    for name in ("temperature", "top_p", "top_logprobs"):
        assert name not in sent


def test_luna_none_effort_still_forwards_temperature() -> None:
    client = _Client(_response("ok"))
    adapter = OpenAIAdapter(client, model="gpt-6-luna")

    async def main() -> None:
        await adapter.complete(
            [ChatMessage(role="user", content="task")],
            [],
            {"temperature": 0, "reasoning_effort": "none", "max_tokens": 64},
        )

    anyio.run(main, backend="asyncio")
    sent = client.responses.kwargs
    assert sent is not None
    assert sent["model"] == "gpt-6-luna"
    assert sent["reasoning"] == {"effort": "none"}
    assert sent["temperature"] == 0
    assert sent["max_output_tokens"] == 64
    assert "top_p" not in sent
    assert "top_logprobs" not in sent


def test_missing_usage_stays_null_and_tool_calls_are_parsed() -> None:
    call = _function_call("get_logs", '{"incident_id":"F011"}', "call_logs")
    client = _Client(_response(None, tool_calls=[call], usage=None))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> object:
        return await adapter.complete(
            [ChatMessage(role="user", content="Investigate incident F011.")],
            [],
            {},
        )

    completion = anyio.run(main, backend="asyncio")
    assert completion.content == ""  # type: ignore[attr-defined]
    assert completion.usage is None  # type: ignore[attr-defined]
    assert completion.tool_calls == (  # type: ignore[attr-defined]
        ToolRequest(name="get_logs", arguments={"incident_id": "F011"}, call_id="call_logs"),
    )
    assert "tools" not in (client.responses.kwargs or {})


def test_non_json_tool_arguments_are_rejected() -> None:
    client = _Client(_response("", tool_calls=[_function_call("get_logs", "not-json", "call_logs")]))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete([ChatMessage(role="user", content="task")], [], {})
        assert caught.value.stage == "response_parsing"
        assert caught.value.exception_type == "ValueError"
        assert "not-json" not in str(caught.value)

    anyio.run(main, backend="asyncio")


def test_from_env_uses_the_supplied_mapping_and_does_not_load_dotenv() -> None:
    loaded = {"count": 0}
    constructed: list[str] = []
    clients: list[_Client] = []

    def load_env() -> None:
        loaded["count"] += 1

    def factory(api_key: str) -> _Client:
        constructed.append(api_key)
        client = _Client(_response("ok"))
        clients.append(client)
        return client

    missing = pytest.raises(RuntimeError, match="OPENAI_API_KEY")
    with missing:
        OpenAIAdapter.from_env(environ={}, load_env=load_env, client_factory=factory)
    assert loaded["count"] == 0
    assert constructed == []
    with pytest.raises(RuntimeError, match="PARALLAX_MODEL"):
        OpenAIAdapter.from_env(
            environ={"OPENAI_API_KEY": "test-key"},
            load_env=load_env,
            client_factory=factory,
        )
    assert constructed == []
    adapter = OpenAIAdapter.from_env(
        environ={"OPENAI_API_KEY": "test-key", "PARALLAX_MODEL": "gpt-test"},
        load_env=load_env,
        client_factory=factory,
    )

    async def main() -> object:
        return await adapter.complete([ChatMessage(role="user", content="task")], [], {})

    completion = anyio.run(main, backend="asyncio")
    assert completion.content == "ok"  # type: ignore[attr-defined]
    assert loaded["count"] == 0
    assert constructed == ["test-key"]
    assert clients[0].responses.kwargs is not None
    assert clients[0].responses.kwargs["model"] == "gpt-test"


def test_adapter_module_imports_openai_and_not_the_oracle() -> None:
    path = ROOT / "src" / "parallax" / "openai_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
            modules.add(node.module)
    assert "openai" in modules
    assert "dotenv" in modules
    assert "parallax.oracle" not in modules
    assert "parallax.fixtures" not in modules
    assert "parallax.scenarios" not in modules
    assert "call_tool" not in path.read_text(encoding="utf-8")
    assert "chat.completions" not in path.read_text(encoding="utf-8")


def test_provider_ids_are_preserved_for_the_three_tools() -> None:
    calls = [
        _function_call("get_metrics", '{"incident_id":"F000"}', "c1"),
        _function_call("get_logs", '{"incident_id":"F000"}', "c2"),
        _function_call("get_deployment", '{"incident_id":"F000"}', "c3"),
    ]
    client = _Client(_response("", tool_calls=calls, response_id="resp_123"))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> object:
        first = await adapter.complete([ChatMessage(role="user", content="Investigate incident F000.")], [], {})
        requests = first.tool_calls  # type: ignore[attr-defined]
        continued = [
            ChatMessage(role="user", content="Investigate incident F000."),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=requests,
                response_id="resp_123",
                continuation=first.continuation,  # type: ignore[attr-defined]
            ),
            ChatMessage(role="tool", content="{}", tool_name="get_metrics", call_id="c1"),
            ChatMessage(role="tool", content="{}", tool_name="get_logs", call_id="c2"),
            ChatMessage(role="tool", content="{}", tool_name="get_deployment", call_id="c3"),
        ]
        return await adapter.complete(continued, [], {})

    completion = anyio.run(main, backend="asyncio")
    assert completion.response_id == "resp_123"  # type: ignore[attr-defined]
    sent = client.responses.history[1]
    assert sent["previous_response_id"] == "resp_123"
    assert sent["input"] == [
        {"type": "function_call_output", "call_id": "c1", "output": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "{}"},
        {"type": "function_call_output", "call_id": "c3", "output": "{}"},
    ]
    assert "Investigate incident F000." not in str(sent["input"])


def test_completions_do_not_reuse_another_transcript() -> None:
    client = _Client(_response("ok", response_id="resp_a"))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        await adapter.complete([ChatMessage(role="user", content="UNIQUE-ARM-A")], [], {})
        await adapter.complete([ChatMessage(role="user", content="UNIQUE-ARM-B")], [], {})

    anyio.run(main, backend="asyncio")
    first = str(client.responses.history[0]["input"])
    second = str(client.responses.history[1]["input"])
    assert "UNIQUE-ARM-A" in first
    assert "UNIQUE-ARM-B" not in first
    assert "UNIQUE-ARM-B" in second
    assert "UNIQUE-ARM-A" not in second
    assert "previous_response_id" not in client.responses.history[1]


def test_non_object_tool_arguments_are_rejected() -> None:
    client = _Client(_response("", tool_calls=[_function_call("get_logs", "[1]", "call_logs")]))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete([ChatMessage(role="user", content="task")], [], {})
        assert caught.value.stage == "response_parsing"
        assert caught.value.exception_type == "TypeError"
        assert "[1]" not in str(caught.value)

    anyio.run(main, backend="asyncio")


def test_previous_response_keeps_replay_arms_apart() -> None:
    client = _Client(_response("ok", response_id="resp_next"))
    adapter = OpenAIAdapter(client, model="gpt-test")
    call = ToolRequest(name="get_metrics", arguments={"incident_id": "F000"}, call_id="c1")

    async def main() -> None:
        await adapter.complete(
            [
                ChatMessage(role="user", content="UNIQUE-ARM-A"),
                ChatMessage(role="assistant", content="", tool_calls=(call,), response_id="resp_a"),
                ChatMessage(role="tool", content="OBS-A", tool_name="get_metrics", call_id="c1"),
            ],
            [],
            {},
        )
        await adapter.complete(
            [
                ChatMessage(role="user", content="UNIQUE-ARM-B"),
                ChatMessage(role="assistant", content="", tool_calls=(call,), response_id="resp_b"),
                ChatMessage(role="tool", content="OBS-B", tool_name="get_metrics", call_id="c1"),
            ],
            [],
            {},
        )

    anyio.run(main, backend="asyncio")
    first = client.responses.history[0]
    second = client.responses.history[1]
    assert first["previous_response_id"] == "resp_a"
    assert first["input"] == [{"type": "function_call_output", "call_id": "c1", "output": "OBS-A"}]
    assert second["previous_response_id"] == "resp_b"
    assert second["input"] == [{"type": "function_call_output", "call_id": "c1", "output": "OBS-B"}]
    assert "OBS-A" not in str(second["input"])
    assert "UNIQUE-ARM-A" not in str(second["input"])


def test_reasoning_items_continue_when_the_response_id_is_absent() -> None:
    reasoning = SimpleNamespace(type="reasoning", id="rs_1", summary=[], content=None, encrypted_content=None, status=None)
    call = _function_call("get_metrics", '{"incident_id":"F000"}', "c1")
    client = _Client(_response(None, tool_calls=[call], prefix=[reasoning], response_id=None))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> object:
        first = await adapter.complete([ChatMessage(role="user", content="task")], [], {})
        continued = [
            ChatMessage(role="user", content="task"),
            ChatMessage(
                role="assistant",
                content=first.content,  # type: ignore[attr-defined]
                tool_calls=first.tool_calls,  # type: ignore[attr-defined]
                continuation=first.continuation,  # type: ignore[attr-defined]
            ),
            ChatMessage(role="tool", content="OBS", tool_name="get_metrics", call_id="c1"),
        ]
        await adapter.complete(continued, [], {})
        return first

    first = anyio.run(main, backend="asyncio")
    assert first.continuation[0]["type"] == "reasoning"  # type: ignore[attr-defined]
    assert first.continuation[0]["id"] == "rs_1"  # type: ignore[attr-defined]
    sent = client.responses.history[1]["input"]
    assert isinstance(sent, list)
    assert sent[1]["type"] == "reasoning"
    assert sent[1]["id"] == "rs_1"
    assert sent[2]["type"] == "function_call"
    assert sent[2]["call_id"] == "c1"
    assert sent[3] == {"type": "function_call_output", "call_id": "c1", "output": "OBS"}
    assert "previous_response_id" not in client.responses.history[1]


def test_partial_usage_stays_null() -> None:
    usage = SimpleNamespace(input_tokens=3, output_tokens=None)
    client = _Client(_response("ok", usage=usage))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> object:
        return await adapter.complete([ChatMessage(role="user", content="task")], [], {})

    completion = anyio.run(main, backend="asyncio")
    assert completion.usage is None  # type: ignore[attr-defined]


def test_failed_response_raises_without_a_fabricated_completion() -> None:
    error = SimpleNamespace(code="server_error", message="sk-should-not-surface")
    client = _Client(_response(None, error=error, status="failed"))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete([ChatMessage(role="user", content="task")], [], {})
        assert caught.value.stage == "response_parsing"
        assert caught.value.error_code == "server_error"
        assert caught.value.parameter == "provider_status"
        assert "sk-" not in str(caught.value)

    anyio.run(main, backend="asyncio")


def test_provider_exception_propagates() -> None:
    client = _Client(RuntimeError("down"))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete([ChatMessage(role="user", content="task")], [], {})
        assert caught.value.stage == "request"
        assert caught.value.exception_type == "RuntimeError"
        assert "down" not in str(caught.value)

    anyio.run(main, backend="asyncio")


class _APIStatus(Exception):
    def __init__(self) -> None:
        super().__init__("Unsupported parameter. sk-secret raw body")
        self.status_code = 400
        self.request_id = "req_abc"
        self.body = {
            "error": {
                "message": "sk-secret",
                "param": "temperature",
                "code": "unsupported_parameter",
            }
        }


def test_request_errors_keep_status_code_and_drop_the_body() -> None:
    client = _Client(_APIStatus())
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete([ChatMessage(role="user", content="task")], [], {})
        failure = caught.value
        assert failure.stage == "request"
        assert failure.exception_type == "_APIStatus"
        assert failure.status_code == 400
        assert failure.error_code == "unsupported_parameter"
        assert failure.request_id == "req_abc"
        assert failure.parameter == "temperature"
        assert "sk-" not in str(failure)
        assert "raw body" not in str(failure)

    anyio.run(main, backend="asyncio")


def test_luna_request_errors_stay_classified_without_credentials() -> None:
    client = _Client(_APIStatus())
    adapter = OpenAIAdapter(client, model="gpt-6-luna")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete(
                [ChatMessage(role="user", content="task")],
                [{"name": "get_logs", "description": "logs", "input_schema": {"type": "object"}}],
                {"temperature": 0, "top_p": 1, "top_logprobs": 3, "reasoning_effort": "low"},
            )
        failure = caught.value
        assert failure.stage == "request"
        assert failure.exception_type == "_APIStatus"
        assert failure.status_code == 400
        assert failure.error_code == "unsupported_parameter"
        assert failure.request_id == "req_abc"
        assert failure.parameter == "temperature"
        assert "sk-" not in str(failure)
        assert "raw body" not in str(failure)

    anyio.run(main, backend="asyncio")
    sent = client.responses.kwargs
    assert sent is not None
    assert sent["model"] == "gpt-6-luna"
    assert sent["reasoning"] == {"effort": "low"}
    for name in ("temperature", "top_p", "top_logprobs", "api_key", "authorization"):
        assert name not in sent
    assert "sk-" not in str(sent)


def test_continuation_errors_are_labeled_before_the_request() -> None:
    client = _Client(_response("ok"))
    adapter = OpenAIAdapter(client, model="gpt-test")

    async def main() -> None:
        with pytest.raises(ProviderCallError) as caught:
            await adapter.complete(
                [ChatMessage(role="tool", content="OBS", tool_name="get_metrics")],
                [],
                {},
            )
        assert caught.value.stage == "continuation"
        assert client.responses.history == []

    anyio.run(main, backend="asyncio")
