"""Optional OpenAI adapter.

The agent loop, replay runner, and fixture store do not import this
module. Install it with the ``live`` extra. Tests inject a client.
Nothing in this module chooses an endpoint or reads a key unless
``from_env`` is called without an explicit environment mapping.

Completions use the Responses API. A follow-up call sends
``previous_response_id`` from the assistant message in the transcript
and only the new ``function_call_output`` items. The adapter does not
keep its own response cursor, so two replay arms cannot share one.
"""

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from parallax.agent import ChatMessage, ModelCompletion, ModelUsage, ToolRequest
from parallax.trace import ProviderErrorMetadata

_STAGES = frozenset({"request", "response_parsing", "continuation", "agent_controller"})
_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SECRET_MARKERS = ("sk-", "bearer", "api_key", "authorization")
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_SAMPLING_FIELDS = ("temperature", "top_p", "top_logprobs")
# These GPT-6 models reject sampling at every effort, including none.
_SAMPLING_NEVER = ("gpt-6-astra", "gpt-6.1")


def project_root() -> Path:
    """Directory that contains this project's ``pyproject.toml``."""

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "parallax").is_dir():
            return candidate
    raise RuntimeError("PARALLAX project root was not found")


def load_project_env(root: Path | None = None) -> bool:
    """Load ``<project-root>/.env``. Existing process variables stay in place."""

    directory = project_root() if root is None else root
    return bool(load_dotenv(directory / ".env"))


class OpenAIAdapter:
    """ModelAdapter backed by the Responses API.

    ``omitted_params`` lists model parameters that were not sent on the
    last call. ``max_tokens`` is sent as ``max_output_tokens``. ``seed``
    is not a Responses parameter. ``reasoning_effort`` is sent as
    ``reasoning.effort`` and is left unset when the caller omitted it.
    ``temperature``, ``top_p``, and ``top_logprobs`` are left out of the
    request while reasoning is active.
    """

    def __init__(self, client: Any, *, model: str) -> None:
        self._client = client
        self._model = model
        self.omitted_params: tuple[str, ...] = ()

    @property
    def model(self) -> str:
        """Provider model id configured for this adapter."""

        return self._model

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        load_env: Callable[[], None] | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> "OpenAIAdapter":
        """Build a client from ``OPENAI_API_KEY`` and ``PARALLAX_MODEL``.

        An explicit ``environ`` mapping is used as given. The process
        environment and ``.env`` files are left untouched in that case.
        A missing key or model raises before a client is constructed.
        """

        if environ is None:
            if load_env is None:
                load_project_env()
            else:
                load_env()
            source: Mapping[str, str] = os.environ
        else:
            source = environ
        key = source.get("OPENAI_API_KEY", "").strip()
        model = source.get("PARALLAX_MODEL", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if not model:
            raise RuntimeError("PARALLAX_MODEL is not set")
        factory = client_factory if client_factory is not None else (lambda api_key: AsyncOpenAI(api_key=api_key))
        return cls(factory(key), model=model)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
    ) -> ModelCompletion:
        """Request one response. Usage stays null when the client omits it."""

        forwarded, omitted = _select_params(self._model, params)
        self.omitted_params = omitted
        try:
            provider_input, previous_response_id = _provider_input(messages)
        except ProviderCallError:
            raise
        except Exception as exc:  # noqa: BLE001 — classify every failure without keeping its text
            raise _failure_from_exception(exc, "continuation") from None
        kwargs: dict[str, Any] = {
            "input": provider_input,
            "model": self._model,
            **forwarded,
        }
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id
        provider_tools = _provider_tools(tools)
        if provider_tools:
            kwargs["tools"] = provider_tools
        try:
            response = await self._client.responses.create(**kwargs)
        except ProviderCallError:
            raise
        except Exception as exc:  # noqa: BLE001 — classify every failure without keeping its text
            raise _failure_from_exception(exc, "request") from None
        try:
            _raise_if_failed(response)
            content, tool_calls, continuation = _parse_output(getattr(response, "output", None))
        except ProviderCallError:
            raise
        except Exception as exc:  # noqa: BLE001 — classify every failure without keeping its text
            raise _failure_from_exception(exc, "response_parsing") from None
        response_id = getattr(response, "id", None)
        return ModelCompletion(
            content=content,
            tool_calls=tool_calls,
            usage=_usage(response),
            response_id=response_id if isinstance(response_id, str) and response_id else None,
            continuation=continuation,
        )


def low_reasoning_effort_supported(model: str) -> bool:
    """True when this model id accepts ``reasoning.effort="low"``."""

    return model.strip().lower().startswith("gpt-6")


def _select_params(model: str, params: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build Responses kwargs. Incompatible sampling keys are absent, not null."""

    effort = _configured_effort(params)
    forwarded: dict[str, Any] = {}
    consumed: set[str] = set()
    if "max_tokens" in params:
        forwarded["max_output_tokens"] = params["max_tokens"]
        consumed.add("max_tokens")
    if effort is not None:
        forwarded["reasoning"] = {"effort": effort}
        consumed.add("reasoning_effort")
    if not _sampling_forbidden(model, effort):
        for name in _SAMPLING_FIELDS:
            if name in params:
                forwarded[name] = params[name]
                consumed.add(name)
    omitted = tuple(name for name in params if name not in consumed)
    return forwarded, omitted


def _configured_effort(params: Mapping[str, Any]) -> str | None:
    effort = params.get("reasoning_effort")
    if isinstance(effort, str) and effort in _EFFORTS:
        return effort
    return None


def _sampling_forbidden(model: str, effort: str | None) -> bool:
    """True when temperature, top_p, and top_logprobs must not be sent."""

    name = model.strip().lower()
    if name.startswith(_SAMPLING_NEVER):
        return True
    if effort is None:
        # GPT-6 reasons at its server default unless effort is none.
        return name.startswith("gpt-6")
    return effort != "none"


def _provider_input(messages: Sequence[ChatMessage]) -> tuple[list[dict[str, Any]], str | None]:
    latest = _latest_assistant(messages)
    if latest is not None and latest.response_id:
        pending = _pending_ids(latest, 0)
        items = [
            _function_output(message, pending)
            for message in _items_after(messages, latest)
        ]
        return items, latest.response_id
    payload: list[dict[str, Any]] = []
    pending: list[str] = []
    for index, message in enumerate(messages):
        if message.role in {"system", "user", "developer"}:
            payload.append({"role": message.role, "content": message.content})
            continue
        if message.role == "assistant":
            pending = _emit_assistant(message, index, payload)
            continue
        if message.role == "tool":
            payload.append(_function_output(message, pending))
            continue
        raise ValueError(f"unsupported message role: {message.role}")
    return payload, None


def _latest_assistant(messages: Sequence[ChatMessage]) -> ChatMessage | None:
    latest = None
    for message in messages:
        if message.role == "assistant":
            latest = message
    return latest


def _items_after(messages: Sequence[ChatMessage], latest: ChatMessage) -> list[ChatMessage]:
    found = False
    rows: list[ChatMessage] = []
    for message in messages:
        if message is latest:
            found = True
            continue
        if not found:
            continue
        if message.role != "tool":
            raise ValueError("continuation item is not a tool result")
        rows.append(message)
    return rows


def _emit_assistant(message: ChatMessage, index: int, payload: list[dict[str, Any]]) -> list[str]:
    if message.continuation:
        payload.extend(dict(item) for item in message.continuation)
        return [
            str(item["call_id"])
            for item in message.continuation
            if item.get("type") == "function_call" and isinstance(item.get("call_id"), str)
        ]
    if message.content and not message.tool_calls:
        payload.append({"role": "assistant", "content": message.content})
        return []
    if message.content:
        payload.append({"role": "assistant", "content": message.content})
    return _emit_function_calls(message, index, payload)


def _emit_function_calls(message: ChatMessage, index: int, payload: list[dict[str, Any]]) -> list[str]:
    pending: list[str] = []
    for call_index, call in enumerate(message.tool_calls):
        call_id = call.call_id or f"call_{index}_{call_index}"
        pending.append(call_id)
        payload.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": call.name,
                "arguments": json.dumps(call.arguments, sort_keys=True, separators=(",", ":")),
            }
        )
    return pending


def _pending_ids(message: ChatMessage, index: int) -> list[str]:
    if message.continuation:
        return [
            str(item["call_id"])
            for item in message.continuation
            if item.get("type") == "function_call" and isinstance(item.get("call_id"), str)
        ]
    return [call.call_id or f"call_{index}_{call_index}" for call_index, call in enumerate(message.tool_calls)]


def _function_output(message: ChatMessage, pending: list[str]) -> dict[str, Any]:
    if message.role != "tool":
        raise ValueError("continuation item is not a tool result")
    if message.call_id:
        if message.call_id in pending:
            pending.remove(message.call_id)
        call_id = message.call_id
    elif not pending:
        raise ValueError("tool message has no matching assistant tool call")
    else:
        call_id = pending.pop(0)
    return {"type": "function_call_output", "call_id": call_id, "output": message.content}


def _provider_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "type": "function",
            "name": tool["name"],
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            "strict": False,
        }
        description = tool.get("description")
        if isinstance(description, str) and description:
            function["description"] = description
        rows.append(function)
    return rows


def _parse_output(output: Any) -> tuple[str, tuple[ToolRequest, ...], tuple[dict[str, Any], ...]]:
    if not output:
        return "", (), ()
    texts: list[str] = []
    requests: list[ToolRequest] = []
    continuation: list[dict[str, Any]] = []
    for item in output:
        kind = getattr(item, "type", None)
        if kind == "message":
            texts.append(_message_text(item))
            continue
        if kind == "function_call":
            requests.append(_tool_request(item))
            continuation.append(_function_call_item(item))
            continue
        if kind == "reasoning":
            continuation.append(_reasoning_item(item))
    return "".join(texts), tuple(requests), tuple(continuation)


def _tool_request(item: Any) -> ToolRequest:
    arguments_text = getattr(item, "arguments", None)
    if not isinstance(arguments_text, str):
        raise TypeError("tool arguments are not a string")
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments are not JSON") from exc
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments are not an object")
    name = getattr(item, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("tool call is missing a name")
    call_id = getattr(item, "call_id", None)
    return ToolRequest(
        name=name,
        arguments=arguments,
        call_id=call_id if isinstance(call_id, str) and call_id else None,
    )


def _function_call_item(item: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "function_call",
        "call_id": getattr(item, "call_id", None),
        "name": getattr(item, "name", None),
        "arguments": getattr(item, "arguments", None),
    }
    item_id = getattr(item, "id", None)
    if isinstance(item_id, str) and item_id:
        row["id"] = item_id
    status = getattr(item, "status", None)
    if isinstance(status, str) and status:
        row["status"] = status
    return row


def _reasoning_item(item: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "reasoning",
        "id": getattr(item, "id", None),
        "summary": _jsonable(getattr(item, "summary", [])),
    }
    for name in ("content", "encrypted_content", "status"):
        value = getattr(item, name, None)
        if value is not None:
            row[name] = _jsonable(value)
    return row


def _message_text(item: Any) -> str:
    parts: list[str] = []
    for part in getattr(item, "content", None) or []:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True, by_alias=True)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _usage(response: Any) -> ModelUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "output_tokens", None)
    if prompt is None or completion is None:
        return None
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    return ModelUsage(input_tokens=prompt, output_tokens=completion)


class ProviderCallError(Exception):
    """Provider failure whose text contains no request or response body."""

    def __init__(
        self,
        *,
        stage: str,
        exception_type: str,
        status_code: int | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        parameter: str | None = None,
    ) -> None:
        self.stage = stage if stage in _STAGES else "agent_controller"
        self.exception_type = exception_type if exception_type.isidentifier() else "Exception"
        self.status_code = status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else None
        self.error_code = _safe_token(error_code, 64)
        self.request_id = _safe_token(request_id, 128)
        self.parameter = _safe_token(parameter, 64)
        self.metadata = ProviderErrorMetadata(
            stage=self.stage,
            exception_class=self.exception_type,
            http_status=self.status_code,
            error_code=self.error_code,
            parameter=self.parameter,
            request_id=self.request_id,
        )
        super().__init__(self.public_text())

    def public_text(self) -> str:
        parts = [f"stage={self.stage}", f"exception={self.exception_type}"]
        if self.status_code is not None:
            parts.append(f"http_status={self.status_code}")
        if self.error_code is not None:
            parts.append(f"error_code={self.error_code}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        if self.parameter is not None:
            parts.append(f"parameter={self.parameter}")
        return " ".join(parts)


def _safe_token(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) > limit or not _TOKEN.fullmatch(value):
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return None
    return value


def _failure_from_exception(exc: BaseException, stage: str) -> ProviderCallError:
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    error_code = None
    parameter = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        details = err if isinstance(err, dict) else body
        error_code = details.get("code")
        parameter = details.get("param")
    if error_code is None:
        error_code = getattr(exc, "code", None)
    if parameter is None:
        parameter = getattr(exc, "param", None)
    return ProviderCallError(
        stage=stage,
        exception_type=type(exc).__name__,
        status_code=status if isinstance(status, int) else None,
        error_code=error_code if isinstance(error_code, str) else None,
        request_id=request_id if isinstance(request_id, str) else None,
        parameter=parameter if isinstance(parameter, str) else None,
    )


def _raise_if_failed(response: Any) -> None:
    error = getattr(response, "error", None)
    status = getattr(response, "status", None)
    if error is None and status != "failed":
        return
    code = getattr(error, "code", None) if error is not None else None
    raise ProviderCallError(
        stage="response_parsing",
        exception_type="RuntimeError",
        error_code=code if isinstance(code, str) else None,
        parameter="provider_status",
    )
