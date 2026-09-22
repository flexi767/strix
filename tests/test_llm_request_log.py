from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, cast

import httpx
import litellm
import pytest
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tool import FunctionTool
from agents.usage import Usage
from litellm.llms.anthropic.common_utils import AnthropicError
from openai import APIStatusError, APITimeoutError, PermissionDeniedError
from openai.types.responses import Response, ResponseCompletedEvent, ResponseCreatedEvent
from openai.types.responses.response import IncompleteDetails

from strix.llm import request_log


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from strix.llm.request_log import LlmRequestEvent


ANTHROPIC_BLOCK_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error","message":"Output blocked by '
    'content filtering policy"},"request_id":"req_011CVBodyOnly00"}'
)


@pytest.fixture
def captured() -> Iterator[list[LlmRequestEvent]]:
    events: list[LlmRequestEvent] = []
    request_log.register_sink(events.append)
    try:
        yield events
    finally:
        request_log.unregister_sink(events.append)


@pytest.fixture(autouse=True)
def _reset_context() -> Iterator[None]:
    token = request_log.bind_call_context(None, None)
    try:
        yield
    finally:
        request_log.reset_call_context(token)


def _anthropic_kwargs(
    exc: BaseException | None,
    *,
    headers: dict[str, str] | None = None,
    stream: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    slo: dict[str, Any] = {
        "model": "claude-sonnet-4-5",
        "custom_llm_provider": "anthropic",
        "api_base": "https://api.anthropic.com",
        "stream": stream,
        "litellm_call_id": "call-1",
        "hidden_params": {
            "additional_headers": {f"llm_provider-{k}": v for k, v in (headers or {}).items()},
            "response_cost": 0.0123,
        },
    }
    if exc is not None:
        slo["error_information"] = {
            "error_code": str(getattr(exc, "status_code", "")),
            "error_class": type(exc).__name__,
            "llm_provider": "anthropic",
            "error_message": error_message or str(exc),
        }
    kwargs: dict[str, Any] = {
        "model": "anthropic/claude-sonnet-4-5",
        "custom_llm_provider": "anthropic",
        "litellm_call_id": "call-1",
        "stream": stream,
        "standard_logging_object": slo,
        "litellm_params": {"api_base": "https://api.anthropic.com", "api_key": "sk-ant-secret"},
        "messages": [{"role": "user", "content": "SECRET PROMPT"}],
    }
    if exc is not None:
        kwargs["exception"] = exc
    return kwargs


class _FakeUsage:
    prompt_tokens = 120
    completion_tokens = 30
    total_tokens = 150
    prompt_tokens_details = None
    cache_read_input_tokens = 100


class _FakeResponse:
    id = "msg_01abc"
    usage = _FakeUsage()
    _hidden_params: ClassVar[dict[str, Any]] = {}
    choices: ClassVar[list[Any]] = [{"message": {"content": "SECRET COMPLETION"}}]


def _anthropic_error(
    status: int, body: str, headers: dict[str, str] | None = None
) -> AnthropicError:
    return AnthropicError(status, body, headers=httpx.Headers(headers or {}))


# --------------------------------------------------------------------------- #
# request-id extraction                                                        #
# --------------------------------------------------------------------------- #


def test_header_request_id_accepts_raw_and_litellm_prefixed_names() -> None:
    assert request_log.request_id_from_headers({"Request-Id": "req_a"}) == "req_a"
    assert request_log.request_id_from_headers({"llm_provider-request-id": "req_b"}) == "req_b"
    assert request_log.request_id_from_headers({"x-request-id": "req_c"}) == "req_c"
    assert request_log.request_id_from_headers({"x-amzn-requestid": "abc-123"}) == "abc-123"
    assert request_log.request_id_from_headers({"content-type": "json"}) is None
    assert request_log.request_id_from_headers({"request-id": "   "}) is None
    assert request_log.request_id_from_headers(None) is None


def test_header_request_id_prefers_provider_id_over_cdn_ray() -> None:
    headers = {"cf-ray": "8f0-FRA", "request-id": "req_real"}
    assert request_log.request_id_from_headers(headers) == "req_real"


def test_body_request_id_extraction() -> None:
    assert request_log.request_id_from_text(ANTHROPIC_BLOCK_BODY) == "req_011CVBodyOnly00"
    assert request_log.request_id_from_text("request_id=req_abcdef") == "req_abcdef"
    assert request_log.request_id_from_text("no id here") is None
    assert request_log.request_id_from_text(None) is None


def test_api_host_never_leaks_path_or_query() -> None:
    assert request_log.api_host("https://gw.corp.example/v1?key=abc") == "gw.corp.example"
    assert request_log.api_host("gw.corp.example:8443/v1") == "gw.corp.example"
    assert request_log.api_host("") is None
    assert request_log.api_host(None) is None


# --------------------------------------------------------------------------- #
# redaction                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
        "sk-proj-abcdefghijklmnopqrstuvwxyz",
        "Bearer abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAABCDEFGHIJKLMNOP",
        "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ01234",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "xoxb-1234567890-abcdefghij",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_redact_removes_credential_shapes(secret: str) -> None:
    redacted = request_log.redact(f"upstream said: {secret} (401)")
    assert secret not in redacted
    assert "[REDACTED]" in redacted
    assert "(401)" in redacted


def test_redact_key_value_forms_keep_the_key_name() -> None:
    text = 'invalid "api_key": "abcd1234", authorization: Basic Zm9vOmJhcg== token=abcdef99'
    redacted = request_log.redact(text)
    assert "abcd1234" not in redacted
    assert "Zm9vOmJhcg==" not in redacted
    assert "abcdef99" not in redacted
    assert '"api_key": "[REDACTED]"' in redacted


def test_clean_error_message_truncates_and_redacts() -> None:
    exc = RuntimeError("x" * 5000 + " sk-ant-api03-abcdefghijklmnop")
    message = request_log.clean_error_message(exc)
    assert len(message) <= request_log.ERROR_MESSAGE_MAX_CHARS
    assert "sk-ant" not in message
    assert request_log.clean_error_message(RuntimeError("")) == "RuntimeError"


# --------------------------------------------------------------------------- #
# LiteLLM route                                                                #
# --------------------------------------------------------------------------- #


def test_litellm_success_event_carries_header_request_id_usage_and_no_content() -> None:
    kwargs = _anthropic_kwargs(None, headers={"request-id": "req_ok_123456"})
    kwargs["response_cost"] = 0.0123
    start = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    end = start + timedelta(milliseconds=850)

    event = request_log.event_from_litellm(kwargs, _FakeResponse(), start, end, outcome="success")

    assert event.route == "litellm"
    assert event.outcome == "success"
    assert event.status_code == 200
    assert event.provider == "anthropic"
    assert event.model == "anthropic/claude-sonnet-4-5"
    assert event.api_host == "api.anthropic.com"
    assert event.provider_request_id == "req_ok_123456"
    assert event.response_id == "msg_01abc"
    assert event.call_id == "call-1"
    assert event.duration_ms == 850
    assert (event.input_tokens, event.output_tokens, event.total_tokens) == (120, 30, 150)
    assert event.cached_input_tokens == 100
    assert event.cost_usd == pytest.approx(0.0123)
    assert event.error_type is None and event.error_message is None
    serialized = str(event.to_dict())
    assert "SECRET PROMPT" not in serialized
    assert "SECRET COMPLETION" not in serialized
    assert "sk-ant-secret" not in serialized


def test_litellm_success_falls_back_to_response_hidden_headers() -> None:
    kwargs = _anthropic_kwargs(None)

    class _Response(_FakeResponse):
        _hidden_params: ClassVar[dict[str, Any]] = {
            "additional_headers": {"llm_provider-request-id": "req_hidden_1"}
        }

    event = request_log.event_from_litellm(kwargs, _Response(), None, None, outcome="success")
    assert event.provider_request_id == "req_hidden_1"


def test_litellm_anthropic_block_prefers_header_id_over_body_id() -> None:
    exc = _anthropic_error(400, ANTHROPIC_BLOCK_BODY, {"request-id": "req_HeaderWins01"})
    kwargs = _anthropic_kwargs(exc, headers={"request-id": "req_HeaderWins01"})

    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")

    assert event.outcome == "error"
    assert event.status_code == 400
    assert event.provider_request_id == "req_HeaderWins01"
    assert event.error_type == "AnthropicError"
    assert event.error_message is not None
    assert "content filtering policy" in event.error_message
    assert "req_011CVBodyOnly00" in event.error_message


def test_litellm_anthropic_block_falls_back_to_body_id() -> None:
    exc = _anthropic_error(400, ANTHROPIC_BLOCK_BODY)
    kwargs = _anthropic_kwargs(exc)

    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")

    assert event.provider_request_id == "req_011CVBodyOnly00"
    assert event.status_code == 400


def test_litellm_failure_uses_exception_headers_when_slo_has_none() -> None:
    exc = _anthropic_error(529, "overloaded", {"request-id": "req_from_exc_hdr"})
    kwargs = _anthropic_kwargs(exc)

    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")

    assert event.provider_request_id == "req_from_exc_hdr"
    assert event.status_code == 529


def test_litellm_failure_without_any_request_id_is_none_not_fabricated() -> None:
    exc = _anthropic_error(500, "internal error")
    event = request_log.event_from_litellm(
        _anthropic_kwargs(exc), None, None, None, outcome="error"
    )
    assert event.provider_request_id is None
    assert event.error_message == "internal error"


def test_litellm_failure_without_exception_object_uses_error_information() -> None:
    kwargs = _anthropic_kwargs(None)
    kwargs["standard_logging_object"]["error_information"] = {
        "error_code": "429",
        "error_class": "RateLimitError",
        "llm_provider": "anthropic",
        "error_message": 'rate limited "request_id": "req_slo_only01"',
    }
    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")
    assert event.status_code == 429
    assert event.error_type == "RateLimitError"
    assert event.provider_request_id == "req_slo_only01"


def test_litellm_timeout_is_an_error_event_with_no_status() -> None:
    exc = APITimeoutError(httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    kwargs = _anthropic_kwargs(exc)
    kwargs["standard_logging_object"]["error_information"]["error_code"] = ""
    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")
    assert event.outcome == "error"
    assert event.status_code is None
    assert event.error_type == "APITimeoutError"
    assert event.provider_request_id is None


def test_litellm_failure_message_is_redacted() -> None:
    exc = _anthropic_error(401, "invalid x-api-key: sk-ant-api03-abcdefghijklmnopqrstu")
    event = request_log.event_from_litellm(
        _anthropic_kwargs(exc), None, None, None, outcome="error"
    )
    assert event.error_message is not None
    assert "sk-ant-api03" not in event.error_message
    assert "[REDACTED]" in event.error_message


def test_litellm_streaming_flag_and_call_id() -> None:
    kwargs = _anthropic_kwargs(None, headers={"request-id": "req_stream1"}, stream=True)
    event = request_log.event_from_litellm(kwargs, _FakeResponse(), None, None, outcome="success")
    assert event.streaming is True
    assert event.provider_request_id == "req_stream1"

    exc = _anthropic_error(400, ANTHROPIC_BLOCK_BODY)
    failed = request_log.event_from_litellm(
        _anthropic_kwargs(exc, stream=True), None, None, None, outcome="error"
    )
    assert failed.streaming is True
    assert failed.provider_request_id == "req_011CVBodyOnly00"


def test_litellm_event_carries_bound_agent_context_and_retry_attempt() -> None:
    token = request_log.bind_call_context("agent-7", "Recon")
    try:
        request_log.set_retry_attempt(2)
        event = request_log.event_from_litellm(
            _anthropic_kwargs(None), _FakeResponse(), None, None, outcome="success"
        )
    finally:
        request_log.reset_call_context(token)
    assert (event.agent_id, event.agent_name, event.retry_attempt) == ("agent-7", "Recon", 2)
    assert request_log.current_call_context().agent_id is None


@pytest.mark.asyncio
async def test_call_context_is_isolated_between_tasks() -> None:
    seen: dict[str, str | None] = {}

    async def run(agent_id: str) -> None:
        token = request_log.bind_call_context(agent_id, None)
        try:
            await asyncio.sleep(0)
            seen[agent_id] = request_log.current_call_context().agent_id
        finally:
            request_log.reset_call_context(token)

    await asyncio.gather(run("a"), run("b"))
    assert seen == {"a": "a", "b": "b"}
    assert request_log.current_call_context().agent_id is None


@pytest.mark.asyncio
async def test_litellm_logger_dispatches_and_isolates_sink_failures(
    captured: list[LlmRequestEvent], caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_event: LlmRequestEvent) -> None:
        raise RuntimeError("sink down")

    request_log.register_sink(boom)
    try:
        logger = request_log._build_litellm_logger()
        with caplog.at_level(logging.ERROR, logger="strix.llm.request_log"):
            await logger.async_log_success_event(
                _anthropic_kwargs(None, headers={"request-id": "req_dispatch"}),
                _FakeResponse(),
                None,
                None,
            )
            await logger.async_log_failure_event(
                _anthropic_kwargs(_anthropic_error(400, ANTHROPIC_BLOCK_BODY)), None, None, None
            )
    finally:
        request_log.unregister_sink(boom)

    assert [e.outcome for e in captured] == ["success", "error"]
    assert captured[0].provider_request_id == "req_dispatch"
    assert captured[1].provider_request_id == "req_011CVBodyOnly00"
    assert sum("sink down" in r.getMessage() or "failed" in r.getMessage() for r in caplog.records)


def test_dispatch_swallows_malformed_callback_payloads(captured: list[LlmRequestEvent]) -> None:
    request_log._dispatch(cast("Any", None), None, None, None, outcome="success")
    assert captured == []


def test_install_is_idempotent_and_registers_one_litellm_callback() -> None:
    request_log.install()
    request_log.install()
    ours = [cb for cb in litellm.callbacks if type(cb).__name__ == "_StrixRequestLogger"]
    assert len(ours) == 1


def test_log_line_sink_formats_without_content(caplog: pytest.LogCaptureFixture) -> None:
    exc = _anthropic_error(400, ANTHROPIC_BLOCK_BODY, {"request-id": "req_line01"})
    event = request_log.event_from_litellm(
        _anthropic_kwargs(exc, headers={"request-id": "req_line01"}),
        None,
        None,
        None,
        outcome="error",
    )
    with caplog.at_level(logging.DEBUG, logger="strix.llm.request_log"):
        request_log._log_line_sink(event)
    line = caplog.records[-1].getMessage()
    assert "request_id=req_line01" in line
    assert "status=400" in line
    assert "provider=anthropic" in line
    assert "SECRET PROMPT" not in line


# --------------------------------------------------------------------------- #
# sizes, timing, finish reason, rate-limit headers                             #
# --------------------------------------------------------------------------- #


def test_rate_limit_headers_keep_only_rate_limit_names() -> None:
    picked = request_log.rate_limit_from_headers(
        {
            "llm_provider-anthropic-ratelimit-requests-remaining": "49",
            "Anthropic-RateLimit-Tokens-Reset": "2026-09-19T12:00:00Z",
            "x-ratelimit-limit-requests": 5000,
            "Retry-After": "12",
            "authorization": "Bearer sk-ant-secret",
            "x-api-key": "sk-ant-secret",
            "set-cookie": "session=abc",
            "request-id": "req_x",
            "content-type": "application/json",
        }
    )
    assert picked == {
        "anthropic-ratelimit-requests-remaining": "49",
        "anthropic-ratelimit-tokens-reset": "2026-09-19T12:00:00Z",
        "x-ratelimit-limit-requests": "5000",
        "retry-after": "12",
    }
    assert "secret" not in str(picked)
    assert request_log.rate_limit_from_headers({"content-type": "json"}) is None
    assert request_log.rate_limit_from_headers(None) is None


def test_rate_limit_headers_are_bounded() -> None:
    headers = {f"x-ratelimit-h{i}": "v" * 500 for i in range(100)}
    picked = request_log.rate_limit_from_headers(headers)
    assert picked is not None
    assert len(picked) == request_log.RATE_LIMIT_MAX_HEADERS
    assert all(len(v) <= 64 for v in picked.values())


def test_json_size_counts_utf8_bytes_of_compact_json() -> None:
    assert request_log.json_size({"a": "é"}) == len('{"a":"é"}'.encode())
    assert request_log.json_size(None) is None
    assert request_log.json_size(_openai_response("r")) is not None
    assert request_log.json_size(object()) is not None  # default=str fallback


def test_litellm_success_carries_sizes_finish_reason_and_rate_limit() -> None:
    kwargs = _anthropic_kwargs(
        None,
        headers={
            "request-id": "req_ok",
            "anthropic-ratelimit-requests-remaining": "49",
            "anthropic-ratelimit-tokens-remaining": "39000",
            "x-api-key": "sk-ant-secret",
        },
    )
    kwargs["optional_params"] = {"max_tokens": 4096, "temperature": 0}

    class _Choice:
        finish_reason = "tool_calls"

    class _Response(_FakeResponse):
        choices: ClassVar[list[Any]] = [_Choice()]

    event = request_log.event_from_litellm(kwargs, _Response(), None, None, outcome="success")

    expected_request = request_log.json_size(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "SECRET PROMPT"}],
            "max_tokens": 4096,
            "temperature": 0,
        }
    )
    assert event.request_bytes == expected_request
    assert event.response_bytes is not None and event.response_bytes > 0
    assert event.finish_reason == "tool_calls"
    assert event.rate_limit == {
        "anthropic-ratelimit-requests-remaining": "49",
        "anthropic-ratelimit-tokens-remaining": "39000",
    }
    assert event.time_to_first_token_ms is None
    assert "sk-ant-secret" not in str(event.to_dict())
    assert "SECRET PROMPT" not in str(event.to_dict())


def test_litellm_streaming_time_to_first_token_from_completion_start() -> None:
    kwargs = _anthropic_kwargs(None, stream=True)
    start = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    kwargs["completion_start_time"] = start + timedelta(milliseconds=420)
    end = start + timedelta(seconds=3)

    event = request_log.event_from_litellm(kwargs, _FakeResponse(), start, end, outcome="success")

    assert event.streaming is True
    assert event.time_to_first_token_ms == 420
    assert event.duration_ms == 3000

    no_first = request_log.event_from_litellm(
        _anthropic_kwargs(None, stream=True), _FakeResponse(), start, end, outcome="success"
    )
    assert no_first.time_to_first_token_ms is None


def test_litellm_failure_carries_status_body_size_and_rate_limit_from_exception() -> None:
    body = '{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'
    exc = _anthropic_error(
        429,
        body,
        {
            "request-id": "req_429",
            "retry-after": "7",
            "anthropic-ratelimit-requests-remaining": "0",
            "x-api-key": "sk-ant-secret",
        },
    )
    kwargs = _anthropic_kwargs(exc)
    kwargs["optional_params"] = {"max_tokens": 10}

    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")

    assert event.status_code == 429
    assert event.provider_request_id == "req_429"
    assert event.request_bytes is not None and event.request_bytes > 0
    assert event.response_bytes == len(body.encode())
    assert event.rate_limit == {
        "retry-after": "7",
        "anthropic-ratelimit-requests-remaining": "0",
    }
    assert event.finish_reason is None
    assert "sk-ant-secret" not in str(event.to_dict())


def test_litellm_failure_without_headers_has_no_rate_limit() -> None:
    exc = APITimeoutError(httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    kwargs = _anthropic_kwargs(exc)
    kwargs["standard_logging_object"]["error_information"]["error_code"] = ""
    event = request_log.event_from_litellm(kwargs, None, None, None, outcome="error")
    assert event.rate_limit is None
    assert event.response_bytes is not None  # size of the timeout message text


def test_to_dict_and_log_line_include_new_fields() -> None:
    kwargs = _anthropic_kwargs(None, headers={"request-id": "req_ok"}, stream=True)
    start = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    kwargs["completion_start_time"] = start + timedelta(milliseconds=100)
    event = request_log.event_from_litellm(
        kwargs, _FakeResponse(), start, start + timedelta(seconds=1), outcome="success"
    )
    data = event.to_dict()
    for key in (
        "request_bytes",
        "response_bytes",
        "time_to_first_token_ms",
        "finish_reason",
        "rate_limit",
    ):
        assert key in data
    assert data["time_to_first_token_ms"] == 100


# --------------------------------------------------------------------------- #
# Native OpenAI route                                                          #
# --------------------------------------------------------------------------- #


def _openai_response(response_id: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        output=[],
        usage=usage or Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        response_id=response_id,
    )


def _completed_event(response_id: str) -> ResponseCompletedEvent:
    response = Response(
        id=response_id,
        created_at=0,
        model="gpt-5",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return ResponseCompletedEvent(response=response, sequence_number=1, type="response.completed")


class _Inner(Model):
    def __init__(
        self,
        *,
        response: ModelResponse | None = None,
        exc: BaseException | None = None,
        stream_events: list[Any] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._response = response
        self._exc = exc
        self._stream_events = stream_events or []
        self._fail_after = fail_after
        self.closed = False

    async def get_response(self, *_args: Any, **_kwargs: Any) -> ModelResponse:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response

    async def stream_response(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        for index, event in enumerate(self._stream_events):
            if self._fail_after is not None and index == self._fail_after:
                assert self._exc is not None
                raise self._exc
            yield event

    async def close(self) -> None:
        self.closed = True


_CALL_ARGS: tuple[Any, ...] = (None, "hi", None, [], None, [], None)
_CALL_KWARGS: dict[str, Any] = {
    "previous_response_id": None,
    "conversation_id": None,
    "prompt": None,
}


def _openai_status_error(status: int, request_id: str | None, body: str = "") -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    headers = {"x-request-id": request_id} if request_id else {}
    response = httpx.Response(status, request=request, headers=headers, text=body)
    if status == 403:
        return PermissionDeniedError(body or "denied", response=response, body=None)
    return APIStatusError(body or f"status {status}", response=response, body=None)


@pytest.mark.asyncio
async def test_openai_route_success_event(captured: list[LlmRequestEvent]) -> None:
    model = request_log.RequestLoggingModel(
        _Inner(response=_openai_response("resp_123")),
        model_name="gpt-5",
        provider="openai",
        base_url=None,
    )
    result = await model.get_response(*_CALL_ARGS, **_CALL_KWARGS)

    assert result.response_id == "resp_123"
    assert len(captured) == 1
    event = captured[0]
    assert event.route == "openai"
    assert event.provider == "openai"
    assert event.api_host == "api.openai.com"
    assert event.outcome == "success"
    assert event.status_code == 200
    assert event.response_id == "resp_123"
    assert event.provider_request_id is None
    assert (event.input_tokens, event.output_tokens, event.total_tokens) == (10, 5, 15)
    assert event.streaming is False
    assert model.model == "gpt-5"


@pytest.mark.asyncio
async def test_openai_route_blocked_request_keeps_header_request_id(
    captured: list[LlmRequestEvent],
) -> None:
    exc = _openai_status_error(403, "req_openai_blocked", "content policy violation")
    model = request_log.RequestLoggingModel(
        _Inner(exc=exc),
        model_name="gpt-5",
        provider="openai",
        base_url="https://gateway.corp.example/v1?token=abc",
    )
    with pytest.raises(PermissionDeniedError):
        await model.get_response(*_CALL_ARGS, **_CALL_KWARGS)

    event = captured[0]
    assert event.outcome == "error"
    assert event.status_code == 403
    assert event.provider_request_id == "req_openai_blocked"
    assert event.error_type == "PermissionDeniedError"
    assert event.api_host == "gateway.corp.example"
    assert "token=abc" not in str(event.to_dict())


@pytest.mark.asyncio
async def test_openai_route_timeout_event(captured: list[LlmRequestEvent]) -> None:
    exc = APITimeoutError(httpx.Request("POST", "https://api.openai.com/v1/responses"))
    model = request_log.RequestLoggingModel(
        _Inner(exc=exc), model_name="gpt-5", provider="openai", base_url=None
    )
    with pytest.raises(APITimeoutError):
        await model.get_response(*_CALL_ARGS, **_CALL_KWARGS)
    assert captured[0].outcome == "error"
    assert captured[0].status_code is None
    assert captured[0].error_type == "APITimeoutError"
    assert captured[0].provider_request_id is None


@pytest.mark.asyncio
async def test_openai_route_cancellation_is_not_logged(captured: list[LlmRequestEvent]) -> None:
    model = request_log.RequestLoggingModel(
        _Inner(exc=asyncio.CancelledError()), model_name="gpt-5", provider="openai", base_url=None
    )
    with pytest.raises(asyncio.CancelledError):
        await model.get_response(*_CALL_ARGS, **_CALL_KWARGS)
    assert captured == []


@pytest.mark.asyncio
async def test_openai_route_streaming_success(captured: list[LlmRequestEvent]) -> None:
    created = ResponseCreatedEvent(
        response=_completed_event("resp_stream").response,
        sequence_number=0,
        type="response.created",
    )
    model = request_log.RequestLoggingModel(
        _Inner(stream_events=[created, _completed_event("resp_stream")]),
        model_name="gpt-5",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api",
    )
    events = [e async for e in model.stream_response(*_CALL_ARGS, **_CALL_KWARGS)]

    assert len(events) == 2
    assert len(captured) == 1
    event = captured[0]
    assert event.streaming is True
    assert event.outcome == "success"
    assert event.response_id == "resp_stream"
    assert event.provider == "openai-codex"
    assert event.api_host == "chatgpt.com"


@pytest.mark.asyncio
async def test_openai_route_streaming_failure_midstream(captured: list[LlmRequestEvent]) -> None:
    exc = _openai_status_error(500, "req_mid_stream", "upstream reset")
    created = ResponseCreatedEvent(
        response=_completed_event("resp_x").response, sequence_number=0, type="response.created"
    )
    model = request_log.RequestLoggingModel(
        _Inner(stream_events=[created, created], exc=exc, fail_after=1),
        model_name="gpt-5",
        provider="openai",
        base_url=None,
    )
    received = 0
    with pytest.raises(APIStatusError):
        async for _ in model.stream_response(*_CALL_ARGS, **_CALL_KWARGS):
            received += 1

    assert received == 1
    assert len(captured) == 1
    assert captured[0].streaming is True
    assert captured[0].outcome == "error"
    assert captured[0].status_code == 500
    assert captured[0].provider_request_id == "req_mid_stream"
    assert captured[0].time_to_first_token_ms is not None
    assert captured[0].response_bytes == len(b"upstream reset")


@pytest.mark.asyncio
async def test_openai_route_success_carries_request_and_response_sizes(
    captured: list[LlmRequestEvent],
) -> None:
    tool = FunctionTool(
        name="lookup",
        description="Look something up",
        params_json_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        on_invoke_tool=_noop_tool,
    )
    model = request_log.RequestLoggingModel(
        _Inner(response=_openai_response("resp_sized")),
        model_name="gpt-5",
        provider="openai",
        base_url=None,
    )
    args = list(_CALL_ARGS)
    args[0] = "SYSTEM SECRET INSTRUCTIONS"
    args[3] = [tool]
    await model.get_response(*args, **_CALL_KWARGS)

    event = captured[0]
    expected = request_log.json_size(
        {
            "instructions": "SYSTEM SECRET INSTRUCTIONS",
            "input": "hi",
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": tool.params_json_schema,
                }
            ],
        }
    )
    assert event.request_bytes == expected
    assert event.response_bytes == request_log.json_size([])
    assert event.time_to_first_token_ms is None
    assert event.finish_reason is None
    assert event.rate_limit is None
    assert "SYSTEM SECRET INSTRUCTIONS" not in str(event.to_dict())


@pytest.mark.asyncio
async def test_openai_route_error_carries_rate_limit_headers_and_body_size(
    captured: list[LlmRequestEvent],
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    body = '{"error":{"message":"Rate limit reached","type":"tokens"}}'
    response = httpx.Response(
        429,
        request=request,
        headers={
            "x-request-id": "req_429_openai",
            "x-ratelimit-limit-tokens": "30000",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "6ms",
            "retry-after": "1",
            "openai-organization": "org-secret",
        },
        text=body,
    )
    exc = APIStatusError("rate limited", response=response, body=None)
    model = request_log.RequestLoggingModel(
        _Inner(exc=exc), model_name="gpt-5", provider="openai", base_url=None
    )
    with pytest.raises(APIStatusError):
        await model.get_response(*_CALL_ARGS, **_CALL_KWARGS)

    event = captured[0]
    assert event.status_code == 429
    assert event.provider_request_id == "req_429_openai"
    assert event.rate_limit == {
        "x-ratelimit-limit-tokens": "30000",
        "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-tokens": "6ms",
        "retry-after": "1",
    }
    assert event.response_bytes is not None and event.response_bytes > 0
    assert "org-secret" not in str(event.to_dict())


@pytest.mark.asyncio
async def test_openai_route_streaming_ttft_and_finish_reason(
    captured: list[LlmRequestEvent],
) -> None:
    created = ResponseCreatedEvent(
        response=_completed_event("resp_fin").response,
        sequence_number=0,
        type="response.created",
    )
    completed = _completed_event("resp_fin")
    completed.response.status = "incomplete"
    completed.response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
    model = request_log.RequestLoggingModel(
        _Inner(stream_events=[created, completed]),
        model_name="gpt-5",
        provider="openai",
        base_url=None,
    )
    async for _ in model.stream_response(*_CALL_ARGS, **_CALL_KWARGS):
        pass

    event = captured[0]
    assert event.streaming is True
    assert event.time_to_first_token_ms is not None
    assert event.time_to_first_token_ms <= event.duration_ms
    assert event.finish_reason == "incomplete:max_output_tokens"

    captured.clear()
    plain = request_log.RequestLoggingModel(
        _Inner(stream_events=[created, _completed_event("resp_done")]),
        model_name="gpt-5",
        provider="openai",
        base_url=None,
    )
    async for _ in plain.stream_response(*_CALL_ARGS, **_CALL_KWARGS):
        pass
    assert captured[0].finish_reason is None or captured[0].finish_reason == "completed"


async def _noop_tool(_ctx: Any, _args: str) -> str:
    return ""


@pytest.mark.asyncio
async def test_openai_route_delegates_close() -> None:
    inner = _Inner(response=_openai_response("r"))
    model = request_log.RequestLoggingModel(
        inner, model_name="gpt-5", provider="openai", base_url=None
    )
    await model.close()
    assert inner.closed is True


# --------------------------------------------------------------------------- #
# failure_text                                                                 #
# --------------------------------------------------------------------------- #


def test_failure_text_appends_header_id_for_openai_errors() -> None:
    exc = _openai_status_error(403, "req_hdr_only", "blocked")
    text = request_log.failure_text(exc)
    assert text.endswith("[provider request id: req_hdr_only]")
    assert "blocked" in text


def test_failure_text_does_not_duplicate_body_id() -> None:
    exc = _anthropic_error(400, ANTHROPIC_BLOCK_BODY)
    text = request_log.failure_text(exc)
    assert text.count("req_011CVBodyOnly00") == 1
    assert "[provider request id" not in text


def test_failure_text_uses_litellm_exception_headers_and_redacts() -> None:
    exc = _anthropic_error(
        401, "bad key sk-ant-api03-abcdefghijklmnopqrstu", {"request-id": "req_exc_hdr"}
    )
    text = request_log.failure_text(exc)
    assert "sk-ant-api03" not in text
    assert text.endswith("[provider request id: req_exc_hdr]")


def test_failure_text_plain_exception_unchanged() -> None:
    assert request_log.failure_text(RuntimeError("boom")) == "boom"
    assert request_log.failure_text(RuntimeError("")) == "RuntimeError"
