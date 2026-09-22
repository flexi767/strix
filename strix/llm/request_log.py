"""Structured, secret-free record of every model provider call.

One :class:`LlmRequestEvent` per HTTP attempt against a provider, whatever the
route (LiteLLM or the native OpenAI client) and whatever the outcome. Retries
are separate events; a streamed call is one event once the stream settles.
Events carry the provider's own request identifier when the provider returns
one (Anthropic ``request-id``, OpenAI ``x-request-id``), which is what a
provider's support team asks for when a call was blocked or misbehaved.

The typed fields are the common ground every provider shares (status, ids,
timing, sizes, tokens, cost). Everything else a provider or SDK reports about
the attempt travels free-form: the response headers minus credential-bearing
names, and a ``details`` object with the request parameters, the full usage
object, the SDK's hidden parameters and the error body. Content (prompts,
completions, tool arguments) and credentials are removed from both, values are
redacted and every string, list and object is bounded. Raw request and
response bodies are never captured.

Sinks are plain callables. The built-in sink writes one log line per event;
deployments register their own (a database, a queue) with
:func:`register_sink`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit

from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tool import FunctionTool
from agents.usage import Usage
from openai import APIError, APIStatusError
from openai.types.responses import ResponseCompletedEvent
from pydantic import BaseModel


if TYPE_CHECKING:
    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import TResponseInputItem, TResponseStreamEvent
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from agents.retry import ModelRetryAdvice, ModelRetryAdviceRequest
    from agents.tool import Tool
    from openai.types.responses import ResponsePromptParam


logger = logging.getLogger(__name__)

Outcome = Literal["success", "error"]
Route = Literal["litellm", "openai"]

ERROR_MESSAGE_MAX_CHARS = 2000


@dataclass(frozen=True)
class LlmCallContext:
    """Who is making the call, and which replay of the turn this is."""

    agent_id: str | None = None
    agent_name: str | None = None
    retry_attempt: int = 0


@dataclass(frozen=True)
class LlmRequestEvent:
    """One attempt against a model provider."""

    call_id: str
    route: Route
    provider: str | None
    model: str
    api_host: str | None
    streaming: bool
    outcome: Outcome
    status_code: int | None
    provider_request_id: str | None
    response_id: str | None
    error_type: str | None
    error_message: str | None
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    retry_attempt: int = 0
    # Serialized JSON body sizes, not on-the-wire bytes.
    request_bytes: int | None = None
    response_bytes: int | None = None
    # Streamed calls only.
    time_to_first_token_ms: int | None = None
    finish_reason: str | None = None
    # Every response header the provider sent, name-normalized, minus
    # credential-bearing names; values redacted and capped.
    response_headers: dict[str, str] | None = None
    # Free-form: request parameters, full usage object, SDK hidden params,
    # error body ... with content and credentials removed and sizes bounded.
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["finished_at"] = self.finished_at.isoformat()
        return data


LlmRequestSink = Callable[[LlmRequestEvent], None]

_sinks: list[LlmRequestSink] = []
_DEFAULT_CALL_CONTEXT = LlmCallContext()
_call_context: ContextVar[LlmCallContext] = ContextVar("strix_llm_call_context")


def bind_call_context(agent_id: str | None, agent_name: str | None) -> Token[LlmCallContext]:
    """Attribute every provider call made from this task tree to one agent.

    Bind in the task that owns the agent's run loop, not in an SDK hook: the
    SDK awaits hooks through ``asyncio.gather``, whose child tasks copy the
    context and cannot write it back.
    """
    return _call_context.set(LlmCallContext(agent_id=agent_id, agent_name=agent_name))


def reset_call_context(token: Token[LlmCallContext]) -> None:
    _call_context.reset(token)


def set_retry_attempt(attempt: int) -> None:
    """Stamp subsequent calls with the turn-replay number (0 = first try)."""
    _call_context.set(replace(current_call_context(), retry_attempt=attempt))


def current_call_context() -> LlmCallContext:
    return _call_context.get(_DEFAULT_CALL_CONTEXT)


def register_sink(sink: LlmRequestSink) -> None:
    """Add a sink. Sinks must return quickly and never raise; a raising sink is logged and kept."""
    if sink not in _sinks:
        _sinks.append(sink)
    install()


def unregister_sink(sink: LlmRequestSink) -> None:
    with contextlib.suppress(ValueError):
        _sinks.remove(sink)


def emit(event: LlmRequestEvent) -> None:
    for sink in tuple(_sinks):
        try:
            sink(event)
        except Exception:
            logger.exception("LLM request log sink %r failed", sink)


# --------------------------------------------------------------------------- #
# Redaction                                                                    #
# --------------------------------------------------------------------------- #

# Credential shapes a provider or gateway error may echo back. Broad by design:
# a redacted-but-useless message costs nothing, a leaked key costs a rotation.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-(?:ant-|proj-|or-v1-)?[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(
        r"(?i)([\"']?(?:api[_-]?key|x-api-key|authorization|api[_-]?token|secret|password)"
        r"[\"']?\s*[:=]\s*[\"']?)(?:(?:bearer|basic)\s+)?[^\s\"',;&]{4,}"
    ),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)=)[^&\s\"']{4,}"),
)


def redact(text: str) -> str:
    """Replace anything credential-shaped in ``text``."""
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def clean_error_message(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    message = redact(message)
    if len(message) > ERROR_MESSAGE_MAX_CHARS:
        message = message[: ERROR_MESSAGE_MAX_CHARS - 1] + "…"
    return message


# --------------------------------------------------------------------------- #
# Provider request-id extraction                                               #
# --------------------------------------------------------------------------- #

# Provider-side request identifiers, in the order they are trusted. Anthropic
# and most gateways: ``request-id``; OpenAI/Azure: ``x-request-id``; Bedrock:
# ``x-amzn-requestid``; Vertex/Gemini: ``x-goog-request-id``. LiteLLM prefixes
# stored provider headers with ``llm_provider-``; both spellings are accepted.
_REQUEST_ID_HEADERS: tuple[str, ...] = (
    "request-id",
    "x-request-id",
    "x-amzn-requestid",
    "x-amz-request-id",
    "x-goog-request-id",
    "cf-ray",
)
_LITELLM_HEADER_PREFIX = "llm_provider-"

# Anthropic error bodies carry the id even when a gateway strips the header.
_BODY_REQUEST_ID = re.compile(r"[\"']?request_id[\"']?\s*[:=]\s*[\"']?(req_[A-Za-z0-9_-]{6,})")


def request_id_from_headers(headers: Mapping[str, Any] | None) -> str | None:
    if not headers:
        return None
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(value, str) or not value.strip():
            continue
        name = str(key).lower().removeprefix(_LITELLM_HEADER_PREFIX)
        normalized.setdefault(name, value.strip())
    for name in _REQUEST_ID_HEADERS:
        if name in normalized:
            return normalized[name]
    return None


def request_id_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = _BODY_REQUEST_ID.search(text)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# Free-form telemetry: response headers, details, raw bodies                   #
# --------------------------------------------------------------------------- #

FINISH_REASON_MAX_CHARS = 128

# Response headers are kept whole, except names that carry a credential. The
# list is a deny-list on purpose: a provider's new telemetry header should show
# up without a code change, a leaked credential must not.
_DENIED_HEADER_FRAGMENTS: tuple[str, ...] = (
    "auth",  # authorization, proxy-authorization, www-authenticate, x-auth-token
    "cookie",
    "api-key",
    "apikey",
    "secret",
    "password",
    "credential",
    "signature",
    "session",
)
HEADERS_MAX_COUNT = 64
HEADER_VALUE_MAX_CHARS = 512

# Keys of a request/response structure whose value is content: the prompt, the
# completion, tool schemas and arguments. ``details`` is metadata about the
# exchange, never the exchange itself; the bodies are a separate opt-in.
_CONTENT_KEYS: frozenset[str] = frozenset(
    {
        "messages",
        "message",
        "content",
        "text",
        "input",
        "output",
        "instructions",
        "system",
        "prompt",
        "completion",
        "choices",
        "delta",
        "tool_calls",
        "function_call",
        "arguments",
        "tools",
        "functions",
        "thinking_blocks",
        "reasoning_content",
        "citations",
        "image_url",
        "data",
        "b64_json",
        "embedding",
        "audio",
        "complete_input_dict",
        "original_response",
    }
)
# Keys whose value is or may hold a credential, whatever the provider calls it.
_SECRET_KEY_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "secret",
    "password",
    "credential",
    "cookie",
    "headers",  # request header blocks carry the auth header
    "access_token",
    "refresh_token",
    "id_token",
    "bearer",
)
DETAILS_MAX_BYTES = 16 * 1024
DETAILS_MAX_DEPTH = 6
DETAILS_MAX_ITEMS = 32
DETAILS_MAX_STRING = 256


def _normalize_header_name(key: object) -> str:
    return str(key).lower().removeprefix(_LITELLM_HEADER_PREFIX).replace("_", "-").strip()


def headers_from_response(headers: Mapping[str, Any] | None) -> dict[str, str] | None:
    """The provider's response headers, minus credential-bearing names, redacted and capped."""
    if not headers:
        return None
    picked: dict[str, str] = {}
    for key, value in headers.items():
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            continue
        text = str(value).strip()
        if not text:
            continue
        name = _normalize_header_name(key)
        if not name or any(fragment in name for fragment in _DENIED_HEADER_FRAGMENTS):
            continue
        picked.setdefault(name, redact(text)[:HEADER_VALUE_MAX_CHARS])
        if len(picked) >= HEADERS_MAX_COUNT:
            break
    return picked or None


def _strip_url_secrets(text: str) -> str:
    """Drop the query and fragment of anything URL-shaped; gateways key on them."""
    if "://" not in text:
        return text
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    return parts._replace(query="", fragment="").geturl()


def _sanitize_value(value: object, depth: int) -> object:  # noqa: PLR0911
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact(_strip_url_secrets(value))[:DETAILS_MAX_STRING]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if depth >= DETAILS_MAX_DEPTH:
        return "…"
    if isinstance(value, BaseModel):
        return _sanitize_value(value.model_dump(exclude_none=True), depth)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _sanitize_value(dataclasses.asdict(value), depth)
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for raw_key, item in cast("Mapping[object, object]", value).items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _CONTENT_KEYS or any(f in lowered for f in _SECRET_KEY_FRAGMENTS):
                continue
            cleaned = _sanitize_value(item, depth + 1)
            if cleaned is None or cleaned in ({}, []):
                continue
            out[key[:DETAILS_MAX_STRING]] = cleaned
            if len(out) >= DETAILS_MAX_ITEMS:
                break
        return out
    if isinstance(value, Sequence | set | frozenset):
        items = list(cast("Sequence[object]", value))[:DETAILS_MAX_ITEMS]
        return [_sanitize_value(item, depth + 1) for item in items]
    return redact(str(value))[:DETAILS_MAX_STRING]


def sanitize_details(value: object) -> dict[str, Any] | None:
    """A bounded, content-free, credential-free copy of ``value`` (a mapping), or None.

    Content keys and secret-shaped keys are dropped by name, strings are
    redacted and capped, nesting, item counts and total size are bounded. When
    the copy is still too large the biggest top-level entries go first and
    their names are listed under ``_dropped``.
    """
    cleaned = _sanitize_value(value, 0)
    if not isinstance(cleaned, dict):
        return None
    details = cast("dict[str, Any]", cleaned)
    details = {k: v for k, v in details.items() if v not in (None, {}, [])}
    if not details:
        return None
    size = json_size(details) or 0
    if size <= DETAILS_MAX_BYTES:
        return details
    dropped: list[str] = []
    by_size = sorted(details, key=lambda k: json_size(details[k]) or 0, reverse=True)
    for key in by_size:
        if size <= DETAILS_MAX_BYTES:
            break
        size -= json_size(details.pop(key)) or 0
        dropped.append(key)
    details["_dropped"] = dropped
    return details


def merge_details(*parts: tuple[str, object]) -> dict[str, Any] | None:
    """``{name: sanitize(value)}`` for every non-empty part, bounded as one object."""
    merged: dict[str, Any] = {}
    for name, value in parts:
        if value is None:
            continue
        merged[name] = value
    return sanitize_details(merged)


def json_text(value: object) -> str | None:
    """``value`` as compact JSON text; None when it cannot be serialized."""
    if value is None:
        return None
    try:
        if isinstance(value, BaseModel):
            return value.model_dump_json(exclude_none=True)
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return json.dumps(value, default=str, separators=(",", ":"), ensure_ascii=False)
    except Exception:  # noqa: BLE001 - telemetry is best-effort
        return None


def json_size(value: object) -> int | None:
    """Byte length of ``value`` serialized as compact JSON; None when it cannot be serialized."""
    text = json_text(value)
    return None if text is None else len(text.encode("utf-8"))


def api_host(url: str | None) -> str | None:
    """Hostname of an endpoint URL. Never the path or query, which may hold keys."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        host = urlsplit(url if "://" in url else f"//{url}").hostname
    except ValueError:
        return None
    return host or None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else None


def _to_datetime(value: object, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    return fallback


# --------------------------------------------------------------------------- #
# LiteLLM route                                                                #
# --------------------------------------------------------------------------- #


def _litellm_usage(response: object) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt = _int_or_none(getattr(usage, "prompt_tokens", None))
    completion = _int_or_none(getattr(usage, "completion_tokens", None))
    total = _int_or_none(getattr(usage, "total_tokens", None))
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _int_or_none(getattr(details, "cached_tokens", None)) if details else None
    if cached is None:
        cached = _int_or_none(getattr(usage, "cache_read_input_tokens", None))
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "cached_input_tokens": cached,
        "total_tokens": total,
    }


def _finish_reason(value: object) -> str | None:
    text = _str_or_none(value)
    return text[:FINISH_REASON_MAX_CHARS] if text else None


def _litellm_finish_reason(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    first: object = cast("list[object]", choices)[0]
    return _finish_reason(getattr(first, "finish_reason", None))


def _litellm_request_body(kwargs: Mapping[str, Any]) -> object:
    """The request as LiteLLM sent it, for its size only.

    The provider payload LiteLLM built (``additional_args.complete_input_dict``)
    when the adapter recorded it; otherwise model, messages and the provider
    params, which is the same information before translation.
    """
    additional = _mapping(kwargs.get("additional_args"))
    payload = _mapping(additional.get("complete_input_dict")) if additional else None
    if payload:
        return payload
    body: dict[str, Any] = {"model": kwargs.get("model")}
    messages = kwargs.get("messages")
    if messages is not None:
        body["messages"] = messages
    optional = _mapping(kwargs.get("optional_params"))
    if optional:
        body.update(optional)
    return body


def _count(value: object) -> int | None:
    return len(cast("Sequence[object]", value)) if isinstance(value, Sequence | Mapping) else None


def _litellm_request_details(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    optional = _mapping(kwargs.get("optional_params")) or {}
    request: dict[str, Any] = dict(optional)
    counts = {
        "message_count": _count(kwargs.get("messages")),
        "tool_count": _count(optional.get("tools") or optional.get("functions")),
    }
    request.update({k: v for k, v in counts.items() if v is not None})
    return request


def _exception_details(exc: BaseException | None) -> dict[str, Any] | None:
    """What the SDK exception says beyond its message: the error body and its typed fields."""
    if exc is None:
        return None
    error: dict[str, Any] = {}
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping | str):
        error["body"] = body if isinstance(body, Mapping) else _mapping(_parse_json(body)) or body
    for name in ("code", "param", "type", "llm_provider", "max_retries", "num_retries"):
        value = getattr(exc, name, None)
        if isinstance(value, str | int) and not isinstance(value, bool):
            error[name] = value
    return error or None


def _parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _exception_headers(exc: BaseException | None) -> Mapping[str, Any] | None:
    """Provider response headers LiteLLM keeps on a mapped exception.

    Where they live depends on the provider adapter: ``litellm_response_headers``
    for the HTTP-handler adapters (Anthropic), ``headers`` for the OpenAI SDK
    adapters, and the wrapped ``response`` otherwise. The first block that
    names a request id wins; failing that, the first non-empty block.
    """
    if exc is None:
        return None
    candidates = (
        getattr(exc, "litellm_response_headers", None),
        getattr(exc, "headers", None),
        getattr(getattr(exc, "response", None), "headers", None),
    )
    first: Mapping[str, Any] | None = None
    for raw in candidates:
        headers = _headers_mapping(raw)
        if not headers:
            continue
        if request_id_from_headers(headers):
            return headers
        first = first or headers
    return first


def _exception_body(exc: BaseException) -> object:
    """The response body an SDK exception carries, else its message text."""
    body = getattr(exc, "body", None)
    if body is not None:
        return body
    text = getattr(exc, "message", None)
    return text if isinstance(text, str) and text else str(exc)


def _exception_body_size(exc: BaseException) -> int | None:
    return json_size(_exception_body(exc))


def _headers_mapping(headers: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", headers) if isinstance(headers, Mapping) else None


def event_from_litellm(
    kwargs: Mapping[str, Any],
    response: object,
    start_time: object,
    end_time: object,
    *,
    outcome: Outcome,
) -> LlmRequestEvent:
    """Build the event from a LiteLLM success/failure callback payload."""
    now = datetime.now(UTC)
    started = _to_datetime(start_time, now)
    finished = _to_datetime(end_time, now)
    slo = _mapping(kwargs.get("standard_logging_object")) or {}
    hidden = _mapping(slo.get("hidden_params")) or {}
    litellm_params = _mapping(kwargs.get("litellm_params")) or {}
    error_info = _mapping(slo.get("error_information")) or {}
    exc = kwargs.get("exception")
    exc = exc if isinstance(exc, BaseException) else None

    provider = (
        _str_or_none(kwargs.get("custom_llm_provider"))
        or _str_or_none(slo.get("custom_llm_provider"))
        or _str_or_none(error_info.get("llm_provider"))
        or _str_or_none(litellm_params.get("custom_llm_provider"))
    )
    model = _str_or_none(kwargs.get("model")) or _str_or_none(slo.get("model")) or "unknown"
    host = api_host(
        _str_or_none(hidden.get("api_base"))
        or _str_or_none(slo.get("api_base"))
        or _str_or_none(litellm_params.get("api_base"))
    )
    streaming = bool(kwargs.get("stream")) or bool(slo.get("stream"))
    call_id = (
        _str_or_none(kwargs.get("litellm_call_id"))
        or _str_or_none(slo.get("litellm_call_id"))
        or str(uuid.uuid4())
    )

    response_headers = _mapping(hidden.get("additional_headers"))
    response_hidden = _mapping(getattr(response, "_hidden_params", None)) or {}
    hidden_headers = _mapping(response_hidden.get("additional_headers"))
    request_id = request_id_from_headers(response_headers) or request_id_from_headers(
        hidden_headers
    )
    headers = headers_from_response(response_headers) or headers_from_response(hidden_headers)

    ttft_ms: int | None = None
    if streaming:
        first = kwargs.get("completion_start_time")
        if first is not None:
            first_at = _to_datetime(first, started)
            ttft_ms = max(0, int((first_at - started).total_seconds() * 1000))

    context = current_call_context()
    base = LlmRequestEvent(
        call_id=call_id,
        route="litellm",
        provider=provider,
        model=model,
        api_host=host,
        streaming=streaming,
        outcome=outcome,
        status_code=None,
        provider_request_id=request_id,
        response_id=None,
        error_type=None,
        error_message=None,
        started_at=started,
        finished_at=finished,
        duration_ms=max(0, int((finished - started).total_seconds() * 1000)),
        agent_id=context.agent_id,
        agent_name=context.agent_name,
        retry_attempt=context.retry_attempt,
        request_bytes=json_size(_litellm_request_body(kwargs)),
        time_to_first_token_ms=ttft_ms,
        response_headers=headers,
    )
    if outcome == "success":
        return _litellm_success(base, kwargs, response, hidden, slo)
    return _litellm_failure(base, kwargs, exc, error_info, slo)


def _litellm_response_details(response: object) -> dict[str, Any] | None:
    """Everything LiteLLM's normalized response says about the call except the choices."""
    if isinstance(response, BaseModel):
        summary: dict[str, Any] = response.model_dump(exclude={"choices"}, exclude_none=True)
    else:
        summary = {}
        for name in ("id", "created", "model", "object", "system_fingerprint", "service_tier"):
            value = getattr(response, name, None)
            if value is not None:
                summary[name] = value
        usage = getattr(response, "usage", None)
        if usage is not None:
            summary["usage"] = usage
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first: object = cast("list[object]", choices)[0]
        summary["choice"] = {
            "finish_reason": getattr(first, "finish_reason", None),
            "provider_specific_fields": getattr(first, "provider_specific_fields", None),
        }
        summary["choice_count"] = len(cast("list[object]", choices))
    return summary or None


def _litellm_hidden_details(hidden: Mapping[str, Any], slo: Mapping[str, Any]) -> dict[str, Any]:
    """LiteLLM's own bookkeeping for the call (model id, cache hit, overhead, cost basis)."""
    litellm_info: dict[str, Any] = {
        k: v for k, v in hidden.items() if k not in {"additional_headers", "api_base"}
    }
    for name in ("cache_hit", "saved_cache_cost", "model_group", "model_id"):
        value = slo.get(name)
        if value is not None:
            litellm_info.setdefault(name, value)
    return litellm_info


def _litellm_success(
    base: LlmRequestEvent,
    kwargs: Mapping[str, Any],
    response: object,
    hidden: Mapping[str, Any],
    slo: Mapping[str, Any],
) -> LlmRequestEvent:
    usage = _litellm_usage(response)
    cost = _float_or_none(kwargs.get("response_cost"))
    if cost is None:
        cost = _float_or_none(hidden.get("response_cost"))
    return replace(
        base,
        status_code=200,
        response_id=_str_or_none(getattr(response, "id", None)),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=cost,
        response_bytes=json_size(response),
        finish_reason=_litellm_finish_reason(response),
        details=merge_details(
            ("request", _litellm_request_details(kwargs)),
            ("response", _litellm_response_details(response)),
            ("litellm", _litellm_hidden_details(hidden, slo)),
        ),
    )


def _litellm_failure(
    base: LlmRequestEvent,
    kwargs: Mapping[str, Any],
    exc: BaseException | None,
    error_info: Mapping[str, Any],
    slo: Mapping[str, Any],
) -> LlmRequestEvent:
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    request_id = base.provider_request_id
    headers = base.response_headers
    response_bytes: int | None = None
    if exc is not None:
        status_code = _int_or_none(getattr(exc, "status_code", None))
        error_type = type(exc).__name__
        error_message = clean_error_message(exc)
        exc_headers = _exception_headers(exc)
        request_id = request_id or request_id_from_headers(exc_headers)
        headers = headers or headers_from_response(exc_headers)
        response_bytes = _exception_body_size(exc)
    if status_code is None:
        code = error_info.get("error_code")
        status_code = _int_or_none(code)
        if status_code is None and isinstance(code, str) and code.isdigit():
            status_code = int(code)
    error_type = error_type or _str_or_none(error_info.get("error_class")) or "Exception"
    if error_message is None:
        raw = _str_or_none(error_info.get("error_message")) or _str_or_none(slo.get("error_str"))
        error_message = redact(raw)[:ERROR_MESSAGE_MAX_CHARS] if raw else error_type
    request_id = request_id or request_id_from_text(error_message)
    hidden = _mapping(slo.get("hidden_params")) or {}
    return replace(
        base,
        status_code=status_code,
        provider_request_id=request_id,
        error_type=error_type,
        error_message=error_message,
        response_bytes=response_bytes,
        response_headers=headers,
        details=merge_details(
            ("request", _litellm_request_details(kwargs)),
            ("error", _exception_details(exc) or dict(error_info) or None),
            ("litellm", _litellm_hidden_details(hidden, slo)),
        ),
    )


def _build_litellm_logger() -> Any:
    from litellm.integrations.custom_logger import CustomLogger

    class _StrixRequestLogger(CustomLogger):
        """Forwards each LiteLLM attempt to the registered sinks.

        Only the async handlers are implemented: the engine calls
        ``acompletion`` exclusively, and LiteLLM schedules these on the calling
        task, so the agent context bound there still applies. Sync fallbacks
        run in a thread pool and would lose it.
        """

        async def async_log_success_event(
            self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _dispatch(kwargs, response_obj, start_time, end_time, outcome="success")

        async def async_log_failure_event(
            self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _dispatch(kwargs, response_obj, start_time, end_time, outcome="error")

    return _StrixRequestLogger()


def _dispatch(
    kwargs: Mapping[str, Any], response: object, start: object, end: object, *, outcome: Outcome
) -> None:
    try:
        event = event_from_litellm(kwargs, response, start, end, outcome=outcome)
    except Exception:
        logger.exception("could not build LLM request event from LiteLLM callback")
        return
    emit(event)


_litellm_logger: Any | None = None


def install() -> None:
    """Attach the LiteLLM capture (idempotent) and the default log-line sink."""
    global _litellm_logger  # noqa: PLW0603
    if _log_line_sink not in _sinks:
        _sinks.insert(0, _log_line_sink)
    if _litellm_logger is not None:
        return
    import litellm

    capture = _build_litellm_logger()
    _litellm_logger = capture
    # litellm types this list with a bare Callable, which strict pyright cannot resolve.
    callbacks = litellm.callbacks  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if capture not in callbacks:
        callbacks.append(capture)  # pyright: ignore[reportUnknownMemberType]


def _log_line_sink(event: LlmRequestEvent) -> None:
    level = logging.DEBUG if event.outcome == "success" else logging.WARNING
    logger.log(
        level,
        "llm_request route=%s provider=%s model=%s host=%s outcome=%s status=%s "
        "request_id=%s response_id=%s stream=%s duration_ms=%d ttft_ms=%s "
        "req_bytes=%s res_bytes=%s finish=%s "
        "in=%s out=%s cached=%s cost=%s agent=%s attempt=%d%s",
        event.route,
        event.provider or "-",
        event.model,
        event.api_host or "-",
        event.outcome,
        event.status_code if event.status_code is not None else "-",
        event.provider_request_id or "-",
        event.response_id or "-",
        "1" if event.streaming else "0",
        event.duration_ms,
        event.time_to_first_token_ms if event.time_to_first_token_ms is not None else "-",
        event.request_bytes if event.request_bytes is not None else "-",
        event.response_bytes if event.response_bytes is not None else "-",
        event.finish_reason or "-",
        event.input_tokens if event.input_tokens is not None else "-",
        event.output_tokens if event.output_tokens is not None else "-",
        event.cached_input_tokens if event.cached_input_tokens is not None else "-",
        f"{event.cost_usd:.6f}" if event.cost_usd is not None else "-",
        event.agent_id or "-",
        event.retry_attempt,
        f" error={event.error_type}: {event.error_message}" if event.outcome == "error" else "",
    )


# --------------------------------------------------------------------------- #
# Native OpenAI route                                                          #
# --------------------------------------------------------------------------- #


def _openai_error_fields(exc: BaseException) -> tuple[int | None, str | None]:
    """(status_code, provider_request_id) from an OpenAI SDK exception."""
    status: int | None = None
    request_id: str | None = None
    if isinstance(exc, APIStatusError):
        status = _int_or_none(exc.status_code)
        request_id = _str_or_none(exc.request_id)
        if request_id is None:
            request_id = request_id_from_headers(_openai_error_headers(exc))
    if request_id is None:
        request_id = request_id_from_text(str(exc))
    return status, request_id


def _openai_error_headers(exc: BaseException) -> Mapping[str, Any] | None:
    if not isinstance(exc, APIStatusError):
        return None
    headers = exc.response.headers
    try:
        return dict(headers.items())
    except Exception:  # noqa: BLE001 - httpx.Headers, or a test double without items()
        return _headers_mapping(headers)


def _openai_usage(response: ModelResponse) -> dict[str, int | None]:
    usage = response.usage
    details = usage.input_tokens_details
    return {
        "input_tokens": _int_or_none(usage.input_tokens),
        "output_tokens": _int_or_none(usage.output_tokens),
        "cached_input_tokens": _int_or_none(details.cached_tokens) if details else None,
        "total_tokens": _int_or_none(usage.total_tokens),
    }


class RequestLoggingModel(Model):
    """Wrap a Model whose calls do not pass through LiteLLM.

    The native OpenAI Responses / Chat Completions routes (``openai/…``, the
    ChatGPT subscription backend) never hit ``litellm.acompletion``, so the
    LiteLLM capture does not see them. This wrapper records the same event
    around each ``get_response`` and ``stream_response``.
    """

    def __init__(self, inner: Model, *, model_name: str, provider: str, base_url: str | None):
        self._inner = inner
        self._model_name = model_name
        self._provider = provider
        self._host = api_host(base_url) or ("api.openai.com" if provider == "openai" else None)

    @property
    def model(self) -> str:
        return self._model_name

    async def close(self) -> None:
        await self._inner.close()

    def get_retry_advice(self, request: ModelRetryAdviceRequest) -> ModelRetryAdvice | None:
        return self._inner.get_retry_advice(request)

    def _event(
        self,
        *,
        started: datetime,
        started_mono: float,
        streaming: bool,
        response: ModelResponse | None,
        exc: BaseException | None,
        request: _OpenAiRequest,
        first_event_mono: float | None = None,
        finish_reason: str | None = None,
        raw_response: object = None,
    ) -> LlmRequestEvent:
        finished = datetime.now(UTC)
        duration_ms = max(0, int((time.monotonic() - started_mono) * 1000))
        ttft_ms = (
            max(0, int((first_event_mono - started_mono) * 1000))
            if streaming and first_event_mono is not None
            else None
        )
        context = current_call_context()
        base = LlmRequestEvent(
            call_id=str(uuid.uuid4()),
            route="openai",
            provider=self._provider,
            model=self._model_name,
            api_host=self._host,
            streaming=streaming,
            outcome="success",
            status_code=200,
            provider_request_id=None,
            response_id=None,
            error_type=None,
            error_message=None,
            started_at=started,
            finished_at=finished,
            duration_ms=duration_ms,
            agent_id=context.agent_id,
            agent_name=context.agent_name,
            retry_attempt=context.retry_attempt,
            request_bytes=request.size,
            time_to_first_token_ms=ttft_ms,
            finish_reason=finish_reason,
        )
        if exc is None:
            usage = _openai_usage(response) if response is not None else {}
            response_id = response.response_id if response is not None else None
            body_source = (
                raw_response
                if raw_response is not None
                else (response.output if response is not None else None)
            )
            return replace(
                base,
                response_id=response_id,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cached_input_tokens=usage.get("cached_input_tokens"),
                total_tokens=usage.get("total_tokens"),
                response_bytes=json_size(body_source),
                details=merge_details(
                    ("request", request.details),
                    ("response", _openai_response_details(response, raw_response)),
                ),
            )
        status, request_id = _openai_error_fields(exc)
        return replace(
            base,
            outcome="error",
            status_code=status,
            provider_request_id=request_id,
            error_type=type(exc).__name__,
            error_message=clean_error_message(exc),
            response_bytes=_exception_body_size(exc),
            response_headers=headers_from_response(_openai_error_headers(exc)),
            details=merge_details(
                ("request", request.details),
                ("error", _exception_details(exc)),
            ),
        )

    @staticmethod
    def _request(
        system_instructions: str | None,
        input: object,  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
    ) -> _OpenAiRequest:
        serialized_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params_json_schema,
            }
            if isinstance(tool, FunctionTool)
            else {"name": tool.name}
            for tool in tools
        ]
        body: dict[str, Any] = {
            "instructions": system_instructions,
            "input": input,
            "tools": serialized_tools,
        }
        settings = _model_settings_dict(model_settings)
        body.update({k: v for k, v in settings.items() if v is not None})
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        if conversation_id:
            body["conversation_id"] = conversation_id
        details: dict[str, Any] = dict(settings)
        details["input_items"] = _count(input) if not isinstance(input, str) else 1
        details["tool_count"] = len(tools)
        details["previous_response_id"] = previous_response_id
        details["conversation_id"] = conversation_id
        return _OpenAiRequest(size=json_size(body), details=details)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        started, started_mono = datetime.now(UTC), time.monotonic()
        request = self._request(
            system_instructions,
            input,
            model_settings,
            tools,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
        )
        try:
            response = await self._inner.get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            emit(
                self._event(
                    started=started,
                    started_mono=started_mono,
                    streaming=False,
                    response=None,
                    exc=exc,
                    request=request,
                )
            )
            raise
        emit(
            self._event(
                started=started,
                started_mono=started_mono,
                streaming=False,
                response=response,
                exc=None,
                request=request,
            )
        )
        return response

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        started, started_mono = datetime.now(UTC), time.monotonic()
        request = self._request(
            system_instructions,
            input,
            model_settings,
            tools,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
        )
        completed: ModelResponse | None = None
        raw_response: object = None
        first_event_mono: float | None = None
        finish_reason: str | None = None
        try:
            async for event in self._inner.stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            ):
                if first_event_mono is None:
                    first_event_mono = time.monotonic()
                if isinstance(event, ResponseCompletedEvent):
                    finish_reason = _openai_finish_reason(event)
                    raw_usage = event.response.usage
                    usage = Usage()
                    if raw_usage is not None:
                        usage = Usage(
                            requests=1,
                            input_tokens=raw_usage.input_tokens,
                            output_tokens=raw_usage.output_tokens,
                            total_tokens=raw_usage.total_tokens,
                            input_tokens_details=raw_usage.input_tokens_details,
                            output_tokens_details=raw_usage.output_tokens_details,
                        )
                    completed = ModelResponse(output=[], usage=usage, response_id=event.response.id)
                    raw_response = event.response
                yield event
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            emit(
                self._event(
                    started=started,
                    started_mono=started_mono,
                    streaming=True,
                    response=None,
                    exc=exc,
                    request=request,
                    first_event_mono=first_event_mono,
                )
            )
            raise
        emit(
            self._event(
                started=started,
                started_mono=started_mono,
                streaming=True,
                response=completed,
                exc=None,
                request=request,
                first_event_mono=first_event_mono,
                finish_reason=finish_reason,
                raw_response=raw_response,
            )
        )


@dataclass(frozen=True)
class _OpenAiRequest:
    """What the wrapper knows about one native request before it is sent."""

    size: int | None
    details: dict[str, Any]


def _model_settings_dict(model_settings: ModelSettings) -> dict[str, Any]:
    try:
        return dict(model_settings.to_json_dict())
    except Exception:  # noqa: BLE001 - settings are telemetry here, never required
        return {}


def _openai_response_details(
    response: ModelResponse | None, raw_response: object
) -> dict[str, Any] | None:
    """The Responses API object minus its output, or the SDK usage when that is all there is."""
    if isinstance(raw_response, BaseModel):
        return raw_response.model_dump(
            exclude={"output", "instructions", "tools", "text"}, exclude_none=True
        )
    if response is None:
        return None
    return {"usage": response.usage, "output_items": len(response.output)}


def _openai_finish_reason(event: ResponseCompletedEvent) -> str | None:
    """``completed`` / ``incomplete:<reason>`` from a Responses API terminal event."""
    status = _str_or_none(event.response.status)
    details = event.response.incomplete_details
    reason = details.reason if details is not None else None
    if status == "incomplete" and reason:
        return _finish_reason(f"incomplete:{reason}")
    return _finish_reason(status)


def failure_text(exc: BaseException) -> str:
    """The message stored as an agent's failure reason, with the provider request id appended.

    LiteLLM's Anthropic mapping keeps the error body (which carries
    ``request_id``) but drops the ``request-id`` header; OpenAI errors carry
    only the header. Either way the id lands in the text an operator reads.
    """
    text = redact(str(exc)) or type(exc).__name__
    request_id: str | None = None
    if isinstance(exc, APIError):
        _, request_id = _openai_error_fields(exc)
    request_id = request_id or request_id_from_headers(_exception_headers(exc))
    if request_id and request_id not in text:
        text = f"{text} [provider request id: {request_id}]"
    return text


__all__ = [
    "ERROR_MESSAGE_MAX_CHARS",
    "LlmCallContext",
    "LlmRequestEvent",
    "LlmRequestSink",
    "RequestLoggingModel",
    "api_host",
    "bind_call_context",
    "clean_error_message",
    "current_call_context",
    "emit",
    "event_from_litellm",
    "failure_text",
    "headers_from_response",
    "install",
    "json_size",
    "json_text",
    "merge_details",
    "redact",
    "register_sink",
    "request_id_from_headers",
    "request_id_from_text",
    "reset_call_context",
    "sanitize_details",
    "set_retry_attempt",
    "unregister_sink",
]
