from __future__ import annotations

import json
from typing import Any, AsyncIterator

from app.errors import anthropic_error
from app.translate import map_openai_finish_reason


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first_choice(chunk: Any) -> Any:
    choices = _get(chunk, "choices", None)
    if isinstance(choices, list) and choices:
        return choices[0]
    return None


def _extract_usage(chunk: Any) -> tuple[int | None, int | None]:
    usage = _get(chunk, "usage", None)
    if usage is None:
        choice = _first_choice(chunk)
        usage = _get(choice, "usage", None)

    if usage is None:
        return None, None

    input_tokens = _get(usage, "prompt_tokens", None)
    output_tokens = _get(usage, "completion_tokens", None)
    return input_tokens, output_tokens


def sse(event: str, data: dict) -> bytes:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


async def openai_stream_to_anthropic_events(
    stream: Any,
    model: str,
) -> AsyncIterator[bytes]:
    message_id = "msg_proxy"
    message_started = False
    content_started = False

    finish_reason: str | None = None
    stop_sequence: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    async def emit_message_start() -> AsyncIterator[bytes]:
        nonlocal message_started
        if message_started:
            return
        message_started = True
        yield sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            },
        )

    async def emit_content_start() -> AsyncIterator[bytes]:
        nonlocal content_started
        if content_started:
            return
        content_started = True
        yield sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

    async def emit_content_stop_if_needed() -> AsyncIterator[bytes]:
        nonlocal content_started
        if not content_started:
            return
        content_started = False
        yield sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )

    async def emit_terminal_events() -> AsyncIterator[bytes]:
        resolved_stop_reason = map_openai_finish_reason(finish_reason) or "end_turn"
        resolved_stop_sequence = stop_sequence if stop_sequence is not None else None
        resolved_input_tokens = input_tokens if isinstance(input_tokens, int) else 0
        resolved_output_tokens = output_tokens if isinstance(output_tokens, int) else 0

        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": resolved_stop_reason,
                    "stop_sequence": resolved_stop_sequence,
                },
                "usage": {
                    "input_tokens": resolved_input_tokens,
                    "output_tokens": resolved_output_tokens,
                },
            },
        )
        yield sse("message_stop", {"type": "message_stop"})

    async def emit_error_event(error_type: str, message: str) -> AsyncIterator[bytes]:
        async for event in emit_content_stop_if_needed():
            yield event
        yield sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
        )

    async def iterate_stream(source: Any) -> AsyncIterator[Any]:
        if hasattr(source, "__aiter__"):
            async for item in source:
                yield item
            return

        if hasattr(source, "__iter__"):
            for item in source:
                yield item
            return

        raise TypeError("Stream object is not iterable")

    try:
        async for chunk in iterate_stream(stream):
            if isinstance(chunk, str):
                if chunk.strip() == "[DONE]":
                    break
                continue

            maybe_id = _get(chunk, "id", None)
            if isinstance(maybe_id, str) and maybe_id:
                message_id = maybe_id

            async for event in emit_message_start():
                yield event
            async for event in emit_content_start():
                yield event

            choice = _first_choice(chunk)
            if choice is not None:
                maybe_finish_reason = _get(choice, "finish_reason", None)
                if isinstance(maybe_finish_reason, str) and maybe_finish_reason:
                    finish_reason = maybe_finish_reason
                    if finish_reason == "tool_calls":
                        async for event in emit_error_event(
                            "invalid_request_error",
                            "Tools are unsupported in v1",
                        ):
                            yield event
                        return

                maybe_stop_sequence = _get(choice, "stop_sequence", None)
                if isinstance(maybe_stop_sequence, str):
                    stop_sequence = maybe_stop_sequence

                delta = _get(choice, "delta", None)
                content_delta = _get(delta, "content", None)

                if isinstance(content_delta, str) and content_delta:
                    async for event in emit_content_start():
                        yield event
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "text_delta",
                                "text": content_delta,
                            },
                        },
                    )

            chunk_input_tokens, chunk_output_tokens = _extract_usage(chunk)
            if isinstance(chunk_input_tokens, int):
                input_tokens = chunk_input_tokens
            if isinstance(chunk_output_tokens, int):
                output_tokens = chunk_output_tokens

        if not message_started:
            async for event in emit_message_start():
                yield event
            async for event in emit_content_start():
                yield event

        async for event in emit_content_stop_if_needed():
            yield event
        async for event in emit_terminal_events():
            yield event
    except Exception as exc:
        _, body = anthropic_error(500, "api_error", str(exc) or "Streaming failed")
        async for event in emit_error_event(
            body["error"]["type"],
            body["error"]["message"],
        ):
            yield event
