from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def _make_stream_behavior(items: Iterable[Any]):
    values = list(items)

    def _behavior(**_: Any):
        async def _gen():
            for value in values:
                if isinstance(value, Exception):
                    raise value
                yield value

        return _gen()

    return _behavior


def _make_sync_stream_behavior(items: Iterable[Any]):
    values = list(items)

    def _behavior(**_: Any):
        return iter(values)

    return _behavior


def _read_sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    chunks = [chunk for chunk in body.split("\n\n") if chunk.strip()]

    for chunk in chunks:
        event_name: str | None = None
        data_value: dict | None = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_value = json.loads(line[len("data: ") :])
        if event_name is not None and data_value is not None:
            events.append((event_name, data_value))

    return events


def test_stream_event_order_and_payload_shapes(make_test_client):
    stream_items = [
        {"id": "chatcmpl-stream-1", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": ""}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
        "[DONE]",
    ]
    client, _ = make_test_client(_make_stream_behavior(stream_items))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _read_sse_events(response.text)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    message_start = events[0][1]
    assert message_start["type"] == "message_start"
    assert message_start["message"]["id"] == "chatcmpl-stream-1"
    assert message_start["message"]["type"] == "message"
    assert message_start["message"]["role"] == "assistant"
    assert message_start["message"]["model"] == "gpt-5.3-codex"
    assert message_start["message"]["content"] == []

    block_start = events[1][1]
    assert block_start == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }

    first_delta = events[2][1]
    second_delta = events[3][1]
    assert first_delta == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hel"},
    }
    assert second_delta == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "lo"},
    }

    message_delta = events[5][1]
    assert message_delta == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    assert events[6][1] == {"type": "message_stop"}


def test_stream_fallback_defaults_when_terminal_metadata_missing(make_test_client):
    stream_items = [
        {
            "id": "chatcmpl-stream-defaults",
            "choices": [{"delta": {"role": "assistant"}}],
        },
        {"choices": [{"delta": {"content": "done"}}]},
        "[DONE]",
    ]
    client, _ = make_test_client(_make_stream_behavior(stream_items))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    events = _read_sse_events(response.text)

    message_delta = events[-2][1]
    assert message_delta["delta"] == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }
    assert message_delta["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert events[-1] == ("message_stop", {"type": "message_stop"})


def test_stream_error_emits_anthropic_error_event(make_test_client):
    stream_items = [
        {"id": "chatcmpl-stream-error", "choices": [{"delta": {"content": "x"}}]},
        RuntimeError("stream exploded"),
    ]
    client, _ = make_test_client(_make_stream_behavior(stream_items))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    events = _read_sse_events(response.text)
    names = [name for name, _ in events]
    assert "content_block_stop" in names
    assert names.index("content_block_stop") < names.index("error")
    assert events[-1][0] == "error"
    assert events[-1][1]["type"] == "error"
    assert events[-1][1]["error"]["type"] == "api_error"
    assert "stream exploded" in events[-1][1]["error"]["message"]


def test_stream_tool_calls_finish_emits_invalid_request_error_and_no_terminal_message_events(
    make_test_client,
):
    stream_items = [
        {
            "id": "chatcmpl-stream-toolcalls",
            "choices": [{"delta": {"role": "assistant"}}],
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    client, _ = make_test_client(_make_stream_behavior(stream_items))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    events = _read_sse_events(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "error"
    assert "content_block_stop" in names
    assert names.index("content_block_stop") < names.index("error")
    assert "message_delta" not in names
    assert "message_stop" not in names
    assert events[-1][1] == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Tools are unsupported in v1",
        },
    }


def test_stream_no_text_delta_still_emits_content_block_and_normal_terminal_order(
    make_test_client,
):
    stream_items = [
        {
            "id": "chatcmpl-stream-no-text",
            "choices": [{"delta": {"role": "assistant"}}],
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0},
        },
        "[DONE]",
    ]
    client, _ = make_test_client(_make_stream_behavior(stream_items))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    events = _read_sse_events(response.text)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[3][1] == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": 3, "output_tokens": 0},
    }


def test_stream_accepts_sync_iterator_streams(make_test_client):
    stream_items = [
        {"id": "chatcmpl-sync-stream", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "sync-ok"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
        "[DONE]",
    ]
    client, _ = make_test_client(_make_sync_stream_behavior(stream_items))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    events = _read_sse_events(response.text)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"]["text"] == "sync-ok"
