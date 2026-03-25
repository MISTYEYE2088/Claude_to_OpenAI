from __future__ import annotations

import json
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_openai_client, get_settings
from app.translate import IMAGE_UNSUPPORTED_ERROR_MESSAGE
from tests.conftest import (
    FakeOpenAIClient,
    FakeUpstreamApiError,
    FakeUpstreamConnectionError,
    FakeUpstreamTimeoutError,
)


class FakeHeaders:
    def __init__(self, data: dict[str, str]):
        self._data = data

    def get(self, key: str, default=None):
        return self._data.get(key, default)


class AttrObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_nonstream_success_returns_anthropic_message_envelope(make_test_client):
    upstream_response = {
        "id": "chatcmpl-success-1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "hello from upstream"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "chatcmpl-success-1",
        "type": "message",
        "role": "assistant",
        "model": "gpt-5.3-codex",
        "content": [{"type": "text", "text": "hello from upstream"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 34},
    }
    assert len(fake_openai_client.calls) == 1
    assert fake_openai_client.calls[0]["model"] == "gpt-5.3-codex"


def test_nonstream_accepts_object_like_upstream_response(make_test_client):
    upstream_response = AttrObject(
        id="chatcmpl-object-like",
        choices=[
            AttrObject(
                finish_reason="stop",
                message=AttrObject(content="hello from object response"),
            )
        ],
        usage=AttrObject(prompt_tokens=7, completion_tokens=8),
    )
    client, _ = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-object-like"
    assert response.json()["content"] == [
        {"type": "text", "text": "hello from object response"}
    ]
    assert response.json()["usage"] == {"input_tokens": 7, "output_tokens": 8}


def test_nonstream_image_request_is_translated_for_upstream(make_test_client):
    upstream_response = {
        "id": "chatcmpl-image-1",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AA==",
                            },
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    translated_messages = fake_openai_client.calls[0]["messages"]
    assert translated_messages[0]["content"][0]["type"] == "image_url"
    assert (
        translated_messages[0]["content"][0]["image_url"]["url"]
        == "data:image/png;base64,AA=="
    )


def test_nonstream_usage_defaults_to_zero_when_upstream_usage_missing(make_test_client):
    upstream_response = {
        "id": "chatcmpl-usage-defaults",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": None,
    }
    client, _ = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_nonstream_usage_is_normalized_for_non_integer_values(make_test_client):
    upstream_response = {
        "id": "chatcmpl-usage-normalized",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": "13", "completion_tokens": True},
    }
    client, _ = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"] == {"input_tokens": 13, "output_tokens": 0}


def test_nonstream_usage_clamps_negative_values_to_zero(make_test_client):
    upstream_response = {
        "id": "chatcmpl-usage-negative-clamped",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": -9, "completion_tokens": "-4"},
    }
    client, _ = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_route_allows_missing_max_tokens(make_test_client):
    upstream_response = {
        "id": "chatcmpl-missing-max-tokens",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert len(fake_openai_client.calls) == 1
    assert "max_tokens" not in fake_openai_client.calls[0]


def test_route_force_model_overrides_incoming_model(settings):
    settings_with_forced_model = settings.__class__(
        openai_api_key=settings.openai_api_key,
        upstream_openai_base_url=settings.upstream_openai_base_url,
        default_model="configured-model",
        host=settings.host,
        port=settings.port,
        force_model=True,
        log_queries=settings.log_queries,
    )
    upstream_response = {
        "id": "chatcmpl-force-model-override",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    try:
        fake_openai_client = FakeOpenAIClient(upstream_response)
        app.dependency_overrides[get_settings] = lambda: settings_with_forced_model
        app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "incoming-model-should-be-ignored",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["model"] == "configured-model"
    assert fake_openai_client.calls[0]["model"] == "configured-model"


@pytest.mark.parametrize(
    "model_payload",
    [
        {},
        {"model": None},
        {"model": ""},
        {"model": "   "},
        {"model": 123},
    ],
)
def test_route_returns_canonical_400_for_all_nonforce_no_model_variants(
    make_test_client, model_payload
):
    upstream_response = {
        "id": "chatcmpl-nonforce-no-model",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            **model_payload,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "invalid_request_error",
        "message": "model is required (set payload.model or config default_model)",
    }
    assert fake_openai_client.calls == []


def test_route_uses_configured_default_model_when_caller_model_is_invalid(settings):
    settings_with_default_model = settings.__class__(
        openai_api_key=settings.openai_api_key,
        upstream_openai_base_url=settings.upstream_openai_base_url,
        default_model="gpt-from-default-model",
        host=settings.host,
        port=settings.port,
    )
    upstream_response = {
        "id": "chatcmpl-config-default-model",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }

    try:
        fake_openai_client = FakeOpenAIClient(upstream_response)
        app.dependency_overrides[get_settings] = lambda: settings_with_default_model
        app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-from-default-model"
    assert fake_openai_client.calls[0]["model"] == "gpt-from-default-model"


def test_route_rejects_missing_messages(make_test_client):
    client, _ = make_test_client({})

    response = client.post(
        "/v1/messages",
        json={"model": "gpt-5.3-codex", "max_tokens": 32},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "invalid_request_error",
        "message": "messages is required",
    }


def test_route_rejects_non_array_messages(make_test_client):
    client, _ = make_test_client({})

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "invalid_request_error",
        "message": "messages must be an array",
    }


def test_route_logs_full_payload_to_queries_jsonl_when_enabled(
    settings, tmp_path, monkeypatch
):
    settings_with_query_logging = settings.__class__(
        openai_api_key=settings.openai_api_key,
        upstream_openai_base_url=settings.upstream_openai_base_url,
        default_model=settings.default_model,
        host=settings.host,
        port=settings.port,
        force_model=settings.force_model,
        log_queries=True,
    )
    request_payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"n": 1, "text": "caf\u00e9"},
    }
    upstream_response = {
        "id": "chatcmpl-log-full-payload",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    monkeypatch.chdir(tmp_path)

    try:
        fake_openai_client = FakeOpenAIClient(upstream_response)
        app.dependency_overrides[get_settings] = lambda: settings_with_query_logging
        app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
        client = TestClient(app)
        response = client.post("/v1/messages", json=request_payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    log_path = tmp_path / "logs" / "queries.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["path"] == "/v1/messages"
    assert record["payload"] == request_payload
    assert isinstance(record["timestamp"], str)
    assert record["timestamp"].endswith("Z")
    assert "caf\\u00e9" in lines[0]


def test_route_logging_failure_does_not_fail_request(settings, monkeypatch, tmp_path):
    settings_with_query_logging = settings.__class__(
        openai_api_key=settings.openai_api_key,
        upstream_openai_base_url=settings.upstream_openai_base_url,
        default_model=settings.default_model,
        host=settings.host,
        port=settings.port,
        force_model=settings.force_model,
        log_queries=True,
    )
    upstream_response = {
        "id": "chatcmpl-log-failure",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    open_called = {"value": False}

    original_open = type(tmp_path).open

    def failing_open(path_obj, *args, **kwargs):
        if path_obj.name == "queries.jsonl":
            open_called["value"] = True
            raise OSError("simulated log write failure")
        return original_open(path_obj, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", failing_open)
    monkeypatch.chdir(tmp_path)

    try:
        fake_openai_client = FakeOpenAIClient(upstream_response)
        app.dependency_overrides[get_settings] = lambda: settings_with_query_logging
        app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "gpt-5.3-codex",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert open_called["value"] is True
    assert response.status_code == 200


def test_route_logs_before_field_validation(settings, tmp_path, monkeypatch):
    settings_with_query_logging = settings.__class__(
        openai_api_key=settings.openai_api_key,
        upstream_openai_base_url=settings.upstream_openai_base_url,
        default_model=settings.default_model,
        host=settings.host,
        port=settings.port,
        force_model=settings.force_model,
        log_queries=True,
    )
    request_payload = {"model": "gpt-5.3-codex"}

    monkeypatch.chdir(tmp_path)

    try:
        fake_openai_client = FakeOpenAIClient({})
        app.dependency_overrides[get_settings] = lambda: settings_with_query_logging
        app.dependency_overrides[get_openai_client] = lambda: fake_openai_client
        client = TestClient(app)
        response = client.post("/v1/messages", json=request_payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "messages is required"

    log_path = tmp_path / "logs" / "queries.jsonl"
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["path"] == "/v1/messages"
    assert record["payload"] == request_payload


def test_route_forwards_zero_max_tokens(make_test_client):
    upstream_response = {
        "id": "chatcmpl-zero-max-tokens",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake_openai_client.calls[0]["max_tokens"] == 0


def test_route_forwards_negative_max_tokens(make_test_client):
    upstream_response = {
        "id": "chatcmpl-negative-max-tokens",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": -1,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake_openai_client.calls[0]["max_tokens"] == -1


def test_route_forwards_non_integer_max_tokens(make_test_client):
    upstream_response = {
        "id": "chatcmpl-string-max-tokens",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": "32",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake_openai_client.calls[0]["max_tokens"] == "32"


def test_route_rejects_malformed_json_body(make_test_client):
    client, _ = make_test_client({})

    response = client.post(
        "/v1/messages",
        content='{"model":"gpt-5.3-codex", bad-json',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Request body must be valid JSON",
        },
    }


def test_local_validation_errors_omit_proxy_object(make_test_client):
    client, _ = make_test_client({})

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "system", "content": "not-allowed"}],
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "proxy" not in body


def test_stream_true_returns_event_stream_response(make_test_client):
    def stream_behavior(**_):
        async def gen():
            yield {"id": "chatcmpl-stream-from-nonstream-test", "choices": []}
            yield "[DONE]"

        return gen()

    client, _ = make_test_client(stream_behavior)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_delta" in response.text
    assert "event: message_stop" in response.text


def test_missing_stream_defaults_to_nonstream_behavior(make_test_client):
    upstream_response = {
        "id": "chatcmpl-missing-stream",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "stream" not in fake_openai_client.calls[0]


def test_wrong_type_stream_is_forwarded_unchanged_to_upstream(make_test_client):
    upstream_response = {
        "id": "chatcmpl-stream-wrong-type",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    client, fake_openai_client = make_test_client(upstream_response)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "stream": "yes",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake_openai_client.calls[0]["stream"] == "yes"


def test_upstream_api_error_includes_proxy_upstream_status(make_test_client):
    client, _ = make_test_client(
        FakeUpstreamApiError(
            status_code=429,
            message="rate limited",
            code="rate_limit_exceeded",
            request_id="req_up_1",
        )
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 429
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "rate limited"},
        "proxy": {"upstream_status": 429, "request_id": "req_up_1"},
    }


def test_upstream_api_error_proxy_omits_request_id_when_unavailable(make_test_client):
    client, _ = make_test_client(
        FakeUpstreamApiError(
            status_code=400,
            message="invalid temperature",
            code="invalid_request",
        )
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "temperature": "very-hot",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "invalid temperature"},
        "proxy": {"upstream_status": 400},
    }


def test_upstream_api_error_forwards_invalid_optional_tuning_values(make_test_client):
    client, fake_openai_client = make_test_client(
        FakeUpstreamApiError(
            status_code=400,
            message="top_p must be between 0 and 1",
            code="invalid_request",
            request_id="req_up_optional_1",
        )
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "temperature": "not-a-number",
            "top_p": -3,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert fake_openai_client.calls[0]["temperature"] == "not-a-number"
    assert fake_openai_client.calls[0]["top_p"] == -3
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "top_p must be between 0 and 1"},
        "proxy": {"upstream_status": 400, "request_id": "req_up_optional_1"},
    }


def test_network_failure_maps_to_502(make_test_client):
    client, _ = make_test_client(FakeUpstreamConnectionError("connect failed"))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "Network error contacting upstream",
        },
        "proxy": {"upstream_status": 502},
    }


def test_timeout_maps_to_504(make_test_client):
    client, _ = make_test_client(FakeUpstreamTimeoutError("timed out"))

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Upstream request timed out"},
        "proxy": {"upstream_status": 504},
    }


def test_image_unsupported_override_for_known_upstream_error_patterns(make_test_client):
    client, _ = make_test_client(
        FakeUpstreamApiError(
            status_code=422,
            message="This model does not support image input",
            code="image_not_supported",
        )
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": IMAGE_UNSUPPORTED_ERROR_MESSAGE},
        "proxy": {"upstream_status": 422},
    }


def test_unknown_upstream_exception_not_mapped_to_network_error(settings):
    class BoomError(Exception):
        pass

    try:
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_openai_client] = lambda: type(
            "BrokenClient",
            (),
            {
                "chat": type(
                    "BrokenChat",
                    (),
                    {
                        "completions": type(
                            "BrokenCompletions",
                            (),
                            {
                                "create": staticmethod(
                                    lambda **_: (_ for _ in ()).throw(BoomError("boom"))
                                )
                            },
                        )()
                    },
                )()
            },
        )()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "gpt-5.3-codex",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.text == "Internal Server Error"


def test_httpx_transport_error_with_network_cause_maps_to_502(make_test_client):
    transport_error = httpx.TransportError("transport failed")
    transport_error.__cause__ = OSError("network unreachable")
    client, _ = make_test_client(transport_error)

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Network error contacting upstream"


def test_extracts_request_id_from_mapping_like_headers(make_test_client):
    class UpstreamResponse:
        status_code = 429
        headers = FakeHeaders({"x-request-id": "req_from_headers_obj"})

        @staticmethod
        def json():
            return {"error": {"message": "rate limited"}}

    class UpstreamError(Exception):
        def __init__(self):
            super().__init__("rate limited")
            self.status_code = 429
            self.response = UpstreamResponse()

    client, _ = make_test_client(UpstreamError())

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 429
    assert response.json()["proxy"] == {
        "upstream_status": 429,
        "request_id": "req_from_headers_obj",
    }


def test_non_string_request_id_is_treated_as_unavailable(make_test_client):
    client, _ = make_test_client(
        FakeUpstreamApiError(
            status_code=429,
            message="rate limited",
            code="rate_limit_exceeded",
            request_id=12345,
        )
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 429
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "rate limited"},
        "proxy": {"upstream_status": 429},
    }
