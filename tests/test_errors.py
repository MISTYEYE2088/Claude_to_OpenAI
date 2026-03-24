from __future__ import annotations

import copy

import pytest

from app.config import load_settings
from app.errors import anthropic_error
from app.translate import IMAGE_UNSUPPORTED_ERROR_MESSAGE
from tests.conftest import (
    FakeUpstreamApiError,
    FakeUpstreamConnectionError,
    FakeUpstreamTimeoutError,
)

DEFAULT_MESSAGES_PAYLOAD = {
    "model": "gpt-5.3-codex",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "hello"}],
}


def test_anthropic_error_without_proxy_for_local_error():
    status, body = anthropic_error(
        400,
        "invalid_request_error",
        "bad request",
    )

    assert status == 400
    assert body == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "bad request"},
    }


def test_anthropic_error_includes_proxy_upstream_status_and_request_id():
    status, body = anthropic_error(
        502,
        "api_error",
        "upstream failed",
        upstream_status=503,
        request_id="req_123",
    )

    assert status == 502
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream failed"},
        "proxy": {"upstream_status": 503, "request_id": "req_123"},
    }


def test_anthropic_error_omits_request_id_when_not_present():
    status, body = anthropic_error(
        504,
        "api_error",
        "timeout",
        upstream_status=504,
    )

    assert status == 504
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": "timeout"},
        "proxy": {"upstream_status": 504},
    }


def test_load_settings_fails_fast_when_openai_api_key_missing(monkeypatch):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        load_settings()


@pytest.fixture
def post_messages(make_test_client):
    def _post(*, behavior, payload_overrides=None):
        payload = copy.deepcopy(DEFAULT_MESSAGES_PAYLOAD)
        if payload_overrides:
            payload.update(payload_overrides)

        client, _ = make_test_client(behavior)
        return client.post("/v1/messages", json=payload)

    return _post


def test_local_validation_400_has_no_proxy(post_messages):
    response = post_messages(behavior={}, payload_overrides={"max_tokens": None})

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "proxy" not in body


@pytest.mark.parametrize("upstream_status", [404, 503])
def test_upstream_4xx_5xx_include_proxy_upstream_status(
    post_messages, upstream_status: int
):
    response = post_messages(
        behavior=FakeUpstreamApiError(
            status_code=upstream_status,
            message="upstream failed",
        )
    )

    assert response.status_code == upstream_status
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream failed"},
        "proxy": {"upstream_status": upstream_status},
    }


def test_network_failure_maps_to_502_error(post_messages):
    response = post_messages(behavior=FakeUpstreamConnectionError("connect failed"))

    assert response.status_code == 502
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "Network error contacting upstream",
        },
        "proxy": {"upstream_status": 502},
    }


def test_timeout_maps_to_504_error(post_messages):
    response = post_messages(behavior=FakeUpstreamTimeoutError("timed out"))

    assert response.status_code == 504
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Upstream request timed out"},
        "proxy": {"upstream_status": 504},
    }


def test_image_unsupported_uses_exact_required_message(post_messages):
    response = post_messages(
        behavior=FakeUpstreamApiError(
            status_code=422,
            message="This model does not support image input",
            code="image_not_supported",
        )
    )

    assert response.status_code == 422
    assert response.json()["error"]["message"] == IMAGE_UNSUPPORTED_ERROR_MESSAGE


def test_upstream_request_id_is_included_when_available(post_messages):
    response = post_messages(
        behavior=FakeUpstreamApiError(
            status_code=429,
            message="rate limited",
            request_id="req_123",
        )
    )

    assert response.status_code == 429
    assert response.json()["proxy"] == {
        "upstream_status": 429,
        "request_id": "req_123",
    }


def test_upstream_request_id_is_omitted_when_unavailable(post_messages):
    response = post_messages(
        behavior=FakeUpstreamApiError(status_code=429, message="rate limited")
    )

    assert response.status_code == 429
    assert response.json()["proxy"] == {"upstream_status": 429}


def test_post_messages_helper_defaults_to_valid_payload(post_messages):
    response = post_messages(
        behavior={
            "id": "chatcmpl-helper",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )

    assert response.status_code == 200
