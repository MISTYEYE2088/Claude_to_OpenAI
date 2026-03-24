from __future__ import annotations

import pytest

from tests.conftest import FakeUpstreamApiError
from tests.conftest import FakeUpstreamConnectionError, FakeUpstreamTimeoutError


def test_models_success_maps_fields_and_fallbacks(make_test_client):
    upstream_models = {
        "data": [
            {
                "id": "gpt-5.3-codex",
                "name": "GPT 5.3 Codex",
                "created": 1735689600,
            },
            {
                "id": "gpt-4.1-mini",
                "display_name": "GPT-4.1 Mini",
                "created_at": "2025-02-01T03:04:05Z",
            },
            {
                "id": "gpt-fallbacks-only",
            },
        ]
    }
    client, fake_openai_client = make_test_client(
        {},
        models_behavior=upstream_models,
    )

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "id": "gpt-5.3-codex",
                "type": "model",
                "display_name": "GPT 5.3 Codex",
                "created_at": "2025-01-01T00:00:00Z",
            },
            {
                "id": "gpt-4.1-mini",
                "type": "model",
                "display_name": "GPT-4.1 Mini",
                "created_at": "2025-02-01T03:04:05Z",
            },
            {
                "id": "gpt-fallbacks-only",
                "type": "model",
                "display_name": "gpt-fallbacks-only",
                "created_at": "1970-01-01T00:00:00Z",
            },
        ]
    }
    assert fake_openai_client.model_calls == [{}]


def test_models_upstream_api_error_maps_status_and_proxy_metadata(make_test_client):
    client, _ = make_test_client(
        {},
        models_behavior=FakeUpstreamApiError(
            status_code=503,
            message="upstream unavailable",
            request_id="req_models_1",
        ),
    )

    response = client.get("/v1/models")

    assert response.status_code == 503
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream unavailable"},
        "proxy": {"upstream_status": 503, "request_id": "req_models_1"},
    }


def test_models_timeout_error_maps_to_504(make_test_client):
    client, _ = make_test_client(
        {},
        models_behavior=FakeUpstreamTimeoutError("timed out"),
    )

    response = client.get("/v1/models")

    assert response.status_code == 504
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Upstream request timed out"},
        "proxy": {"upstream_status": 504},
    }


def test_models_network_error_maps_to_502(make_test_client):
    client, _ = make_test_client(
        {},
        models_behavior=FakeUpstreamConnectionError("connect failed"),
    )

    response = client.get("/v1/models")

    assert response.status_code == 502
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "Network error contacting upstream",
        },
        "proxy": {"upstream_status": 502},
    }


def test_models_extracts_error_message_and_request_id_from_response_object(
    make_test_client,
):
    class UpstreamResponse:
        status_code = 429
        headers = {"x-request-id": "req_models_from_headers"}

        @staticmethod
        def json():
            return {"error": {"message": "rate limited from response"}}

    class UpstreamError(Exception):
        def __init__(self):
            super().__init__("")
            self.response = UpstreamResponse()

    client, _ = make_test_client({}, models_behavior=UpstreamError())

    response = client.get("/v1/models")

    assert response.status_code == 429
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "rate limited from response"},
        "proxy": {"upstream_status": 429, "request_id": "req_models_from_headers"},
    }


@pytest.mark.parametrize("bad_status", ["503", 99, 600])
def test_models_invalid_status_does_not_override_timeout_or_network_classification(
    make_test_client,
    bad_status,
):
    class UpstreamError(Exception):
        def __init__(self):
            super().__init__("connect failed")
            self.status_code = bad_status

    client, _ = make_test_client({}, models_behavior=UpstreamError())

    response = client.get("/v1/models")

    assert response.status_code == 502
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "Network error contacting upstream",
        },
        "proxy": {"upstream_status": 502},
    }
