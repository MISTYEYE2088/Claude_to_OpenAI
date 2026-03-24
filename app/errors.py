from __future__ import annotations


def anthropic_error(
    status: int,
    err_type: str,
    message: str,
    upstream_status: int | None = None,
    request_id: str | None = None,
) -> tuple[int, dict]:
    body: dict = {"type": "error", "error": {"type": err_type, "message": message}}
    if upstream_status is not None:
        proxy: dict = {"upstream_status": upstream_status}
        if request_id:
            proxy["request_id"] = request_id
        body["proxy"] = proxy
    return status, body


def local_validation_error(message: str) -> tuple[int, dict]:
    return anthropic_error(400, "invalid_request_error", message)


def upstream_api_error(
    message: str,
    *,
    upstream_status: int,
    request_id: str | None = None,
) -> tuple[int, dict]:
    return anthropic_error(
        upstream_status,
        "api_error",
        message,
        upstream_status=upstream_status,
        request_id=request_id,
    )


def network_error(
    message: str = "Network error contacting upstream",
) -> tuple[int, dict]:
    return anthropic_error(502, "api_error", message, upstream_status=502)


def timeout_error(message: str = "Upstream request timed out") -> tuple[int, dict]:
    return anthropic_error(504, "api_error", message, upstream_status=504)
