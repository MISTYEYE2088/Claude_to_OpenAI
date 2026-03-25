from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.client import build_openai_client
from app.config import Settings, load_settings
from app.errors import (
    anthropic_error,
    local_validation_error,
    network_error,
    timeout_error,
    upstream_api_error,
)
from app.translate import (
    IMAGE_UNSUPPORTED_ERROR_MESSAGE,
    is_image_unsupported_upstream_error,
    map_anthropic_request_to_openai,
    map_openai_nonstream_to_anthropic,
)
from app.streaming import openai_stream_to_anthropic_events

app = FastAPI(title="OpenAI to Claude Proxy")


@app.on_event("startup")
async def _validate_settings_on_startup() -> None:
    get_settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def get_openai_client(settings: Settings = Depends(get_settings)) -> Any:
    return build_openai_client(
        api_key=settings.openai_api_key,
        base_url=settings.upstream_openai_base_url,
    )


def _json_error_response(status: int, body: dict) -> JSONResponse:
    return JSONResponse(status_code=status, content=body)


def _extract_upstream_error_info(
    exc: Exception,
) -> tuple[int | None, str, str | None, str | None]:
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)

    message = getattr(exc, "message", None)
    code = getattr(exc, "code", None)

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if status is None and response_status is not None:
            status = response_status
        if request_id is None:
            headers = getattr(response, "headers", None)
            if hasattr(headers, "get"):
                request_id = headers.get("x-request-id")

        body = None
        body_method = getattr(response, "json", None)
        if callable(body_method):
            try:
                body = body_method()
            except Exception:
                body = None

        if isinstance(body, dict):
            error_obj = body.get("error")
            if isinstance(error_obj, dict):
                if message is None:
                    maybe_message = error_obj.get("message")
                    if isinstance(maybe_message, str) and maybe_message:
                        message = maybe_message
                if code is None:
                    maybe_code = error_obj.get("code")
                    if isinstance(maybe_code, str) and maybe_code:
                        code = maybe_code

    if not isinstance(message, str) or not message:
        message = str(exc) or "Upstream API request failed"

    if not isinstance(code, str):
        code = None

    if not isinstance(request_id, str) or not request_id:
        request_id = None

    if (
        not isinstance(status, int)
        or isinstance(status, bool)
        or not (100 <= status <= 599)
    ):
        status = None

    return status, message, code, request_id


def _map_upstream_exception_to_error_response(
    exc: Exception,
    *,
    apply_image_unsupported_override: bool,
) -> tuple[int, dict]:
    status, message, code, request_id = _extract_upstream_error_info(exc)

    if status is not None:
        if apply_image_unsupported_override and is_image_unsupported_upstream_error(
            upstream_status=status,
            message=message,
            code=code,
        ):
            message = IMAGE_UNSUPPORTED_ERROR_MESSAGE

        return upstream_api_error(
            message,
            upstream_status=status,
            request_id=request_id,
        )

    if _is_timeout_error(exc):
        return timeout_error()

    if _is_network_error(exc):
        return network_error()

    raise exc


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    return isinstance(exc, TimeoutError)


def _is_network_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True

    fragments = ("connect", "network", "transport", "connection")

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}".lower()
        if any(fragment in text for fragment in fragments):
            return True
        current = current.__cause__ or current.__context__

    return False


def _validate_request_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload must be an object"
    if "messages" not in payload:
        return "messages is required"
    if not isinstance(payload.get("messages"), list):
        return "messages must be an array"
    return None


def _resolve_model(payload: dict[str, Any], settings: Settings) -> str | None:
    if settings.force_model:
        return settings.default_model

    caller_model = payload.get("model")
    if isinstance(caller_model, str) and caller_model.strip():
        return caller_model

    if settings.default_model:
        return settings.default_model

    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        attrs = vars(value)
        if isinstance(attrs, dict):
            return attrs
    return {}


def _to_rfc3339(value: Any) -> str:
    fallback = "1970-01-01T00:00:00Z"

    if isinstance(value, bool) or value is None:
        return fallback

    if isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except Exception:
            return fallback

    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback

        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            return fallback

    return fallback


def _map_models_response(upstream_response: Any) -> dict[str, list[dict[str, str]]]:
    payload = _as_mapping(upstream_response)
    raw_data = payload.get("data") if isinstance(payload, dict) else None
    models = raw_data if isinstance(raw_data, list) else []

    mapped: list[dict[str, str]] = []
    for model in models:
        model_obj = _as_mapping(model)
        model_id_raw = model_obj.get("id")
        if not isinstance(model_id_raw, str) or not model_id_raw:
            continue

        display_name = model_obj.get("display_name") or model_obj.get("name")
        if not isinstance(display_name, str) or not display_name:
            display_name = model_id_raw

        created_value = model_obj.get("created_at")
        if created_value is None:
            created_value = model_obj.get("created")

        mapped.append(
            {
                "id": model_id_raw,
                "type": "model",
                "display_name": display_name,
                "created_at": _to_rfc3339(created_value),
            }
        )

    return {"data": mapped}


def _append_query_log(payload: dict[str, Any], path: str = "/v1/messages") -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "path": path,
        "payload": payload,
    }
    log_path = Path.cwd() / "logs" / "queries.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _fetch_and_map_models(client: Any) -> tuple[int, dict]:
    try:
        upstream_response = client.models.list()
    except Exception as exc:
        return _map_upstream_exception_to_error_response(
            exc,
            apply_image_unsupported_override=False,
        )

    return 200, _map_models_response(upstream_response)


@app.get("/v1/models")
async def list_models(client: Any = Depends(get_openai_client)) -> JSONResponse:
    status, body = _fetch_and_map_models(client)
    return _json_error_response(status, body)


@app.get("/ready")
async def readiness(client: Any = Depends(get_openai_client)) -> JSONResponse:
    status, body = _fetch_and_map_models(client)
    if status != 200:
        return _json_error_response(status, body)
    return _json_error_response(200, {"status": "ready"})


@app.post("/v1/messages")
async def messages(
    request: Request,
    settings: Settings = Depends(get_settings),
    client: Any = Depends(get_openai_client),
) -> Any:
    try:
        payload = await request.json()
    except Exception:
        status, body = local_validation_error("Request body must be valid JSON")
        return _json_error_response(status, body)

    if settings.log_queries and isinstance(payload, dict):
        try:
            _append_query_log(payload)
        except Exception:
            pass

    validation_error = _validate_request_payload(payload)
    if validation_error is not None:
        status, body = local_validation_error(validation_error)
        return _json_error_response(status, body)

    resolved_model = _resolve_model(payload, settings)
    if resolved_model is None:
        status, body = local_validation_error(
            "model is required (set payload.model or config default_model)"
        )
        return _json_error_response(status, body)

    normalized_payload = dict(payload)
    normalized_payload["model"] = resolved_model

    try:
        upstream_request = map_anthropic_request_to_openai(normalized_payload)
    except ValueError as exc:
        status, body = local_validation_error(str(exc))
        return _json_error_response(status, body)

    if normalized_payload.get("stream") is True:
        try:
            upstream_stream = client.chat.completions.create(**upstream_request)
        except Exception as exc:
            status_code, body = _map_upstream_exception_to_error_response(
                exc,
                apply_image_unsupported_override=True,
            )
            return _json_error_response(status_code, body)

        return StreamingResponse(
            openai_stream_to_anthropic_events(upstream_stream, resolved_model),
            media_type="text/event-stream",
        )

    try:
        upstream_response = client.chat.completions.create(**upstream_request)
    except Exception as exc:
        status_code, body = _map_upstream_exception_to_error_response(
            exc,
            apply_image_unsupported_override=True,
        )
        return _json_error_response(status_code, body)

    try:
        response_body = map_openai_nonstream_to_anthropic(
            upstream_response,
            resolved_model,
        )
    except ValueError as exc:
        status, body = local_validation_error(str(exc))
        return _json_error_response(status, body)

    return _json_error_response(200, response_body)
