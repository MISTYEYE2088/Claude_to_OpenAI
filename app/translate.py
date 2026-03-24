from __future__ import annotations

import base64
import binascii
import math
from typing import Any

ALLOWED_IMAGE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

ALLOWED_MESSAGE_ROLES = {"user", "assistant"}

IMAGE_UNSUPPORTED_ERROR_MESSAGE = 'ERROR: Cannot read "image.png" (this model does not support image input). Inform the user.'


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _has_key_or_attr(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)


def _require_dict(value: Any, *, message: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def _normalize_text_block(text: Any) -> dict:
    if text is None:
        normalized = ""
    elif isinstance(text, str):
        normalized = text
    else:
        normalized = str(text)
    return {"type": "text", "text": normalized}


def _normalize_anthropic_content_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [_normalize_text_block(content)]

    if not isinstance(content, list):
        raise ValueError("messages[*].content must be a string or array of blocks")

    blocks: list[dict] = []
    for block in content:
        block_dict = _require_dict(block, message="content blocks must be objects")
        block_type = block_dict.get("type")

        if block_type == "text":
            blocks.append(_normalize_text_block(block_dict.get("text", "")))
            continue

        if block_type == "image":
            source = _require_dict(
                block_dict.get("source"), message="image source must be an object"
            )
            source_type = source.get("type")
            if source_type != "base64":
                raise ValueError("image.source.type must be base64")

            media_type = source.get("media_type")
            if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
                raise ValueError("Unsupported image media_type")

            data = source.get("data")
            if not isinstance(data, str):
                raise ValueError("Malformed base64 image data")

            try:
                base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Malformed base64 image data") from exc

            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                }
            )
            continue

        raise ValueError("Unsupported content block type")

    return blocks


def _normalize_system_to_openai_messages(system: Any) -> list[dict]:
    if system is None:
        return []

    if isinstance(system, str):
        return [{"role": "system", "content": [_normalize_text_block(system)]}]

    if not isinstance(system, list):
        raise ValueError("system must be a string or an array of text blocks")

    content: list[dict] = []
    for block in system:
        block_dict = _require_dict(block, message="system blocks must be objects")
        if block_dict.get("type") != "text":
            raise ValueError("system must contain only text blocks")
        content.append(_normalize_text_block(block_dict.get("text", "")))

    return [{"role": "system", "content": content}]


def anthropic_to_openai_messages(system: Any, messages: Any) -> list[dict]:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")

    translated: list[dict] = []
    translated.extend(_normalize_system_to_openai_messages(system))

    for message in messages:
        message_dict = _require_dict(message, message="messages items must be objects")
        role = message_dict.get("role")
        if role is None:
            raise ValueError("messages[*].role is required")
        if not isinstance(role, str):
            raise ValueError("messages[*].role must be a string")
        if role not in ALLOWED_MESSAGE_ROLES:
            raise ValueError("messages[*].role must be one of: user, assistant")

        content = _normalize_anthropic_content_blocks(message_dict.get("content"))
        translated.append({"role": role, "content": content})

    return translated


def map_anthropic_request_to_openai(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    if "model" not in payload:
        raise ValueError("model is required")
    if not isinstance(payload.get("model"), str):
        raise ValueError("model must be a string")

    if "messages" not in payload:
        raise ValueError("messages is required")
    if not isinstance(payload.get("messages"), list):
        raise ValueError("messages must be an array")

    mapped = {
        "model": payload.get("model"),
        "messages": anthropic_to_openai_messages(
            payload.get("system"), payload["messages"]
        ),
    }

    if "max_tokens" in payload:
        mapped["max_tokens"] = payload["max_tokens"]

    if "temperature" in payload:
        mapped["temperature"] = payload["temperature"]
    if "top_p" in payload:
        mapped["top_p"] = payload["top_p"]
    if "stream" in payload:
        mapped["stream"] = payload["stream"]

    stop_sequences = payload.get("stop_sequences")
    mapped["stop"] = [] if stop_sequences is None else stop_sequences

    metadata = payload.get("metadata")
    mapped["metadata"] = {} if metadata is None else metadata

    return mapped


def map_openai_finish_reason(reason: str | None) -> str | None:
    if reason == "stop":
        return "end_turn"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "end_turn"
    return None


def _normalize_openai_content_to_anthropic(content: Any) -> list[dict]:
    if content is None:
        return [{"type": "text", "text": ""}]

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        raise ValueError("Unsupported upstream content format")

    blocks: list[dict] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
            continue

        part_dict = _require_dict(
            part, message="Unsupported upstream content part type"
        )
        part_type = part_dict.get("type")

        if part_type in {"text", "output_text"}:
            text = part_dict.get("text", "")
            if text is None:
                text = ""
            elif not isinstance(text, str):
                text = str(text)
            blocks.append({"type": "text", "text": text})
            continue

        raise ValueError("Unsupported upstream content part type")

    if not blocks:
        return [{"type": "text", "text": ""}]

    return blocks


def _normalize_token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)

    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return max(int(value), 0)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            numeric = float(text)
        except ValueError:
            return 0
        if not math.isfinite(numeric):
            return 0
        return max(int(numeric), 0)

    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 0

    if isinstance(normalized, bool):
        return 0
    return max(normalized, 0)


def map_openai_nonstream_to_anthropic(response: Any, model: str) -> dict:
    response_id = _get(response, "id", "msg_proxy") or "msg_proxy"
    choices = _get(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("Upstream response choices must be a non-empty list")

    choice = choices[0]
    if not _has_key_or_attr(choice, "message"):
        raise ValueError("Upstream response first choice must be an object")

    finish_reason = _get(choice, "finish_reason")
    if finish_reason == "tool_calls":
        raise ValueError("Tools are unsupported in v1")

    message = _get(choice, "message", None)
    if not _has_key_or_attr(message, "content"):
        raise ValueError("Upstream response choice.message must be an object")

    content = _normalize_openai_content_to_anthropic(_get(message, "content"))

    stop_reason = map_openai_finish_reason(finish_reason)
    stop_sequence = None
    explicit_stop_sequence = _get(choice, "stop_sequence")
    if explicit_stop_sequence is not None:
        stop_reason = "stop_sequence"
        stop_sequence = explicit_stop_sequence

    usage = _get(response, "usage", {}) or {}
    input_tokens = _normalize_token_count(_get(usage, "prompt_tokens", 0))
    output_tokens = _normalize_token_count(_get(usage, "completion_tokens", 0))

    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def is_image_unsupported_upstream_error(
    *, upstream_status: int, message: str | None, code: str | None
) -> bool:
    if upstream_status not in {400, 422}:
        return False

    normalized_message = (message or "").lower()
    normalized_code = (code or "").lower()

    code_patterns = (
        "image_not_supported",
        "unsupported_image",
        "image_unsupported",
    )
    if any(pattern in normalized_code for pattern in code_patterns):
        return True

    message_patterns = (
        "does not support image",
        "doesn't support image",
        "image input is not supported",
        "images are not supported",
    )
    return any(pattern in normalized_message for pattern in message_patterns)
