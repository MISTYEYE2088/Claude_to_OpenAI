from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

DEFAULT_UPSTREAM_OPENAI_BASE_URL = "https://us.12888888.xyz:8317/v1"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    upstream_openai_base_url: str = DEFAULT_UPSTREAM_OPENAI_BASE_URL
    default_model: str | None = None
    host: str = "127.0.0.1"
    port: int = 8181


def _parse_port(raw_port: str | int) -> int:
    if isinstance(raw_port, bool):
        raise ValueError("port must be an integer between 1 and 65535")

    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError("port must be an integer between 1 and 65535") from exc

    if port < 1 or port > 65535:
        raise ValueError("port must be between 1 and 65535")

    return port


def _parse_default_model(raw_model: str | None) -> str | None:
    if raw_model is None:
        return None

    normalized = raw_model.strip()
    if not normalized:
        return None

    return normalized


def _parse_host(raw_host: str) -> str:
    if not isinstance(raw_host, str):
        raise ValueError("host must be a non-empty string")

    normalized = raw_host.strip()
    if not normalized:
        raise ValueError("host must be a non-empty string")

    return normalized


def _parse_upstream_openai_base_url(raw_url: str) -> str:
    normalized = raw_url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("upstream_openai_base_url must be an absolute HTTP(S) URL")
    return normalized


def _read_config_file(config_path: Path) -> dict[str, object]:
    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {config_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")

    return data


def resolve_default_config_path(
    *,
    is_frozen: bool,
    executable: str | None = None,
) -> Path:
    if is_frozen:
        exe_path = Path(executable or sys.executable)
        return exe_path.parent / "config" / "config.json"

    return Path.cwd() / "config" / "config.json"


def _resolve_is_frozen(is_frozen: bool | None) -> bool:
    if is_frozen is not None:
        return is_frozen
    return bool(getattr(sys, "frozen", False))


def load_settings(
    *,
    config_path: Path | None = None,
    is_frozen: bool | None = None,
    executable: str | None = None,
) -> Settings:
    load_dotenv()
    resolved_is_frozen = _resolve_is_frozen(is_frozen)

    if config_path is not None:
        resolved_config_path = config_path
        config_payload = _read_config_file(resolved_config_path)
    else:
        resolved_config_path = resolve_default_config_path(
            is_frozen=resolved_is_frozen,
            executable=executable,
        )
        config_payload = _read_config_file(resolved_config_path)

        if resolved_is_frozen and not config_payload:
            fallback_path = Path.cwd() / "config" / "config.json"
            if fallback_path != resolved_config_path:
                fallback_payload = _read_config_file(fallback_path)
                if fallback_payload:
                    config_payload = fallback_payload

    openai_api_key_env = config_payload.get("openai_api_key_env", "OPENAI_API_KEY")
    if not isinstance(openai_api_key_env, str) or not openai_api_key_env.strip():
        raise ValueError("openai_api_key_env must be a non-empty string")
    openai_api_key_env = openai_api_key_env.strip()

    upstream_openai_base_url = config_payload.get(
        "upstream_openai_base_url", DEFAULT_UPSTREAM_OPENAI_BASE_URL
    )
    if not isinstance(upstream_openai_base_url, str):
        raise ValueError("upstream_openai_base_url must be a string")

    default_model = config_payload.get("default_model", None)
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("default_model must be a string or null")

    host = config_payload.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")

    raw_port = config_payload.get("port", 8181)

    upstream_openai_base_url = os.getenv(
        "UPSTREAM_OPENAI_BASE_URL", upstream_openai_base_url
    )
    default_model = os.getenv("DEFAULT_MODEL", default_model)
    host = os.getenv("HOST", host)
    raw_port = os.getenv("PORT", raw_port)

    api_key = os.getenv(openai_api_key_env, "")
    if not api_key:
        raise ValueError(f"{openai_api_key_env} is required")

    return Settings(
        openai_api_key=api_key,
        upstream_openai_base_url=_parse_upstream_openai_base_url(
            upstream_openai_base_url
        ),
        default_model=_parse_default_model(default_model),
        host=_parse_host(host),
        port=_parse_port(raw_port),
    )
