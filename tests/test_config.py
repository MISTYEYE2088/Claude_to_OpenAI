from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import (
    DEFAULT_UPSTREAM_OPENAI_BASE_URL,
    load_settings,
    resolve_default_config_path,
)


def _write_config(path: Path, **overrides: object) -> None:
    payload = {
        "openai_api_key_env": "OPENAI_API_KEY",
        "upstream_openai_base_url": "https://example.invalid/v1",
        "default_model": "gpt-5.3-codex",
        "host": "127.0.0.1",
        "port": 8181,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_default_config_path_source_mode_uses_cwd(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    assert (
        resolve_default_config_path(is_frozen=False)
        == tmp_path / "config" / "config.json"
    )


def test_resolve_default_config_path_frozen_mode_uses_executable_dir(tmp_path: Path):
    executable = tmp_path / "bin" / "proxy.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")

    assert resolve_default_config_path(is_frozen=True, executable=str(executable)) == (
        executable.parent / "config" / "config.json"
    )


def test_load_settings_frozen_mode_falls_back_to_cwd_config_when_exe_config_missing(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cwd = tmp_path / "cwd"
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    _write_config(cwd / "config" / "config.json")

    settings = load_settings(
        is_frozen=True, executable=str(tmp_path / "bin" / "proxy.exe")
    )

    assert settings.openai_api_key == "test-key"
    assert settings.upstream_openai_base_url == "https://example.invalid/v1"


def test_load_settings_explicit_config_path_wins(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cwd = tmp_path / "cwd"
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    _write_config(tmp_path / "cwd" / "config" / "config.json", host="0.0.0.0")
    explicit_path = tmp_path / "custom" / "settings.json"
    _write_config(explicit_path, host="127.0.0.2")

    settings = load_settings(
        config_path=explicit_path,
        is_frozen=True,
        executable=str(tmp_path / "bin" / "proxy.exe"),
    )

    assert settings.host == "127.0.0.2"


def test_load_settings_env_overrides_non_secret_fields(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("UPSTREAM_OPENAI_BASE_URL", "https://override.invalid/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9191")
    _write_config(tmp_path / "config" / "config.json")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(is_frozen=False)

    assert settings.openai_api_key == "test-key"
    assert settings.upstream_openai_base_url == "https://override.invalid/v1"
    assert settings.default_model == "gpt-4o-mini"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9191


@pytest.mark.parametrize("host_override", ["", "   "])
def test_load_settings_rejects_empty_or_whitespace_host_env_override(
    monkeypatch,
    tmp_path: Path,
    host_override: str,
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HOST", host_override)
    _write_config(tmp_path / "config" / "config.json", host="127.0.0.1")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="host"):
        load_settings(is_frozen=False)


def test_load_settings_reads_secret_from_openai_api_key_env_only(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key")
    monkeypatch.setenv("APP_KEY", "correct-key")
    _write_config(tmp_path / "config" / "config.json", openai_api_key_env="APP_KEY")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(is_frozen=False)

    assert settings.openai_api_key == "correct-key"


def test_load_settings_validates_upstream_base_url_is_absolute_http_url(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_config(tmp_path / "config" / "config.json", upstream_openai_base_url="/v1")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="upstream_openai_base_url"):
        load_settings(is_frozen=False)


def test_load_settings_empty_default_model_normalizes_to_none(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_config(tmp_path / "config" / "config.json", default_model="   ")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(is_frozen=False)

    assert settings.default_model is None


def test_load_settings_ignores_unknown_json_keys(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_config(tmp_path / "config" / "config.json", extra="ignored")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(is_frozen=False)

    assert settings.upstream_openai_base_url == "https://example.invalid/v1"


@pytest.mark.parametrize("value", [None, ""])
def test_load_settings_fails_when_resolved_api_key_is_missing_or_empty(
    monkeypatch,
    tmp_path: Path,
    value: str | None,
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    _write_config(tmp_path / "config" / "config.json", openai_api_key_env="APP_KEY")
    monkeypatch.chdir(tmp_path)

    if value is None:
        monkeypatch.delenv("APP_KEY", raising=False)
    else:
        monkeypatch.setenv("APP_KEY", value)

    with pytest.raises(ValueError, match="APP_KEY"):
        load_settings(is_frozen=False)


def test_load_settings_fails_clearly_for_invalid_port(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_config(tmp_path / "config" / "config.json", port="not-a-port")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="port"):
        load_settings(is_frozen=False)


def test_load_settings_fails_clearly_for_malformed_json(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    config_path = tmp_path / "config" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{ this is malformed", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_settings(is_frozen=False)


def test_load_settings_defaults_to_existing_non_secret_defaults(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_config(
        tmp_path / "config" / "config.json",
        upstream_openai_base_url=DEFAULT_UPSTREAM_OPENAI_BASE_URL,
        default_model=None,
        host="127.0.0.1",
        port=8181,
    )
    monkeypatch.chdir(tmp_path)

    settings = load_settings(is_frozen=False)

    assert settings.upstream_openai_base_url == DEFAULT_UPSTREAM_OPENAI_BASE_URL
    assert settings.default_model is None
    assert settings.host == "127.0.0.1"
    assert settings.port == 8181


def test_default_upstream_base_url_is_public_safe_default() -> None:
    assert DEFAULT_UPSTREAM_OPENAI_BASE_URL == "https://api.openai.com/v1"
