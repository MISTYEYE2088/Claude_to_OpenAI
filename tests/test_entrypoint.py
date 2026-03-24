from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings


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


def test_main_passes_cli_config_path_to_settings_loader(monkeypatch, tmp_path: Path):
    from app import entrypoint

    explicit_path = tmp_path / "custom" / "settings.json"
    calls: dict[str, object] = {}

    def fake_load_settings(*, config_path=None):
        calls["config_path"] = config_path
        return Settings(
            openai_api_key="test-key",
            upstream_openai_base_url="https://example.invalid/v1",
            default_model=None,
            host="127.0.0.1",
            port=8181,
        )

    def fake_uvicorn_run(app_target: str, *, host: str, port: int):
        calls["app_target"] = app_target
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr(entrypoint, "load_settings", fake_load_settings)
    monkeypatch.setattr(entrypoint.uvicorn, "run", fake_uvicorn_run)

    result = entrypoint.main(["--config", str(explicit_path)])

    assert result == 0
    assert calls["config_path"] == explicit_path
    assert calls["app_target"] == "app.main:app"
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8181


def test_main_without_config_uses_cwd_config_resolution_in_source_mode(
    monkeypatch,
    tmp_path: Path,
):
    from app import entrypoint

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "config" / "config.json", host="0.0.0.0", port=9191)

    calls: dict[str, object] = {}

    def fake_uvicorn_run(app_target: str, *, host: str, port: int):
        calls["app_target"] = app_target
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr(entrypoint.uvicorn, "run", fake_uvicorn_run)

    result = entrypoint.main([])

    assert result == 0
    assert calls["app_target"] == "app.main:app"
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9191


def test_main_starts_uvicorn_with_resolved_host_port(monkeypatch):
    from app import entrypoint

    calls: dict[str, object] = {}

    def fake_load_settings(*, config_path=None):
        calls["config_path"] = config_path
        return Settings(
            openai_api_key="test-key",
            upstream_openai_base_url="https://example.invalid/v1",
            default_model=None,
            host="1.2.3.4",
            port=4321,
        )

    def fake_uvicorn_run(app_target: str, *, host: str, port: int):
        calls["app_target"] = app_target
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr(entrypoint, "load_settings", fake_load_settings)
    monkeypatch.setattr(entrypoint.uvicorn, "run", fake_uvicorn_run)

    result = entrypoint.main([])

    assert result == 0
    assert calls["config_path"] is None
    assert calls["app_target"] == "app.main:app"
    assert calls["host"] == "1.2.3.4"
    assert calls["port"] == 4321
