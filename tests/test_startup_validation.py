from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_settings


@pytest.fixture
def clear_get_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_startup_fails_when_force_model_true_without_default_model(
    monkeypatch,
    tmp_path: Path,
    clear_get_settings_cache,
):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.json").write_text(
        '{"force_model": true, "default_model": "   "}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_model.*force_model"):
        with TestClient(app):
            pass
