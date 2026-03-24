from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_openai_client, get_settings


class FakeUpstreamApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.request_id = request_id


class FakeUpstreamTimeoutError(Exception):
    pass


class FakeUpstreamConnectionError(Exception):
    pass


@dataclass
class FakeCompletionsAPI:
    behavior: Any
    calls: list[dict]

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.behavior, Exception):
            raise self.behavior
        if callable(self.behavior):
            callback = self.behavior
            return callback(**kwargs)
        return self.behavior


@dataclass
class FakeModelsAPI:
    behavior: Any
    calls: list[dict]

    def list(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.behavior, Exception):
            raise self.behavior
        if callable(self.behavior):
            callback = self.behavior
            return callback(**kwargs)
        return self.behavior


class FakeOpenAIClient:
    def __init__(self, behavior: Any, models_behavior: Any = None) -> None:
        completions = FakeCompletionsAPI(behavior=behavior, calls=[])
        models = FakeModelsAPI(
            behavior={"data": []} if models_behavior is None else models_behavior,
            calls=[],
        )
        self._completions = completions
        self._models = models
        self.chat = type("FakeChat", (), {"completions": completions})()
        self.models = models

    @property
    def calls(self) -> list[dict]:
        return self._completions.calls

    @property
    def model_calls(self) -> list[dict]:
        return self._models.calls


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        upstream_openai_base_url="https://example.invalid/v1",
        default_model=None,
        host="127.0.0.1",
        port=8181,
    )


@pytest.fixture
def make_test_client(
    settings: Settings,
) -> Callable[..., tuple[TestClient, FakeOpenAIClient]]:
    def _factory(
        behavior: Any,
        *,
        models_behavior: Any = None,
    ) -> tuple[TestClient, FakeOpenAIClient]:
        fake_client = FakeOpenAIClient(behavior, models_behavior=models_behavior)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_openai_client] = lambda: fake_client
        client = TestClient(app)
        return client, fake_client

    yield _factory
    app.dependency_overrides.clear()
