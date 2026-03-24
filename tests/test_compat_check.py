from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "compat_check.py"
    spec = importlib.util.spec_from_file_location("compat_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._payload


def test_compat_check_succeeds_with_expected_endpoint_shapes(monkeypatch, capsys):
    module = _load_module()

    def fake_urlopen(request, timeout=10):
        url = request.full_url
        method = request.get_method()

        if url.endswith("/ready"):
            assert method == "GET"
            return _FakeResponse(200, {"status": "ok"})
        if url.endswith("/v1/models"):
            assert method == "GET"
            return _FakeResponse(200, {"data": [{"id": "gpt-4o-mini"}]})
        if url.endswith("/v1/messages"):
            assert method == "POST"
            return _FakeResponse(
                200,
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "claude-3-5-sonnet",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    code = module.main(["--base-url", "http://compat.test"])
    output = capsys.readouterr().out

    assert code == 0
    assert "PASS" in output


def test_compat_check_fails_when_anthropic_envelope_keys_are_missing(
    monkeypatch, capsys
):
    module = _load_module()

    def fake_urlopen(request, timeout=10):
        url = request.full_url
        if url.endswith("/ready"):
            return _FakeResponse(200, {"status": "ok"})
        if url.endswith("/v1/models"):
            return _FakeResponse(200, {"data": [{"id": "gpt-4o-mini"}]})
        if url.endswith("/v1/messages"):
            return _FakeResponse(
                200,
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    code = module.main(["--base-url", "http://compat.test"])
    output = capsys.readouterr().out

    assert code != 0
    assert "FAIL" in output
    assert "missing required Anthropic envelope keys" in output


def test_compat_check_prints_clear_failure_output(monkeypatch, capsys):
    module = _load_module()

    def fake_urlopen(request, timeout=10):
        url = request.full_url
        if url.endswith("/ready"):
            return _FakeResponse(503, {"status": "down"})
        if url.endswith("/v1/models"):
            return _FakeResponse(200, {"data": [{"id": "gpt-4o-mini"}]})
        if url.endswith("/v1/messages"):
            return _FakeResponse(
                200,
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "claude-3-5-sonnet",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    code = module.main(["--base-url", "http://compat.test"])
    output = capsys.readouterr().out

    assert code != 0
    assert "FAIL" in output
    assert "/ready" in output
    assert "expected status 200" in output
