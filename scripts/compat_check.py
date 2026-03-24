"""Compatibility and release-gating verification for proxy endpoints.

Security checks to run before a public release:
- gitleaks detect --source . --no-banner
- gitleaks git --no-banner
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8181"
REQUIRED_MESSAGE_KEYS = {
    "id",
    "type",
    "role",
    "content",
    "model",
    "stop_reason",
    "usage",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Anthropic compatibility surface on /ready, /v1/models, and "
            "/v1/messages."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser.parse_args(argv)


def _request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    body = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=request_headers,
        method=method,
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read()
        text = raw.decode("utf-8") if raw else ""
        parsed = json.loads(text) if text else None
        return response.status, parsed


def _append_issue(issues: list[str], message: str) -> None:
    issues.append(f"- FAIL: {message}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base_url = args.base_url.rstrip("/")
    issues: list[str] = []

    try:
        ready_status, ready_payload = _request_json(
            method="GET", url=f"{base_url}/ready"
        )
        if ready_status != 200:
            _append_issue(
                issues,
                f"/ready expected status 200, got {ready_status}",
            )
        if not isinstance(ready_payload, dict) or "status" not in ready_payload:
            _append_issue(issues, "/ready response missing required 'status' field")
    except Exception as exc:
        _append_issue(issues, f"/ready request failed: {exc}")

    try:
        models_status, models_payload = _request_json(
            method="GET",
            url=f"{base_url}/v1/models",
        )
        if models_status != 200:
            _append_issue(
                issues,
                f"/v1/models expected status 200, got {models_status}",
            )
        if not isinstance(models_payload, dict) or not isinstance(
            models_payload.get("data"), list
        ):
            _append_issue(
                issues, "/v1/models response missing required list field 'data'"
            )
    except Exception as exc:
        _append_issue(issues, f"/v1/models request failed: {exc}")

    try:
        messages_status, messages_payload = _request_json(
            method="POST",
            url=f"{base_url}/v1/messages",
            payload={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers={"x-api-key": "compat-check", "anthropic-version": "2023-06-01"},
        )
        if messages_status != 200:
            _append_issue(
                issues,
                f"/v1/messages expected status 200, got {messages_status}",
            )
        if not isinstance(messages_payload, dict):
            _append_issue(issues, "/v1/messages response body is not a JSON object")
        else:
            missing = sorted(REQUIRED_MESSAGE_KEYS - set(messages_payload))
            if missing:
                _append_issue(
                    issues,
                    "missing required Anthropic envelope keys: " + ", ".join(missing),
                )
    except Exception as exc:
        _append_issue(issues, f"/v1/messages request failed: {exc}")

    if issues:
        print("Compatibility check summary: FAIL")
        for issue in issues:
            print(issue)
        return 1

    print("Compatibility check summary: PASS")
    print("- PASS: /ready, /v1/models, and /v1/messages checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
