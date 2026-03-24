from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx
import pytest


RUN_LIVE_IMAGE_TEST_ENV = "RUN_LIVE_IMAGE_TEST"
LIVE_PROXY_BASE_URL_ENV = "LIVE_PROXY_BASE_URL"
LIVE_IMAGE_MODEL_ENV = "LIVE_IMAGE_MODEL"


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(
    not _env_enabled(RUN_LIVE_IMAGE_TEST_ENV),
    reason=(
        "Live test disabled. Set RUN_LIVE_IMAGE_TEST=1 "
        "(optionally set LIVE_PROXY_BASE_URL and LIVE_IMAGE_MODEL)."
    ),
)
def test_live_proxy_describes_png_image() -> None:
    base_url = os.getenv(LIVE_PROXY_BASE_URL_ENV, "http://127.0.0.1:8181")
    model = os.getenv(LIVE_IMAGE_MODEL_ENV, "gpt-5.3-codex")
    image_path = Path(__file__).with_name("black_1980x1080.png")

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one short sentence.",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    }

    assert "max_tokens" not in payload

    response = httpx.post(f"{base_url}/v1/messages", json=payload, timeout=120)
    assert response.status_code == 200, response.text

    body = response.json()
    content = body.get("content") or []
    text_blocks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        block_text = block.get("text", "")
        text_blocks.append(str(block_text))

    text = " ".join(text_blocks)

    assert text.strip(), body
