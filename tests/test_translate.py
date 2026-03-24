import pytest

from app.translate import (
    map_anthropic_request_to_openai,
    map_openai_finish_reason,
    map_openai_nonstream_to_anthropic,
)


def test_normalizes_string_message_content_to_text_block():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_rejects_unknown_content_block_type():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "audio", "source": {"data": "AA=="}}],
            }
        ],
    }

    with pytest.raises(ValueError, match="Unsupported content block type"):
        map_anthropic_request_to_openai(payload)


def test_rejects_non_base64_image_source_type():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "media_type": "image/png",
                            "data": "AA==",
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="image.source.type must be base64"):
        map_anthropic_request_to_openai(payload)


def test_rejects_disallowed_image_media_type():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/tiff",
                            "data": "AA==",
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="Unsupported image media_type"):
        map_anthropic_request_to_openai(payload)


def test_rejects_malformed_image_base64():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "%%bad%%",
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="Malformed base64 image data"):
        map_anthropic_request_to_openai(payload)


def test_rejects_non_text_system_blocks():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "system": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AA==",
                },
            }
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }

    with pytest.raises(ValueError, match="system must contain only text blocks"):
        map_anthropic_request_to_openai(payload)


def test_forwards_metadata_when_present():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"trace_id": "abc123"},
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["metadata"] == {"trace_id": "abc123"}


def test_omits_max_tokens_when_missing():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = map_anthropic_request_to_openai(payload)

    assert "max_tokens" not in result


def test_omits_temperature_and_top_p_when_missing():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = map_anthropic_request_to_openai(payload)

    assert "temperature" not in result
    assert "top_p" not in result


def test_maps_stop_sequences_to_openai_stop():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
        "stop_sequences": ["END"],
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["stop"] == ["END"]


def test_maps_missing_stop_sequences_to_empty_stop_list():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["stop"] == []


def test_maps_null_stop_sequences_to_empty_stop_list():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "stop_sequences": None,
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["stop"] == []


def test_maps_missing_metadata_to_empty_object():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["metadata"] == {}


def test_maps_null_metadata_to_empty_object():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": None,
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["metadata"] == {}


def test_passes_optional_values_through_unchanged():
    payload = {
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 0,
        "temperature": "0.6",
        "top_p": -1,
        "stop_sequences": ["END"],
        "metadata": {"trace_id": "abc123"},
    }

    result = map_anthropic_request_to_openai(payload)

    assert result["max_tokens"] == 0
    assert result["temperature"] == "0.6"
    assert result["top_p"] == -1
    assert result["stop"] == ["END"]
    assert result["metadata"] == {"trace_id": "abc123"}


def test_nonstream_normalizes_string_content():
    upstream = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "hello from upstream"},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["content"] == [{"type": "text", "text": "hello from upstream"}]


def test_nonstream_normalizes_array_text_like_content():
    upstream = {
        "id": "chatcmpl-2",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {"type": "output_text", "text": "hello"},
                        {"type": "text", "text": " world"},
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 2},
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": " world"},
    ]


def test_nonstream_normalizes_null_content_to_empty_text():
    upstream = {
        "id": "chatcmpl-3",
        "choices": [{"finish_reason": "stop", "message": {"content": None}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["content"] == [{"type": "text", "text": ""}]


@pytest.mark.parametrize(
    ("upstream_reason", "anthropic_reason"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("content_filter", "end_turn"),
        ("random_reason", None),
        (None, None),
    ],
)
def test_finish_reason_mapping(upstream_reason, anthropic_reason):
    assert map_openai_finish_reason(upstream_reason) == anthropic_reason


def test_tool_calls_finish_reason_is_unsupported_in_v1():
    upstream = {
        "id": "chatcmpl-4",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"content": "hi"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    with pytest.raises(ValueError, match="Tools are unsupported in v1"):
        map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")


def test_nonstream_rejects_unsupported_content_part_type():
    upstream = {
        "id": "chatcmpl-5",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {"type": "output_text", "text": "ok"},
                        {"type": "tool_call", "name": "x"},
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    with pytest.raises(ValueError, match="Unsupported upstream content part type"):
        map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")


def test_stop_sequence_only_set_when_explicitly_signaled_upstream():
    upstream = {
        "id": "chatcmpl-6",
        "choices": [
            {
                "finish_reason": "stop",
                "stop_sequence": "END",
                "message": {"content": "done"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["stop_reason"] == "stop_sequence"
    assert result["stop_sequence"] == "END"


def test_stop_sequence_not_set_without_explicit_signal():
    upstream = {
        "id": "chatcmpl-7",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "done"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["stop_reason"] == "end_turn"
    assert result["stop_sequence"] is None


def test_rejects_message_with_missing_role():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [{"content": "hello"}],
    }

    with pytest.raises(ValueError, match=r"messages\[\*\]\.role is required"):
        map_anthropic_request_to_openai(payload)


def test_rejects_message_with_non_string_role():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [{"role": 123, "content": "hello"}],
    }

    with pytest.raises(ValueError, match=r"messages\[\*\]\.role must be a string"):
        map_anthropic_request_to_openai(payload)


def test_rejects_message_with_unsupported_role():
    payload = {
        "model": "gpt-5.3-codex",
        "max_tokens": 32,
        "messages": [{"role": "system", "content": "hello"}],
    }

    with pytest.raises(
        ValueError, match=r"messages\[\*\]\.role must be one of: user, assistant"
    ):
        map_anthropic_request_to_openai(payload)


def test_nonstream_validates_choices_must_be_non_empty_list():
    upstream = {
        "id": "chatcmpl-shape-1",
        "choices": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    with pytest.raises(
        ValueError, match="Upstream response choices must be a non-empty list"
    ):
        map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")


def test_nonstream_validates_first_choice_must_be_object():
    upstream = {
        "id": "chatcmpl-shape-2",
        "choices": ["bad-choice"],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    with pytest.raises(
        ValueError, match="Upstream response first choice must be an object"
    ):
        map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")


def test_nonstream_validates_message_must_be_object():
    upstream = {
        "id": "chatcmpl-shape-3",
        "choices": [{"finish_reason": "stop", "message": "bad-message"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    with pytest.raises(
        ValueError, match="Upstream response choice.message must be an object"
    ):
        map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")


def test_nonstream_treats_text_part_none_as_empty_string():
    upstream = {
        "id": "chatcmpl-none-text",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {"type": "text", "text": None},
                        {"type": "output_text", "text": "ok"},
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 1},
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["content"] == [
        {"type": "text", "text": ""},
        {"type": "text", "text": "ok"},
    ]


def test_nonstream_usage_token_normalization_for_string_and_bool_values():
    upstream = {
        "id": "chatcmpl-usage-norm-1",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": "17",
            "completion_tokens": True,
        },
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["usage"] == {"input_tokens": 17, "output_tokens": 0}


def test_nonstream_usage_token_normalization_for_invalid_values():
    upstream = {
        "id": "chatcmpl-usage-norm-2",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": "not-a-number",
            "completion_tokens": {"bad": "shape"},
        },
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_nonstream_usage_token_normalization_clamps_negative_values_to_zero():
    upstream = {
        "id": "chatcmpl-usage-norm-3",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": -7,
            "completion_tokens": "-3",
        },
    }

    result = map_openai_nonstream_to_anthropic(upstream, "gpt-5.3-codex")

    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
