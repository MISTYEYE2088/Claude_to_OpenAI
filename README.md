# OpenAI to Claude Proxy

Anthropic-compatible local proxy that accepts Claude-style `/v1/messages` requests and forwards them to an OpenAI-compatible upstream API.

## Purpose

This project provides a small compatibility layer so Anthropic-format clients can call an OpenAI-compatible backend through a Claude-style API surface.

## Binary Usage

The runtime reads settings from `config/config.json` by default.

1. Copy `config/config.example.json` to `config/config.json`.
2. Adjust values in `config/config.json` for your environment.
3. Start the binary:

```bash
./openai-to-claude-proxy --config config/config.json
```

## API Key Environment Variable

Set the API key variable before starting the service:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

`config/config.json` uses `openai_api_key_env` to choose which environment variable name to read.

## Optional: Run From Source (Python and pip)

If you want to run from source instead of a packaged binary:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m app.entrypoint --config config/config.json
```

## Linux Compatibility

For Linux binaries built on GitHub Ubuntu runners, compatibility follows that runner image's glibc baseline (for example, Ubuntu 22.04 uses glibc 2.35). Use a runner or build environment that matches your deployment target.
