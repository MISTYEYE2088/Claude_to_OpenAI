from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from app.config import load_settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI to Claude proxy")
    parser.add_argument("--config", dest="config", help="Path to config JSON file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config) if args.config else None
    if args.config and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    settings = load_settings(config_path=config_path)
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
