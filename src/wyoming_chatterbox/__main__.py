"""Command-line entry point for the Wyoming Chatterbox server."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.server.server import run_server


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(settings: Settings) -> None:
    """Configure root logging based on settings."""
    handler = logging.StreamHandler(sys.stderr)
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


def main() -> None:
    """Load settings, configure logging and run the server."""
    settings = Settings()
    setup_logging(settings)
    try:
        asyncio.run(run_server(settings))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass


if __name__ == "__main__":
    main()
