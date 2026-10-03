"""Structured logging.

Two output modes:
  * ``console`` -- human-readable, colourless, good for local development
  * ``json``    -- one JSON object per line, for log aggregators in production

Every log record emitted inside a request carries the ``request_id`` bound by
:mod:`app.core.middleware`, which is what makes a single request traceable
across async boundaries.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, TextIO

# ContextVar, not a global: each asyncio task gets its own request id.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class _ConsoleFormatter(logging.Formatter):
    """``LEVEL  logger.name  message  key=value`` -- scannable, greppable."""

    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        base = f"{record.levelname:<8} {record.name:<28} {record.getMessage()}"
        return f"{base}  {extras}" if extras else base


class _JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Pull caller-supplied structured fields off the record."""
    extra = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
    request_id = request_id_ctx.get()
    if request_id != "-":
        extra.setdefault("request_id", request_id)
    return extra


def configure_logging(
    level: str = "INFO", fmt: str = "console", stream: TextIO | None = None
) -> None:
    """Install a root handler. Idempotent: safe to call on every app start.

    Defaults to stdout, which is right for a server whose only output is logs.
    The CLI passes stderr instead so stdout stays a clean JSON stream and
    ``ai-app ask ... | jq`` works.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _ConsoleFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party libraries are noisy at DEBUG; keep them at WARNING unless the
    # app itself is configured for DEBUG.
    noisy = logging.DEBUG if level.upper() == "DEBUG" else logging.WARNING
    for name in ("uvicorn.access", "uvicorn.error", "httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(noisy)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
