"""Domain errors and the FastAPI handlers that turn them into HTTP responses.

Raise :class:`AppError` subclasses from business logic. They are translated by
:func:`register_exception_handlers` into a consistent JSON envelope, so no
endpoint ever has to hand-build an error response.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for expected, client-facing failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AppError):
    """The caller sent something unusable."""

    status_code = 422
    code = "validation_error"


class NotFoundError(AppError):
    """A referenced resource does not exist."""

    status_code = 404
    code = "not_found"


class ForbiddenError(AppError):
    """The current deployment does not permit this operation."""

    status_code = 403
    code = "forbidden"


class PayloadTooLargeError(AppError):
    """Input exceeded a configured size limit."""

    status_code = 413
    code = "payload_too_large"


class RateLimitError(AppError):
    """Too many requests from this client."""

    status_code = 429
    code = "rate_limited"


class ProviderError(AppError):
    """An upstream model provider failed. Retriable by the caller."""

    status_code = 502
    code = "provider_error"


class IngestionError(AppError):
    """A document could not be ingested."""

    status_code = 422
    code = "ingestion_error"


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {"code": code, "message": message},
        "request_id": request_id_ctx.get(),
    }
    if details:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach exception handlers that emit the standard error envelope."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error code=%s message=%s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code, content=_envelope(exc.code, exc.message, exc.details)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Project to the three fields a client can act on. The raw error list
        # carries a `ctx` mapping that can contain live exception objects, which
        # are not JSON serialisable -- and would leak internals anyway.
        issues = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        logger.info("request_validation_failed issues=%d", len(issues))
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error", "Request body failed validation.", {"issues": issues}
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Log the traceback server-side; return only an opaque message to the
        # client so internals are never leaked in production.
        logger.exception("unhandled_exception type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "An unexpected error occurred."),
        )
