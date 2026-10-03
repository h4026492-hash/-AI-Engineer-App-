"""HTTP middleware: request IDs, access logs, latency, and rate limiting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger, request_id_ctx

logger = get_logger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, bind it to the log context, and log one access line.

    Honours an inbound ``X-Request-ID`` so IDs survive across services and load
    balancers; otherwise mints one. The ID is echoed back on the response.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "request_failed",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            raise
        finally:
            request_id_ctx.reset(token)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "client": request.client.host if request.client else "unknown",
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client rate limiter for ``/v1`` routes.

    Deliberately in-process: it protects a single instance and is dependency
    free. For a multi-replica deployment, swap the counter for Redis -- the
    middleware's interface does not change.
    """

    def __init__(self, app: ASGIApp, *, requests_per_minute: int) -> None:
        super().__init__(app)
        self.limit = requests_per_minute
        self.window_seconds = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self.limit <= 0 or not request.url.path.startswith("/v1"):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] >= self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.limit:
            retry_after = int(self.window_seconds - (now - bucket[0])) + 1
            return Response(
                status_code=429,
                content=(b'{"error":{"code":"rate_limited","message":"Too many requests."}}'),
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)
