"""Liveness, readiness, and configuration endpoints.

``/health`` answers as long as the process is up -- that is what an
orchestrator's liveness probe should hit, because restarting a process whose
dependencies are briefly slow makes things worse.

``/health/ready`` reports whether the service can actually serve traffic, which
is what a readiness probe and a load balancer should use.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response

from app import __version__
from app.api.deps import AppState, get_app_state
from app.core.logging import get_logger
from app.schemas.common import ApiResponse, HealthResponse

logger = get_logger(__name__)
router = APIRouter(tags=["operational"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 whenever the process is able to handle requests.",
)
async def liveness(state: Annotated[AppState, Depends(get_app_state)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=state.settings.app_env.value,
        checks={"process": "ok"},
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Returns 200 when the service can serve questions, 503 when it cannot.",
)
async def readiness(
    state: Annotated[AppState, Depends(get_app_state)], response: Response
) -> HealthResponse:
    checks: dict[str, str] = {
        "process": "ok",
        "provider": state.provider,
        "index": "bound" if state.store.is_bound else "unbound",
        "indexed_chunks": str(len(state.store)),
    }
    # An empty index is not a failure: ingestion may not have run yet. Only an
    # unconfigured provider would make the service unable to answer.
    ready = state.provider != ""
    checks["ready"] = "yes" if ready else "no"
    if not ready:
        response.status_code = 503
    return HealthResponse(
        status="ready" if ready else "not_ready",
        version=__version__,
        environment=state.settings.app_env.value,
        checks=checks,
    )


@router.get(
    "/v1/config",
    response_model=ApiResponse[dict[str, Any]],
    summary="Effective configuration",
    description="Non-secret configuration in force, including the active provider and retrieval settings.",
)
async def config(state: Annotated[AppState, Depends(get_app_state)]) -> ApiResponse[dict[str, Any]]:
    public = state.settings.public_config()
    public["indexed_chunks"] = len(state.store)
    public["indexed_documents"] = len(state.store.documents())
    public["embedder"] = state.pipeline.embedder_name
    public["offline_mode"] = state.offline
    return ApiResponse(data=public)
