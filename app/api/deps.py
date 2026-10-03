"""Dependency wiring.

FastAPI's dependency injection is the seam that makes the API testable:
``app.dependency_overrides`` swaps the real pipeline for a test double without
touching a single route handler.

Everything expensive (model clients, the vector index) is built once in the
lifespan and stored on ``app.state``; these accessors just hand it out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.rag.pipeline import RAGPipeline
    from app.rag.vectorstore import VectorStore


@dataclass(slots=True)
class AppState:
    """Everything the request handlers need, built at startup."""

    settings: Settings
    pipeline: RAGPipeline
    store: VectorStore
    provider: str
    offline: bool


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state  # type: ignore[no-any-return]


def get_settings_dep(request: Request) -> Settings:
    """Settings as a dependency, so tests can override configuration per request."""
    state: AppState = request.app.state.app_state
    return state.settings


def get_pipeline(request: Request) -> RAGPipeline:
    return get_app_state(request).pipeline


def get_store(request: Request) -> VectorStore:
    return get_app_state(request).store


__all__ = [
    "AppState",
    "get_app_state",
    "get_pipeline",
    "get_settings",
    "get_settings_dep",
    "get_store",
]
