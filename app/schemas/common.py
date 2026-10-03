"""Request/response models shared across routers."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Uniform success envelope.

    Keeping every 2xx response shaped the same way means clients can share one
    parser, and adding metadata (pagination, timings) never breaks them.
    """

    model_config = ConfigDict(json_schema_extra={"example": {"data": {}, "meta": {}}})

    data: T
    meta: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the process is up")
    version: str
    environment: str
    checks: dict[str, str] = Field(default_factory=dict)
