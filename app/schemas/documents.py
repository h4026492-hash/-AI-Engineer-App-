"""Models for document ingestion and retrieval endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentIngestRequest(BaseModel):
    """A single document to index."""

    content: str = Field(min_length=1, description="Raw text of the document")
    source: str = Field(
        min_length=1, description="Human-readable origin, e.g. 'policies/refunds.md'"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "content must contain at least one non-whitespace character"
            raise ValueError(msg)
        return value


class DocumentIngestResponse(BaseModel):
    document_id: str
    source: str
    chunks: int
    characters: int


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunks: int
    characters: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filter_source: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
