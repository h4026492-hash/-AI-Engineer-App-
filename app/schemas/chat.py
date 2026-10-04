"""Models for the chat (question answering) endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One retrieved chunk that influenced the answer."""

    chunk_id: str
    document_id: str
    source: str = Field(description="Document name or URL the chunk came from")
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity to the query")
    text: str = Field(description="The chunk text as handed to the model")
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Natural-language question to answer from the indexed corpus",
        examples=["What is the refund policy?"],
    )
    top_k: int | None = Field(default=None, ge=1, le=50, description="Override the default top_k")
    filter_source: str | None = Field(default=None, description="Only retrieve from this source")
    include_context: bool = Field(
        default=True, description="Return the retrieved chunks in the response"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "How do I reset my password?", "top_k": 4, "include_context": True}
            ]
        }
    }


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = Field(
        description="False when no retrieved evidence supports the response. This also "
        "covers deterministic safety handoffs that bypass retrieval."
    )
    provider: Literal["echo", "openai"]
    latency_ms: float = Field(ge=0.0)
    tokens_in: int = Field(ge=0, description="Approximate prompt tokens")
    tokens_out: int = Field(ge=0, description="Approximate completion tokens")
