"""API models for ephemeral local OCR previews."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Text extracted from one PDF page or one image."""

    page_number: int = Field(ge=1)
    method: Literal["embedded-text", "ocr"]
    text: str


class OcrPreviewResponse(BaseModel):
    """Raw text only; never represents a clinical interpretation."""

    media_type: Literal["application/pdf", "image/png", "image/jpeg"]
    pages: list[ExtractedPage]
    character_count: int = Field(ge=0)
    truncated: bool
