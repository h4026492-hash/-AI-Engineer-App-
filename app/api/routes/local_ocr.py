"""Explicitly opt-in OCR preview for trusted loopback development only."""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.deps import AppState, get_app_state
from app.config import Environment
from app.core.errors import (
    ForbiddenError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from app.core.ocr import (
    MAX_DOCUMENT_BYTES,
    DocumentTextError,
    OcrEngineUnavailable,
    extract_document_text,
    identify_media_type,
)
from app.schemas.common import ApiResponse
from app.schemas.ocr import ExtractedPage as ExtractedPageModel
from app.schemas.ocr import OcrPreviewResponse

router = APIRouter(prefix="/v1/local", tags=["local-only OCR"])
ALLOWED_CONTENT_TYPES = {"application/pdf", "image/png", "image/jpeg", "application/octet-stream"}
BYTES_PER_MIB = 1024 * 1024


def _size_limit_message(limit: int) -> str:
    if limit % BYTES_PER_MIB == 0:
        return f"Files must be {limit // BYTES_PER_MIB} MiB or smaller."
    return f"Files must be {limit:,} bytes or smaller."


def is_local_ocr_request(request: Request, state: AppState) -> bool:
    """Fail closed unless opted in from a loopback client and local host."""
    if not state.settings.allow_local_document_ocr:
        return False
    if state.settings.app_env is not Environment.DEVELOPMENT:
        return False

    hostname = (request.url.hostname or "").lower().rstrip(".")
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        return False

    client = request.client
    if client is None:
        return False
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        return False


async def _read_limited_body(request: Request, limit: int) -> bytes:
    """Read a request body incrementally so an oversized upload is not buffered."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise ValidationError("Content-Length must be a valid integer.") from exc
        if declared_length < 0:
            raise ValidationError("Content-Length cannot be negative.")
        if declared_length > limit:
            raise PayloadTooLargeError(_size_limit_message(limit))

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(_size_limit_message(limit))
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/ocr-preview",
    response_model=ApiResponse[OcrPreviewResponse],
    summary="Extract text from a document in local development only",
    description=(
        "Accepts one small PDF, PNG, or JPEG and returns extracted text only. "
        "The endpoint is disabled by default, only enabled for opted-in "
        "development requests from localhost, does not store or index the file, "
        "and never calls a language model. OCR can misread medical text."
    ),
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in sorted(ALLOWED_CONTENT_TYPES - {"application/octet-stream"})
            },
        }
    },
    responses={
        403: {"description": "OCR is disabled or the request is not local development."},
        413: {"description": "The upload exceeds the configured byte limit."},
        415: {"description": "The request is not a PDF, PNG, or JPEG."},
        422: {"description": "The document could not be read or exceeds a page limit."},
        503: {"description": "The local Tesseract executable is unavailable."},
    },
)
async def preview_local_document_text(
    request: Request,
    response: Response,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ApiResponse[OcrPreviewResponse]:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    if not is_local_ocr_request(request, state):
        raise ForbiddenError(
            "Local document extraction is disabled here. It is available only when explicitly "
            "enabled in development and accessed through localhost."
        )

    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedMediaTypeError("Choose a PDF, PNG, or JPEG file.")

    limit = min(state.settings.max_local_ocr_bytes, MAX_DOCUMENT_BYTES)
    content = await _read_limited_body(request, limit)
    if not content:
        raise ValidationError("Choose a non-empty PDF, PNG, or JPEG file.")

    try:
        detected_type = identify_media_type(content)
    except DocumentTextError as exc:
        raise ValidationError(str(exc)) from exc
    if content_type != "application/octet-stream" and content_type != detected_type:
        raise UnsupportedMediaTypeError("The selected file type does not match its contents.")

    try:
        extracted = await asyncio.to_thread(extract_document_text, content)
    except OcrEngineUnavailable as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    except DocumentTextError as exc:
        raise ValidationError(str(exc)) from exc

    preview_response = OcrPreviewResponse(
        media_type=extracted.media_type,
        pages=[
            ExtractedPageModel(
                page_number=page.page_number,
                method=page.method,
                text=page.text,
            )
            for page in extracted.pages
        ],
        character_count=extracted.character_count,
        truncated=extracted.truncated,
    )
    return ApiResponse(
        data=preview_response,
        meta={"ephemeral": True, "indexed": False, "model_called": False},
    )
