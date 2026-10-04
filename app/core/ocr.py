"""Ephemeral, local-only text extraction for user-supplied medical documents.

This module extracts text only. It does not interpret results, add content to the
RAG index, call a language model, or write the input or extracted text to disk.
The API layer must enforce the explicit local-development and loopback checks.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image, ImageOps

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 5
MAX_IMAGE_PIXELS = 16_000_000
MAX_EXTRACTED_CHARACTERS = 40_000
TESSERACT_TIMEOUT_SECONDS = 10

MediaType = Literal["application/pdf", "image/png", "image/jpeg"]
ExtractionMethod = Literal["embedded-text", "ocr"]


class DocumentTextError(Exception):
    """The document is unsupported, malformed, or exceeds a safe limit."""


class OcrEngineUnavailable(Exception):
    """Local Tesseract OCR is not installed or could not be run."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one page or one image."""

    page_number: int
    method: ExtractionMethod
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Ephemeral extraction result; callers must not persist it."""

    media_type: MediaType
    pages: tuple[ExtractedPage, ...]
    truncated: bool

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


def identify_media_type(content: bytes) -> MediaType:
    """Identify supported files by their signature rather than their name."""
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise DocumentTextError("Choose a PDF, PNG, or JPEG file.")


def _normalise_text(text: str) -> str:
    without_nuls = text.replace("\x00", "")
    return "\n".join(line.rstrip() for line in without_nuls.splitlines()).strip()


def _require_tesseract() -> None:
    if shutil.which("tesseract") is None:
        raise OcrEngineUnavailable(
            "Local OCR needs the Tesseract engine. Install it on this computer, "
            "then restart the development server."
        )


def _ocr_image(image: Image.Image) -> str:
    """Run OCR via stdin/stdout, avoiding temporary files for medical images."""
    _require_tesseract()
    image_buffer = io.BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6"],
            input=image_buffer.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=TESSERACT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrEngineUnavailable("The local OCR engine could not process this image.") from exc
    if result.returncode != 0:
        raise OcrEngineUnavailable("The local OCR engine could not process this image.")
    return _normalise_text(result.stdout.decode("utf-8", errors="replace"))


def _check_image_dimensions(image: Image.Image) -> None:
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise DocumentTextError(
            f"The image is too large to process safely (maximum {MAX_IMAGE_PIXELS:,} pixels)."
        )


def _extract_image(content: bytes, media_type: MediaType) -> ExtractedDocument:
    try:
        with Image.open(io.BytesIO(content)) as source:
            expected_format = "PNG" if media_type == "image/png" else "JPEG"
            if source.format != expected_format:
                raise DocumentTextError("The image type does not match its file contents.")
            _check_image_dimensions(source)
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except DocumentTextError:
        raise
    except Exception as exc:
        raise DocumentTextError(
            "The image could not be read. Try a clear PNG or JPEG image."
        ) from exc

    text = _ocr_image(image)
    return _apply_character_limit(media_type, [ExtractedPage(1, "ocr", text)])


def _ocr_pdf_page(page: pdfium.PdfPage) -> str:
    width, height = page.get_size()
    if width <= 0 or height <= 0:
        return ""
    scale = min(2.0, math.sqrt(MAX_IMAGE_PIXELS / (width * height)))
    if scale < 0.25:
        raise DocumentTextError("A scanned PDF page is too large to process safely.")

    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil()
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise DocumentTextError("A scanned PDF page is too large to process safely.")
        image.load()
        converted = image.convert("RGB")
    except DocumentTextError:
        raise
    except Exception as exc:
        raise DocumentTextError("A scanned PDF page could not be prepared for OCR.") from exc
    finally:
        bitmap.close()
    return _ocr_image(converted)


def _extract_pdf(content: bytes) -> ExtractedDocument:
    try:
        document = pdfium.PdfDocument(content)
    except Exception as exc:
        raise DocumentTextError(
            "The PDF could not be opened. Password-protected or malformed PDFs are not supported."
        ) from exc

    with document:
        page_count = len(document)
        if page_count == 0:
            raise DocumentTextError("The PDF has no pages to read.")
        if page_count > MAX_PDF_PAGES:
            raise DocumentTextError(f"Use a PDF with {MAX_PDF_PAGES} pages or fewer.")

        pages: list[ExtractedPage] = []
        for page_index in range(page_count):
            page = None
            try:
                page = document[page_index]
                text_page = page.get_textpage()
                try:
                    text = _normalise_text(text_page.get_text_range())
                finally:
                    text_page.close()
                if text:
                    method: ExtractionMethod = "embedded-text"
                else:
                    text = _ocr_pdf_page(page)
                    method = "ocr"
            except (DocumentTextError, OcrEngineUnavailable):
                raise
            except Exception as exc:
                raise DocumentTextError("A PDF page could not be read.") from exc
            finally:
                if page is not None:
                    page.close()
            pages.append(ExtractedPage(page_index + 1, method, text))

    return _apply_character_limit("application/pdf", pages)


def _apply_character_limit(media_type: MediaType, pages: list[ExtractedPage]) -> ExtractedDocument:
    remaining = MAX_EXTRACTED_CHARACTERS
    limited_pages: list[ExtractedPage] = []
    truncated = False
    for page in pages:
        if len(page.text) > remaining:
            limited_pages.append(
                ExtractedPage(page.page_number, page.method, page.text[:remaining])
            )
            truncated = True
            break
        limited_pages.append(page)
        remaining -= len(page.text)
    return ExtractedDocument(media_type, tuple(limited_pages), truncated)


def extract_document_text(content: bytes) -> ExtractedDocument:
    """Extract selectable PDF text and/or OCR text from a small local document.

    The function deliberately returns raw extracted text only. It performs no
    clinical interpretation and has no filesystem, vector-store, or model calls.
    """
    if not content:
        raise DocumentTextError("Choose a non-empty PDF, PNG, or JPEG file.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentTextError(
            f"Files must be {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB or smaller."
        )

    media_type = identify_media_type(content)
    if media_type == "application/pdf":
        return _extract_pdf(content)
    return _extract_image(content, media_type)
