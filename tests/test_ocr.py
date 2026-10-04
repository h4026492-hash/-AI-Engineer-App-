"""Safety and extraction tests for the ephemeral local OCR preview."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.config import Settings
from app.core.ocr import (
    MAX_DOCUMENT_BYTES,
    DocumentTextError,
    ExtractedDocument,
    extract_document_text,
)
from app.main import create_app


def make_pdf(text: str | None = None, pages: int = 1) -> bytes:
    """Build a tiny synthetic PDF without patient data."""
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=letter)
    for page_number in range(pages):
        if text:
            document.drawString(72, 720, f"{text} {page_number + 1}")
        document.showPage()
    document.save()
    return buffer.getvalue()


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_embedded_pdf_text_is_extracted_without_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_ocr(_: Image.Image) -> str:
        pytest.fail("OCR should not run for a text-based PDF")

    monkeypatch.setattr("app.core.ocr._ocr_image", unexpected_ocr)

    result = extract_document_text(make_pdf("Synthetic reference range example"))

    assert result.media_type == "application/pdf"
    assert result.pages[0].method == "embedded-text"
    assert "Synthetic reference range example" in result.pages[0].text
    assert result.character_count > 0
    assert result.truncated is False


def test_scanned_pdf_uses_ocr_and_keeps_page_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.ocr._ocr_image", lambda _: "synthetic OCR sample")

    result = extract_document_text(make_pdf())

    assert result.pages[0].method == "ocr"
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == "synthetic OCR sample"


def test_png_uses_ocr_without_persisting_or_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.ocr._ocr_image", lambda _: "synthetic bottle-label text")

    result = extract_document_text(make_png())

    assert result.media_type == "image/png"
    assert result.pages[0].method == "ocr"
    assert result.pages[0].text == "synthetic bottle-label text"


def test_tesseract_receives_image_bytes_over_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import ocr

    calls: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls["command"] = command
        calls.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"synthetic decoded text\n")

    monkeypatch.setattr(ocr.shutil, "which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    result = extract_document_text(make_png())

    assert result.pages[0].text == "synthetic decoded text"
    assert calls["command"] == ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6"]
    assert calls["input"].startswith(b"\x89PNG\r\n\x1a\n")
    assert calls["stdout"] == ocr.subprocess.PIPE
    assert calls["stderr"] == ocr.subprocess.DEVNULL
    assert calls["check"] is False


def test_unsupported_or_oversized_input_is_rejected() -> None:
    with pytest.raises(DocumentTextError, match="Choose a PDF"):
        extract_document_text(b"not a supported document")

    with pytest.raises(DocumentTextError, match="8 MB or smaller"):
        extract_document_text(b"%PDF-" + b"x" * MAX_DOCUMENT_BYTES)


def test_pdf_page_limit_is_enforced_before_extraction() -> None:
    with pytest.raises(DocumentTextError, match="5 pages or fewer"):
        extract_document_text(make_pdf(pages=6))


def test_character_limit_is_applied() -> None:
    from app.core import ocr

    pages = [ocr.ExtractedPage(1, "embedded-text", "x" * (ocr.MAX_EXTRACTED_CHARACTERS + 10))]
    result: ExtractedDocument = ocr._apply_character_limit("application/pdf", pages)  # noqa: SLF001

    assert result.character_count == ocr.MAX_EXTRACTED_CHARACTERS
    assert result.truncated is True


def test_default_configuration_blocks_local_ocr(client: TestClient) -> None:
    response = client.post(
        "/v1/local/ocr-preview",
        content=make_pdf("Synthetic sample"),
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def _test_settings(tmp_path: Path, **updates: Any) -> Settings:
    defaults: dict[str, Any] = {
        "_env_file": None,
        "app_env": "development",
        "llm_provider": "echo",
        "rag_embed_dim": 256,
        "vector_store_path": str(tmp_path / "vectorstore"),
        "seed_on_startup": False,
        "rate_limit_per_minute": 0,
        "allow_local_document_ocr": True,
    }
    defaults.update(updates)
    return Settings(**defaults)  # type: ignore[call-arg]


def test_opted_in_loopback_preview_is_ephemeral(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    application = create_app(settings)
    with TestClient(
        application,
        base_url="http://localhost",
        client=("127.0.0.1", 50001),
    ) as local_client:
        response = local_client.post(
            "/v1/local/ocr-preview",
            content=make_pdf("Synthetic lab heading"),
            headers={"Content-Type": "application/pdf"},
        )
        service_info = local_client.get("/service-info")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["data"]["pages"][0]["text"].startswith("Synthetic lab heading")
    assert body["meta"] == {"ephemeral": True, "indexed": False, "model_called": False}
    assert service_info.json()["local_document_ocr_available"] is True


def test_local_ocr_is_denied_for_non_loopback_and_production(
    tmp_path: Path,
) -> None:
    development_app = create_app(_test_settings(tmp_path / "dev"))
    with TestClient(
        development_app,
        base_url="http://example.test",
        client=("192.0.2.20", 50001),
    ) as external_client:
        external_response = external_client.post(
            "/v1/local/ocr-preview",
            content=make_pdf("Synthetic sample"),
            headers={"Content-Type": "application/pdf"},
        )
    production_app = create_app(_test_settings(tmp_path / "prod", app_env="production"))
    with TestClient(
        production_app,
        base_url="http://localhost",
        client=("127.0.0.1", 50002),
    ) as production_client:
        production_response = production_client.post(
            "/v1/local/ocr-preview",
            content=make_pdf("Synthetic sample"),
            headers={"Content-Type": "application/pdf"},
        )

    assert external_response.status_code == 403
    assert production_response.status_code == 403


def test_local_ocr_reports_missing_engine_and_never_indexes_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _test_settings(tmp_path)
    application = create_app(settings)
    monkeypatch.setattr("app.core.ocr.shutil.which", lambda _: None)
    with TestClient(
        application,
        base_url="http://localhost",
        client=("127.0.0.1", 50004),
    ) as local_client:
        response = local_client.post(
            "/v1/local/ocr-preview",
            content=make_png(),
            headers={"Content-Type": "image/png"},
        )
        documents = local_client.get("/v1/documents").json()["data"]

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert documents == []


def test_local_ocr_rejects_oversized_request_before_processing(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, max_local_ocr_bytes=1024)
    application = create_app(settings)
    with TestClient(
        application,
        base_url="http://localhost",
        client=("127.0.0.1", 50003),
    ) as local_client:
        response = local_client.post(
            "/v1/local/ocr-preview",
            content=b"%PDF-" + b"x" * 1024,
            headers={"Content-Type": "application/pdf"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
