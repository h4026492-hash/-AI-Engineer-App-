"""Shared fixtures.

Every fixture here is offline: no API keys, no network. The app fixture builds a
real application with the deterministic provider, so tests exercise the same
code path the server runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.deps import AppState
from app.config import Settings
from app.llm.echo_provider import EchoChatModel
from app.llm.hashing_embedder import HashingEmbedder
from app.main import create_app
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore

EMBED_DIM = 256

SAMPLE_REFUNDS = """\
# Refund Policy

You can request a refund within 30 days of your first payment for a full refund.
After the 30-day window, payments are non-refundable.

Annual plans may be refunded in full within 14 days of purchase.
"""

SAMPLE_SECURITY = """\
# Security

All data in transit is encrypted with TLS 1.3. Data at rest is encrypted with
AES-256.

Encryption keys are held in a hardware security module and rotated every 90 days.
"""


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings isolated from the developer's environment and `.env` file."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_env="development",
        debug=True,
        log_level="WARNING",
        log_format="console",
        llm_provider="echo",
        rag_chunk_size=400,
        rag_chunk_overlap=50,
        rag_top_k=3,
        rag_min_score=0.05,
        rag_embed_dim=EMBED_DIM,
        vector_store_path=str(tmp_path / "vectorstore"),
        seed_on_startup=False,
        rate_limit_per_minute=0,
        cors_origins="http://testserver",
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=EMBED_DIM)


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(dimension=EMBED_DIM)


@pytest.fixture
def pipeline(store: VectorStore, embedder: HashingEmbedder) -> RAGPipeline:
    return RAGPipeline(
        chat=EchoChatModel(),
        embedder=embedder,
        store=store,
        chunk_size=400,
        chunk_overlap=50,
        top_k=3,
        min_score=0.05,
    )


@pytest.fixture
def seeded_pipeline(pipeline: RAGPipeline) -> RAGPipeline:
    """A pipeline with two documents already indexed."""
    asyncio.run(pipeline.ingest(SAMPLE_REFUNDS, source="refunds.md"))
    asyncio.run(pipeline.ingest(SAMPLE_SECURITY, source="security.md"))
    return pipeline


@pytest.fixture
def app_state(test_settings: Settings, seeded_pipeline: RAGPipeline) -> AppState:
    return AppState(
        settings=test_settings,
        pipeline=seeded_pipeline,
        store=seeded_pipeline.store,
        provider="echo",
        offline=True,
    )


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    """A TestClient running the real app, with the test's pre-seeded state.

    Startup runs normally (middleware, exception handlers, lifespan), then the
    app state is swapped for the fixture's pre-seeded pipeline so tests do not
    depend on startup seeding.
    """
    application = create_app(app_state.settings)
    with TestClient(application) as test_client:
        application.state.app_state = app_state
        yield test_client


@pytest.fixture
def live_client(test_settings: Settings) -> Iterator[TestClient]:
    """A TestClient that runs the real lifespan, seeding included.

    Used for the handful of tests that specifically verify startup behaviour.
    """
    application = create_app(test_settings)
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """A temporary corpus directory for seeding tests."""
    directory = tmp_path / "sample_docs"
    directory.mkdir()
    (directory / "refunds.md").write_text(SAMPLE_REFUNDS, encoding="utf-8")
    (directory / "security.md").write_text(SAMPLE_SECURITY, encoding="utf-8")
    (directory / "empty.md").write_text("   \n", encoding="utf-8")
    (directory / "ignored.pdf").write_bytes(b"not markdown")
    return directory


def json_envelope(body: Any) -> dict[str, Any]:
    """Unwrap the standard success envelope, asserting its shape."""
    assert "data" in body, f"missing 'data' key in {body!r}"
    assert "meta" in body, f"missing 'meta' key in {body!r}"
    return body
