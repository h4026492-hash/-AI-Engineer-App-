"""Application factory and lifespan.

``create_app`` is the single entry point: it configures logging, builds the model
stack, loads or creates the vector index, wires middleware and routers, and
returns a ready ASGI app. ``app.main:app`` (module level) exists for uvicorn,
but tests build their own app through this function.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.api.deps import AppState
from app.api.routes import chat, documents, health, local_ocr
from app.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.llm.factory import build_stack
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore

logger = get_logger(__name__)

SAMPLE_DOCS_DIR = Path("data/sample_docs")
SAMPLE_SOURCE_METADATA: dict[str, dict[str, str]] = {
    "lab-results.md": {
        "publisher": "MedlinePlus / U.S. National Library of Medicine",
        "source_url": "https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/",
    },
    "medication-safety.md": {
        "publisher": "U.S. Food and Drug Administration",
        "source_url": "https://www.fda.gov/drugs/resources-drugs/drug-interactions-what-you-should-know",
        "additional_source_url": "https://www.fda.gov/consumers/consumer-updates/5-medication-safety-tips-older-adults",
    },
}


async def seed_sample_documents(pipeline: RAGPipeline, directory: Path = SAMPLE_DOCS_DIR) -> int:
    """Index every ``*.md``/``*.txt`` file in ``directory``. Returns count ingested."""
    if not directory.is_dir():
        return 0
    count = 0
    for path in sorted(p for p in directory.iterdir() if p.suffix in {".md", ".txt"}):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        metadata: dict[str, Any] = {"seeded": True}
        metadata.update(SAMPLE_SOURCE_METADATA.get(path.name, {}))
        await pipeline.ingest(text, source=path.name, metadata=metadata)
        count += 1
    if count:
        logger.info("sample_documents_seeded count=%d directory=%s", count, directory)
    return count


def _build_state(settings: Settings) -> tuple[AppState, Any]:
    """Construct the model stack, index, and pipeline. Returns state + stack."""
    stack = build_stack(settings)

    # The offline embedder has a known width; a hosted embedder's width is only
    # known after a call, so the index adopts whatever the first insert has.
    dimension = settings.rag_embed_dim if stack.offline else 0
    store = VectorStore.load(settings.vector_store_path, dimension=dimension)
    if not store.is_bound and stack.offline:
        store = VectorStore(dimension=settings.rag_embed_dim)

    pipeline = RAGPipeline(
        chat=stack.chat,
        embedder=stack.embedder,
        store=store,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    state = AppState(
        settings=settings,
        pipeline=pipeline,
        store=store,
        provider=stack.provider.value,
        offline=stack.offline,
    )
    return state, stack


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state, stack = _build_state(settings)
        app.state.app_state = state

        if settings.seed_on_startup and state.store.is_empty():
            await seed_sample_documents(state.pipeline)
            if not settings.is_production:
                state.store.save(settings.vector_store_path)

        logger.info(
            "app_started name=%s env=%s provider=%s chunks=%d",
            settings.app_name,
            settings.app_env.value,
            state.provider,
            len(state.store),
            extra={"chunks": len(state.store)},
        )
        try:
            yield
        finally:
            # Persist the index so the next start does not re-embed everything.
            try:
                state.store.save(settings.vector_store_path)
            except OSError:  # pragma: no cover - read-only filesystem
                logger.exception("index_persist_failed")
            await stack.aclose()
            logger.info("app_stopped")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "A privacy-first health education demo that explains general medical "
            "information in plain language with citations to public sources. It is "
            "not a diagnostic tool and does not provide personal medication guidance."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    if settings.rate_limit_per_minute > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(local_ocr.router)
    app.include_router(chat.router)

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """Serve the single-page, accessible web demo."""
        return FileResponse(Path(__file__).parent / "static" / "index.html", media_type="text/html")

    @app.get("/service-info", tags=["operational"], summary="Non-sensitive service information")
    async def service_info(request: Request) -> JSONResponse:
        state: AppState = app.state.app_state
        return JSONResponse(
            {
                "name": settings.app_name,
                "version": __version__,
                "environment": settings.app_env.value,
                "provider": state.provider,
                "offline_mode": state.offline,
                "indexed_chunks": len(state.store),
                "local_document_ocr_available": local_ocr.is_local_ocr_request(request, state),
                "docs": "/docs",
                "health": "/health",
                "mode": (
                    "read-only education demo"
                    if not settings.allow_document_management
                    else "trusted local document-management mode"
                ),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug and not config.is_production,
    )
