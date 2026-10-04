"""Family MedGuard: a source-grounded health-education demo.

The package is organised by layer rather than by feature, so each layer can be
tested and swapped independently:

    app.config   -- settings, loaded once from env / .env
    app.core     -- cross-cutting concerns: logging, errors, middleware
    app.schemas  -- request/response models (the public API contract)
    app.llm      -- chat + embedding providers behind a protocol
    app.rag      -- chunking, embeddings, vector store, retrieval pipeline
    app.api      -- FastAPI routers and dependency wiring
    app.main     -- application factory
"""

from importlib.metadata import PackageNotFoundError, version

try:  # installed as a package
    __version__ = version("ai-engineer-app")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.2.0"

__all__ = ["__version__"]
