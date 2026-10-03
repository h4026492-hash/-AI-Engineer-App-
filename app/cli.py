"""Command-line interface: seed the index, ask questions, inspect the store.

    ai-app seed                 # index data/sample_docs
    ai-app ask "question"       # one question, with citations
    ai-app docs                 # list what is indexed
    ai-app search "query"       # raw retrieval, no generation
    ai-app run                  # start the API server

The CLI shares the exact same pipeline as the HTTP API, so what you debug here
is what the service does in production.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.main import (
    _build_state,
    seed_sample_documents,
)

logger = get_logger("app.cli")


def _print_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


async def _ask(question: str, top_k: int | None) -> int:
    settings = get_settings()
    state, stack = _build_state(settings)
    try:
        result = await state.pipeline.ask(question, top_k=top_k)
    finally:
        await stack.aclose()

    payload: dict[str, Any] = {
        "question": result.question,
        "answer": result.answer,
        "grounded": result.grounded,
        "latency_ms": result.latency_ms,
        "citations": [
            {
                "source": hit.chunk.source,
                "score": round(hit.score, 4),
                "text": hit.chunk.text[:200] + ("..." if len(hit.chunk.text) > 200 else ""),
            }
            for hit in result.hits
        ],
    }
    if not result.hits:
        # The server seeds on boot; the CLI does not, so an empty index here
        # usually means "you have not run seed yet" rather than "no match".
        payload["hint"] = "Index is empty. Run `ai-app seed` to index data/sample_docs."
    _print_json(payload)
    return 0


async def _search(query: str, top_k: int) -> int:
    settings = get_settings()
    state, stack = _build_state(settings)
    try:
        hits = await state.pipeline.search(query, top_k=top_k)
    finally:
        await stack.aclose()

    if not hits and state.store.is_empty():
        _print_json({"hits": [], "hint": "Index is empty. Run `ai-app seed` first."})
        return 0

    _print_json(
        [
            {"source": hit.chunk.source, "score": round(hit.score, 4), "text": hit.chunk.text}
            for hit in hits
        ]
    )
    return 0


async def _seed() -> int:
    settings = get_settings()
    state, stack = _build_state(settings)
    try:
        count = await seed_sample_documents(state.pipeline, Path("data/sample_docs"))
        state.store.save(settings.vector_store_path)
    finally:
        await stack.aclose()
    _print_json({"seeded_documents": count, "indexed_chunks": len(state.store)})
    return 0


async def _docs() -> int:
    settings = get_settings()
    state, stack = _build_state(settings)
    try:
        documents = state.store.documents()
    finally:
        await stack.aclose()
    _print_json(
        {
            "count": len(documents),
            "chunks": len(state.store),
            "embedder": state.pipeline.embedder_name,
            "documents": [
                {
                    "document_id": doc.document_id,
                    "source": doc.source,
                    "chunks": doc.chunks,
                    "characters": doc.characters,
                }
                for doc in documents
            ],
        }
    )
    return 0


def _run() -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-app", description="RAG service command line interface"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Answer a question against the index")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)

    search = subparsers.add_parser("search", help="Show retrieval results without generating")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)

    subparsers.add_parser("seed", help="Index data/sample_docs into the vector store")
    subparsers.add_parser("docs", help="List indexed documents")
    subparsers.add_parser("run", help="Start the API server")
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    # Logs to stderr so stdout carries only the JSON payload -- otherwise
    # `ai-app ask ... | jq` fails to parse.
    configure_logging(settings.log_level, settings.log_format, stream=sys.stderr)
    args = build_parser().parse_args(argv)

    if args.command == "ask":
        return asyncio.run(_ask(args.question, args.top_k))
    if args.command == "search":
        return asyncio.run(_search(args.query, args.top_k))
    if args.command == "seed":
        return asyncio.run(_seed())
    if args.command == "docs":
        return asyncio.run(_docs())
    if args.command == "run":
        return _run()

    # Unreachable through the parser (subparsers are required, so argparse exits
    # first) but keeps this function total for direct callers and for the day a
    # new subcommand is added to build_parser without a branch here.
    msg = f"unknown command: {args.command}"
    raise ValueError(msg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
