"""The RAG pipeline: ingest, retrieve, ground, answer.

This is the orchestration layer. It owns no model specifics and no HTTP
concerns -- it takes text in and produces grounded answers with citations out.
Both the REST API and the CLI drive it, which keeps their behaviour identical.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.llm.base import ChatMessage, ChatModel, Embedder, approximate_token_count, stream_text
from app.rag.chunking import Chunk, chunk_text, content_hash, document_id_for, normalise
from app.rag.prompting import build_messages
from app.rag.vectorstore import ScoredChunk, VectorStore

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    document_id: str
    source: str
    chunks: int
    characters: int
    replaced_chunks: int = 0
    changed: bool = True


@dataclass(slots=True)
class Answer:
    """A grounded answer plus everything needed to audit or display it."""

    question: str
    answer: str
    hits: list[ScoredChunk] = field(default_factory=list)
    grounded: bool = False
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def citations(self) -> list[dict[str, Any]]:
        """Serialise the retrieved chunks for the API response."""
        return [
            {
                "chunk_id": hit.chunk.chunk_id,
                "document_id": hit.chunk.document_id,
                "source": hit.chunk.source,
                "score": round(hit.score, 4),
                "text": hit.chunk.text,
                "metadata": hit.chunk.metadata,
            }
            for hit in self.hits
        ]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One event from :meth:`RAGPipeline.ask_stream`.

    ``event`` is the SSE event name and ``data`` is JSON-serialisable, so the
    transport layer only has to encode -- it never decides structure.
    """

    event: str
    data: dict[str, Any]


class RAGPipeline:
    """Coordinates chunking, embedding, retrieval, and generation."""

    def __init__(
        self,
        *,
        chat: ChatModel,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 4,
        min_score: float = 0.05,
        max_answer_tokens: int = 1024,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            raise ValueError(msg)
        self._chat = chat
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._min_score = min_score
        self._max_answer_tokens = max_answer_tokens
        # Embedding and generation both touch shared state / external APIs;
        # serialising ingest keeps re-ingestion of one document atomic.
        self._ingest_lock = asyncio.Lock()

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    @property
    def provider_name(self) -> str:
        return self._chat.name

    async def ingest(
        self,
        content: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Chunk, embed, and index one document.

        The source is the document's identity. Re-ingesting a source replaces
        whatever was indexed for it, so an edited document cannot leave a stale
        version behind. If the text is byte-identical to what is already
        indexed, embedding is skipped entirely and ``changed`` is False.
        """
        normalised = normalise(content)
        if not normalised:
            raise IngestionError("Document contains no indexable text.")

        document_id = document_id_for(source)
        fingerprint = content_hash(normalised)

        chunks: list[Chunk] = chunk_text(
            normalised,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
            document_id=document_id,
            source=source,
            metadata=metadata,
        )
        if not chunks:
            raise IngestionError("Document produced no chunks after normalisation.")

        async with self._ingest_lock:
            existing = self._store.content_hash_for(document_id)
            if existing == fingerprint:
                logger.info(
                    "document_unchanged document_id=%s source=%s",
                    document_id,
                    source,
                    extra={"document_id": document_id, "source": source},
                )
                return IngestResult(
                    document_id=document_id,
                    source=source,
                    chunks=len(chunks),
                    characters=len(normalised),
                    replaced_chunks=0,
                    changed=False,
                )

            replaced = self._store.delete_document(document_id)
            vectors = await self._embedder.embed([chunk.text for chunk in chunks])
            inserted = self._store.add(chunks, vectors, embedder_name=self._embedder.name)

        logger.info(
            "document_ingested document_id=%s source=%s chunks=%d replaced=%d",
            document_id,
            source,
            inserted,
            replaced,
            extra={"document_id": document_id, "source": source, "replaced": replaced},
        )
        return IngestResult(
            document_id=document_id,
            source=source,
            chunks=inserted,
            characters=len(normalised),
            replaced_chunks=replaced,
            changed=True,
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter_source: str | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve the most relevant chunks for ``query``."""
        if not query.strip():
            return []
        if self._store.is_empty():
            return []

        query_vector = (await self._embedder.embed([query]))[0]
        return self._store.search(
            query_vector,
            top_k=top_k or self._top_k,
            filter_source=filter_source,
            min_score=self._min_score,
        )

    async def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filter_source: str | None = None,
    ) -> Answer:
        """Retrieve relevant context and answer ``question`` from it."""
        started = time.perf_counter()
        hits = await self.search(question, top_k=top_k, filter_source=filter_source)
        transcript = build_messages(question, hits)
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in transcript]  # type: ignore[arg-type]

        result = await self._chat.complete(
            messages,
            max_tokens=self._max_answer_tokens,
            temperature=0.0,
        )

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "question_answered grounded=%s hits=%d latency_ms=%.1f provider=%s",
            bool(hits),
            len(hits),
            latency_ms,
            self._chat.name,
            extra={"grounded": bool(hits), "hits": len(hits), "provider": self._chat.name},
        )
        return Answer(
            question=question,
            answer=result.text,
            hits=hits,
            grounded=bool(hits),
            latency_ms=round(latency_ms, 2),
            tokens_in=result.usage.prompt_tokens,
            tokens_out=result.usage.completion_tokens,
        )

    async def ask_stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filter_source: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Same as :meth:`ask`, but emits progress events as they become known.

        Event order is guaranteed: ``sources`` (so a client can render citations
        before the first word), zero or more ``delta``, then ``done``. The
        ``delta`` pieces are the provider's real streaming deltas -- this is not
        a completed answer chopped up client-side.
        """
        started = time.perf_counter()
        hits = await self.search(question, top_k=top_k, filter_source=filter_source)
        transcript = build_messages(question, hits)
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in transcript]  # type: ignore[arg-type]

        yield StreamEvent(
            event="sources",
            data={
                "grounded": bool(hits),
                "citations": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "document_id": hit.chunk.document_id,
                        "source": hit.chunk.source,
                        "score": round(hit.score, 4),
                    }
                    for hit in hits
                ],
            },
        )

        pieces: list[str] = []
        async for delta in stream_text(self._chat, messages):
            pieces.append(delta)
            yield StreamEvent(event="delta", data={"text": delta})

        answer = "".join(pieces)
        prompt_tokens = sum(approximate_token_count(m.content) for m in messages)
        yield StreamEvent(
            event="done",
            data={
                "answer": answer,
                "grounded": bool(hits),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "provider": self._chat.name,
                "tokens_in": prompt_tokens,
                "tokens_out": approximate_token_count(answer),
            },
        )
