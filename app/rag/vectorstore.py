"""In-memory vector store with cosine similarity and JSONL/npy persistence.

Vectors are stored L2-normalised, so cosine similarity reduces to a dot product
and a single matrix multiplication scores the whole corpus. This is brute force
and linear in corpus size -- perfectly fine to ~100k chunks, which is far beyond
a demo corpus. The interface (``add`` / ``search`` / ``delete_document``) matches
what you would implement against pgvector, Qdrant, or Chroma, so swapping the
backend touches one module.

All mutation happens under a lock: ingestion and search can run concurrently
inside one process.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from app.core.logging import get_logger
from app.rag.chunking import Chunk

logger = get_logger(__name__)

_METADATA_FILE = "chunks.jsonl"
_VECTORS_FILE = "vectors.npy"
_INDEX_FILE = "index.json"


@dataclass(slots=True)
class ScoredChunk:
    """A chunk paired with its similarity score for a given query."""

    chunk: Chunk
    score: float


@dataclass(slots=True)
class DocumentInfo:
    """Bookkeeping for one ingested document."""

    document_id: str
    source: str
    chunks: int
    characters: int
    metadata: dict[str, Any] = field(default_factory=dict)


def normalise_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalise each row. Zero rows are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return cast(NDArray[np.float32], (matrix / safe).astype(np.float32))


class VectorStore:
    """Thread-safe in-memory index of embedded chunks."""

    def __init__(self, dimension: int = 0) -> None:
        if dimension < 0:
            msg = f"dimension must be non-negative, got {dimension}"
            raise ValueError(msg)
        # dimension == 0 means "unbound": the width is adopted from the first
        # batch of vectors inserted. This lets an OpenAI embedder (whose width
        # is only known after a live call) work with no extra configuration.
        self._dimension = dimension
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._vectors: NDArray[np.float32] = np.zeros((0, dimension), dtype=np.float32)
        self._embedder_name = ""

    # --- Introspection -------------------------------------------------

    @property
    def dimension(self) -> int:
        """Vector width. ``0`` until the first insert binds it."""
        return self._dimension

    @property
    def is_bound(self) -> bool:
        return self._dimension > 0

    @property
    def embedder_name(self) -> str:
        return self._embedder_name

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    def is_empty(self) -> bool:
        return len(self) == 0

    def documents(self) -> list[DocumentInfo]:
        """Aggregate the index into per-document summaries."""
        with self._lock:
            grouped: dict[str, DocumentInfo] = {}
            for chunk in self._chunks:
                info = grouped.get(chunk.document_id)
                if info is None:
                    info = DocumentInfo(
                        document_id=chunk.document_id,
                        source=chunk.source,
                        chunks=0,
                        characters=0,
                        metadata=dict(chunk.metadata),
                    )
                    grouped[chunk.document_id] = info
                info.chunks += 1
                info.characters += chunk.characters
            return sorted(grouped.values(), key=lambda item: item.source)

    # --- Mutation ------------------------------------------------------

    def add(
        self,
        chunks: list[Chunk],
        vectors: NDArray[np.float32] | Sequence[Sequence[float]],
        *,
        embedder_name: str = "",
    ) -> int:
        """Insert ``chunks`` with their ``vectors``. Returns rows inserted."""
        if not chunks:
            return 0
        # The Embedder protocol returns nested lists; callers may also pass an
        # ndarray. Coerce once here so both work.
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            msg = f"vectors must be 2-dimensional, got shape {matrix.shape}"
            raise ValueError(msg)
        if len(chunks) != matrix.shape[0]:
            msg = f"chunk/vector count mismatch: {len(chunks)} vs {matrix.shape[0]}"
            raise ValueError(msg)
        if self.is_bound and matrix.shape[1] != self._dimension:
            msg = f"vector dimension {matrix.shape[1]} does not match store dimension {self._dimension}"
            raise ValueError(msg)

        with self._lock:
            normalised = normalise_rows(matrix)
            if not self.is_bound:
                self._dimension = int(matrix.shape[1])
            if self._vectors.shape[0] == 0:
                self._vectors = normalised
            else:
                self._vectors = np.vstack([self._vectors, normalised])
            self._chunks.extend(chunks)
            if embedder_name:
                self._embedder_name = embedder_name
            return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to ``document_id``. Returns rows removed."""
        with self._lock:
            keep = [
                index
                for index, chunk in enumerate(self._chunks)
                if chunk.document_id != document_id
            ]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[index] for index in keep]
                self._vectors = (
                    self._vectors[keep]
                    if keep
                    else np.zeros((0, self._dimension), dtype=np.float32)
                )
            return removed

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._vectors = np.zeros((0, self._dimension), dtype=np.float32)

    def content_hash_for(self, document_id: str) -> str | None:
        """Return the stored content hash for a document, if it is indexed.

        Lets ingestion skip re-embedding a document whose text has not changed,
        which is the common case when a sync job re-reads a whole corpus.
        """
        with self._lock:
            for chunk in self._chunks:
                if chunk.document_id == document_id:
                    return str(chunk.metadata.get("content_hash", "")) or None
            return None

    # --- Query ---------------------------------------------------------

    def search(
        self,
        vector: list[float] | NDArray[np.float32],
        *,
        top_k: int = 4,
        filter_source: str | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        """Return the ``top_k`` most similar chunks, highest score first."""
        with self._lock:
            if not self._chunks or not self.is_bound:
                return []

            query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            if query.shape[1] != self._dimension:
                msg = f"query dimension {query.shape[1]} does not match store dimension {self._dimension}"
                raise ValueError(msg)
            query = normalise_rows(query)

            scores = (self._vectors @ query.T).ravel()

            mask = np.ones(len(self._chunks), dtype=bool)
            if filter_source is not None:
                mask = np.array(
                    [chunk.source == filter_source for chunk in self._chunks], dtype=bool
                )
            mask &= scores >= min_score

            candidate_indices = np.flatnonzero(mask)
            if candidate_indices.size == 0:
                return []

            order = candidate_indices[np.argsort(-scores[candidate_indices], kind="stable")]
            return [
                ScoredChunk(chunk=self._chunks[int(i)], score=float(scores[int(i)]))
                for i in order[:top_k]
            ]

    # --- Persistence ---------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the index to ``directory``. Creates it if needed."""
        path = Path(directory)
        with self._lock:
            path.mkdir(parents=True, exist_ok=True)
            with (path / _METADATA_FILE).open("w", encoding="utf-8") as handle:
                for chunk in self._chunks:
                    handle.write(
                        json.dumps(
                            {
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "source": chunk.source,
                                "index": chunk.index,
                                "text": chunk.text,
                                "metadata": chunk.metadata,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            np.save(path / _VECTORS_FILE, self._vectors)
            (path / _INDEX_FILE).write_text(
                json.dumps(
                    {
                        "dimension": self._dimension,
                        "count": len(self._chunks),
                        "embedder": self._embedder_name,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("vectorstore_saved count=%d path=%s", len(self._chunks), path)
            return path

    @classmethod
    def load(cls, directory: str | Path, *, dimension: int = 0) -> VectorStore:
        """Load a persisted index. Returns an empty store if none exists.

        A persisted index written by a different embedder is not silently
        reused -- dimension mismatch means every similarity score would be
        meaningless, so the index is discarded with a warning. Pass
        ``dimension=0`` to adopt whatever width is on disk.
        """
        path = Path(directory)
        store = cls(dimension=dimension)
        metadata_path = path / _METADATA_FILE
        vectors_path = path / _VECTORS_FILE

        if not metadata_path.exists() or not vectors_path.exists():
            return store

        vectors = np.load(vectors_path).astype(np.float32)
        stored_dimension = vectors.shape[1] if vectors.ndim == 2 else 0
        if vectors.ndim != 2 or (dimension > 0 and stored_dimension != dimension):
            logger.warning(
                "discarding_persisted_index stored_dim=%s expected_dim=%d path=%s",
                stored_dimension or "unknown",
                dimension,
                path,
            )
            return store

        chunks: list[Chunk] = []
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    chunks.append(
                        Chunk(
                            chunk_id=raw["chunk_id"],
                            document_id=raw["document_id"],
                            source=raw["source"],
                            text=raw["text"],
                            index=int(raw["index"]),
                            metadata=raw.get("metadata", {}),
                        )
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A corrupt or partially written index must not take the service
            # down at boot: log it and start empty so ingestion can rebuild.
            logger.warning("persisted_index_unreadable path=%s error=%s; starting empty", path, exc)
            return VectorStore(dimension=dimension)

        if len(chunks) != len(vectors):
            logger.warning(
                "persisted_index_corrupt chunks=%d vectors=%d; starting empty",
                len(chunks),
                len(vectors),
            )
            return store

        store._chunks = chunks
        store._vectors = normalise_rows(vectors)
        store._dimension = stored_dimension
        index_path = path / _INDEX_FILE
        if index_path.exists():
            store._embedder_name = json.loads(index_path.read_text(encoding="utf-8")).get(
                "embedder", ""
            )
        logger.info("vectorstore_loaded count=%d path=%s", len(chunks), path)
        return store
