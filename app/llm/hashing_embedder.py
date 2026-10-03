"""Feature-hashing embedder.

Maps text to a fixed-dimension unit vector using signed hashing over word
unigrams and bigrams. It captures lexical overlap, which is enough for keyword
style retrieval, needs no model download, and -- crucially -- is stable across
processes and machines, so a persisted index stays valid.

It is not a semantic embedder. Synonyms and paraphrase will not match. Swap in
:class:`app.llm.openai_provider.OpenAIEmbedder` (or any real model) when you
need semantic recall; the :class:`app.rag.vectorstore.VectorStore` does not care
which produced the vectors, as long as the dimension matches.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from app.core.text import tokenize


def _terms(text: str) -> list[str]:
    """Stemmed unigrams plus stemmed bigrams.

    Bigrams are what let the embedder distinguish "30 day refund" from a chunk
    that merely mentions "refund" somewhere unrelated.
    """
    words = tokenize(text)
    bigrams = [f"{a}_{b}" for a, b in pairwise(words)]
    return words + bigrams


class HashingEmbedder:
    """Deterministic hashed-bag-of-n-grams embedder."""

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 8:
            msg = f"dimension must be at least 8, got {dimension}"
            raise ValueError(msg)
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text).tolist() for text in texts]

    def embed_sync(self, texts: list[str]) -> NDArray[np.float32]:
        """Synchronous batch embedding. Used by bulk ingestion and tests."""
        return np.stack([self._embed_one(text) for text in texts], dtype=np.float32)

    def _embed_one(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self._dimension, dtype=np.float32)
        terms = _terms(text)
        if not terms:
            return vector

        for term in terms:
            # Two hashes from one digest: one picks the bucket, one picks the
            # sign. Signed hashing cancels out a large share of collision bias.
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimension
            sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector
