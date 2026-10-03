"""Document chunking.

Chunking is the highest-leverage knob in a RAG system: retrieve the wrong span
and no model can recover. This implementation packs on natural boundaries
(paragraphs, then sentences) and only hard-splits when a single unit exceeds the
budget, so a chunk rarely begins or ends mid-clause.

Consecutive chunks overlap by ``overlap`` characters, which stops an answer that
straddles a boundary from being unrecoverable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass(slots=True)
class Chunk:
    """One retrievable unit of a document."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def characters(self) -> int:
        return len(self.text)


def normalise(text: str) -> str:
    """Normalise line endings and collapse runs of horizontal whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def document_id_for(source: str) -> str:
    """Stable ID for a logical document, derived from its source.

    Keyed on ``source`` alone, deliberately: the source is what identifies a
    document across revisions. Keying on content instead would mean an edited
    document got a new ID, and re-ingesting it would leave the previous version
    in the index alongside the new one -- two documents competing for the same
    questions, one of them stale.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def content_hash(text: str) -> str:
    """Fingerprint of a document's text, used to skip redundant re-embedding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _atomic_units(text: str, chunk_size: int) -> list[str]:
    """Break text into the smallest pieces that still respect boundaries.

    Returns paragraphs, sentences, or hard character slices -- always in
    document order and never longer than ``chunk_size``.
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                # No usable boundary: fall back to a hard slice.
                for start in range(0, len(sentence), chunk_size):
                    piece = sentence[start : start + chunk_size].strip()
                    if piece:
                        units.append(piece)
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    """Last ``overlap`` characters of ``text``, trimmed to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    document_id: str | None = None,
    source: str = "inline",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Raises ``ValueError`` if ``overlap >= chunk_size``, which would loop forever
    and produces no usable chunks anyway.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)
    if overlap < 0:
        msg = f"overlap must be non-negative, got {overlap}"
        raise ValueError(msg)
    if overlap >= chunk_size:
        msg = f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        raise ValueError(msg)

    normalised = normalise(text)
    if not normalised:
        return []

    resolved_id = document_id or document_id_for(source)
    base_metadata = {**(metadata or {}), "content_hash": content_hash(normalised)}

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_length = 0

    def flush() -> None:
        if not buffer:
            return
        body = "\n\n".join(buffer).strip()
        if not body:
            return
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{resolved_id}:{index:04d}",
                document_id=resolved_id,
                source=source,
                text=body,
                index=index,
                metadata=dict(base_metadata),
            )
        )
        buffer.clear()

    for unit in _atomic_units(normalised, chunk_size):
        # +2 accounts for the "\n\n" joiner.
        projected = buffer_length + len(unit) + (2 if buffer else 0)
        if buffer and projected > chunk_size:
            previous_text = "\n\n".join(buffer)
            flush()
            carry = _overlap_tail(previous_text, overlap)
            if carry:
                buffer.append(carry)
                buffer_length = len(carry)
        buffer.append(unit)
        buffer_length = len("\n\n".join(buffer))

    flush()
    return chunks
