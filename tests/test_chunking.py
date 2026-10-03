"""Chunking: boundary behaviour, overlap, determinism, and edge cases."""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.rag.chunking import Chunk, chunk_text, content_hash, document_id_for, normalise


def test_short_text_yields_single_chunk() -> None:
    chunks = chunk_text("A short policy note.", chunk_size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "A short policy note."
    assert chunks[0].index == 0


def test_empty_and_whitespace_input_yields_no_chunks() -> None:
    assert chunk_text("", chunk_size=400, overlap=50) == []
    assert chunk_text("   \n\t  \n", chunk_size=400, overlap=50) == []


def test_long_text_is_split_into_multiple_chunks() -> None:
    text = "\n\n".join(f"Paragraph number {i} with enough words to matter." for i in range(20))
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 + 20 for chunk in chunks)


def test_chunk_ids_are_ordered_and_unique() -> None:
    text = "\n\n".join(f"Sentence set {i}. More detail here." for i in range(30))
    chunks = chunk_text(text, chunk_size=150, overlap=20)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_overlap_carries_text_between_chunks() -> None:
    text = "Alpha policy. " * 40 + "\n\n" + "Beta policy. " * 40
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1

    def collapse(value: str) -> str:
        # Chunks join units with blank lines; compare on normalised whitespace.
        return " ".join(value.split())

    # The tail of each chunk should reappear at the head of the next.
    for previous, following in pairwise(chunks):
        tail_words = previous.text.split()[-3:]
        assert " ".join(tail_words) in collapse(following.text)


def test_chunks_respect_paragraph_boundaries() -> None:
    text = "First paragraph about refunds.\n\nSecond paragraph about security."
    chunks = chunk_text(text, chunk_size=400, overlap=0)
    assert len(chunks) == 1
    assert "First paragraph" in chunks[0].text and "Second paragraph" in chunks[0].text


def test_oversized_sentence_is_hard_split() -> None:
    text = "word " * 500  # one long run with no sentence terminators
    chunks = chunk_text(text.strip(), chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 115 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        chunk_text("text", chunk_size=100, overlap=100)


def test_invalid_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("text", chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_text("text", chunk_size=100, overlap=-1)


def test_normalise_collapses_whitespace_and_line_endings() -> None:
    assert normalise("a\r\nb\t\tc   ") == "a\nb c"
    assert normalise("  \n  ") == ""


def test_document_id_is_stable_per_source() -> None:
    # Identity follows the source, so an edited document keeps its id and
    # replaces the old version rather than sitting next to it.
    assert document_id_for("a.md") == document_id_for("a.md")
    assert document_id_for("a.md") != document_id_for("b.md")
    assert len(document_id_for("a.md")) == 16


def test_content_hash_tracks_text_changes() -> None:
    assert content_hash("same") == content_hash("same")
    assert content_hash("same") != content_hash("different")
    assert len(content_hash("same")) == 16


def test_metadata_is_attached_to_every_chunk() -> None:
    text = "\n\n".join(f"Section {i} with a reasonable amount of text." for i in range(15))
    chunks = chunk_text(
        text,
        chunk_size=150,
        overlap=20,
        source="guide.md",
        metadata={"team": "support", "version": 3},
    )
    assert len(chunks) > 1
    assert all(chunk.metadata["team"] == "support" for chunk in chunks)
    assert all(chunk.metadata["version"] == 3 for chunk in chunks)
    assert all(chunk.source == "guide.md" for chunk in chunks)


def test_explicit_document_id_is_honoured() -> None:
    chunks = chunk_text("Some text.", chunk_size=400, overlap=0, document_id="custom-id")
    assert chunks[0].document_id == "custom-id"
    assert chunks[0].chunk_id.startswith("custom-id:")


def test_chunk_characters_property_matches_text() -> None:
    chunk = Chunk(chunk_id="a:0", document_id="a", source="a.md", text="hello", index=0)
    assert chunk.characters == 5
