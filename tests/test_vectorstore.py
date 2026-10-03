"""Vector store: ranking, filtering, deletion, and persistence round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.rag.chunking import Chunk
from app.rag.vectorstore import VectorStore, normalise_rows

DIM = 8


def make_chunk(
    index: int, text: str = "text", document_id: str = "doc-1", source: str = "a.md"
) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:{index:04d}",
        document_id=document_id,
        source=source,
        text=text,
        index=index,
        metadata={"i": index},
    )


def vec(*values: float) -> np.ndarray:
    array = np.zeros(DIM, dtype=np.float32)
    for position, value in enumerate(values):
        array[position] = value
    return array


@pytest.fixture
def populated() -> VectorStore:
    store = VectorStore(dimension=DIM)
    store.add(
        [make_chunk(0, "alpha"), make_chunk(1, "beta", document_id="doc-2", source="b.md")],
        np.stack([vec(1.0), vec(0.0, 1.0)]),
        embedder_name="test-embedder",
    )
    return store


def test_empty_store_reports_zero_and_returns_no_hits(populated: VectorStore) -> None:
    assert len(VectorStore(dimension=DIM)) == 0
    assert VectorStore(dimension=DIM).is_empty()
    assert len(populated) == 2
    assert not populated.is_empty()


def test_search_ranks_by_cosine_similarity() -> None:
    store = VectorStore(dimension=DIM)
    store.add(
        [make_chunk(0, "a"), make_chunk(1, "b"), make_chunk(2, "c")],
        np.stack([vec(1.0), vec(1.0, 1.0), vec(0.0, 1.0)]),
    )
    hits = store.search(vec(1.0), top_k=3)
    assert [hit.chunk.text for hit in hits] == ["a", "b", "c"]
    # Exact match first, orthogonal last.
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[-1].score == pytest.approx(0.0, abs=1e-6)


def test_top_k_limits_result_count() -> None:
    store = VectorStore(dimension=DIM)
    store.add([make_chunk(i) for i in range(5)], np.stack([vec(1.0) for _ in range(5)]))
    assert len(store.search(vec(1.0), top_k=2)) == 2


def test_min_score_filters_weak_matches() -> None:
    store = VectorStore(dimension=DIM)
    store.add([make_chunk(0), make_chunk(1)], np.stack([vec(1.0), vec(0.0, 1.0)]))
    hits = store.search(vec(1.0), top_k=5, min_score=0.5)
    assert len(hits) == 1
    assert hits[0].chunk.index == 0


def test_filter_source_restricts_the_corpus(populated: VectorStore) -> None:
    hits = populated.search(vec(1.0, 1.0), top_k=5, filter_source="b.md")
    assert len(hits) == 1
    assert hits[0].chunk.source == "b.md"


def test_filter_source_with_no_match_returns_empty(populated: VectorStore) -> None:
    assert populated.search(vec(1.0), top_k=5, filter_source="missing.md") == []


def test_search_on_empty_store_returns_empty_list() -> None:
    assert VectorStore(dimension=DIM).search(vec(1.0), top_k=3) == []


def test_delete_document_removes_only_that_document(populated: VectorStore) -> None:
    assert populated.delete_document("doc-2") == 1
    assert len(populated) == 1
    assert populated.delete_document("doc-2") == 0
    assert [doc.document_id for doc in populated.documents()] == ["doc-1"]


def test_clear_empties_the_store(populated: VectorStore) -> None:
    populated.clear()
    assert len(populated) == 0
    assert populated.search(vec(1.0), top_k=3) == []


def test_documents_aggregates_chunk_counts() -> None:
    store = VectorStore(dimension=DIM)
    store.add(
        [
            make_chunk(0, document_id="d1"),
            make_chunk(1, document_id="d1"),
            make_chunk(0, document_id="d2"),
        ],
        np.stack([vec(1.0), vec(0.0, 1.0), vec(1.0, 1.0)]),
    )
    documents = {doc.document_id: doc for doc in store.documents()}
    assert documents["d1"].chunks == 2
    assert documents["d2"].chunks == 1


def test_dimension_mismatch_on_add_is_rejected() -> None:
    store = VectorStore(dimension=DIM)
    with pytest.raises(ValueError, match="does not match store dimension"):
        store.add([make_chunk(0)], np.zeros((1, DIM + 1), dtype=np.float32))


def test_chunk_vector_count_mismatch_is_rejected() -> None:
    store = VectorStore(dimension=DIM)
    with pytest.raises(ValueError, match="count mismatch"):
        store.add([make_chunk(0), make_chunk(1)], np.zeros((1, DIM), dtype=np.float32))


def test_query_dimension_mismatch_is_rejected(populated: VectorStore) -> None:
    with pytest.raises(ValueError, match="query dimension"):
        populated.search([1.0, 2.0], top_k=1)


def test_unbound_store_adopts_width_from_first_insert() -> None:
    store = VectorStore()
    assert not store.is_bound
    assert store.search([1.0, 0.0, 0.0], top_k=1) == []

    store.add([make_chunk(0)], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert store.is_bound
    assert store.dimension == 3
    assert len(store.search([1.0, 0.0, 0.0], top_k=1)) == 1


def test_unbound_store_rejects_inconsistent_second_batch() -> None:
    store = VectorStore()
    store.add([make_chunk(0)], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="does not match store dimension"):
        store.add([make_chunk(1)], np.array([[1.0, 0.0]], dtype=np.float32))


def test_save_and_load_round_trip(tmp_path: Path, populated: VectorStore) -> None:
    path = tmp_path / "index"
    populated.save(path)

    restored = VectorStore.load(path, dimension=DIM)
    assert len(restored) == len(populated)
    assert restored.embedder_name == "test-embedder"
    assert [doc.source for doc in restored.documents()] == [
        doc.source for doc in populated.documents()
    ]

    original = populated.search(vec(1.0), top_k=2)
    after = restored.search(vec(1.0), top_k=2)
    assert [hit.chunk.chunk_id for hit in after] == [hit.chunk.chunk_id for hit in original]
    assert after[0].score == pytest.approx(original[0].score, abs=1e-6)
    assert after[0].chunk.text == original[0].chunk.text


def test_load_missing_directory_returns_empty_store(tmp_path: Path) -> None:
    store = VectorStore.load(tmp_path / "nope", dimension=DIM)
    assert len(store) == 0


def test_load_with_wrong_dimension_discards_index(tmp_path: Path, populated: VectorStore) -> None:
    path = tmp_path / "index"
    populated.save(path)
    # An index from a different embedder would produce meaningless scores, so it
    # must be discarded rather than truncated or padded.
    assert len(VectorStore.load(path, dimension=DIM * 2)) == 0


def test_load_without_expected_dimension_adopts_stored_width(
    tmp_path: Path, populated: VectorStore
) -> None:
    path = tmp_path / "index"
    populated.save(path)
    restored = VectorStore.load(path)
    assert restored.dimension == DIM
    assert len(restored) == 2


def test_corrupted_index_is_discarded(tmp_path: Path, populated: VectorStore) -> None:
    path = tmp_path / "index"
    populated.save(path)
    (path / "chunks.jsonl").write_text('{"chunk_id": "x:0"}\n', encoding="utf-8")
    assert len(VectorStore.load(path, dimension=DIM)) == 0


def test_normalise_rows_leaves_zero_rows_alone() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalised = normalise_rows(matrix)
    assert normalised[0] == pytest.approx([0.6, 0.8], abs=1e-6)
    assert normalised[1] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_vectors_are_normalised_on_insert() -> None:
    store = VectorStore(dimension=DIM)
    store.add([make_chunk(0)], np.stack([vec(3.0, 4.0)]))
    # Cosine similarity with the same direction must be exactly 1.
    assert store.search(vec(3.0, 4.0), top_k=1)[0].score == pytest.approx(1.0, abs=1e-6)


def test_negative_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        VectorStore(dimension=-1)
