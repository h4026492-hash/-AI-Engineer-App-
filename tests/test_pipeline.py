"""Pipeline: ingestion, retrieval, grounding, citations, streaming, persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.core.errors import IngestionError
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore
from tests.conftest import SAMPLE_REFUNDS


def test_ingest_returns_counts_and_indexes_the_document(pipeline: RAGPipeline) -> None:
    result = asyncio.run(pipeline.ingest(SAMPLE_REFUNDS, source="refunds.md"))
    assert result.chunks >= 1
    assert result.characters == len(SAMPLE_REFUNDS.strip())
    assert result.replaced_chunks == 0
    assert len(pipeline.store) == result.chunks
    assert [doc.source for doc in pipeline.store.documents()] == ["refunds.md"]


def test_reingesting_identical_content_is_a_no_op(pipeline: RAGPipeline) -> None:
    first = asyncio.run(pipeline.ingest(SAMPLE_REFUNDS, source="refunds.md"))
    second = asyncio.run(pipeline.ingest(SAMPLE_REFUNDS, source="refunds.md"))

    assert second.document_id == first.document_id
    assert second.changed is False, "unchanged text must be detected, not re-embedded"
    assert second.replaced_chunks == 0
    assert len(pipeline.store) == first.chunks, "re-ingest must not accumulate chunks"


def test_reingesting_changed_content_updates_the_document(pipeline: RAGPipeline) -> None:
    asyncio.run(pipeline.ingest(SAMPLE_REFUNDS, source="refunds.md"))
    before = len(pipeline.store)

    result = asyncio.run(
        pipeline.ingest(
            SAMPLE_REFUNDS + "\n\nDisputes are escalated within 48 hours.", source="refunds.md"
        )
    )

    assert result.changed is True
    assert result.replaced_chunks == before
    assert len(pipeline.store.documents()) == 1, "an edit must not leave a stale copy behind"

    hits = asyncio.run(pipeline.search("How are disputes escalated?", top_k=5))
    assert any("Disputes are escalated within 48 hours." in hit.chunk.text for hit in hits)


def test_ingest_rejects_blank_content(pipeline: RAGPipeline) -> None:
    with pytest.raises(IngestionError, match="no indexable text"):
        asyncio.run(pipeline.ingest("   \n\t ", source="blank.md"))


def test_search_ranks_the_relevant_document_first(seeded_pipeline: RAGPipeline) -> None:
    hits = asyncio.run(seeded_pipeline.search("How are refunds handled?"))
    assert hits, "expected at least one hit"
    assert hits[0].chunk.source == "refunds.md"

    other = asyncio.run(seeded_pipeline.search("Is data encrypted at rest?"))
    assert other[0].chunk.source == "security.md"


def test_search_respects_top_k(seeded_pipeline: RAGPipeline) -> None:
    assert len(asyncio.run(seeded_pipeline.search("encrypted data", top_k=1))) == 1


def test_search_filters_by_source(seeded_pipeline: RAGPipeline) -> None:
    hits = asyncio.run(seeded_pipeline.search("encryption", filter_source="security.md"))
    assert hits and all(hit.chunk.source == "security.md" for hit in hits)

    missing = asyncio.run(seeded_pipeline.search("encryption", filter_source="nope.md"))
    assert missing == []


def test_search_on_empty_index_returns_empty(pipeline: RAGPipeline) -> None:
    assert asyncio.run(pipeline.search("anything")) == []


def test_search_on_blank_query_returns_empty(seeded_pipeline: RAGPipeline) -> None:
    assert asyncio.run(seeded_pipeline.search("   ")) == []


def test_ask_returns_a_grounded_answer_with_citations(seeded_pipeline: RAGPipeline) -> None:
    result = asyncio.run(seeded_pipeline.ask("How long do I have to request a refund?"))

    assert result.grounded is True
    assert result.hits, "a grounded answer must carry the chunks it used"
    assert result.hits[0].chunk.source == "refunds.md"
    assert "30 days" in result.answer
    # The "30 days" must come from the retrieved document, not from the prompt
    # template's own example. Guard against the answer quoting its instructions.
    assert "precise assistant" not in result.answer
    assert "[1]" not in result.answer
    assert result.tokens_in > 0
    assert result.latency_ms >= 0.0


def test_citation_numbers_line_up_with_the_citations_array(seeded_pipeline: RAGPipeline) -> None:
    result = asyncio.run(seeded_pipeline.ask("How long do I have to request a refund?"))
    citations = result.citations()

    assert len(citations) == len(result.hits)
    # Same order, same ids -- so "[2]" in an answer means citations[1].
    assert [c["chunk_id"] for c in citations] == [hit.chunk.chunk_id for hit in result.hits]
    assert citations[0]["source"] == result.hits[0].chunk.source
    assert 0.0 <= citations[0]["score"] <= 1.0


def test_ask_on_empty_index_is_not_grounded(pipeline: RAGPipeline) -> None:
    result = asyncio.run(pipeline.ask("What is the refund policy?"))

    assert result.grounded is False
    assert result.hits == []
    assert result.citations() == []
    assert result.answer, "must still return a usable message"


def test_grounded_flag_tracks_whether_evidence_was_retrieved(seeded_pipeline: RAGPipeline) -> None:
    # `grounded` is an objective statement about retrieval, not a model opinion:
    # it must be true exactly when chunks were handed to the model.
    for question in ("How long do I have to request a refund?", "What is the capital of France?"):
        result = asyncio.run(seeded_pipeline.ask(question))
        assert result.grounded is bool(result.hits)


def test_ask_respects_filter_source(seeded_pipeline: RAGPipeline) -> None:
    result = asyncio.run(seeded_pipeline.ask("encryption", filter_source="security.md"))
    assert all(hit.chunk.source == "security.md" for hit in result.hits)


def test_ask_stream_emits_sources_then_deltas_then_done(seeded_pipeline: RAGPipeline) -> None:
    async def collect() -> list[tuple[str, dict[str, object]]]:
        return [
            (event.event, event.data)
            async for event in seeded_pipeline.ask_stream("How long do I have to request a refund?")
        ]

    events = asyncio.run(collect())
    names = [name for name, _ in events]

    assert names[0] == "sources"
    assert names[-1] == "done"
    assert "delta" in names
    # Order is strict: no delta before sources, no delta after done.
    assert names == ["sources", *(["delta"] * names.count("delta")), "done"]

    sources = events[0][1]
    assert sources["grounded"] is True
    assert isinstance(sources["citations"], list)

    streamed = "".join(str(data["text"]) for name, data in events if name == "delta")
    done = events[-1][1]
    assert str(done["answer"]) == streamed
    assert done["grounded"] is True
    assert str(done["provider"]) == "echo"


def test_ask_stream_matches_ask(seeded_pipeline: RAGPipeline) -> None:
    question = "How long do I have to request a refund?"
    batched = asyncio.run(seeded_pipeline.ask(question))

    async def stream_answer() -> str:
        parts: list[str] = []
        async for event in seeded_pipeline.ask_stream(question):
            if event.event == "delta":
                parts.append(str(event.data["text"]))
        return "".join(parts)

    assert asyncio.run(stream_answer()) == batched.answer


def test_ask_stream_on_empty_index_reports_no_sources(pipeline: RAGPipeline) -> None:
    async def collect() -> list[tuple[str, dict[str, object]]]:
        return [(e.event, e.data) async for e in pipeline.ask_stream("anything at all")]

    events = asyncio.run(collect())
    assert events[0][0] == "sources"
    assert events[0][1]["grounded"] is False
    assert events[0][1]["citations"] == []


def test_pipeline_rejects_overlap_at_or_above_chunk_size(store: VectorStore) -> None:
    from app.llm.echo_provider import EchoChatModel
    from app.llm.hashing_embedder import HashingEmbedder

    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        RAGPipeline(
            chat=EchoChatModel(),
            embedder=HashingEmbedder(dimension=64),
            store=store,
            chunk_size=100,
            chunk_overlap=100,
        )


def test_pipeline_exposes_provider_and_embedder_names(seeded_pipeline: RAGPipeline) -> None:
    assert seeded_pipeline.provider_name == "echo"
    assert seeded_pipeline.embedder_name.startswith("hashing-")


def test_index_survives_a_save_and_reload(seeded_pipeline: RAGPipeline, tmp_path: Path) -> None:
    path = tmp_path / "store"
    seeded_pipeline.store.save(path)

    restored = VectorStore.load(path, dimension=seeded_pipeline.store.dimension)
    assert len(restored) == len(seeded_pipeline.store)

    question = "How is data encrypted at rest?"
    original = asyncio.run(seeded_pipeline.search(question))
    after = restored.search(
        # Re-embed the query with the same embedder the index was built with,
        # and apply the same relevance floor the pipeline uses.
        asyncio.run(seeded_pipeline._embedder.embed([question]))[0],  # noqa: SLF001
        top_k=3,
        min_score=0.05,
    )
    assert [hit.chunk.chunk_id for hit in after] == [hit.chunk.chunk_id for hit in original]
    assert after[0].chunk.text == original[0].chunk.text
