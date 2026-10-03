"""The prompt <-> provider seam.

This is the contract that silently broke once already: ``build_messages`` decides
where retrieved context lives, and a provider that selects evidence from context
has to agree on where it starts. Pinning it here means the two cannot drift.
"""

from __future__ import annotations

from app.llm.base import CONTEXT_HEADER, ChatMessage, extract_context
from app.rag.chunking import Chunk
from app.rag.prompting import (
    NO_CONTEXT_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    format_context,
)
from app.rag.vectorstore import ScoredChunk


def make_hit(text: str, source: str = "policy.md", score: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id="abc:0000", document_id="abc", source=source, text=text, index=0),
        score=score,
    )


def test_context_header_is_the_shared_marker() -> None:
    assert CONTEXT_HEADER == "CONTEXT:"


def test_prompt_is_versioned() -> None:
    # Bumping this is how a prompt change becomes visible in review and in logs.
    assert PROMPT_VERSION == "v1"


def test_format_context_numbers_chunks_from_one() -> None:
    rendered = format_context([make_hit("First."), make_hit("Second.", source="other.md")])
    assert rendered.index("[1]") < rendered.index("[2]")
    assert "source: policy.md" in rendered
    assert "source: other.md" in rendered
    assert "First." in rendered and "Second." in rendered


def test_format_context_of_no_hits_is_empty() -> None:
    assert format_context([]) == ""


def test_context_lives_in_the_system_message_not_the_user_message() -> None:
    messages = build_messages(
        "What is the refund window?", [make_hit("Refunds close after 30 days.")]
    )

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "What is the refund window?"
    assert "Refunds close after 30 days." in messages[0]["content"]
    assert "Refunds close after 30 days." not in messages[1]["content"]


def test_extract_context_round_trips_the_evidence() -> None:
    messages = build_messages("question", [make_hit("Refunds close after 30 days.")])
    assert extract_context(messages[0]["content"]) == (
        "[1] (source: policy.md, chunk abc:0000, similarity 0.800)\nRefunds close after 30 days."
    )


def test_extract_context_excludes_the_instructions() -> None:
    messages = build_messages("question", [make_hit("Evidence here.")])
    context = extract_context(messages[0]["content"])

    assert SYSTEM_PROMPT not in context
    assert "precise assistant" not in context


def test_extract_context_without_a_header_is_empty() -> None:
    assert extract_context("just instructions, no evidence") == ""
    assert extract_context(NO_CONTEXT_SYSTEM_PROMPT) == ""


def test_no_hits_produces_the_no_context_prompt() -> None:
    messages = build_messages("What is the refund window?", [])
    assert messages[0]["content"] == NO_CONTEXT_SYSTEM_PROMPT
    assert CONTEXT_HEADER not in messages[0]["content"]


def test_citation_numbers_match_the_hits_order() -> None:
    hits = [
        make_hit("Alpha.", source="a.md", score=0.9),
        make_hit("Beta.", source="b.md", score=0.4),
    ]
    rendered = format_context(hits)

    # The number a model cites must line up with the citations array returned to
    # the caller, which is built from this same ordered list.
    assert rendered.split("[")[1].startswith("1] (source: a.md")
    assert rendered.split("[")[2].startswith("2] (source: b.md")


def test_built_messages_are_provider_agnostic_dicts() -> None:
    messages = build_messages("q", [make_hit("Evidence.")])
    mapped = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]  # type: ignore[arg-type]
    assert mapped[0].role == "system"
    assert mapped[1].content == "q"
