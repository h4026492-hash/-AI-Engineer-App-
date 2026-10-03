"""Prompt construction.

Prompts live in one module so they can be versioned, reviewed in a diff, and
A/B tested. The grounding rules below are the contract that keeps the model from
inventing facts: answer only from the supplied context, cite it, and say so
plainly when the context does not cover the question.
"""

from __future__ import annotations

from app.llm.base import CONTEXT_HEADER, citation_header
from app.rag.vectorstore import ScoredChunk

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are a precise assistant that answers questions using only the CONTEXT below.

Rules:
1. Base every claim on the CONTEXT. Never use outside knowledge to fill gaps.
2. Cite the sources you used with bracketed numbers, e.g. "within 30 days [1]".
3. If the CONTEXT does not contain the answer, reply exactly:
   "I could not find that information in the provided documents."
4. Do not speculate, hedge at length, or pad the answer.
5. If sources disagree, say so and attribute each position to its source.

Answer in the same language as the question.
"""

NO_CONTEXT_SYSTEM_PROMPT = """\
You are a precise assistant. No documents have been indexed yet, so you have no
grounding context. Reply exactly:
"The knowledge base is empty. Ingest a document first, then ask again."
"""


def format_context(hits: list[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered, citation-friendly block.

    Numbers here are what the model is told to cite, so they must line up with
    the ``citations`` array returned to the caller (1-based, same order).
    """
    if not hits:
        return ""
    blocks = [
        citation_header(position, hit.chunk.source, hit.chunk.chunk_id, hit.score)
        + f"\n{hit.chunk.text}"
        for position, hit in enumerate(hits, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[ScoredChunk]) -> list[dict[str, str]]:
    """Assemble the transcript sent to the chat model.

    Retrieved context lives in the system message under ``CONTEXT_HEADER``, and
    the user message carries the question alone. Keeping them apart matters: a
    provider that selects evidence from context (the offline extractive model)
    must not treat the instruction text as a citable fact, and the header is how
    it tells the two apart.

    Returns plain dicts (role/content) so this module stays independent of any
    provider SDK; :class:`app.rag.pipeline.RAGPipeline` maps them to
    :class:`app.llm.base.ChatMessage`.
    """
    context = format_context(hits)
    if not context:
        system = NO_CONTEXT_SYSTEM_PROMPT
    else:
        system = f"{SYSTEM_PROMPT}\n\n{CONTEXT_HEADER}\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]
