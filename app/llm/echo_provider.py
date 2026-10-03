"""Deterministic, offline chat model.

This is not a placeholder that returns a fixed string -- it performs extractive
question answering over the context it is given. That has three practical
consequences, which are the reason it exists:

1. ``make run`` and the test suite work with no API key and no network.
2. Answers are reproducible, so assertions in tests are stable over time.
3. The full retrieval path is genuinely exercised: if citations are wrong, the
   offline answer is wrong too, so integration tests catch retrieval bugs.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from app.core.text import token_set
from app.llm.base import (
    ChatMessage,
    ChatResult,
    ChatUsage,
    approximate_token_count,
    extract_context,
    is_citation_header,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

NO_ANSWER = "I could not find anything relevant in the indexed documents to answer that."


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]


class EchoChatModel:
    """Extractive QA over context. Deterministic; no I/O."""

    def __init__(self, *, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    @property
    def name(self) -> str:
        return "echo"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        question = ""
        system = ""
        for message in reversed(messages):
            if message.role == "user" and not question:
                question = message.content
            if message.role == "system" and not system:
                system = message.content
            if question and system:
                break

        # Only the part after CONTEXT_HEADER is evidence. Taking the whole
        # system message would let the prompt's own examples ("within 30 days
        # [1]") be quoted back as though they came from a document.
        answer = self._answer(question, extract_context(system))
        usage = ChatUsage(
            prompt_tokens=sum(approximate_token_count(m.content) for m in messages),
            completion_tokens=approximate_token_count(answer),
        )
        return ChatResult(text=answer, usage=usage, model=self.name, finish_reason="stop")

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Emit the extractive answer sentence by sentence.

        The answer is computed in full first (it is a selection over the
        context, not an autoregressive decode), then released in sentence-sized
        deltas so streaming endpoints behave the same as with a real model.
        """
        result = await self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        for index, sentence in enumerate(_sentences(result.text) or [result.text]):
            yield sentence if index == 0 else f" {sentence}"

    def _answer(self, question: str, context: str) -> str:
        """Pick the sentences from ``context`` most overlapping with the question."""
        query_terms = token_set(question)
        if not query_terms:
            return NO_ANSWER

        candidates: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(_sentences(context)):
            # Skip provenance headers: a filename in "[1] (source: refunds.md...)"
            # would otherwise match on the word "refund" and be quoted as an answer.
            if is_citation_header(sentence):
                continue
            overlap = len(query_terms & token_set(sentence))
            if overlap:
                # Tie-break on earlier position: documents usually state the
                # main rule before the exceptions.
                candidates.append((overlap, -index, sentence))

        if not candidates:
            return NO_ANSWER

        candidates.sort(reverse=True)
        selected = candidates[: self._max_sentences]
        # Restore document order so the answer reads naturally.
        selected.sort(key=lambda item: -item[1])
        return " ".join(sentence for _, _, sentence in selected)
