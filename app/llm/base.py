"""Protocols that every model backend must satisfy.

These are ``typing.Protocol`` rather than ABCs so a provider can be any object
with the right shape -- including a test double defined inline.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]

# Marker separating prompt instructions from retrieved context inside the system
# message. This is part of the model-interface contract rather than a prompt
# detail: a provider that reasons over context (the offline extractive model,
# and any future reranker) needs to know where the evidence starts, so that
# instruction text is never mistaken for a citable fact.
CONTEXT_HEADER = "CONTEXT:"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn of a chat transcript."""

    role: Role
    content: str


@dataclass(slots=True)
class ChatUsage:
    """Token accounting for one completion.

    Counts are approximate unless the provider reports exact figures; they are
    used for cost dashboards and for spotting runaway prompts, not for billing.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ChatResult:
    """A completion plus the usage that produced it."""

    text: str
    usage: ChatUsage = field(default_factory=ChatUsage)
    model: str = ""
    finish_reason: str = "stop"


@runtime_checkable
class ChatModel(Protocol):
    """Anything that can turn a transcript into an answer."""

    @property
    def name(self) -> str:
        """Stable identifier surfaced in responses and logs."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Return a completion for ``messages``."""

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield the completion as text deltas.

        Optional at runtime: :func:`stream_text` falls back to a single delta
        from :meth:`complete` for providers that do not implement it.
        """


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into fixed-length vectors."""

    @property
    def name(self) -> str:
        """Stable identifier, recorded on every stored chunk."""

    @property
    def dimension(self) -> int:
        """Length of the vectors produced."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``. Implementations must preserve input order."""


def extract_context(system_content: str) -> str:
    """Return the retrieved-context portion of a system message.

    The inverse of ``app.rag.prompting.build_messages``. Anything before
    ``CONTEXT_HEADER`` is prompt instruction text and is deliberately excluded,
    so a provider can never quote its own instructions back as though they were
    sourced evidence.
    """
    marker = f"\n{CONTEXT_HEADER}\n"
    position = system_content.find(marker)
    if position == -1:
        return ""
    return system_content[position + len(marker) :].strip()


_CITATION_HEADER = re.compile(r"^\[\d+\] \(source: .*?\)$")


def citation_header(index: int, source: str, chunk_id: str, score: float) -> str:
    """Render the one-line provenance header that precedes a context block.

    Both halves of this contract live together on purpose: the prompt builder
    writes the header, and a provider that selects evidence recognises and skips
    it. Otherwise metadata such as a filename gets quoted back as an answer.
    """
    return f"[{index}] (source: {source}, chunk {chunk_id}, similarity {score:.3f})"


def is_citation_header(line: str) -> bool:
    """True for a provenance header line rather than a line of evidence."""
    return bool(_CITATION_HEADER.match(line.strip()))


def approximate_token_count(text: str) -> int:
    """Rough token estimate for providers that do not report usage.

    Uses the widely-cited ~4 characters per English token heuristic. Good enough
    for observability; do not use for billing.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


async def stream_text(
    model: ChatModel,
    messages: list[ChatMessage],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> AsyncIterator[str]:
    """Stream a completion from ``model``, with a non-streaming fallback.

    Providers that implement ``stream`` are used directly. Those that do not
    get their full completion emitted as one delta, so callers can always rely
    on the same interface.
    """
    stream = getattr(model, "stream", None)
    if stream is not None:
        async for delta in stream(messages, max_tokens=max_tokens, temperature=temperature):
            if delta:
                yield delta
        return

    result = await model.complete(messages, max_tokens=max_tokens, temperature=temperature)
    if result.text:
        yield result.text
