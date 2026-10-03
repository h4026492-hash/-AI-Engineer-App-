"""OpenAI-compatible chat and embedding providers.

Works against the OpenAI API and against anything speaking the same protocol
(Azure OpenAI, LiteLLM, vLLM, Ollama) by setting ``OPENAI_BASE_URL``.

Two behaviours worth calling out:

* **Retries** -- the SDK retries transport errors and 429/5xx with exponential
  backoff. We configure it explicitly instead of trusting the default.
* **Error translation** -- every SDK exception becomes a
  :class:`app.core.errors.ProviderError`, so callers never import the SDK and
  the API layer can render a consistent 502 without leaking upstream details.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.llm.base import ChatMessage, ChatResult, ChatUsage

logger = get_logger(__name__)


def to_sdk_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """Map internal messages to the SDK's typed parameter union.

    Spelled out per role rather than built generically so each dict literal has
    a single literal ``role`` value, which is what lets mypy match it to the
    right member of the SDK's union.
    """
    out: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message.role == "system":
            out.append({"role": "system", "content": message.content})
        elif message.role == "assistant":
            out.append({"role": "assistant", "content": message.content})
        else:
            out.append({"role": "user", "content": message.content})
    return out


class OpenAIChatModel:
    """Chat completions via the OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            msg = "OPENAI_API_KEY is required when llm_provider='openai'"
            raise ValueError(msg)
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=to_sdk_messages(messages),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except (APIStatusError, APIConnectionError, OpenAIError) as exc:
            raise ProviderError(
                f"Chat completion failed: {type(exc).__name__}",
                details={"model": self._model},
            ) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        return ChatResult(
            text=text,
            usage=ChatUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            ),
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream completion deltas from the OpenAI API."""
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=to_sdk_messages(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta
        except (APIStatusError, APIConnectionError, OpenAIError) as exc:
            raise ProviderError(
                f"Streaming chat completion failed: {type(exc).__name__}",
                details={"model": self._model},
            ) from exc

    async def aclose(self) -> None:
        await self._client.close()


class OpenAIEmbedder:
    """Text embeddings via the OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            msg = "OPENAI_API_KEY is required when llm_provider='openai'"
            raise ValueError(msg)
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            msg = "dimension is unknown until the first embed() call resolves it"
            raise RuntimeError(msg)
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(model=self._model, input=texts)
        except (APIStatusError, APIConnectionError, OpenAIError) as exc:
            raise ProviderError(
                f"Embedding request failed: {type(exc).__name__}",
                details={"model": self._model},
            ) from exc

        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(texts):
            msg = f"expected {len(texts)} vectors, provider returned {len(vectors)}"
            raise ProviderError(msg, details={"model": self._model})

        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def usage_from_response(self, response: Any) -> int:  # pragma: no cover - passthrough
        """Expose token usage when callers want to track embedding spend."""
        return int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)

    async def aclose(self) -> None:
        await self._client.close()
