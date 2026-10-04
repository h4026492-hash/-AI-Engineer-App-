"""Providers: protocol conformance, offline behaviour, and stack composition."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.config import Provider, Settings
from app.core.errors import ProviderError
from app.llm.base import (
    CONTEXT_HEADER,
    ChatMessage,
    ChatModel,
    ChatResult,
    ChatUsage,
    Embedder,
    approximate_token_count,
    stream_text,
)
from app.llm.echo_provider import NO_ANSWER, EchoChatModel
from app.llm.factory import build_stack
from app.llm.hashing_embedder import HashingEmbedder

# --- HashingEmbedder -------------------------------------------------------


def test_embedder_conforms_to_the_protocol() -> None:
    assert isinstance(HashingEmbedder(dimension=64), Embedder)


def test_embedder_produces_unit_vectors_of_the_requested_width() -> None:
    import numpy as np

    vectors = asyncio.run(HashingEmbedder(dimension=128).embed(["hello world", "goodbye"]))
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == 128
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_embedder_is_deterministic_across_instances() -> None:
    first = asyncio.run(HashingEmbedder(dimension=64).embed(["the quick brown fox"]))
    second = asyncio.run(HashingEmbedder(dimension=64).embed(["the quick brown fox"]))
    assert first == second, "a persisted index depends on embeddings being stable"


def test_embedder_similar_texts_score_higher_than_unrelated_ones() -> None:
    import numpy as np

    embedder = HashingEmbedder(dimension=512)
    query, near, far = asyncio.run(
        embedder.embed(
            [
                "How do I request a refund for my subscription?",
                "You can request a refund for a subscription within 30 days.",
                "The quarterly earnings call is scheduled for Tuesday morning.",
            ]
        )
    )
    assert float(np.dot(query, near)) > float(np.dot(query, far))


def test_embedder_preserves_input_order() -> None:
    embedder = HashingEmbedder(dimension=64)
    texts = ["alpha only", "beta only", "gamma only"]
    vectors = asyncio.run(embedder.embed(texts))
    individually = [asyncio.run(embedder.embed([text]))[0] for text in texts]
    assert vectors == individually


def test_embedder_handles_empty_and_whitespace_input() -> None:
    embedder = HashingEmbedder(dimension=32)
    assert asyncio.run(embedder.embed([])) == []
    vectors = asyncio.run(embedder.embed(["", "   "]))
    assert len(vectors) == 2
    assert all(all(value == 0.0 for value in vector) for vector in vectors)


def test_embedder_rejects_tiny_dimensions() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        HashingEmbedder(dimension=4)


def test_embed_sync_matches_async_embed() -> None:
    import numpy as np

    embedder = HashingEmbedder(dimension=64)
    texts = ["alpha", "beta gamma"]
    sync = embedder.embed_sync(texts)
    async_result = asyncio.run(embedder.embed(texts))
    assert np.allclose(sync, np.array(async_result), atol=1e-6)


# --- EchoChatModel ---------------------------------------------------------


def test_chat_model_conforms_to_the_protocol() -> None:
    assert isinstance(EchoChatModel(), ChatModel)


def _contextual(
    context: str, *, instructions: str = "Answer only from the context."
) -> list[ChatMessage]:
    """Build a transcript in the shape ``app.rag.prompting.build_messages`` emits."""
    return [
        ChatMessage(role="system", content=f"{instructions}\n\n{CONTEXT_HEADER}\n{context}"),
        ChatMessage(role="user", content="How long do I have to request a refund?"),
    ]


def test_echo_model_extracts_from_context() -> None:
    result = asyncio.run(
        EchoChatModel().complete(_contextual("Refunds are issued within 30 days of purchase."))
    )
    assert "Refunds are issued within 30 days of purchase." in result.text
    assert result.model == "echo"
    assert result.usage.total_tokens > 0


def test_echo_model_does_not_return_markdown_headings_as_answer() -> None:
    context = (
        "# Reading medicine labels and asking safer questions\n\n"
        "## What a medicine label can tell you\n\n"
        "A nonprescription medicine label includes active ingredients, uses, warnings, and directions.\n\n"
        "## Where to ask\n\n"
        "Ask a pharmacist about questions on a medicine label."
    )
    messages = [
        ChatMessage(role="system", content=f"Instructions.\n\n{CONTEXT_HEADER}\n{context}"),
        ChatMessage(role="user", content="What should I ask about a medicine label?"),
    ]

    answer = asyncio.run(EchoChatModel().complete(messages)).text

    assert "#" not in answer
    assert "Ask a pharmacist about questions on a medicine label." in answer


def test_echo_model_never_quotes_its_own_instructions() -> None:
    # Regression guard: evidence is only the text after CONTEXT_HEADER. Using the
    # whole system message lets the prompt's own examples be quoted back as
    # though they were sourced facts.
    instructions = 'Cite sources like "within 30 days [1]" when you use them.'
    messages = _contextual("The office closes at five.", instructions=instructions)
    answer = asyncio.run(EchoChatModel().complete(messages)).text

    assert "Cite sources" not in answer
    assert "[1]" not in answer
    assert "Answer only from the context." not in answer


def test_echo_model_declines_when_context_is_irrelevant() -> None:
    messages = [
        ChatMessage(
            role="system",
            content=f"Instructions.\n\n{CONTEXT_HEADER}\nThe earnings call is Tuesday.",
        ),
        ChatMessage(role="user", content="What is the capital of France?"),
    ]
    assert asyncio.run(EchoChatModel().complete(messages)).text == NO_ANSWER


def test_echo_model_declines_without_a_context_header() -> None:
    # No CONTEXT_HEADER means there is no evidence at all; instructions alone
    # must not be treated as a source.
    messages = [
        ChatMessage(role="system", content="Answer only from the context you are given."),
        ChatMessage(role="user", content="What is the refund policy?"),
    ]
    assert asyncio.run(EchoChatModel().complete(messages)).text == NO_ANSWER


def test_echo_model_declines_without_a_system_message() -> None:
    messages = [ChatMessage(role="user", content="What is the refund policy?")]
    assert asyncio.run(EchoChatModel().complete(messages)).text == NO_ANSWER


def test_echo_model_prefers_highest_overlap_and_restores_document_order() -> None:
    context = (
        "Unrelated opening sentence about the weather.\n"
        "Refunds are issued within 30 days of the purchase date.\n"
        "Another unrelated sentence about parking.\n"
        "Annual plan refunds close after 14 days."
    )
    messages = [
        ChatMessage(role="system", content=f"Instructions.\n\n{CONTEXT_HEADER}\n{context}"),
        ChatMessage(role="user", content="refund purchase days"),
    ]
    answer = asyncio.run(EchoChatModel(max_sentences=2).complete(messages)).text
    # Both refund sentences selected, in document order.
    assert answer.index("Refunds are issued") < answer.index("Annual plan refunds")
    assert "parking" not in answer


def test_echo_model_is_deterministic() -> None:
    messages = [
        ChatMessage(
            role="system",
            content=f"Instructions.\n\n{CONTEXT_HEADER}\nSupport answers email within one business day.",
        ),
        ChatMessage(role="user", content="How fast does support answer email?"),
    ]
    model = EchoChatModel()
    assert asyncio.run(model.complete(messages)).text == asyncio.run(model.complete(messages)).text


def test_echo_stream_yields_the_same_text_as_complete() -> None:
    messages = [
        ChatMessage(
            role="system", content=f"Instructions.\n\n{CONTEXT_HEADER}\nKeys rotate every 90 days."
        ),
        ChatMessage(role="user", content="How often do keys rotate?"),
    ]
    model = EchoChatModel()
    complete = asyncio.run(model.complete(messages)).text

    async def collect() -> str:
        return "".join([piece async for piece in model.stream(messages)])

    assert asyncio.run(collect()) == complete


# --- stream_text fallback --------------------------------------------------


class CompleteOnlyModel:
    """A provider that implements `complete` but not `stream`."""

    @property
    def name(self) -> str:
        return "complete-only"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        return ChatResult(text="fallback answer", usage=ChatUsage(10, 2), model=self.name)


def test_stream_text_falls_back_to_a_single_delta_for_non_streaming_models() -> None:
    async def collect() -> list[str]:
        return [
            piece async for piece in stream_text(CompleteOnlyModel(), [ChatMessage("user", "hi")])
        ]

    assert asyncio.run(collect()) == ["fallback answer"]


def test_stream_text_uses_native_streaming_when_available() -> None:
    class StreamingModel(CompleteOnlyModel):
        async def stream(
            self,
            messages: list[ChatMessage],
            *,
            max_tokens: int = 1024,
            temperature: float = 0.0,
        ) -> AsyncIterator[str]:
            for piece in ("one ", "two", ""):
                yield piece

    async def collect() -> list[str]:
        return [piece async for piece in stream_text(StreamingModel(), [ChatMessage("user", "hi")])]

    # Empty deltas are filtered; non-empty ones pass through in order.
    assert asyncio.run(collect()) == ["one ", "two"]


# --- Factory ---------------------------------------------------------------


def test_build_stack_defaults_to_offline() -> None:
    settings = Settings(_env_file=None, llm_provider="echo", rag_embed_dim=64)  # type: ignore[call-arg]
    stack = build_stack(settings)
    assert stack.offline is True
    assert stack.provider is Provider.ECHO
    assert stack.chat.name == "echo"
    assert stack.embedder.dimension == 64


def test_openai_without_a_key_falls_back_to_offline() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider="openai",
        openai_api_key="",
        rag_embed_dim=64,
    )
    stack = build_stack(settings)
    assert stack.offline is True, "must boot rather than crash-loop on a missing key"
    assert stack.provider is Provider.ECHO


def test_openai_with_a_key_builds_real_providers() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider="openai",
        openai_api_key="sk-test-not-a-real-key",
        openai_chat_model="gpt-test",
        openai_embedding_model="embed-test",
        rag_embed_dim=64,
    )
    stack = build_stack(settings)
    assert stack.offline is False
    assert stack.provider is Provider.OPENAI
    assert stack.chat.name == "gpt-test"
    assert stack.embedder.name == "embed-test"


def test_openai_providers_reject_an_empty_key() -> None:
    from app.llm.openai_provider import OpenAIChatModel, OpenAIEmbedder

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        OpenAIChatModel(api_key="", model="gpt-test")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        OpenAIEmbedder(api_key="", model="embed-test")


def test_openai_embedder_dimension_is_unknown_before_first_call() -> None:
    from app.llm.openai_provider import OpenAIEmbedder

    embedder = OpenAIEmbedder(api_key="sk-test", model="embed-test")
    with pytest.raises(RuntimeError, match="unknown until the first embed"):
        _ = embedder.dimension
    assert asyncio.run(embedder.embed([])) == []


def test_stack_aclose_is_safe_for_providers_without_it() -> None:
    settings = Settings(_env_file=None, llm_provider="echo")  # type: ignore[call-arg]
    stack = build_stack(settings)
    asyncio.run(stack.aclose())  # offline providers define no aclose


def test_chat_provider_maps_a_successful_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.llm.openai_provider import OpenAIChatModel

    model = OpenAIChatModel(api_key="sk-test", model="gpt-test")

    async def ok(**kwargs: object) -> SimpleNamespace:
        assert kwargs["model"] == "gpt-test"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="hi there"), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            model="gpt-test",
        )

    monkeypatch.setattr(model._client.chat.completions, "create", ok)  # noqa: SLF001
    result = asyncio.run(model.complete([ChatMessage("user", "hi")]))

    assert result.text == "hi there"
    assert result.usage.total_tokens == 7
    assert result.finish_reason == "stop"
    assert result.model == "gpt-test"


def test_chat_provider_streams_deltas_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.llm.openai_provider import OpenAIChatModel

    model = OpenAIChatModel(api_key="sk-test", model="gpt-test")

    async def ok(**kwargs: object) -> object:
        assert kwargs["stream"] is True

        async def chunks() -> object:
            for piece in ("Hel", "lo", None):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))]
                )

        return chunks()

    monkeypatch.setattr(model._client.chat.completions, "create", ok)  # noqa: SLF001

    async def collect() -> list[str]:
        return [piece async for piece in model.stream([ChatMessage("user", "hi")])]

    assert asyncio.run(collect()) == ["Hel", "lo"], "empty deltas are skipped"


# --- Provider error translation -------------------------------------------
# The valuable behaviour here is that no SDK exception escapes as-is: callers
# get a ProviderError and never have to import the OpenAI package.


def test_chat_provider_translates_upstream_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    from openai import APIConnectionError

    from app.llm.openai_provider import OpenAIChatModel

    model = OpenAIChatModel(api_key="sk-test", model="gpt-test")

    async def boom(*args: object, **kwargs: object) -> None:
        raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))

    monkeypatch.setattr(model._client.chat.completions, "create", boom)  # noqa: SLF001

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(model.complete([ChatMessage("user", "hello")]))
    assert excinfo.value.code == "provider_error"
    assert excinfo.value.details["model"] == "gpt-test"


def test_embedding_provider_translates_upstream_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    from openai import APIConnectionError

    from app.llm.openai_provider import OpenAIEmbedder

    embedder = OpenAIEmbedder(api_key="sk-test", model="embed-test")

    async def boom(*args: object, **kwargs: object) -> None:
        raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))

    monkeypatch.setattr(embedder._client.embeddings, "create", boom)  # noqa: SLF001

    with pytest.raises(ProviderError):
        asyncio.run(embedder.embed(["hello"]))


def test_embedding_provider_rejects_a_short_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.llm.openai_provider import OpenAIEmbedder

    embedder = OpenAIEmbedder(api_key="sk-test", model="embed-test")

    async def short(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    monkeypatch.setattr(embedder._client.embeddings, "create", short)  # noqa: SLF001

    with pytest.raises(ProviderError, match="expected 2 vectors"):
        asyncio.run(embedder.embed(["one", "two"]))


def test_embedding_provider_records_dimension_from_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from app.llm.openai_provider import OpenAIEmbedder

    embedder = OpenAIEmbedder(api_key="sk-test", model="embed-test")

    async def ok(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.5, 0.5]), SimpleNamespace(embedding=[0.0, 1.0])],
            usage=SimpleNamespace(total_tokens=7),
        )

    monkeypatch.setattr(embedder._client.embeddings, "create", ok)  # noqa: SLF001

    vectors = asyncio.run(embedder.embed(["one", "two"]))
    assert len(vectors) == 2
    assert embedder.dimension == 2
    assert embedder.usage_from_response(SimpleNamespace(usage=SimpleNamespace(total_tokens=7))) == 7


# --- Utilities -------------------------------------------------------------


def test_approximate_token_count() -> None:
    assert approximate_token_count("") == 0
    assert approximate_token_count("abc") == 1
    assert approximate_token_count("a" * 400) == 100


def test_provider_error_carries_code_and_details() -> None:
    error = ProviderError("upstream failed", details={"model": "gpt-test"})
    assert error.code == "provider_error"
    assert error.status_code == 502
    assert error.details["model"] == "gpt-test"
