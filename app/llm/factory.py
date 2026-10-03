"""Composition root for the model layer.

``build_stack`` is the only place that knows which concrete provider classes
exist. Everything downstream receives a :class:`ModelStack` and programs against
the protocols, which is what makes provider swapping a config change rather
than a refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Provider, Settings
from app.core.logging import get_logger
from app.llm.base import ChatModel, Embedder
from app.llm.echo_provider import EchoChatModel
from app.llm.hashing_embedder import HashingEmbedder

logger = get_logger(__name__)


@dataclass(slots=True)
class ModelStack:
    """The chat model and embedder the service runs with."""

    chat: ChatModel
    embedder: Embedder
    provider: Provider
    offline: bool

    async def aclose(self) -> None:
        """Release provider resources on shutdown."""
        for component in (self.chat, self.embedder):
            close = getattr(component, "aclose", None)
            if close is not None:
                await close()


def build_stack(settings: Settings | None = None) -> ModelStack:
    """Instantiate providers according to ``settings.llm_provider``.

    Falls back to the offline stack when ``openai`` is selected but no API key
    is configured, so a misconfigured deployment still boots and serves traffic
    instead of crash-looping. The fallback is logged loudly.
    """
    from app.config import get_settings

    settings = settings or get_settings()

    if settings.llm_provider is Provider.OPENAI and settings.openai_api_key:
        # Imported here so the offline stack never pays the import cost.
        from app.llm.openai_provider import OpenAIChatModel, OpenAIEmbedder

        logger.info(
            "model_stack provider=openai chat=%s embedder=%s",
            settings.openai_chat_model,
            settings.openai_embedding_model,
        )
        return ModelStack(
            chat=OpenAIChatModel(
                api_key=settings.openai_api_key,
                model=settings.openai_chat_model,
                base_url=settings.openai_base_url,
                timeout=settings.request_timeout_seconds,
            ),
            embedder=OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                base_url=settings.openai_base_url,
                timeout=settings.request_timeout_seconds,
            ),
            provider=Provider.OPENAI,
            offline=False,
        )

    if settings.llm_provider is Provider.OPENAI:
        logger.warning(
            "llm_provider=openai but OPENAI_API_KEY is not set; "
            "falling back to the deterministic offline stack"
        )

    logger.info("model_stack provider=echo embedder=hashing-%d (offline)", settings.rag_embed_dim)
    return ModelStack(
        chat=EchoChatModel(),
        embedder=HashingEmbedder(dimension=settings.rag_embed_dim),
        provider=Provider.ECHO,
        offline=True,
    )
