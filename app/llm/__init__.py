"""Model providers: chat completion and embedding behind narrow protocols.

Adding a backend (Anthropic, Gemini, a local vLLM server) means writing one
module that satisfies :class:`ChatModel` and/or :class:`Embedder`, then
registering it in :func:`app.llm.factory.build_stack`. Nothing in the RAG or
API layers changes.
"""

from app.llm.base import ChatMessage, ChatModel, ChatUsage, Embedder, Role
from app.llm.echo_provider import EchoChatModel
from app.llm.factory import ModelStack, build_stack
from app.llm.hashing_embedder import HashingEmbedder

__all__ = [
    "ChatMessage",
    "ChatModel",
    "ChatUsage",
    "EchoChatModel",
    "Embedder",
    "HashingEmbedder",
    "ModelStack",
    "Role",
    "build_stack",
]
