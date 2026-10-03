"""Retrieval-augmented generation: chunking, indexing, retrieval, grounding."""

from app.rag.chunking import Chunk, chunk_text, content_hash, document_id_for, normalise
from app.rag.pipeline import Answer, IngestResult, RAGPipeline
from app.rag.vectorstore import DocumentInfo, ScoredChunk, VectorStore

__all__ = [
    "Answer",
    "Chunk",
    "DocumentInfo",
    "IngestResult",
    "RAGPipeline",
    "ScoredChunk",
    "VectorStore",
    "chunk_text",
    "content_hash",
    "document_id_for",
    "normalise",
]
