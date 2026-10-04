"""Document ingestion, listing, deletion, and raw retrieval."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.deps import AppState, get_app_state
from app.core.errors import ForbiddenError, NotFoundError, PayloadTooLargeError
from app.schemas.common import ApiResponse
from app.schemas.documents import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentSummary,
    SearchHit,
    SearchRequest,
)

router = APIRouter(prefix="/v1", tags=["documents"])


@router.post(
    "/documents",
    response_model=ApiResponse[DocumentIngestResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description=(
        "Chunks, embeds, and indexes a document. The source is the document's "
        "identity: re-ingesting a source replaces the previous version instead of "
        "leaving a stale copy behind, and identical content is detected so "
        "embedding is skipped."
    ),
)
async def ingest_document(
    payload: DocumentIngestRequest,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ApiResponse[DocumentIngestResponse]:
    if not state.settings.allow_document_management:
        raise ForbiddenError("Document changes are disabled in this read-only demo.")

    limit = state.settings.max_ingest_chars
    if len(payload.content) > limit:
        raise PayloadTooLargeError(
            f"Document exceeds the {limit:,} character limit.",
            details={"characters": len(payload.content), "limit": limit},
        )

    result = await state.pipeline.ingest(
        payload.content,
        source=payload.source,
        metadata=payload.metadata,
    )
    return ApiResponse(
        data=DocumentIngestResponse(
            document_id=result.document_id,
            source=result.source,
            chunks=result.chunks,
            characters=result.characters,
        ),
        meta={
            "replaced_chunks": result.replaced_chunks,
            "changed": result.changed,
            "indexed_chunks": len(state.store),
        },
    )


@router.get(
    "/documents",
    response_model=ApiResponse[list[DocumentSummary]],
    summary="List indexed documents",
)
async def list_documents(
    state: Annotated[AppState, Depends(get_app_state)],
) -> ApiResponse[list[DocumentSummary]]:
    documents = [
        DocumentSummary(
            document_id=info.document_id,
            source=info.source,
            chunks=info.chunks,
            characters=info.characters,
            metadata=info.metadata,
        )
        for info in state.store.documents()
    ]
    return ApiResponse(data=documents, meta={"count": len(documents), "chunks": len(state.store)})


@router.delete(
    "/documents/{document_id}",
    response_model=ApiResponse[dict[str, int | str]],
    summary="Delete a document from the index",
)
async def delete_document(
    document_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ApiResponse[dict[str, int | str]]:
    if not state.settings.allow_document_management:
        raise ForbiddenError("Document changes are disabled in this read-only demo.")

    removed = state.store.delete_document(document_id)
    if removed == 0:
        raise NotFoundError(
            f"No document with id '{document_id}' in the index.",
            details={"document_id": document_id},
        )
    return ApiResponse(
        data={"document_id": document_id, "removed_chunks": removed},
        meta={"indexed_chunks": len(state.store)},
    )


@router.post(
    "/search",
    response_model=ApiResponse[list[SearchHit]],
    summary="Raw retrieval (no generation)",
    description=(
        "Returns the chunks a question would retrieve, with similarity scores, "
        "without calling the model. Use this to debug retrieval quality "
        "independently of generation quality."
    ),
)
async def search_documents(
    payload: SearchRequest,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ApiResponse[list[SearchHit]]:
    hits = await state.pipeline.search(
        payload.query,
        top_k=payload.top_k,
        filter_source=payload.filter_source,
    )
    data = [
        SearchHit(
            chunk_id=hit.chunk.chunk_id,
            document_id=hit.chunk.document_id,
            source=hit.chunk.source,
            score=round(hit.score, 4),
            text=hit.chunk.text,
            metadata=hit.chunk.metadata,
        )
        for hit in hits
    ]
    return ApiResponse(data=data, meta={"count": len(data), "indexed_chunks": len(state.store)})
