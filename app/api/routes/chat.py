"""Chat endpoints: grounded question answering, batched and streamed."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import AppState, get_app_state
from app.core.errors import ProviderError, ValidationError
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.common import ApiResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


def _guard_question(question: str, limit: int) -> str:
    stripped = question.strip()
    if not stripped:
        raise ValidationError("question must contain at least one non-whitespace character.")
    if len(stripped) > limit:
        raise ValidationError(
            f"question exceeds the {limit:,} character limit.",
            details={"characters": len(stripped), "limit": limit},
        )
    return stripped


@router.post(
    "/chat",
    response_model=ApiResponse[ChatResponse],
    summary="Ask a question grounded in the indexed documents",
    description=(
        "Retrieves the most relevant chunks and answers from them, returning the "
        "citations that backed the answer. `grounded` is false when nothing scored "
        "above the relevance threshold."
    ),
)
async def chat(
    payload: ChatRequest,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ApiResponse[ChatResponse]:
    question = _guard_question(payload.question, state.settings.max_question_chars)

    result = await state.pipeline.ask(
        question,
        top_k=payload.top_k,
        filter_source=payload.filter_source,
    )

    citations = [Citation(**citation) for citation in result.citations()]
    if not payload.include_context:
        citations = [citation.model_copy(update={"text": ""}) for citation in citations]

    return ApiResponse(
        data=ChatResponse(
            question=question,
            answer=result.answer,
            citations=citations,
            grounded=result.grounded,
            provider=cast(Literal["echo", "openai"], state.provider),
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        ),
        meta={
            "retrieved_chunks": len(result.hits),
            "indexed_chunks": len(state.store),
            "model": state.pipeline.provider_name,
        },
    )


@router.post(
    "/chat/stream",
    summary="Ask a question, streaming the answer as Server-Sent Events",
    description=(
        "Emits `sources` (citations), then `delta` (text pieces) as the model "
        "produces them, then `done` (full answer and timings). Connect with "
        "`EventSource` or read the response as `text/event-stream`."
    ),
    response_class=StreamingResponse,
)
async def chat_stream(
    payload: ChatRequest,
    state: Annotated[AppState, Depends(get_app_state)],
) -> StreamingResponse:
    question = _guard_question(payload.question, state.settings.max_question_chars)
    pipeline = state.pipeline

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in pipeline.ask_stream(
                question,
                top_k=payload.top_k,
                filter_source=payload.filter_source,
            ):
                yield f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
        except ProviderError as exc:
            # The response has already started, so a non-2xx status is no longer
            # possible. Surface the failure as an in-band event instead.
            logger.warning("stream_provider_error message=%s", exc.message)
            error_payload: dict[str, Any] = {"code": exc.code, "message": exc.message}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("stream_unexpected_error")
            yield 'event: error\ndata: {"code":"internal_error","message":"Stream failed."}\n\n'

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering so deltas reach the client immediately.
            "X-Accel-Buffering": "no",
        },
    )
