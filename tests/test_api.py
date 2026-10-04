"""HTTP contract: envelope shape, status codes, errors, SSE, and startup."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.echo_provider import EchoChatModel
from app.llm.hashing_embedder import HashingEmbedder
from app.main import create_app, seed_sample_documents
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore
from tests.conftest import EMBED_DIM, json_envelope

# --- Operational -----------------------------------------------------------


def test_root_serves_web_demo_and_service_info(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Family MedGuard" in page.text
    assert 'id="workspace"' in page.text

    body = client.get("/service-info").json()
    assert body["name"] == "Family MedGuard"
    assert body["provider"] == "echo"
    assert body["offline_mode"] is True
    assert body["docs"] == "/docs"


def test_liveness_is_always_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_index_state(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["provider"] == "echo"
    assert body["checks"]["index"] == "bound"


def test_request_id_is_generated_and_honoured(client: TestClient) -> None:
    assert client.get("/health").headers["X-Request-ID"]

    inbound = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert inbound.headers["X-Request-ID"] == "trace-me-123"

    # The same id must appear in the error envelope for that request.
    failure = client.delete("/v1/documents/nope", headers={"X-Request-ID": "trace-me-456"})
    assert failure.json()["request_id"] == "trace-me-456"


def test_document_management_defaults_to_disabled() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.allow_document_management is False
    assert settings.public_config()["allow_document_management"] is False


def test_config_redacts_secrets(client: TestClient) -> None:
    body = json_envelope(client.get("/v1/config").json())["data"]
    assert body["llm_provider"] == "echo"
    assert body["openai_api_key_set"] is False
    assert "openai_api_key" not in body
    assert body["rag"]["top_k"] == 3
    assert body["indexed_chunks"] > 0


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/chat" in response.json()["paths"]


# --- Documents -------------------------------------------------------------


def test_ingest_and_list_documents(client: TestClient) -> None:
    before = len(json_envelope(client.get("/v1/documents").json())["data"])

    response = client.post(
        "/v1/documents",
        json={
            "content": "Password reset links expire after 30 minutes.",
            "source": "handbook/passwords.md",
            "metadata": {"team": "support"},
        },
    )
    assert response.status_code == 201
    body = json_envelope(response.json())
    assert body["data"]["chunks"] >= 1
    assert body["data"]["source"] == "handbook/passwords.md"

    documents = json_envelope(client.get("/v1/documents").json())["data"]
    assert len(documents) == before + 1
    assert any(doc["source"] == "handbook/passwords.md" for doc in documents)
    assert any(doc["metadata"].get("team") == "support" for doc in documents)


def test_ingest_is_idempotent_for_identical_content(client: TestClient) -> None:
    payload = {"content": "Office hours are 9 to 5.", "source": "handbook/hours.md"}
    first = json_envelope(client.post("/v1/documents", json=payload).json())
    second = json_envelope(client.post("/v1/documents", json=payload).json())

    assert second["data"]["document_id"] == first["data"]["document_id"]
    assert second["meta"]["changed"] is False
    assert second["meta"]["replaced_chunks"] == 0
    assert second["meta"]["indexed_chunks"] == first["meta"]["indexed_chunks"]


def test_ingest_a_revision_replaces_the_previous_version(client: TestClient) -> None:
    source = "handbook/versioned.md"
    client.post(
        "/v1/documents", json={"content": "Support answers within 5 days.", "source": source}
    )
    revision = json_envelope(
        client.post(
            "/v1/documents", json={"content": "Support answers within 1 day.", "source": source}
        ).json()
    )

    assert revision["meta"]["changed"] is True
    assert revision["meta"]["replaced_chunks"] >= 1

    hits = json_envelope(
        client.post(
            "/v1/search", json={"query": "How fast does support answer?", "top_k": 10}
        ).json()
    )["data"]
    assert any("1 day" in hit["text"] for hit in hits)
    assert not any("5 days" in hit["text"] for hit in hits), "the stale revision must be gone"


def test_ingest_rejects_blank_content(client: TestClient) -> None:
    response = client.post("/v1/documents", json={"content": "   ", "source": "x.md"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_ingest_rejects_payload_over_the_limit(client: TestClient) -> None:
    limit = client.app.state.app_state.settings.max_ingest_chars
    response = client.post("/v1/documents", json={"content": "x" * (limit + 1), "source": "big.md"})
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "payload_too_large"
    assert body["error"]["details"]["limit"] == limit


def test_delete_document_removes_it(client: TestClient) -> None:
    created = json_envelope(
        client.post("/v1/documents", json={"content": "Temporary note.", "source": "tmp.md"}).json()
    )["data"]

    deleted = json_envelope(client.delete(f"/v1/documents/{created['document_id']}").json())
    assert deleted["data"]["removed_chunks"] == created["chunks"]

    sources = [doc["source"] for doc in json_envelope(client.get("/v1/documents").json())["data"]]
    assert "tmp.md" not in sources


def test_delete_unknown_document_returns_404(client: TestClient) -> None:
    response = client.delete("/v1/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --- Search and chat -------------------------------------------------------


def test_search_returns_scored_chunks_without_generating(client: TestClient) -> None:
    hits = json_envelope(
        client.post(
            "/v1/search", json={"query": "How is data encrypted at rest?", "top_k": 3}
        ).json()
    )["data"]

    assert hits
    assert hits[0]["source"] == "security.md"
    assert hits[0]["text"]
    assert all(0.0 <= hit["score"] <= 1.0 for hit in hits)
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)


def test_chat_returns_a_grounded_answer_with_citations(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "How long do I have to request a refund?"})
    assert response.status_code == 200

    body = json_envelope(response.json())
    data = body["data"]
    assert data["grounded"] is True
    assert data["citations"]
    assert data["citations"][0]["source"] == "refunds.md"
    assert "30 days" in data["answer"]
    assert data["provider"] == "echo"
    assert data["latency_ms"] >= 0.0
    assert data["tokens_in"] > 0
    assert body["meta"]["retrieved_chunks"] == len(data["citations"])


def test_chat_includes_educational_footer(client: TestClient) -> None:
    data = json_envelope(
        client.post("/v1/chat", json={"question": "What is the refund policy?"}).json()
    )["data"]
    assert "Educational information only" in data["answer"]


def test_personal_medication_question_is_routed_to_safety_gate(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "Can I take ibuprofen with warfarin?"})
    assert response.status_code == 200
    body = json_envelope(response.json())
    assert body["data"]["grounded"] is False
    assert body["data"]["citations"] == []
    assert "can't check whether a medicine is safe" in body["data"]["answer"]
    assert body["meta"]["model"] == "safety-gate"


def test_general_dose_request_is_routed_to_safety_gate(client: TestClient) -> None:
    data = json_envelope(
        client.post(
            "/v1/chat", json={"question": "What dose of acetaminophen should a child take?"}
        ).json()
    )["data"]
    assert data["grounded"] is False
    assert "recommend a dose" in data["answer"]


def test_named_medicine_interaction_is_routed_to_safety_gate(client: TestClient) -> None:
    data = json_envelope(
        client.post("/v1/chat", json={"question": "Can ibuprofen be used with warfarin?"}).json()
    )["data"]
    assert data["grounded"] is False
    assert "compare personal drug interactions" in data["answer"]


def test_personal_lab_question_is_routed_to_safety_gate(client: TestClient) -> None:
    data = json_envelope(
        client.post("/v1/chat", json={"question": "Can you interpret my CBC test result?"}).json()
    )["data"]
    assert data["grounded"] is False
    assert "can't diagnose or interpret" in data["answer"]


def test_current_emergency_terms_get_immediate_handoff(client: TestClient) -> None:
    data = json_envelope(
        client.post("/v1/chat", json={"question": "I am having chest pain right now"}).json()
    )["data"]
    assert data["grounded"] is False
    assert "call 911" in data["answer"]
    assert "do not wait" in data["answer"]


def test_public_read_only_mode_blocks_document_changes(client: TestClient) -> None:
    client.app.state.app_state.settings.allow_document_management = False
    response = client.post(
        "/v1/documents", json={"content": "A demo document.", "source": "demo.md"}
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"

    document_id = client.app.state.app_state.store.documents()[0].document_id
    deletion = client.delete(f"/v1/documents/{document_id}")
    assert deletion.status_code == 403
    assert deletion.json()["error"]["code"] == "forbidden"


def test_chat_can_omit_context_text(client: TestClient) -> None:
    data = json_envelope(
        client.post("/v1/chat", json={"question": "refund policy", "include_context": False}).json()
    )["data"]
    assert data["citations"]
    assert all(citation["text"] == "" for citation in data["citations"])
    assert all(citation["chunk_id"] for citation in data["citations"])


def test_chat_rejects_blank_question(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_chat_rejects_question_over_the_limit(client: TestClient) -> None:
    limit = client.app.state.app_state.settings.max_question_chars
    response = client.post("/v1/chat", json={"question": "q" * (limit + 1)})
    assert response.status_code == 422


def test_chat_rejects_missing_field(client: TestClient) -> None:
    response = client.post("/v1/chat", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_chat_respects_top_k_override(client: TestClient) -> None:
    data = json_envelope(
        client.post("/v1/chat", json={"question": "encryption and refunds", "top_k": 1}).json()
    )["data"]
    assert len(data["citations"]) == 1


def test_chat_filters_by_source(client: TestClient) -> None:
    data = json_envelope(
        client.post(
            "/v1/chat", json={"question": "encryption", "filter_source": "security.md"}
        ).json()
    )["data"]
    assert all(citation["source"] == "security.md" for citation in data["citations"])


def test_unknown_route_returns_the_error_envelope(client: TestClient) -> None:
    response = client.get("/v1/nope")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "http_error"
    assert "request_id" in body


# --- Streaming -------------------------------------------------------------


def test_chat_stream_emits_ordered_sse_events(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/chat/stream", json={"question": "How long do I have to request a refund?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        raw = "".join(response.iter_text())

    events = parse_sse(raw)
    names = [name for name, _ in events]

    assert names[0] == "sources"
    assert names[-1] == "done"
    assert "delta" in names
    # Strict order: no delta before sources, none after done.
    assert names == ["sources", *(["delta"] * names.count("delta")), "done"]

    sources = events[0][1]
    assert sources["grounded"] is True
    assert sources["citations"][0]["source"] == "refunds.md"

    streamed = "".join(payload["text"] for name, payload in events if name == "delta")
    done = events[-1][1]
    assert done["answer"] == streamed
    assert "30 days" in done["answer"]


def test_chat_stream_hands_personal_medication_questions_to_safety_gate(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"question": "Can I take this medicine with my other prescription?"},
    ) as response:
        events = parse_sse("".join(response.iter_text()))

    assert [name for name, _ in events] == ["sources", "delta", "done"]
    assert events[0][1]["grounded"] is False
    assert "pharmacist" in events[-1][1]["answer"]


def test_chat_stream_validates_before_opening_the_stream(client: TestClient) -> None:
    # Guardrails run first, so a bad request is a plain 422, not a stream.
    response = client.post("/v1/chat/stream", json={"question": "  "})
    assert response.status_code == 422


def test_chat_stream_on_empty_index_reports_no_sources(client: TestClient) -> None:
    client.app.state.app_state.store.clear()

    with client.stream("POST", "/v1/chat/stream", json={"question": "anything"}) as response:
        events = parse_sse("".join(response.iter_text()))

    assert events[0][0] == "sources"
    assert events[0][1]["grounded"] is False
    assert events[0][1]["citations"] == []
    assert events[-1][0] == "done"
    assert events[-1][1]["grounded"] is False


# --- Startup, seeding, and rate limiting -----------------------------------


def test_live_startup_with_seeding_disabled(test_settings: Settings) -> None:
    application = create_app(test_settings)
    with TestClient(application) as test_client:
        assert json_envelope(test_client.get("/v1/documents").json())["data"] == []


def test_live_startup_seeds_and_persists_the_index(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_level="WARNING",
        llm_provider="echo",
        rag_embed_dim=EMBED_DIM,
        vector_store_path=str(tmp_path / "vectorstore"),
        seed_on_startup=True,
        rate_limit_per_minute=0,
    )
    (tmp_path / "data" / "sample_docs").mkdir(parents=True)
    (tmp_path / "data" / "sample_docs" / "note.md").write_text(
        "The support desk answers email within one business day.", encoding="utf-8"
    )

    import os

    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        application = create_app(settings)
        with TestClient(application) as test_client:
            documents = json_envelope(test_client.get("/v1/documents").json())["data"]
            assert [doc["source"] for doc in documents] == ["note.md"]
    finally:
        os.chdir(previous)

    # Shutdown persists the index so the next boot does not re-embed.
    assert (tmp_path / "vectorstore" / "chunks.jsonl").exists()
    assert (tmp_path / "vectorstore" / "vectors.npy").exists()


def test_seeding_indexes_only_markdown_and_text_with_content(sample_docs_dir: Path) -> None:
    pipeline = RAGPipeline(
        chat=EchoChatModel(),
        embedder=HashingEmbedder(dimension=EMBED_DIM),
        store=VectorStore(dimension=EMBED_DIM),
        chunk_size=400,
        chunk_overlap=50,
    )

    assert asyncio.run(seed_sample_documents(pipeline, sample_docs_dir)) == 2
    assert sorted(doc.source for doc in pipeline.store.documents()) == ["refunds.md", "security.md"]


def test_seeding_attaches_public_source_metadata(tmp_path: Path, pipeline: RAGPipeline) -> None:
    directory = tmp_path / "health_docs"
    directory.mkdir()
    (directory / "lab-results.md").write_text(
        "Reference ranges differ between labs.", encoding="utf-8"
    )

    assert asyncio.run(seed_sample_documents(pipeline, directory)) == 1
    document = pipeline.store.documents()[0]
    assert document.metadata["publisher"] == "MedlinePlus / U.S. National Library of Medicine"
    assert document.metadata["source_url"] == (
        "https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/"
    )


def test_seeding_a_missing_directory_is_a_noop(tmp_path: Path) -> None:
    pipeline = RAGPipeline(
        chat=EchoChatModel(),
        embedder=HashingEmbedder(dimension=EMBED_DIM),
        store=VectorStore(dimension=EMBED_DIM),
    )
    assert asyncio.run(seed_sample_documents(pipeline, tmp_path / "nope")) == 0


def test_rate_limiter_returns_429_after_the_limit(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_level="WARNING",
        llm_provider="echo",
        rag_embed_dim=EMBED_DIM,
        vector_store_path=str(tmp_path / "vectorstore"),
        seed_on_startup=False,
        rate_limit_per_minute=2,
    )
    application = create_app(settings)
    with TestClient(application) as test_client:
        assert test_client.get("/v1/documents").status_code == 200
        assert test_client.get("/v1/documents").status_code == 200

        limited = test_client.get("/v1/documents")
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "rate_limited"
        assert int(limited.headers["Retry-After"]) > 0

        # Liveness is outside /v1 and stays reachable while rate limited.
        assert test_client.get("/health").status_code == 200


# --- Helpers ---------------------------------------------------------------


def parse_sse(raw: str) -> list[tuple[str, dict[str, object]]]:
    """Parse a text/event-stream body into (event name, data) pairs."""
    events: list[tuple[str, dict[str, object]]] = []
    name: str | None = None
    data: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :].strip()
        elif line.startswith("data: "):
            data.append(line[len("data: ") :])
        elif line == "" and name is not None:
            events.append((name, json.loads("\n".join(data))))
            name, data = None, []
    if name is not None and data:
        events.append((name, json.loads("\n".join(data))))
    return events
