# ai-engineer-app

A production-shaped **retrieval-augmented generation (RAG) service**: ingest your
documents, ask questions in natural language, get answers with citations.

Built the way a real AI service is built — pluggable model providers, a
retrieval layer you can benchmark separately from generation, structured
logging, request IDs, rate limiting, tests, an eval harness, Docker, and CI.
It runs **offline with no API key** out of the box and switches to a hosted
model with one environment variable.

```
┌──────────────────────────────────────────────────────────────────────┐
│  POST /v1/documents          POST /v1/chat        POST /v1/search    │
│        │                           │                     │           │
│        ▼                           ▼                     ▼           │
│  ┌───────────┐            ┌─────────────────────────────────┐        │
│  │  chunking │            │         RAGPipeline             │        │
│  └─────┬─────┘            │  retrieve → ground → generate   │        │
│        ▼                  └───────┬─────────────────┬───────┘        │
│  ┌───────────┐   embed            │                 │                │
│  │ Embedder  │◀───────────────────┘                 │                │
│  └─────┬─────┘                                      ▼                │
│        ▼                                     ┌────────────┐          │
│  ┌───────────────┐        top-k cosine       │ ChatModel  │          │
│  │  VectorStore  │◀──────────────────────────│ (echo |    │          │
│  │  (normalised) │──────────────────────────▶│  openai)   │          │
│  └───────────────┘   scored chunks           └────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
        Protocols: ChatModel, Embedder  ── swap implementations freely
```

---

## Quickstart

```bash
git clone <your-repo-url> && cd ai-engineer-app
python -m venv .venv && source .venv/bin/activate
make install          # or: pip install -e ".[dev]"
make run              # http://localhost:8000  (docs at /docs)
```

No API key needed. The server indexes `data/sample_docs/` on first boot, so
`http://localhost:8000/docs` answers questions immediately.

For the CLI, seed the index once (the CLI does not auto-seed):

```bash
make seed
make ask QUESTION="How long do I have to request a refund?"
```

```json
{
  "question": "How long do I have to request a refund?",
  "answer": "# Refunds and Billing Policy ## Standard refunds You can request a refund within 30 days of your first payment for a full refund, no questions asked.",
  "grounded": true,
  "latency_ms": 0.4,
  "citations": [{ "source": "refunds.md", "score": 0.3324, "text": "..." }]
}
```

The offline provider is **extractive**, so it quotes the source text rather than
paraphrasing. Swap in a real model for generative answers.

### Run it with a real model

```bash
cp .env.example .env
# set OPENAI_API_KEY=sk-... and LLM_PROVIDER=openai
make run
```

Any OpenAI-compatible gateway works (Azure OpenAI, LiteLLM, vLLM, Ollama) —
point `OPENAI_BASE_URL` at it.

---

## API

| Method | Path                     | Purpose                                        |
| ------ | ------------------------ | ---------------------------------------------- |
| `GET`  | `/`                      | Service info                                   |
| `GET`  | `/health`                | Liveness probe                                 |
| `GET`  | `/health/ready`          | Readiness probe (503 if not serving)           |
| `GET`  | `/v1/config`             | Effective config, secrets redacted             |
| `POST` | `/v1/documents`          | Ingest a document                              |
| `GET`  | `/v1/documents`          | List indexed documents                         |
| `DELETE` | `/v1/documents/{id}`   | Remove a document from the index               |
| `POST` | `/v1/search`             | Raw retrieval with scores — no generation      |
| `POST` | `/v1/chat`               | Grounded answer with citations                 |
| `POST` | `/v1/chat/stream`        | Same, streamed as Server-Sent Events           |

Interactive docs: **`/docs`** (Swagger) and **`/redoc`**.

### Ask a question

```bash
curl -s localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the rate limits for the Pro plan?", "top_k": 3}'
```

Actual response from the offline provider against the bundled corpus:

```json
{
  "data": {
    "question": "What are the rate limits for the Pro plan?",
    "answer": "## Rate limits Rate limits are per API token and use a fixed 60-second window. Rate limit state is returned on every response in the `X-RateLimit-Limit`,",
    "citations": [
      { "chunk_id": "948a6ea05dd85fbf:0000", "source": "api-guide.md", "score": 0.2934, "text": "..." },
      { "chunk_id": "948a6ea05dd85fbf:0001", "source": "api-guide.md", "score": 0.1321, "text": "..." },
      { "chunk_id": "c8fac47431246695:0000", "source": "refunds.md", "score": 0.1022, "text": "..." }
    ],
    "grounded": true,
    "provider": "echo",
    "latency_ms": 0.64,
    "tokens_in": 706,
    "tokens_out": 33
  },
  "meta": { "retrieved_chunks": 3, "indexed_chunks": 8, "model": "echo" }
}
```

Every response uses the same envelope: `{"data": ..., "meta": ...}` on success,
`{"error": {"code", "message"}, "request_id": "..."}` on failure. The
`X-Request-ID` header is echoed back and attached to every log line.

### Ingest a document

```bash
curl -s localhost:8000/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"content": "Our office closes at 5pm Central.", "source": "handbook/office.md"}'
```

### Stream an answer

```bash
curl -N localhost:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "How is data encrypted at rest?"}'
```

```
event: sources
data: {"grounded": true, "citations": [{"chunk_id": "013c61324be8fda2:0000", "source": "security.md", "score": 0.2408}]}

event: delta
data: {"text": "# Security and Data Handling All data in transit is encrypted with TLS 1.3."}

event: delta
data: {"text": " Data at rest is encrypted with"}

event: done
data: {"answer": "...", "grounded": true, "latency_ms": 1.36, "provider": "echo", "tokens_in": 502, "tokens_out": 26}
```

Citations arrive before the first word, so a UI can render sources immediately.
The `done` event carries the complete answer: **if you never receive `done`, the
stream did not finish cleanly** (errors after the response has started arrive as
an in-band `error` event, because the status code is already sent).

---

## CLI

```bash
ai-app seed                                  # index data/sample_docs
ai-app ask "How do I reset my password?"     # answer with citations
ai-app search "encryption at rest" --top-k 5 # retrieval only, no generation
ai-app docs                                  # what is currently indexed
ai-app run                                   # start the API server
```

The CLI drives the same `RAGPipeline` as the HTTP API — there is no second
implementation to drift out of sync.

---

## Configuration

Copy `.env.example` to `.env`. Everything has a working default, so an empty
`.env` is fine.

| Variable                | Default               | Notes                                             |
| ----------------------- | --------------------- | ------------------------------------------------- |
| `LLM_PROVIDER`          | `echo`                | `echo` (offline) or `openai`                      |
| `OPENAI_API_KEY`        | —                     | Required only for `openai`                        |
| `OPENAI_CHAT_MODEL`     | `gpt-4o-mini`         | Any chat model your gateway serves                |
| `OPENAI_EMBEDDING_MODEL`| `text-embedding-3-small` |                                               |
| `OPENAI_BASE_URL`       | —                     | Any OpenAI-compatible endpoint                    |
| `RAG_CHUNK_SIZE`        | `800`                 | Characters per chunk                              |
| `RAG_CHUNK_OVERLAP`     | `120`                 | Character overlap between chunks                  |
| `RAG_TOP_K`             | `4`                   | Chunks handed to the model                        |
| `RAG_MIN_SCORE`         | `0.05`                | Discard chunks below this cosine similarity       |
| `RAG_EMBED_DIM`         | `512`                 | Width of the offline hashing embedder             |
| `VECTOR_STORE_PATH`     | `data/vectorstore`    | Where the index is persisted                      |
| `SEED_ON_STARTUP`       | `true`                | Seed sample docs into an empty index              |
| `LOG_FORMAT`            | `console`             | `console` or `json`                               |
| `RATE_LIMIT_PER_MINUTE` | `60`                  | Per client IP; `0` disables                       |
| `MAX_INGEST_CHARS`      | `200000`              | Ingestion guardrail                               |

Secrets are never logged and never appear in `/v1/config`, which reports only
whether a key is set.

---

## Architecture

```
app/
├── config.py            Settings (pydantic-settings), cached
├── main.py              App factory, lifespan, wiring
├── cli.py               Command line interface
├── core/                Cross-cutting: logging, errors, middleware
├── schemas/             Pydantic request/response models (the API contract)
├── llm/                 Providers behind protocols
│   ├── base.py            ChatModel / Embedder protocols
│   ├── echo_provider.py   Deterministic extractive model (offline)
│   ├── hashing_embedder.py Feature-hashing embedder (offline)
│   ├── openai_provider.py OpenAI-compatible chat + embeddings
│   └── factory.py         Composition root
├── rag/                 Retrieval
│   ├── chunking.py        Boundary-aware chunking with overlap
│   ├── vectorstore.py     Cosine index + persistence
│   ├── prompting.py       Prompt templates, versioned
│   └── pipeline.py        ingest / search / ask / ask_stream
└── api/                 FastAPI routers + dependency injection
```

**Layers depend inward.** `api` → `rag` → `llm`. Nothing in `rag` imports
FastAPI; nothing in `llm` imports retrieval. That is what makes each layer
testable on its own and swappable without a refactor.

### Two design decisions worth knowing about

**1. The offline provider is real, not a stub.** `EchoChatModel` performs
extractive question answering over the retrieved context, and `HashingEmbedder`
produces stable lexical vectors. So the whole path — chunking, embedding,
retrieval, grounding, citations — genuinely runs with no API key. Integration
tests catch retrieval regressions, and CI needs no secrets.

**2. Retrieval is separable from generation.** `POST /v1/search` and
`ai-app search` return scored chunks without calling a model. When answers are
wrong, you can tell immediately whether retrieval missed the right chunk or the
model mishandled a chunk it was given. That distinction is most of the work in
tuning a RAG system.

### Swapping a component

* **Model provider** — implement `ChatModel` / `Embedder` in `app/llm/`,
  register it in `build_stack`. Nothing else changes.
* **Vector store** — implement `add` / `search` / `delete_document` against
  pgvector, Qdrant, or Chroma. The pipeline only uses those three.
* **Chunking** — change `chunk_text`. `/v1/search` tells you if it helped.

---

## Development

```bash
make install     # editable install with dev dependencies
make lint        # ruff
make format      # ruff --fix
make typecheck   # mypy --strict
make test        # pytest with coverage
make check       # lint + typecheck + test (what CI runs)
make coverage    # coverage report
```

### Tests

```
tests/
├── test_chunking.py       Chunk boundaries, overlap, edge cases
├── test_vectorstore.py    Cosine ranking, filtering, persistence round-trip
├── test_pipeline.py       End-to-end grounding, citations, empty index
├── test_api.py            HTTP contract, error envelope, SSE streaming
├── test_providers.py      Provider protocols and offline fallback
└── conftest.py            App + pipeline fixtures
```

Tests run entirely offline. Tests needing a live API are marked
`@pytest.mark.integration` and deselected by default.

### Evals

A small golden set of question → expected-source assertions, so retrieval
changes are measured rather than eyeballed:

```bash
make eval
```

Real output from the current default configuration:

```
corpus=data/sample_docs documents=3 chunks=8 embedder=hashing-512

  PASS  How long do I have to request a refund?                     rank 1
  FAIL  Can I get money back on an annual plan after two weeks?     not retrieved
        expected ['refunds.md'], got top hit 'api-guide.md'
  PASS  What happens when my card payment fails?                    rank 1
  PASS  How is data encrypted at rest?                              rank 1
  PASS  Which plans include SAML single sign-on?                    rank 1
  PASS  Where is customer data stored and can it leave the region?  rank 1
  PASS  How many requests per minute does the Pro plan allow?       rank 1
  PASS  How do I paginate through a list endpoint?                  rank 1
  PASS  How do I verify that a webhook really came from you?        rank 2
  PASS  How quickly are customers told about a security incident?   rank 1

9/10 passed (90.0%)  |  MRR 0.850  |  threshold 70%
```

That one failure is real and worth keeping visible: "money back" and "two weeks"
share no tokens with "refunded" and "14 days". A lexical embedder cannot bridge
that gap; a real embedding model can. It is exactly the kind of regression the
harness exists to surface, so fix it by improving the embedder rather than by
deleting the case.

Add cases to `evals/golden_set.json`. `make eval` exits non-zero below
`--threshold`, so it can gate CI.

---

## Docker

```bash
docker compose up --build      # http://localhost:8000
```

```bash
docker build -t ai-engineer-app .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=openai -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$PWD/data:/app/data" ai-engineer-app
```

The image is multi-stage, runs as a non-root user, and exposes a healthcheck
against `/health`. The index persists through the `./data` bind mount.

---

## Production notes

Real gaps, called out rather than hidden:

* **Vector store is in-process.** Fine to roughly 100k chunks; a restart
  reloads from disk. Move to a real vector database before scaling.
* **Rate limiting is per-process.** Correct for one replica; use Redis for a
  fleet. The middleware interface does not change.
* **No auth.** Add API-key or OAuth2 middleware before exposing this publicly.
* **Ingestion is synchronous.** Move to a task queue (Celery, ARQ, Temporal)
  for large documents.
* **No prompt injection defence** beyond the grounding instructions. Treat
  retrieved document text as untrusted input.

See `docs/ARCHITECTURE.md` for the reasoning behind the layering.

---

## Make targets

`make help` lists them all.

| Target             | Action                                            |
| ------------------ | ------------------------------------------------- |
| `make install`     | Editable install with dev dependencies            |
| `make run`         | Start the dev server with reload                  |
| `make seed`        | Index the sample documents                        |
| `make ask`         | Ask one question: `make ask QUESTION="..."`       |
| `make search`      | Raw retrieval: `make search QUERY="..."`          |
| `make demo`        | Boot a server and run a scripted end-to-end walkthrough |
| `make lint`        | Ruff lint + format check                          |
| `make format`      | Ruff autofix + format                             |
| `make typecheck`   | mypy `--strict`                                   |
| `make test`        | pytest with coverage                              |
| `make coverage`    | HTML coverage report                              |
| `make check`       | Lint + typecheck + test (what CI runs)            |
| `make eval`        | Retrieval eval harness                            |
| `make docker-build`| Build the container image                         |
| `make docker-up`   | Run the stack with docker compose                 |
| `make clean`       | Remove caches and build artifacts                 |

Current status: **152 tests passing**, **94% line coverage**, ruff clean,
`mypy --strict` clean across 30 source files, evals at 9/10 (MRR 0.850).

---

## License

MIT — see [LICENSE](LICENSE).
