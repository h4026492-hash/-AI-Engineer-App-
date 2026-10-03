# Architecture

Notes on why the code is shaped the way it is. Read the README for what it
does; this is about the decisions behind it.

## Layering

```
api  ──▶  rag  ──▶  llm
 │         │         │
 └─────────┴─────────┴──▶  core, config, schemas
```

Dependencies point one way. `app/rag` does not import FastAPI, and `app/llm`
does not import retrieval. Two consequences that pay off immediately:

* The pipeline is testable without an HTTP server, and the HTTP layer is
  testable without a model.
* Swapping a component is a change to one module, not a refactor.

`app/core` holds cross-cutting concerns (logging, error types, middleware)
that everyone may use but that use nothing back.

## The provider seam

`ChatModel` and `Embedder` are `typing.Protocol`s, not ABCs. A provider is
anything with the right shape, which is why a test double can be a twelve-line
class inside a test file rather than a registered subclass.

`build_stack` is the only function that knows concrete provider classes exist.
That single choke point is what makes "run this locally with no API key" and
"run this in production against a gateway" the same code path.

A detail worth noting: if `LLM_PROVIDER=openai` but no key is set, the service
**falls back to the offline stack and logs a warning** rather than failing to
boot. A deployment that serves degraded answers with a loud log is usually more
useful than a crash loop, but this is a judgement call — flip it if you would
rather fail fast.

## Why the offline provider does real work

`EchoChatModel` does extractive question answering over the retrieved context,
and `HashingEmbedder` produces stable lexical vectors via signed feature
hashing. Neither is a placeholder returning a constant.

The alternative — a stub that returns `"hello"` — makes the test suite fast and
worthless: it passes whether retrieval is correct or not. With a real extractive
model, a broken chunker or a broken similarity function produces a wrong answer,
and the integration tests catch it. It also means CI needs no secrets and the
repo is runnable five seconds after cloning.

The honest limitation: hashed lexical embeddings do not understand synonyms.
Retrieval quality is meaningfully better with a real embedding model. That is
exactly what the eval harness is for — switch embedders and measure.

## Chunking

Chunks are packed on natural boundaries: paragraphs first, then sentences, then
a hard character slice only when a single unit exceeds the budget. Consecutive
chunks overlap by `RAG_CHUNK_OVERLAP` characters, trimmed to a word boundary.

Overlap exists because answers straddle boundaries. If a refund window is stated
in one sentence and its exception in the next, non-overlapping chunks can
retrieve the rule without the exception — a confidently wrong answer.

Documents are content-addressed: `document_id` is a hash of `source + content`.
Re-ingesting the same document replaces the old chunks instead of duplicating
them, so a re-run of the ingestion job is idempotent.

## The vector store

Vectors are stored L2-normalised, so cosine similarity is a dot product and one
matrix multiplication scores the whole corpus. Brute force, linear in corpus
size, fine to roughly 100k chunks.

The interface is deliberately the shape of a real vector database — `add`,
`search`, `delete_document`. Moving to pgvector or Qdrant means implementing
those three against a client; the pipeline does not change.

One behaviour that is easy to get wrong: **a persisted index whose vector width
does not match the current embedder is discarded with a warning, not silently
used.** Comparing 512-dim vectors against 1536-dim ones is not merely bad, it is
undefined — but a naive implementation will happily truncate or zero-pad and
return plausible-looking garbage.

## Grounding and the `grounded` flag

The prompt instructs the model to answer only from context, cite bracketed
source numbers, and say so plainly when the context is insufficient.

Separately, `grounded: false` is set by the pipeline when nothing scored above
`RAG_MIN_SCORE`. This is not the same as asking the model whether it is
confident — models are unreliable reporters of their own uncertainty. It is an
objective statement about retrieval: *there was no evidence*. Clients should
surface that state distinctly rather than showing an ungrounded answer as if it
were sourced.

Citation numbers in the prompt context and the `citations` array in the response
are built from the same ordered list, so `[2]` in the answer always refers to
`citations[1]`.

## Streaming

`POST /v1/chat/stream` emits `sources`, then `delta` events, then `done`.
Citations arrive before the first word because a UI can render sources
immediately.

The deltas are the provider's real streaming deltas. `stream_text` falls back to
a single delta for providers that do not implement `stream`, so the transport
contract is uniform either way.

Errors after the response has started cannot change the status code — the
headers are already sent. They are emitted as an in-band `error` event instead.
A client that does not listen for it will see a truncated answer, which is why
the `done` event carries the complete text: absence of `done` means the stream
did not finish cleanly.

## Errors

Raise `AppError` subclasses from business logic; never construct an error
response in a handler. Handlers registered in `app/core/errors.py` render the
uniform envelope, attach the request ID, and log at the appropriate level.
Unhandled exceptions log a full traceback server-side and return an opaque
message — internals never reach the client.

## Observability

A request ID is minted per request (or taken from an inbound `X-Request-ID` so
it survives load balancers), stored in a `ContextVar`, attached to every log
line emitted during the request, and echoed on the response header. That is the
minimum for answering "what happened to this one request?" in a concurrent
async service, and it costs one middleware.

`LOG_FORMAT=json` emits one JSON object per line for aggregators; `console` is
human-readable for local work.

## What is deliberately missing

Named so it is a known gap rather than a surprise:

* **No authentication.** Add API-key or OAuth2 middleware before exposing this.
* **In-process rate limiting and vector store.** Both correct for a single
  replica, both wrong for a fleet.
* **Synchronous ingestion.** Large documents should go through a task queue.
* **No prompt injection defence** beyond the grounding instructions. Retrieved
  document text is untrusted input; a document can contain instructions.
* **No semantic evaluation.** The harness measures retrieval deterministically.
  Judging answer quality needs an LLM judge with a rubric, or human review.
