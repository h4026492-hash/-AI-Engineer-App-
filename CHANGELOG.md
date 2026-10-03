# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet.

## [0.1.0] - 2026-10-02

Initial release.

### Added
- FastAPI service with a versioned `/v1` API and a uniform response envelope.
- RAG pipeline: boundary-aware chunking with overlap, cosine retrieval,
  grounded generation, and per-chunk citations.
- Pluggable model layer: `ChatModel` and `Embedder` protocols with an offline
  deterministic implementation (`echo` + feature hashing) and an
  OpenAI-compatible implementation.
- In-memory vector store with numpy cosine scoring and JSONL/npy persistence;
  index reloads on boot and dimension mismatches are rejected rather than
  silently producing meaningless scores.
- Streaming answers over Server-Sent Events (`POST /v1/chat/stream`), with
  provider-side deltas and in-band error events.
- Operational endpoints: liveness, readiness, redacted effective config.
- Cross-cutting infrastructure: structured logging (`console`/`json`), request
  IDs propagated from inbound headers, per-client rate limiting, CORS, and a
  consistent error envelope for every failure path.
- Guardrails: ingestion size limit, question length limit, chunk/overlap
  validation, and grounding threshold (`grounded` flag).
- CLI (`ai-app`) covering seed, ask, search, docs, and run.
- Test suite (offline, no secrets) and a retrieval eval harness with a golden
  set, MRR, and a pass-rate threshold for CI.
- Multi-stage Dockerfile (non-root, healthchecked), docker-compose, Makefile,
  and GitHub Actions CI running lint, typecheck, tests, and evals.

[Unreleased]: https://github.com/your-org/ai-engineer-app/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/ai-engineer-app/releases/tag/v0.1.0
