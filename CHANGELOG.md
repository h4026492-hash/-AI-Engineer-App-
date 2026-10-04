# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Family MedGuard web demo with responsive layout, larger-text control, a
  source-linked chat experience, and clear privacy/emergency notices.
- Small plain-language reference corpus summarized from FDA and MedlinePlus.
- Deterministic handoff responses for selected personal medication, individual
  lab-result, diagnosis, and urgent-symptom questions.
- Public read-only default: document ingestion and deletion are disabled unless
  explicitly enabled for trusted local development.
- Public launch checklist and health-domain retrieval evaluation cases.

### Changed
- App branding and default service metadata now identify Family MedGuard.
- Root path serves the web demo; non-sensitive runtime details are at
  `/service-info`.
- Chat responses carry an educational disclaimer. Streaming responses include
  the same disclaimer before the final `done` event.
- Updated sample questions, CLI defaults, and local walkthrough for the medical
  education demo.
- Added an explicit NumPy typing cast so strict type checking also passes with
  current NumPy and mypy versions.

### Safety
- The supplied confidential blueprint was not copied into the repository.
- No public deployment has been provisioned from this workspace. The project is
  still a portfolio prototype, not a clinical tool.

## [0.1.0] - 2026-10-02

Initial release of the generic retrieval service that this project extends.

### Added
- FastAPI service with a versioned `/v1` API and a uniform response envelope.
- RAG pipeline: boundary-aware chunking with overlap, cosine retrieval,
  grounded generation, and per-chunk citations.
- Pluggable model layer: `ChatModel` and `Embedder` protocols with an offline
  deterministic implementation (`echo` + feature hashing) and an
  OpenAI-compatible implementation.
- In-memory vector store with numpy cosine scoring and JSONL/npy persistence.
- Streaming answers over Server-Sent Events (`POST /v1/chat/stream`).
- Structured logging, request IDs, per-client rate limiting, CORS, and
  consistent errors.
- CLI (`ai-app`), offline tests, retrieval eval harness, Dockerfile,
  docker-compose, Makefile, and GitHub Actions CI.

[Unreleased]: https://github.com/h4026492-hash/-AI-Engineer-App-/compare/main...HEAD
[0.1.0]: https://github.com/h4026492-hash/-AI-Engineer-App-/releases/tag/v0.1.0
