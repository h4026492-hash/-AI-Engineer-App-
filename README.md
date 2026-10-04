# Family MedGuard

**An AI engineering portfolio prototype for plain-language patient education.**

Family MedGuard is a read-only web demo that answers a small set of general
health-information questions from public FDA and MedlinePlus summaries. It uses
retrieval-augmented generation (RAG), shows source links, includes conservative
handoff rules for obvious personal-medication and emergency questions, and runs
locally without a cloud model or API key.

> **Important:** This is an early engineering prototype, not a medical device,
> clinical service, or clinically validated tool. It does not diagnose, interpret
> individual lab results, recommend a dose, check personal drug interactions, or
> provide treatment advice. Do not enter protected or identifying health
> information. For emergencies, call 911 in the U.S. or your local emergency
> number.

## What works in this first slice

- Responsive, keyboard-accessible explainer UI with a larger-text control.
- General explanations grounded in a deliberately small public-source corpus.
- Clickable source links to FDA and MedlinePlus.
- Offline extractive model and local feature-hashing retrieval; no cloud API key
  is needed for the demo.
- Deterministic safety gates for obvious emergency, personal medication, and
  individual lab-result requests. These checks are **not** comprehensive triage.
- Public-demo read-only mode: document ingestion and deletion are disabled by
  default.
- Versioned FastAPI endpoints, structured errors, request IDs, rate limiting,
  Docker, tests, and a retrieval evaluation harness.

The bundled reference summaries cover only basic medicine-label safety and the
meaning of lab reference ranges. There is no PDF/OCR upload, dose calculator,
medication interaction database, patient profile, EHR integration, or saved chat
history.

## Run locally

Requirements: Python 3.11–3.13.

```bash
git clone https://github.com/h4026492-hash/-AI-Engineer-App-.git family-medguard
cd family-medguard
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
make install
make run
```

Open **http://localhost:8000**. The application indexes the sample reference
summaries on startup. It runs in offline mode by default; no credentials are
needed. The interactive API schema is at **http://localhost:8000/docs**.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Accessible web demo |
| `GET` | `/service-info` | Non-sensitive runtime information |
| `GET` | `/health` and `/health/ready` | Liveness and readiness probes |
| `POST` | `/v1/chat` | Source-grounded answer with citations and a safety footer |
| `POST` | `/v1/chat/stream` | Same answer over Server-Sent Events |
| `POST` | `/v1/search` | Inspect retrieval without generation |
| `GET` | `/v1/documents` | List indexed reference documents |
| `POST` / `DELETE` | `/v1/documents` | Disabled unless explicitly enabled for trusted local work |

Example general question:

```bash
curl -s http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does an out-of-range lab result mean in general?", "include_context":false}'
```

Every successful response uses a `{"data": ..., "meta": ...}` envelope. The
answer includes source citations and a visible educational disclaimer. Personal
medication and lab questions may be stopped by the deterministic safety gate
before retrieval; the gate is a prototype safeguard, not a medical classifier.

## Privacy and safe operation

- The default provider is `echo`: a deterministic local extractive model. Keep
  this for the public demo. Do not use this prototype with real patient records.
- Document changes are disabled by default (`ALLOW_DOCUMENT_MANAGEMENT=false`).
  The API has no user authentication; do not enable writes on an internet-facing
  instance.
- The demo does not offer file uploads or patient accounts. Avoid entering names,
  dates of birth, contact information, medication lists, or individual results.
- If you deliberately configure a hosted LLM, submitted questions may be sent to
  that provider. Review its privacy terms and do not send protected health
  information.
- The in-process rate limiter and local vector index are suitable only for a
  small single-process demo, not a public healthcare service.

To enable document management for **trusted local development only**, set
`ALLOW_DOCUMENT_MANAGEMENT=true` in a local `.env`. Never use that setting as a
substitute for authentication or access control.

## Sources included in the demo

- [FDA — Drug Interactions: What You Should Know](https://www.fda.gov/drugs/resources-drugs/drug-interactions-what-you-should-know)
- [FDA — 5 Medication Safety Tips for Older Adults](https://www.fda.gov/consumers/consumer-updates/5-medication-safety-tips-older-adults)
- [MedlinePlus — How to Understand Your Lab Results](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)

The bundled Markdown files are short plain-language summaries. Always consult
the linked source and a qualified clinician for complete or personal guidance.

## Development and quality checks

```bash
make check       # lint + strict type checks + tests
make eval        # deterministic retrieval checks against the bundled corpus
make demo        # scripted local walkthrough
```

Tests run offline and do not call a live model. The retrieval evaluation checks
whether the expected document is found; it does **not** establish clinical
accuracy, safety, or fitness for care.

## Deployment status and next steps

The GitHub repository is public, and this clone is prepared as a local product
iteration. **A public URL has not been deployed from this workspace.** Before
sharing a live demo, review `docs/LAUNCH_CHECKLIST.md`, set production
configuration, and verify every safety and privacy statement against the actual
host. Any real clinical use would require substantially more work: clinical
review, validated sources and updates, privacy/security engineering, user
research and accessibility testing, regulatory assessment, and ongoing
monitoring.

## Architecture

```text
Browser UI
    │ same-origin JSON
    ▼
FastAPI routes ──▶ deterministic medical safety gate
    │                         │ handoff response
    ▼                         └── (no retrieval/model call)
RAG pipeline ──▶ local feature-hashing index ──▶ offline extractive answerer
    │
    └── source metadata / citations to public references
```

Core modules are separated by responsibility: `app/api` defines the HTTP
contract, `app/rag` handles chunking and retrieval, `app/llm` contains pluggable
model providers, and `app/core/medical_safety.py` holds deterministic
prototype-specific handoff rules.

## License

MIT — see [LICENSE](LICENSE).
