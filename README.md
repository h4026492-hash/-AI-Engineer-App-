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
meaning of lab reference ranges. There is no dose calculator, medication
interaction database, patient profile, EHR integration, or saved chat history.
The public demo has no upload feature. A separate, explicitly opt-in OCR preview
is available only to a localhost request in development; it extracts text from
small PDFs/PNG/JPEG files but does not interpret, index, or save them.

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

### Optional local OCR preview

The OCR preview is off by default and is deliberately limited to a development
server accessed through `localhost`/`127.0.0.1`. It accepts PDFs, PNGs, and
JPEGs up to 8 MiB and five PDF pages. Selectable PDF text uses local extraction;
scanned PDF pages and images use the local Tesseract OCR executable (English
only). On macOS, install it with `brew install tesseract`; on Debian/Ubuntu use
`sudo apt-get install tesseract-ocr`. Then set `ALLOW_LOCAL_DOCUMENT_OCR=true`
in your uncommitted `.env` and restart the app. Do not set this on an Arena
preview, a public host, or any production deployment. Use synthetic examples
only, not real patient records. OCR is fallible and only returns raw text; it
does not interpret, index, or persist the file or extracted text.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Accessible web demo |
| `GET` | `/service-info` | Non-sensitive runtime information |
| `GET` | `/health` and `/health/ready` | Liveness and readiness probes |
| `POST` | `/v1/chat` | Source-grounded answer with citations and a safety footer |
| `POST` | `/v1/chat/stream` | Same answer over Server-Sent Events |
| `POST` | `/v1/local/ocr-preview` | Extract text from a small PDF/image (opt-in localhost development only) |
| `POST` | `/v1/search` | Inspect retrieval without generation |
| `GET` | `/v1/documents` | List indexed reference documents |
| `POST` / `DELETE` | `/v1/documents` | Disabled unless explicitly enabled for trusted local work |

When local OCR is explicitly enabled, send a raw file body (not multipart); use
`application/pdf`, `image/png`, or `image/jpeg` as the `Content-Type`. For a
synthetic local test file only:

```bash
curl --data-binary @synthetic-sample.png -H 'Content-Type: image/png' http://localhost:8000/v1/local/ocr-preview
```

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
- The public demo does not expose uploads or patient accounts. The optional
  OCR preview requires `ALLOW_LOCAL_DOCUMENT_OCR=true`, a development
  environment, and a loopback client/host; it is blocked in production and from
  non-local hosts. It processes a file in memory, makes no model call, and does
  not save or index the file/text. Still use synthetic documents only; this
  prototype is not approved for protected health information.
- Avoid entering names, dates of birth, contact details, medication lists, or
  individual results into chat. OCR may misread text, especially medication
  names, numbers, units, and decimal points; verify everything against the
  original. It is not a medical interpretation or decision-support feature.
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

The first Family MedGuard demo has been merged into the public GitHub `main`
branch. This optional OCR iteration is being developed separately and is not yet
published. **A permanent public URL has not been deployed from this workspace.**
Before sharing a live demo, review `docs/LAUNCH_CHECKLIST.md`, set production
configuration, and verify every safety and privacy statement against the actual
host. Keep local OCR disabled on public hosts. Any real clinical use would
require substantially more work: clinical review, validated sources and updates,
privacy/security engineering, user research and accessibility testing,
regulatory assessment, and ongoing monitoring.

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

Localhost development only:
Browser file ──▶ /v1/local/ocr-preview ──▶ in-memory PDF/image text extraction
                                      └── raw text preview; not indexed or sent to a model
```

Core modules are separated by responsibility: `app/api` defines the HTTP
contract, `app/rag` handles chunking and retrieval, `app/llm` contains pluggable
model providers, `app/core/medical_safety.py` holds prototype handoff rules,
and `app/core/ocr.py` implements the local, non-persisting extraction path.

## License

MIT — see [LICENSE](LICENSE).
