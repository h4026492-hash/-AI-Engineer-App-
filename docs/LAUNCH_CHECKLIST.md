# Public-demo launch checklist

This checklist is for a **read-only portfolio demo**, not a live clinical
service. A public URL makes the app available to strangers; it does not make the
app safe for protected health information or medical decision support.

## Current scope

- Public, non-patient-specific education over a tiny reviewed corpus.
- No accounts, real/persistent profiles, saved chat history, cloud document
  uploads, or EHR connection.
- The fixed synthetic profile selector is available only on loopback development
  with the offline provider and is hidden in production.
- Local deterministic provider (`LLM_PROVIDER=echo`) only.
- Document ingestion and deletion disabled (`ALLOW_DOCUMENT_MANAGEMENT=false`).
- Local OCR is disabled (`ALLOW_LOCAL_DOCUMENT_OCR=false`) and is blocked in
  production even if the setting is mistakenly enabled.
- Clear limitation and emergency copy visible in the UI.

Do not deploy if any of these safeguards are absent or if the hosting setup
retains more information than the privacy notice describes.

## Before exposing a demo URL

1. **Keep the attached engineering blueprint private.** The supplied PDF is
   marked confidential. It is not copied into this repository; do not commit it
   to a public repo without preparing a public-safe version and removing anything
   you do not intend to disclose.
2. Set production environment variables:

   ```text
   APP_NAME=Family MedGuard
   APP_ENV=production
   DEBUG=false
   HOST=0.0.0.0
   LLM_PROVIDER=echo
   ALLOW_DOCUMENT_MANAGEMENT=false
   ALLOW_LOCAL_DOCUMENT_OCR=false
   RATE_LIMIT_PER_MINUTE=30
   SEED_ON_STARTUP=true
   ```

   Keep `OPENAI_API_KEY` empty and do not configure a hosted model for this demo.
   The small corpus is seeded at startup; an ephemeral local index is fine for a
   demonstration.
3. Serve behind the host's HTTPS endpoint. Confirm the service binds to the
   platform-provided port, accepts the public preview origin, and exposes only
   the routes required for the demo.
4. Confirm write operations return `403`: `POST /v1/documents` and
   `DELETE /v1/documents/{document_id}`. Confirm `POST /v1/local/ocr-preview`
   also returns `403` in production. `GET /service-info` must report
   `local_synthetic_profiles_available=false` on a public host.
5. Smoke-test `GET /`, `/health`, `/health/ready`, `/service-info`, and the
   sample prompts in the UI. Verify that answers show clickable official sources,
   personal medication questions are handed off, and current-emergency wording
   tells a visitor not to wait for AI.
6. Check host logs, access logs, analytics, crash reporting, and backups. Make
   sure no request bodies or personal health information are retained. The app
   itself does not add chat-history storage, but hosting defaults still need to
   be reviewed.
7. Publish a concise privacy and limitations notice alongside the URL. Avoid
   claims such as “clinically safe,” “validated,” “HIPAA compliant,” or “medical
   advice.”

## Changes required before real patient use

A public prototype is not ready for real-world clinical decisions. Before any
clinical or patient-facing use, seek qualified clinical, privacy, security, and
regulatory review. Likely work includes validated and versioned medical content;
formal hazard analysis and human-factors testing; tested escalation behavior;
secure identity, authorization, encryption, retention/deletion, and incident
response; appropriate contractual and regulatory review; and a monitoring and
update process. Do not treat a regex safety gate or retrieval test as clinical
validation.

## Publishing the code vs. deploying the app

The GitHub repository is public, and the first Family MedGuard demo has been
merged into `main`. The local OCR iteration is on `feature/family-medguard-local-ocr`;
it does not itself deploy the application and has not yet been published. No
permanent cloud service has been provisioned from this workspace. Review and
publish the feature branch separately, and configure a hosting account and
environment variables separately if a live URL is wanted. Never put access
tokens, patient data, or the confidential PDF in the repository.
