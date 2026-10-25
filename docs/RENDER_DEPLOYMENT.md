# Proposed Render deployment (read-only demo)

**Status: configuration only.** Render is a recommended first target because the
repository already has a Docker image, a dynamic `PORT` entrypoint, and a
`/health` endpoint. No Render account, service, credentials, or public URL has
been created or connected from this workspace. The user must review and approve
any hosting provider and the creation of a public service.

## What the Blueprint configures

The root [`render.yaml`](../render.yaml) describes one Docker web service:

- Free compute plan in Render's Ohio region; no database or persistent disk.
- Builds the repository's root `Dockerfile`, serves on Render's supplied port,
  and checks `GET /health`.
- Uses `APP_ENV=production`, the offline `echo` provider, and JSON logs.
- Disables document management and local OCR; keeps the synthetic profile
  selector unavailable in production.
- Uses a 30-request-per-minute in-process limit for `/v1` routes.
- Writes only the reviewed seed-corpus index under `/tmp`; this is ephemeral
  runtime state, not a durable user-data store. No chat history or profiles are
  stored by the app.
- Turns off automatic deploys. A later release must be started manually.
- Leaves `OPENAI_API_KEY` unset and disables CORS because the UI calls the API
  on the same origin.

The Blueprint does **not** itself deploy or provision anything. When using
Render's Blueprint flow, review the proposed changes before approving them. Do
not approve if it proposes resources other than the single web service, or if
the account shows a paid plan, persistent storage, or unreviewed add-ons.

## Before creating a public service

1. Confirm Render is the chosen host and that a public, unauthenticated demo is
   acceptable. The application is a portfolio prototype, not a clinical service;
   visitors must be told not to enter real health information.
2. Review the account's plan, billing, logs, analytics, and data-retention
   settings. The application logger records request method/path, status, client
   address, and timing (not request bodies); the hosting provider may retain
   its own access or infrastructure logs. Do not enable request-body capture,
   analytics, or error reporting that could collect user-entered content.
3. Create the Blueprint instance only after confirming the plan is `free` and
   that the proposed change contains just this one web service. No database or
   persistent disk is needed. Check billing and usage in the provider dashboard
   before approving any changes.
4. After deployment, verify HTTPS and the public UI. Then check:
   - `GET /health` and `GET /health/ready` return success.
   - `GET /service-info` reports `provider: echo`,
     `offline_mode: true`, and
     `local_synthetic_profiles_available: false`.
   - `POST /v1/documents`, `DELETE /v1/documents/{document_id}`, and
     `POST /v1/local/ocr-preview` return `403`.
   - The public UI has no upload/OCR controls or synthetic profile selector;
     sample questions work and responses link to the reviewed public sources.
   - Logs and settings do not retain request bodies or introduce third-party
     model, analytics, or document-storage calls.
5. Keep automatic deploys off until a release has been reviewed. To take the
   service down, delete the Render service in the provider dashboard and verify
   its logs and any related resources are removed according to the account's
   retention controls.

## Free-tier trade-off

Render's Free web services spin down after 15 minutes without traffic and can
take about a minute to start again. The free plan is suitable for a portfolio
demo where occasional cold starts are acceptable, not a reliability commitment.
See Render's [free-instance limits](https://render.com/docs/free) and
[compute-plan details](https://render.com/docs/compute-plans) before creating a
service.

## Official Render references

- [Blueprint YAML specification](https://render.com/docs/blueprint-spec)
- [Docker deployments](https://render.com/docs/docker)
- [HTTP health checks](https://render.com/docs/health-checks)
- [Free instances](https://render.com/docs/free)
