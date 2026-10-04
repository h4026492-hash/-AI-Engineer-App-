#!/usr/bin/env bash
# End-to-end walkthrough against a local server: boots one, exercises endpoints,
# then shuts it down. Run with `make demo` from a trusted development machine.
set -euo pipefail

PORT="${PORT:-8000}"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
PYTHON="${PYTHON:-python}"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; RESET=$'\033[0m'

section() { printf "\n%s==> %s%s\n" "$BOLD" "$1" "$RESET"; }
show()    { printf "%s%s%s\n" "$DIM" "$1" "$RESET"; }

command -v jq >/dev/null 2>&1 && PRETTY="jq ." || PRETTY="cat"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    show "stopping server (pid ${SERVER_PID})"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

section "Starting local demo server on ${BASE}"
# This local walkthrough enables the otherwise-disabled write API to exercise
# ingestion. Never use ALLOW_DOCUMENT_MANAGEMENT=true on a public deployment.
ALLOW_DOCUMENT_MANAGEMENT=true SEED_ON_STARTUP=true "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level warning &
SERVER_PID=$!

for _ in $(seq 1 40); do
  if curl -sf "${BASE}/health" >/dev/null 2>&1; then break; fi
  sleep 0.25
done

if ! curl -sf "${BASE}/health" >/dev/null 2>&1; then
  printf "%sServer failed to start%s\n" "$RED" "$RESET" >&2
  exit 1
fi
printf "%sServer ready%s\n" "$GREEN" "$RESET"

section "GET /health"
curl -sf "${BASE}/health" | $PRETTY

section "GET /service-info"
curl -sf "${BASE}/service-info" | $PRETTY

section "GET /v1/documents  (auto-seeded public reference corpus)"
curl -sf "${BASE}/v1/documents" | $PRETTY

section "POST /v1/documents  (synthetic, trusted local-only example)"
curl -sf -X POST "${BASE}/v1/documents" \
  -H 'Content-Type: application/json' \
  -d '{
        "content": "Demo reference note: the public library information desk is open Monday through Friday, 9 a.m. to 5 p.m.",
        "source": "demo/library-hours.md",
        "metadata": {"synthetic": true}
      }' | $PRETTY

section "POST /v1/search  (retrieval only, with scores)"
curl -sf -X POST "${BASE}/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query": "What does an out-of-range lab result mean?", "top_k": 3}' | $PRETTY

section "POST /v1/chat  (grounded general explanation with citations)"
curl -sf -X POST "${BASE}/v1/chat" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does an out-of-range lab result mean in general?", "top_k": 3}' | $PRETTY

section "POST /v1/chat  (personal medication handoff)"
curl -sf -X POST "${BASE}/v1/chat" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Can I take this medicine with my other prescription?"}' | $PRETTY

section "POST /v1/chat/stream  (Server-Sent Events)"
curl -sfN -X POST "${BASE}/v1/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Why should someone tell a pharmacist about supplements?"}'

section "Public-deployment reminder"
show "Document changes default to disabled. This local script enabled them only to demonstrate ingestion."
show "Keep ALLOW_DOCUMENT_MANAGEMENT=false on any internet-facing demo."

printf "\n%sDemo complete.%s\n" "$GREEN" "$RESET"
