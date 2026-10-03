#!/usr/bin/env bash
# End-to-end walkthrough against a real server: boots one, exercises every
# endpoint, then shuts it down. Run with `make demo`.
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

section "Starting server on ${BASE}"
SEED_ON_STARTUP=true "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level warning &
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

section "GET /v1/config  (secrets redacted)"
curl -sf "${BASE}/v1/config" | $PRETTY

section "GET /v1/documents  (auto-seeded corpus)"
curl -sf "${BASE}/v1/documents" | $PRETTY

section "POST /v1/documents  (ingest a new document)"
curl -sf -X POST "${BASE}/v1/documents" \
  -H 'Content-Type: application/json' \
  -d '{
        "content": "Password resets are requested from the login screen. Reset links expire after 30 minutes and can be used only once.",
        "source": "handbook/passwords.md",
        "metadata": {"team": "support"}
      }' | $PRETTY

section "POST /v1/search  (retrieval only, with scores)"
curl -sf -X POST "${BASE}/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query": "How long is a password reset link valid?", "top_k": 3}' | $PRETTY

section "POST /v1/chat  (grounded answer with citations)"
curl -sf -X POST "${BASE}/v1/chat" \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long is a password reset link valid?", "top_k": 3}' | $PRETTY

section "POST /v1/chat  (a question the corpus cannot answer)"
curl -sf -X POST "${BASE}/v1/chat" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the CEO favourite colour?"}' | $PRETTY

section "POST /v1/chat/stream  (Server-Sent Events)"
curl -sfN -X POST "${BASE}/v1/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the rate limits for the Pro plan?"}'

section "Error envelope  (unknown document)"
curl -s -o /dev/null -w "status=%{http_code}\n" -X DELETE "${BASE}/v1/documents/does-not-exist"
curl -s -X DELETE "${BASE}/v1/documents/does-not-exist" | $PRETTY

printf "\n%sDemo complete.%s\n" "$GREEN" "$RESET"
