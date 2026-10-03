#!/bin/sh
# Container entrypoint: seed an empty index, then hand off to uvicorn.
#
# Seeding is skipped when the index already has content, so a restarted
# container with a mounted volume does not re-embed everything.
set -e

if [ "${SEED_ON_STARTUP:-true}" = "true" ]; then
    echo "[entrypoint] checking index at ${VECTOR_STORE_PATH:-data/vectorstore}"
    if [ ! -f "${VECTOR_STORE_PATH:-data/vectorstore}/chunks.jsonl" ]; then
        echo "[entrypoint] index empty; seeding data/sample_docs"
        python -m app.cli seed || echo "[entrypoint] seeding failed; continuing"
    fi
fi

echo "[entrypoint] starting uvicorn on ${HOST:-0.0.0.0}:${PORT:-8000}"
exec python -m uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers "${UVICORN_WORKERS:-1}" \
    --proxy-headers \
    --forwarded-allow-ips '*'
