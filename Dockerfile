# syntax=docker/dockerfile:1

# --- Stage 1: build a self-contained wheel + dependencies -------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY app ./app

# Build the wheel so runtime installs a real artifact, not a source tree.
RUN pip install --upgrade pip build \
    && python -m build --wheel --outdir /build/dist

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/app/.local/bin:$PATH"

# Non-root user; no shell needed by the container itself.
RUN groupadd --gid 1001 app \
    && useradd --uid 1001 --gid 1001 --create-home --shell /usr/sbin/nologin app

# Local-only OCR uses the offline Tesseract executable. The API stays disabled
# by default and refuses production/non-loopback requests even when installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Runtime data (persisted index) lives here; mount a volume over it.
RUN mkdir -p /app/data && chown -R app:app /app/data

COPY --chown=app:app data/sample_docs /app/data/sample_docs
COPY --chown=app:app docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER app

EXPOSE 8000

# Liveness probe against the endpoint the app actually serves.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,sys,urllib.request; url='http://127.0.0.1:'+os.getenv('PORT','8000')+'/health'; sys.exit(0 if urllib.request.urlopen(url, timeout=2).status == 200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
