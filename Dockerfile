# syntax=docker/dockerfile:1
# Multi-stage build (spec §16.1). Models are downloaded at BUILD time so the free
# tier's cold start is load-only. Per ADR 0004 the serving image omits the
# cross-encoder reranker (RERANK_BACKEND=none) to stay inside the 512 MB envelope.

FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix /install .

# Runtime stage: minimal base, non-root user, models baked in
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed-cache \
    EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    SPARSE_MODEL=Qdrant/bm25
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src
RUN useradd --create-home --shell /usr/sbin/nologin atlas \
    && mkdir -p /opt/fastembed-cache \
    && chown -R atlas:atlas /app /opt/fastembed-cache
USER atlas

# bake the two retrieval models (reranker deliberately excluded — ADR 0004)
RUN python - <<'PY'
import os
from fastembed import SparseTextEmbedding, TextEmbedding

TextEmbedding(model_name=os.environ["EMBED_MODEL"])
SparseTextEmbedding(model_name=os.environ["SPARSE_MODEL"])
print("models cached")
PY

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '10000') + '/healthz')" || exit 1

EXPOSE 10000
CMD ["sh", "-c", "uvicorn atlas.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
