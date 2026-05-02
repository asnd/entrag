# Lean single-stage build (~500MB without local models, ~3GB with).
# By default, only base + dev deps are installed — reranking and embeddings
# are handled by the LiteLLM proxy sidecar.
#
# Set INSTALL_LOCAL=true to include local embedding + reranker models
# (adds PyTorch ~2GB, sentence-transformers).
#
# Docker:
#   docker build -t entrag:dev .
#   docker build --build-arg INSTALL_LOCAL=true -t entrag:dev-local .
#   docker run --rm -p 7860:7860 --env-file .env entrag:dev
#   docker run --rm entrag:dev python -m pytest tests/ -v
#
# Podman:
#   podman build -t entrag:dev .
#   podman build --build-arg INSTALL_LOCAL=true -t entrag:dev-local .
#   podman run --rm -p 7860:7860 --env-file .env entrag:dev
#   podman run --rm entrag:dev python -m pytest tests/ -v

FROM python:3.12-slim

ARG INSTALL_LOCAL=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .

RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e ".[dev]" && \
    if [ "$INSTALL_LOCAL" = "true" ]; then \
        uv pip install -e ".[local]"; \
    fi

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 7860

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/')" || exit 1

CMD ["python", "-m", "src.app"]