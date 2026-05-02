# Lean single-stage build (~500MB instead of 5GB).
# PyTorch, sentence-transformers, and playwright are NOT included.
# Reranking and embeddings are handled by LiteLLM proxy.
#
# Build:
#   podman build --format docker -t entrag:dev .
#
# Run app:
#   podman run --rm -p 7860:7860 --env-file .env entrag:dev
#
# Run tests:
#   podman run --rm entrag:dev python -m pytest tests/ -v

FROM python:3.12-slim

# System dependencies for lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency spec first (layer caching)
COPY pyproject.toml .

# Create venv and install ONLY base + dev dependencies (no torch/playwright)
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e ".[dev]"

# Copy application code
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
