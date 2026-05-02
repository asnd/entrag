# EntRAG

VMware/Broadcom Knowledge Base RAG Application — a Retrieval-Augmented Generation system that scrapes, indexes, and queries Broadcom/VMware KB articles through a Gradio web interface.

## Architecture

EntRAG uses a lightweight app container paired with a LiteLLM proxy sidecar:

- **rag-app** — Gradio web UI (port 7860), LlamaIndex-based ingestion and retrieval, LanceDB vector store
- **litellm** — Proxy for LLM and embedding API calls (OpenAI or other providers)

By default, embeddings and reranking are routed through the LiteLLM proxy to remote APIs. An optional **local models** mode bundles PyTorch and runs models in-process.

## Quick Start

### Prerequisites

- **Docker** (recommended) or **Podman** (both supported)
- OpenAI API key (or another provider supported by LiteLLM)

> **Podman users:** The project provides a `Containerfile` symlink to `Dockerfile`. Simply replace `docker` with `podman` in all commands below.

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set LITELLM_API_KEY and OPENAI_API_KEY
```

### 2. Build and run

**Docker:**
```bash
# Lean image (~500MB) — embeddings via LiteLLM proxy
docker compose up -d

# With local models (~3GB) — embeddings and reranker run in-container
docker compose --profile local-models up -d
```

**Podman:**
```bash
# Lean image
podman compose up -d

# With local models
podman build --build-arg INSTALL_LOCAL=true -t entrag:dev-local .
podman run --rm -p 7860:7860 --env-file .env entrag:dev-local
```

### 3. Use

- App: [http://localhost:7860](http://localhost:7860)
- LiteLLM proxy: [http://localhost:4000](http://localhost:4000)

### Running tests

```bash
# Docker
docker run --rm entrag:dev python -m pytest tests/ -v

# Podman
podman run --rm entrag:dev python -m pytest tests/ -v
```

## Local Models (Optional Feature)

Local models allow the application to run entirely offline — no external API calls for embeddings or reranking. This adds ~2.5GB to the image size due to PyTorch.

For detailed technical documentation, see:
- **[Embedding Models](docs/embedding.md)** — How vector embeddings work, provider options, pipeline flow
- **[Reranking](docs/reranking.md)** — How cross-encoders improve retrieval quality, two-stage architecture

### a) Local Embedding (Vector Indexing)

When local embedding is enabled, the app uses HuggingFace sentence-transformers models to generate vector embeddings in-process instead of calling the LiteLLM proxy / OpenAI API.

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_PROVIDER` | `litellm` | Set to `local` to use in-process HuggingFace models |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | HuggingFace model ID for local embeddings |

**Default model:** `BAAI/bge-large-en-v1.5` — a 335M parameter BGE model producing 1024-dim vectors. It ranks highly on the MTEB benchmark and balances quality with speed for enterprise RAG workloads.

**Quick enable via Docker Compose:**

```bash
docker compose --profile local-models up -d
```

This sets `EMBEDDING_PROVIDER=local` automatically. You can also set it manually in `.env`.

**Enable manually (without Docker):**

```bash
pip install -e ".[local]"
export EMBEDDING_PROVIDER=local
```

> 📖 **Deep dive:** [docs/embedding.md](docs/embedding.md) — embedding pipeline, chunking strategy, switching providers, performance comparison

### b) Local Reranker (Retrieval Quality Assurance)

The reranker is a cross-encoder model that re-scores retrieved documents to improve answer quality. It acts as a second-pass filter after initial vector/keyword retrieval.

| Setting | Default | Description |
|---------|---------|-------------|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for re-scoring |
| `RERANKER_TOP_N` | `5` | Number of top documents after reranking |

**Default model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — a lightweight cross-encoder trained on MS MARCO passage ranking. It takes a query-document pair as input and outputs a relevance score, allowing more precise ranking than vector similarity alone.

**Why it matters:**
1. Initial retrieval fetches `SIMILARITY_TOP_K` (default: 10) candidates via hybrid search
2. The cross-encoder scores each (query, document) pair with full attention over both texts
3. Only the top `RERANKER_TOP_N` (default: 5) documents are passed to the LLM for generation
4. This reduces hallucination by filtering out loosely similar but irrelevant context

The reranker requires the same `.[local]` extra as local embeddings (both depend on `sentence-transformers`):

```bash
pip install -e ".[local]"
```

> 📖 **Deep dive:** [docs/reranking.md](docs/reranking.md) — two-stage retrieval architecture, cross-encoder mechanics, quality impact examples, alternative models

## Configuration

All settings are configured via environment variables or `.env` file. See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_MODEL` | `gpt-4o` | LLM model for generation |
| `EMBEDDING_PROVIDER` | `litellm` | `litellm` or `local` |
| `LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (via LiteLLM) |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `RETRIEVAL_SIMILARITY_TOP_K` | `10` | Candidates before reranking |
| `RETRIEVAL_HYBRID_ALPHA` | `0.7` | Hybrid search weight (0=keyword, 1=vector) |

## Project Structure

```
entrag/
├── src/
│   ├── config.py          # Pydantic-settings configuration
│   ├── app.py             # Gradio web interface
│   ├── scraper/           # Broadcom KB scraper
│   ├── ingestion/          # Document ingestion pipeline
│   └── retrieval/          # Hybrid search + reranking engine
├── scripts/
│   ├── scrape.py           # CLI: entrag-scrape
│   └── ingest.py           # CLI: entrag-ingest
├── tests/                  # Pytest test suite
├── Dockerfile              # Container image (~500MB lean, ~3GB with local models)
├── compose.yaml            # Docker Compose: rag-app + litellm sidecar
├── litellm_config.yaml     # LiteLLM proxy model routing
└── pyproject.toml           # Package definition & dependencies
```

## Optional Dependencies

| Extra | Install | Size | Description |
|-------|---------|------|-------------|
| `local` | `pip install -e ".[local]"` | +2GB | Local embeddings + reranker (PyTorch, sentence-transformers) |
| `playwright` | `pip install -e ".[playwright]"` | +700MB | Browser-based auth scraping |
| `eval` | `pip install -e ".[eval]"` | varies | RAG evaluation framework (ragas) |
| `full` | `pip install -e ".[full]"` | +3GB | All optional dependencies |

## License

Private — internal use only.