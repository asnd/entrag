# EntRAG

VMware/Broadcom Knowledge Base RAG Application — a Retrieval-Augmented Generation system that scrapes, indexes, and queries Broadcom/VMware KB articles through a Gradio web interface.

## Architecture

EntRAG uses a **multi-container architecture** with clear separation of runtime responsibilities:

### Containers by Responsibility

| Container | Purpose | Image Size | Runtime |
|-----------|---------|-----------|---------|
| `rag-app` | Main RAG application (Gradio UI, retrieval, ingestion) | ~500MB | Always on |
| `scraper` | One-shot scraping job (Playwright/browser) | ~700MB | On-demand (`--profile scrape`) |
| `ingest` | One-shot ingestion job (LlamaIndex + LanceDB) | ~500MB | On-demand (`--profile ingest`) |
| `litellm` | Proxy for remote LLM/embedding APIs (OpenAI) | ~200MB | Always on |
| `litellm-local` | Proxy with local models (sentence-transformers) | ~3GB | Optional (`--profile local-models`) |

### Key Design Decisions

- **No monolith**: Main app stays lean (~500MB). Heavy ML models live in their own container.
- **Jobs, not always-on**: Scraping and ingestion run as one-shot containers (not long-running services).
- **LiteLLM as model abstraction**: The app **always calls LiteLLM**. Local vs remote is just a config change in LiteLLM.
- **Shared `data` volume**: Raw HTML + LanceDB mounted into scraper/ingest/app containers.

### Data Flow

```
Step 1: Scrape (one-shot)
  docker compose --profile scrape run --rm scraper search --query "vmware" --max 73
  → Downloads HTML to ./data/raw/

Step 2: Ingest (one-shot)
  docker compose --profile ingest run --rm ingest --source ./data/raw --reset
  → Parses HTML, chunks, embeds via LiteLLM → LanceDB at ./data/lancedb/

Step 3: Serve (always on)
  docker compose up -d
  → Gradio UI at http://localhost:7860
```

## Features & Optimizations

### Implemented
- ✅ **Incremental scraping** — SHA256 checksums detect content changes, skip unchanged articles
- ✅ **HTTP connection pooling** — Tuned connection limits (100 max, 20 keepalive) for faster scraping
- ✅ **Resumable scraping** — Persists state (downloaded/failed articles, checksums) to disk
- ✅ **Container separation** — App, scraper, ingestion, and ML models in separate containers
- ✅ **Dual container support** — Works with both Docker and Podman (`Containerfile` → `Dockerfile` symlink)
- ✅ **Comprehensive testing** — 38+ pytest tests covering config, scraper, parser, and container configs

### Planned (MVP Blockers)
- ⏳ **Gradio UI** — Chat interface with source citations
- ⏳ **Retrieval engine** — Hybrid search (vector + BM25) with reranking

## Quick Start

### Prerequisites

- **Docker** (recommended) or **Podman** (both supported)
- OpenAI API key (or another provider supported by LiteLLM)

> **Podman users:** The project provides `Containerfile` symlink to `Dockerfile.app`. Replace `docker` with `podman` in all commands below.

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set LITELLM_API_KEY and OPENAI_API_KEY
```

### 2. Build and run (Remote Models - Default)

**Docker:**
```bash
# Start main app + LiteLLM proxy (remote APIs)
docker compose up -d

# Scrape articles (one-shot job)
docker compose -f compose.yaml -f compose.jobs.yaml run --rm scraper search --query "vmware" --max 73

# Ingest into LanceDB (one-shot job)
docker compose -f compose.yaml -f compose.jobs.yaml run --rm ingest --source ./data/raw --reset
```

**Podman:**
```bash
podman compose up -d

podman compose -f compose.yaml -f compose.jobs.yaml run --rm scraper search --query "vmware" --max 73

podman compose -f compose.yaml -f compose.jobs.yaml run --rm ingest --source ./data/raw --reset
```

### 3. With Local Models (Optional)

Local models run in a separate LiteLLM container — they do NOT bloat the main app.

**Docker:**
```bash
# Start everything with local embedding service
docker compose -f compose.yaml -f compose.local.yaml up -d

# Or start local model service only
docker compose -f compose.yaml -f compose.local.yaml up litellm-local

# Then point app to local service by uncommenting in .env:
#   LITELLM_BASE_URL=http://litellm-local:4001
#   LITELLM_EMBEDDING_MODEL=local-embedding
```

**Podman:**
```bash
podman compose -f compose.yaml -f compose.local.yaml up -d
```

### 4. Use

- App: [http://localhost:7860](http://localhost:7860)
- LiteLLM proxy (remote): [http://localhost:4000](http://localhost:4000)
- LiteLLM proxy (local): [http://localhost:4001](http://localhost:4001)

### Running tests

```bash
# Docker
docker run --rm entrag-app python -m pytest tests/ -v

# Podman
podman run --rm entrag-app python -m pytest tests/ -v
```

### 2. Build and run (Remote Models - Default)

**Docker:**
```bash
# Start main app + LiteLLM proxy (remote APIs)
docker compose up -d

# Scrape articles (one-shot job)
docker compose --profile scrape run --rm scraper search --query "vmware" --max 73

# Ingest into LanceDB (one-shot job)
docker compose --profile ingest run --rm ingest --source ./data/raw --reset
```

**Podman:**
```bash
podman compose up -d
podman compose --profile scrape run --rm scraper search --query "vmware" --max 73
podman compose --profile ingest run --rm ingest --source ./data/raw --reset
```

### 3. With Local Models (Optional)

Local models run in a separate LiteLLM container (`litellm-local`) — they do NOT bloat the main app.

```bash
# Start everything with local embedding service
docker compose --profile local-models up -d

# Or start local model service only
docker compose --profile local-models up litellm-local
# Then point app to it by uncommenting in .env:
#   LITELLM_BASE_URL=http://litellm-local:4000
#   LITELLM_EMBEDDING_MODEL=local-embedding
```

### 4. Use

- App: [http://localhost:7860](http://localhost:7860)
- LiteLLM proxy (remote): [http://localhost:4000](http://localhost:4000)
- LiteLLM proxy (local): [http://localhost:4001](http://localhost:4001)

### Running tests

```bash
# Docker
docker run --rm entrag-app python -m pytest tests/ -v

# Podman
podman run --rm entrag-app python -m pytest tests/ -v
```

## Local Models (Optional Feature)

Local models allow the application to run entirely offline — no external API calls for embeddings or reranking. The models run inside a **separate LiteLLM container** (`litellm-local`), keeping the main app lean.

For detailed technical documentation, see:
- **[Embedding Models](docs/embedding.md)** — How vector embeddings work, provider options, pipeline flow
- **[Reranking](docs/reranking.md)** — How cross-encoders improve retrieval quality, two-stage architecture

### a) Local Embedding (Vector Indexing)

When local embedding is enabled, LiteLLM loads `sentence-transformers` models in the `litellm-local` container:

| Setting | Default | Description |
|---------|---------|-------------|
| `LITELLM_BASE_URL` | `http://litellm:4000` (remote) or `http://litellm-local:4001` (local) | LiteLLM proxy URL |
| `LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` (remote) or `local-embedding` (local) | Model name |

**Default local model:** `BAAI/bge-large-en-v1.5` — configured in `litellm_config_local.yaml`

**Quick enable via Docker Compose:**
```bash
docker compose --profile local-models up -d
```

> 📖 **Deep dive:** [docs/embedding.md](docs/embedding.md) — embedding pipeline, chunking strategy, switching providers, performance comparison

### b) Local Reranker (Retrieval Quality Assurance)

The reranker is a cross-encoder model that re-scores retrieved documents to improve answer quality. It runs as a second LiteLLM model entry.

| Setting | Default | Description |
|---------|---------|-------------|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for re-scoring |

**Default model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — a lightweight cross-encoder trained on MS MARCO passage ranking.

> 📖 **Deep dive:** [docs/reranking.md](docs/reranking.md) — two-stage retrieval architecture, cross-encoder mechanics, quality impact examples, alternative models

## Configuration

All settings are configured via environment variables or `.env` file. See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://litellm:4000` | LiteLLM proxy URL (remote or local) |
| `LITELLM_MODEL` | `gpt-4o` | LLM model for generation |
| `LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (via LiteLLM) |
| `LANCEDB_PATH` | `./data/lancedb` | LanceDB vector store path |
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
├── docs/                    # ML documentation (embedding, reranking)
├── Dockerfile.app              # Main app image (~500MB)
├── Dockerfile.scraper          # Scraper job image (~700MB)
├── Dockerfile.local-models     # Local ML models image (~3GB)
├── compose.yaml            # Multi-container orchestration
├── litellm_config.yaml     # LiteLLM remote config
├── litellm_config_local.yaml # LiteLLM local models config
└── pyproject.toml           # Package definition & dependencies
```

## Optional Dependencies

| Extra | Install | Size | Description |
|-------|---------|------|-------------|
| `playwright` | `pip install -e ".[playwright]"` | +700MB | Browser-based auth scraping |
| `eval` | `pip install -e ".[eval]"` | varies | RAG evaluation framework (ragas) |
| `full` | `pip install -e ".[full]"` | +1GB | All optional dependencies |

## License

Private — internal use only.
