# EntRAG

EntRAG is a VMware/Broadcom knowledge-base retrieval app. It scrapes KB articles, parses them into structured sections, indexes them in LanceDB, and exposes a Gradio interface that returns grounded KB excerpts with citations.

## What is implemented

- Broadcom KB scraping with resumable state and checksum-based skip logic
- HTML parsing into structured sections such as Symptoms, Cause, and Resolution
- LanceDB-backed ingestion with metadata-preserving chunks
- Hybrid retrieval over the stored index, with lightweight reranking biased toward actionable sections
- A Gradio UI that queries the index instead of returning a placeholder
- Pytest and Ruff coverage for config, scraping, parsing, retrieval, app wiring, and container layout

## Current retrieval behavior

The runtime is retrieval-first:

1. Query embedding is generated through LiteLLM
2. LanceDB hybrid search retrieves candidate chunks
3. A lightweight reranker boosts exact term overlap and section intent
4. The UI returns the strongest KB excerpts and source links

This keeps responses grounded in indexed content and makes retrieval quality observable before adding a full answer-synthesis layer.

## Architecture

### Runtime split

| Component | Responsibility |
| --- | --- |
| `rag-app` | Gradio UI and retrieval |
| `litellm` | Model proxy for embeddings and optional generation |
| `scraper` | One-shot article collection job |
| `ingest` | One-shot parsing + embedding + indexing job |
| `litellm-local` | Optional isolated local-model sidecar |

### Data flow

```text
Broadcom KB search/download
  -> ./data/raw/*.html
  -> parser extracts sections + metadata
  -> ingestion chunks and embeds content
  -> LanceDB stores vectors + metadata
  -> retrieval queries LanceDB and reranks matches
  -> Gradio returns cited excerpts
```

## Repository layout

```text
src/
  app.py                Gradio entrypoint
  config.py             Settings and runtime resolution helpers
  ingestion/            Parsing -> chunking -> LanceDB indexing
  retrieval/            Hybrid search and reranking
  scraper/              Broadcom KB scraper and parser
scripts/
  scrape.py             CLI wrapper for scraping
  ingest.py             CLI wrapper for ingestion
tests/                  Automated coverage
docs/                   Deeper notes on embeddings and reranking
```

## Quick start

### 1. Configure the environment

```bash
cp .env.example .env
```

At minimum set:

- `OPENAI_API_KEY`
- `LITELLM_API_KEY`
- optionally `LITELLM_MASTER_KEY`

The app now validates `LITELLM_API_KEY` before ingestion or retrieval starts.

### 2. Start the always-on services

```bash
docker compose up -d
```

### 3. Scrape articles

```bash
docker compose -f compose.yaml -f compose.jobs.yaml run --rm scraper search \
  --query "ESXi boot failure" \
  --max 25
```

### 4. Build the index

```bash
docker compose -f compose.yaml -f compose.jobs.yaml run --rm ingest \
  --source ./data/raw \
  --reset
```

### 5. Query the app

- UI: http://localhost:7860
- LiteLLM health/API proxy: http://localhost:4000

Ask product, version, error, or symptom-specific questions such as:

- `How do I fix an ESXi host boot failure after a patch?`
- `What are the symptoms of vCenter certificate issues?`
- `Why does NSX Manager fail to start after upgrade?`

## CLI usage

### Scrape

```bash
python -m scripts.scrape search --query "vmware" --max 20
python -m scripts.scrape fetch --numbers 10001,10002
python -m scripts.scrape parse --input ./data/raw
python -m scripts.scrape status
```

### Ingest

```bash
python -m scripts.ingest --source ./data/raw --reset
python -m scripts.ingest --local
```

`--local` switches the ingestion command to the local LiteLLM endpoint/model aliases instead of mutating process environment variables.

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM endpoint used by the app |
| `LITELLM_API_KEY` | placeholder | Auth token used by the app to call LiteLLM |
| `LITELLM_MASTER_KEY` | `sk-entrag-dev` in example only | LiteLLM proxy master key |
| `LITELLM_MODEL` | `gpt-4o` | Optional generation model |
| `LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model alias |
| `LANCEDB_PATH` | `./data/lancedb` | Vector store path |
| `SCRAPER_OUTPUT_DIR` | `./data/raw` | Downloaded article directory |
| `RETRIEVAL_SIMILARITY_TOP_K` | `10` | Candidate pool size from LanceDB |
| `RETRIEVAL_HYBRID_ALPHA` | `0.7` | LanceDB hybrid weighting |
| `RERANKER_TOP_N` | `5` | Final number of displayed matches |

## Security notes

- Do not keep real LiteLLM keys in tracked config files
- LiteLLM master keys are now sourced from environment instead of hardcoded in YAML
- Authenticated scraping is opt-in and should only be used when you understand the legal and operational implications
- Placeholder API keys are rejected before ingestion and retrieval start network calls

## Retrieval quality notes

- Section metadata is stored with each chunk, which lets reranking prefer `resolution` content for fix-oriented questions and `cause` content for root-cause questions
- Hybrid search uses LanceDB first and falls back to vector-only mode if hybrid capabilities are unavailable
- Ingestion opens LanceDB in append mode unless `--reset` is passed, preventing accidental index replacement during normal updates

## Development

### Local checks

```bash
python -m ruff check src/ scripts/ tests/
python -m pytest tests/ -v --tb=short
```

### Test coverage

The test suite covers:

- configuration validation and runtime helpers
- scraper state, search, download, and auth behavior
- parser extraction and serialization
- retrieval ranking and answer formatting
- app error handling
- container and compose layout expectations

## Additional guides

- [docs/embedding.md](docs/embedding.md)
- [docs/reranking.md](docs/reranking.md)
