# Embedding Models in EntRAG

## Overview

Embeddings are dense vector representations of text that capture semantic meaning. In EntRAG, embeddings power the **vector similarity search** component of the retrieval pipeline — allowing the system to find KB articles that are semantically related to a user's query, even when exact keywords don't match.

## How Embeddings Work

### 1. Text to Vector

When a document is ingested, it is:
1. Split into overlapping chunks (typically 512-1024 tokens)
2. Each chunk is passed through an embedding model
3. The model outputs a fixed-length vector (e.g., 1024 dimensions for BGE models)
4. The vector is stored in LanceDB alongside the original text

```
Input: "ESXi host fails to boot after firmware update"
Output: [0.123, -0.456, 0.789, ..., 0.234]  (1024-dim vector)
```

### 2. Query Embedding

When a user asks a question:
1. The query is embedded using the **same model** as document ingestion
2. This ensures queries and documents exist in the same vector space
3. Cosine similarity or dot product is used to find the closest document vectors

### 3. Vector Search

LanceDB performs approximate nearest neighbor (ANN) search:
- Uses vector indexing (HNSW or IVF) for fast retrieval
- Returns top-K most similar chunks by vector distance
- Typical default: `RETRIEVAL_SIMILARITY_TOP_K = 10`

## Embedding Provider Options

EntRAG supports two embedding providers, controlled by `EMBEDDING_PROVIDER`:

### A) LiteLLM Proxy (Remote API) — Default

```
EMBEDDING_PROVIDER=litellm
LITELLM_EMBEDDING_MODEL=text-embedding-3-small
```

**How it works:**
1. Document chunks are sent to the LiteLLM proxy (port 4000)
2. LiteLLM forwards requests to the configured API (e.g., OpenAI)
3. OpenAI `text-embedding-3-small` returns 1536-dim vectors
4. Vectors are stored in LanceDB

**Pros:**
- No local GPU required
- Fast inference (API-based)
- Small container image (~500MB)

**Cons:**
- Requires internet connectivity
- API costs for large document sets
- Latency dependent on network

### B) Local HuggingFace Models

```
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

**How it works:**
1. On first use, the model is downloaded from HuggingFace Hub
2. `llama-index-embeddings-huggingface` loads the model via `sentence-transformers`
3. Inference runs in-process using PyTorch (CPU or CUDA)
4. Vectors are generated locally and stored in LanceDB

**Default model: `BAAI/bge-large-en-v1.5`**
- 335M parameters (BERT-large architecture)
- 1024-dimensional output
- Trained with contrastive learning on diverse text pairs
- Ranked #1 on MTEB leaderboard (at time of release) for English tasks
- Good balance of quality vs. speed for RAG workloads

**Pros:**
- Fully offline capable
- No API costs
- Lower latency after model is loaded
- Data privacy (no external API calls)

**Cons:**
- Large container image (~3GB with PyTorch)
- Slower inference on CPU
- Requires more RAM (~2GB for model + overhead)

## Embedding Pipeline Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw HTML/Text  │────▶│  Parse & Chunk   │────▶│  Embed Chunks   │
│  (KB Articles)  │     │  (512 tok/chunk) │     │  (local or API) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
                                                      ┌─────────────────┐
                                                      │  LanceDB Store  │
                                                      │  (vectors + text)│
                                                      └─────────────────┘

                        ─── Query Time ───

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Query     │────▶│  Embed Query    │────▶│  Vector Search │
│  "boot fails"   │     │  (same model)   │     │  (top-K similar)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
                                                      ┌─────────────────┐
                                                      │  Retrieved Docs │
                                                      │  → Reranker    │
                                                      └─────────────────┘
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `litellm` | `litellm` or `local` |
| `LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Model name for LiteLLM/OpenAI API |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | HuggingFace model ID for local embedding |
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LANCEDB_PATH` | `./data/lancedb` | Where vectors are stored |

## Chunking Strategy

Document chunking is critical for embedding quality:

- **Chunk size**: Typically 512-1024 tokens
- **Overlap**: 10-20% overlap between chunks to preserve context
- **Boundary detection**: Chunks respect section boundaries (Symptoms, Cause, Resolution) when possible

This ensures that each chunk is:
- Small enough for the embedding model's context window
- Large enough to contain coherent information
- Overlapping to avoid splitting related content across chunks

## Vector Similarity Metrics

EntRAG uses **cosine similarity** for vector comparison:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Values range from -1 (opposite) to 1 (identical). In practice, relevant documents score 0.7-0.9+ for well-matched queries.

## Switching Providers

### From LiteLLM to Local

```bash
# 1. Install local dependencies
pip install -e ".[local]"

# 2. Update .env
EMBEDDING_PROVIDER=local
# LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5  # (default)

# 3. Re-ingest documents (vectors from different models are NOT compatible)
rm -rf data/lancedb/
entrag-ingest
```

### From Local to LiteLLM

```bash
# 1. Update .env
EMBEDDING_PROVIDER=litellm

# 2. Re-ingest documents
rm -rf data/lancedb/
entrag-ingest
```

**Important:** Vectors from different embedding models are NOT compatible. Switching providers requires re-ingesting all documents.

## Performance Considerations

| Factor | LiteLLM API | Local (CPU) | Local (GPU) |
|--------|------------|-------------|-------------|
| Embedding speed | ~100 docs/sec (network bound) | ~10 docs/sec | ~100+ docs/sec |
| RAM usage | ~500MB (app only) | ~2.5GB (model) | ~2.5GB + VRAM |
| Cold start | Fast (no model load) | Slow (~30s model load) | Slow (~30s model load) |
| Ongoing latency | ~50-200ms per doc (network) | ~100ms per doc | ~10ms per doc |
