# Reranking in EntRAG — Retrieval Quality Assurance

## Overview

Reranking is a **second-pass filtering step** that dramatically improves retrieval quality. While vector similarity search (embedding-based) finds documents that are broadly semantically related, reranking uses a **cross-encoder model** to precisely score how relevant each document is to the specific query.

This two-stage approach is a best practice in production RAG systems and is the primary mechanism for ensuring answer quality.

## Why Reranking is Essential

### The Problem with Vector Search Alone

Vector similarity search has limitations:

1. **Lexical gaps**: "boot failure" and "startup problem" have similar vectors, but "the system won't start" might score lower despite being semantically identical in context.

2. **False positives**: A document about "ESXi boot process" might rank highly for "ESXi boot failure" because it contains all the keywords, even if it doesn't address failures.

3. **Context mismatch**: Vector search treats the query and document independently. It doesn't understand that "How to fix X" requires a document with a resolution section, not just any document mentioning X.

### How Reranking Solves This

Cross-encoder rerankers take **both the query and document together** as input and output a single relevance score:

```
Input:  Query: "ESXi host fails to boot after firmware update"
        Document: "To resolve boot issues, check firmware compatibility..."

Output: Relevance Score: 0.92  (highly relevant)

Input:  Query: "ESXi host fails to boot after firmware update"
        Document: "This article describes the normal ESXi boot sequence..."

Output: Relevance Score: 0.31  (not relevant despite keyword overlap)
```

By attending to both texts simultaneously, cross-encoders capture nuanced relevance that vector similarity misses.

## The Reranking Pipeline

### Two-Stage Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

Stage 1: COARSE RETRIEVAL (Fast, Approximate)
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│ User Query   │────▶│ Embed Query      │────▶│ Vector Search  │
│ "boot fail"  │     │ (BGE/SBERT)     │     │ (LanceDB ANN)  │
└──────────────┘     └──────────────────┘     └───────┬────────┘
                                                        │
                                                        ▼
                                              ┌────────────────┐
                                              │ Top-K Results  │
                                              │ (K=10 default) │
                                              └───────┬────────┘
                                                      │
                                                      ▼
Stage 2: RERANKING (Precise, Slower)
┌──────────────────────────────────────────────────────────────┐
│ For each of the K documents:                                │
│    score = cross_encoder(query, document_text)               │
│    → Full attention over concatenated [query; document]      │
└──────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌────────────────┐
                                              │ Sort by Score  │
                                              │ Take Top-N     │
                                              │ (N=5 default)  │
                                              └───────┬────────┘
                                                      │
                                                      ▼
                                              ┌────────────────┐
                                              │ To LLM for     │
                                              │ Answer Gen     │
                                              └────────────────┘
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model |
| `RERANKER_TOP_N` | `5` | Number of documents to keep after reranking |
| `RETRIEVAL_SIMILARITY_TOP_K` | `10` | Documents retrieved before reranking |

**Typical flow:**
1. Vector search returns 10 candidates (`SIMILARITY_TOP_K`)
2. Reranker scores all 10 against the query
3. Top 5 by reranker score are passed to the LLM (`RERANKER_TOP_N`)

## The Default Reranker Model

### `cross-encoder/ms-marco-MiniLM-L-6-v2`

This is a lightweight but effective cross-encoder with the following characteristics:

**Architecture:**
- Based on MiniLM (distilled from BERT/ROBERTA)
- 6 transformer layers
- ~80M parameters (vs 335M for BGE embedding model)
- Trained with a classification head on top of `[CLS]` token

**Training Data:**
- MS MARCO passage ranking dataset (500k+ query-passage pairs)
- Each training example: `(query, passage, relevance_label)`
- Optimized with pairwise or listwise ranking loss

**Performance:**
- Fast inference: ~10-20ms per query-document pair on CPU
- Strong ranking quality despite small size
- Well-suited for RAG because MS MARCO covers diverse query types

**Output:**
- Score range: typically 0 to 1 (after sigmoid)
- Higher = more relevant
- Absolute values are less important than relative ranking

## How Cross-Encoders Work (Technical)

### Attention Over Concatenated Input

Unlike embedding models (bi-encoders) that encode query and document separately, cross-encoders process them jointly:

```
Input: [CLS] How to fix ESXi boot failure? [SEP] To resolve boot issues... [SEP]
        ↓
MiniLM Encoder (6 layers of self-attention)
        ↓
[CLS] token representation (768-dim)
        ↓
Linear head → Single relevance score
        ↓
Sigmoid → 0.92 (highly relevant)
```

The self-attention mechanism allows every token in the query to attend to every token in the document, and vice versa. This captures:
- Query-document term interactions
- Contextual relevance beyond keyword matching
- Negation and scope (e.g., "not fixed" vs "fixed")

### Why Cross-Encoders are More Accurate

| Aspect | Bi-Encoder (Embedding) | Cross-Encoder (Reranker) |
|--------|------------------------|--------------------------|
| Encoding | Query and doc separate | Joint encoding |
| Interaction | Dot product (no interaction) | Full attention (deep interaction) |
| Speed | Fast (~10ms per doc) | Slower (~20ms per doc) |
| Accuracy | Good for broad retrieval | Best for precise ranking |
| Use case | Retrieve candidates | Re-rank candidates |

## Enabling Reranking in EntRAG

### Prerequisites

Reranking requires the `.[local]` extra (same as local embeddings):

```bash
pip install -e ".[local]"
# or with Docker:
docker build --build-arg INSTALL_LOCAL=true -t entrag:dev-local .
```

This installs:
- `sentence-transformers` (which provides cross-encoder support)
- PyTorch (required by sentence-transformers)

### Configuration

In `.env`:
```bash
# Reranker is enabled automatically if .[local] is installed
# These are the defaults:
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=5
RETRIEVAL_SIMILARITY_TOP_K=10
```

### Without Reranking

If the `.[local]` extra is not installed:
- The reranker step is **skipped**
- All `SIMILARITY_TOP_K` documents are passed directly to the LLM
- Retrieval still works, but may include less relevant documents
- This is acceptable for basic use cases

## Quality Impact

### Example: Query "ESXi 7.0 U3 boot failure after patch"

**Without reranking** (top 3 passed to LLM):
1. "ESXi 7.0 Update 3 Release Notes" — keyword match, but not a fix
2. "How to boot ESXi from USB" — wrong topic, keyword overlap only
3. "Resolved: Boot failure after 7.0 U3 update" — **correct answer**

**With reranking** (reranker scores, then top 3):
1. "Resolved: Boot failure after 7.0 U3 update" — score 0.94
2. "Troubleshooting ESXi boot issues" — score 0.78
3. "7.0 U3 known issues list" — score 0.65

The reranker correctly identifies that document #3 (from the no-reranking case) is less relevant than other candidates, improving the context quality for the LLM.

## Alternative Reranker Models

You can swap the reranker model by changing `RERANKER_MODEL`:

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fastest | Good | ~80MB | Default, balanced |
| `cross-encoder/ms-marco-BERT-base` | Medium | Better | ~420MB | Higher quality, slower |
| `cross-encoder/ms-marco-roberta-large` | Slow | Best | ~1.3GB | Production, highest quality |

To use an alternative:
```bash
RERANKER_MODEL=cross-encoder/ms-marco-BERT-base
```

The model will be downloaded automatically from HuggingFace Hub on first use.

## Performance Considerations

### Latency Budget

For `RERANKER_TOP_N=5` and `SIMILARITY_TOP_K=10`:
- Reranking 10 docs: ~100-200ms on CPU
- Added to vector search (~50ms) and LLM generation (~2-5s)
- Total latency impact: ~5-10%

This is an excellent trade-off for the quality improvement.

### Batching

The reranker processes documents sequentially by default. For high-throughput scenarios, batching can be enabled in the retrieval module:

```python
# Future enhancement: batch reranking
scores = reranker.predict([(query, doc) for doc in documents], batch_size=32)
```

### GPU Acceleration

If CUDA is available, the reranker can use the GPU:
```python
import torch
if torch.cuda.is_available():
    reranker.model.to("cuda")
```

This reduces per-document scoring to ~2-5ms.

## Troubleshooting

### Reranker not working?

1. Check if `.[local]` is installed:
   ```bash
   python -c "from sentence_transformers import CrossEncoder; print('OK')"
   ```

2. Check `RERANKER_MODEL` is a valid HuggingFace model

3. Check logs for import errors when retrieval initializes

### Slow reranking?

- Reduce `SIMILARITY_TOP_K` (try 5 instead of 10)
- Use the default MiniLM model (fastest)
- Consider GPU acceleration for production workloads

### Poor ranking quality?

- Try a larger reranker model (BERT-base or roberta-large)
- Ensure document chunks are not too short (need enough context)
- Check that the reranker is actually being called (add logging)
