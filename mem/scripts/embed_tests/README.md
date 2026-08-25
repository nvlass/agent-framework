# Embedding Test Scripts

Educational scripts demonstrating text embeddings and how they're used in the memory system.

## Prerequisites

1. Download an embedding model (at least one):
   ```bash
   # Default model: nomic-embed-text (~137MB, fast)
   python scripts/setup_embedding_model.py

   # Optional: Better quality model (~670MB)
   # See "Available Models" section below
   ```

2. Install dependencies (if not already):
   ```bash
   pip install hdbscan
   ```

## Available Models

All scripts support `--model` / `-m` to select the embedding model:

| Model | Size | Dimensions | Quality | Speed |
|-------|------|------------|---------|-------|
| `nomic` (default) | ~137MB | 768 | Good | Fast |
| `mxbai` | ~670MB | 1024 | Better | Slower |

**To download mxbai-embed-large:**
```bash
pip install huggingface_hub
huggingface-cli download mixedbread-ai/mxbai-embed-large-v1 \
  --include 'gguf/mxbai-embed-large-v1-f16.gguf' \
  --local-dir models
mv models/gguf/mxbai-embed-large-v1-f16.gguf models/
```

## Scripts

### 01_basic_embeddings.py

**What it covers:**
- What embeddings are (dense vector representations)
- Generating embeddings with llama.cpp
- Embedding properties (dimension, normalization, statistics)
- Embedding caching for performance

**Key concepts:**
- Embeddings convert text to 768-dimensional vectors
- Similar texts produce similar vectors
- Embeddings are typically normalized (unit length)

```bash
python scripts/embed_tests/01_basic_embeddings.py
python scripts/embed_tests/01_basic_embeddings.py --model mxbai  # compare with better model
```

---

### 02_similarity_exploration.py

**What it covers:**
- How cosine similarity works
- Similarity ranges for different text relationships
- Edge cases (polysemy, negation, word order)
- Practical example: finding similar problems

**Key concepts:**
- Similarity range: -1 to 1 (1 = identical)
- 0.7-0.9 = very similar / same topic
- 0.3-0.5 = weakly related
- 0.0-0.3 = unrelated

```bash
python scripts/embed_tests/02_similarity_exploration.py
python scripts/embed_tests/02_similarity_exploration.py -m mxbai  # compare models
```

---

### 03_clustering_demo.py

**What it covers:**
- Why clustering is useful for memory consolidation
- How HDBSCAN works (vs K-means)
- Interpreting cluster results
- Handling noise/outliers

**Key concepts:**
- HDBSCAN finds natural cluster count (no need to specify k)
- Labels outliers as noise (-1)
- Works well with embeddings of varying density
- `min_cluster_size` controls minimum points per cluster

```bash
python scripts/embed_tests/03_clustering_demo.py
```

---

### 04_full_system_test.py

**What it covers:**
- Complete memory system pipeline
- Storing episodes with embeddings
- Semantic retrieval
- Consolidation (clustering + pattern extraction)
- Pattern-based recommendations

**Key concepts:**
- Memory hierarchy: working -> short-term -> long-term
- Patterns emerge from clusters of similar experiences
- Recommendations use both pattern matching and confidence

```bash
python scripts/embed_tests/04_full_system_test.py
```

---

## Quick Reference

### Embedding Model

- **Model**: nomic-embed-text-v1.5 (Q8_0 quantization)
- **Size**: ~137MB
- **Dimensions**: 768
- **Location**: `models/nomic-embed-text-v1.5.Q8_0.gguf`

### Similarity Guidelines

| Range | Meaning |
|-------|---------|
| 0.9-1.0 | Nearly identical / paraphrases |
| 0.7-0.9 | Very similar / same topic & intent |
| 0.5-0.7 | Related / same domain |
| 0.3-0.5 | Weakly related |
| 0.0-0.3 | Unrelated |

### HDBSCAN Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 5 | Minimum points to form a cluster |
| `min_samples` | None | How conservative noise detection is |
| `metric` | 'euclidean' | Distance metric (also try 'cosine') |

### Cosine Similarity Formula

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

For normalized vectors (||a|| = ||b|| = 1): `similarity = np.dot(a, b)`
