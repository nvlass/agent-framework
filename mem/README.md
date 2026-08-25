# Agent Long-Term Memory System

A complete agent memory system with learning, reflection, and adaptation capabilities for local llama.cpp installation.

## Current Status: Phase 1 Complete ✓

**Phase 1: Foundation - Basic Episodic Memory** is now implemented with:
- ✅ Episode storage and retrieval
- ✅ Embedding generation interface
- ✅ Semantic similarity search
- ✅ SQLite + ChromaDB integration
- ✅ Comprehensive test suite

## Quick Start

### Prerequisites

1. **Python 3.10+** (you have 3.11.14 via asdf)
2. **llama.cpp** (already installed via brew)
3. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **GGUF Embedding Model** (optional for testing, required for production):
   - Download from HuggingFace (e.g., `all-MiniLM-L6-v2`, `bge-small-en`)
   - Or use the mock generator for development

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests to verify everything works
pytest tests/test_basic_memory.py -v

# Try the example (uses mock embeddings)
python example_usage.py
```

## Project Structure

```
mem/
├── src/
│   ├── schema.sql          # Database schema
│   ├── embeddings.py       # Embedding generation interface
│   └── memory_store.py     # Core memory storage
├── tests/
│   └── test_basic_memory.py  # Test suite
├── data/                   # Created automatically
│   ├── agent_memory.db     # SQLite database
│   └── memory_vectors/     # ChromaDB vector store
├── example_usage.py        # Demo script
├── requirements.txt        # Dependencies
└── CLAUDE.md              # Full roadmap
```

## Usage

### Basic Example

```python
from embeddings import EmbeddingGenerator
from memory_store import MemoryStore

# Initialize (with real model)
embedding_gen = EmbeddingGenerator(model_path="path/to/model.gguf")
memory_store = MemoryStore(embedding_generator=embedding_gen)

# Store an episode
episode_id = memory_store.store_episode(
    context="User asked about Python decorators",
    action="Explained with examples and use cases",
    outcome="User understood and applied successfully",
    success_score=0.9,
    tags=["python", "teaching", "decorators"]
)

# Semantic search
results = memory_store.retrieve_episodes(
    query="How do Python decorators work?",
    limit=5
)

for episode, similarity in results:
    print(f"Similarity: {similarity:.2f}")
    print(f"Context: {episode.context}")
    print(f"Action: {episode.action}")
    print()

# Get recent episodes
recent = memory_store.get_recent_episodes(hours=24, limit=10)

# Statistics
stats = memory_store.get_stats()
print(f"Total episodes: {stats['total_episodes']}")
print(f"Average success: {stats['average_success_score']}")
```

### Tag-Based Querying (SQLite JSON)

The system uses SQLite's native JSON functions for efficient tag querying:

```python
# Find episodes by single tag
python_episodes = memory_store.get_episodes_by_tag("python")

# Find episodes with ANY of the specified tags
episodes = memory_store.get_episodes_by_tags(
    ["python", "javascript"],
    match_all=False
)

# Find episodes with ALL specified tags
advanced_python = memory_store.get_episodes_by_tags(
    ["python", "advanced"],
    match_all=True
)

# Get all tags with usage counts
tags = memory_store.get_all_tags()
for tag, count in tags:
    print(f"{tag}: {count} episodes")
```

**Benefits of JSON storage:**
- ✅ SQL-level tag filtering (no need to load all episodes)
- ✅ Automatic JSON validation via CHECK constraint
- ✅ Efficient indexing for tag searches
- ✅ Support for complex queries (AND/OR combinations)

See `example_tag_queries.py` for a complete demonstration.

## Running Tests

```bash
# Run all tests
pytest tests/test_basic_memory.py -v

# Run with coverage
pytest tests/test_basic_memory.py --cov=src

# Run performance tests (slower)
pytest tests/test_basic_memory.py -v -m slow
```

## Phase 1 Success Criteria

- ✅ Can store 100+ episodes
- ✅ Can retrieve relevant episodes via semantic search
- ✅ Retrieval latency < 1 second for 1000 episodes
- ✅ All tests passing

## Next Steps

See `CLAUDE.md` for the complete roadmap. Next phases:

- **Phase 2**: Learning from Outcomes (outcome classification, pattern recognition)
- **Phase 3**: Semantic Memory & Consolidation (knowledge extraction)
- **Phase 4**: Reflection & Meta-Learning (causal reasoning)
- **Phase 5**: Adaptation & Transfer Learning (cross-domain application)
- **Phase 6**: Advanced Features & Polish (production-ready system)

## Architecture

### Storage Layers

1. **SQLite**: Structured data (episodes, metadata, timestamps)
2. **ChromaDB**: Vector embeddings for semantic search
3. **Embedding Generator**: llama.cpp interface for text→vector

### Key Components

- **`Episode`**: Data class representing a single memory
- **`MemoryStore`**: Main interface for storage/retrieval
- **`EmbeddingGenerator`**: Wrapper around llama.cpp embeddings

### Design Principles

- **Local-first**: No external API dependencies
- **Incremental**: Each phase builds on previous
- **Educational**: Well-commented, clear design decisions
- **Practical**: Focus on working implementations

## Configuration

Create a `config.yaml` (coming in Phase 6):

```yaml
memory:
  embedding_model: "path/to/model.gguf"
  llm_model: "path/to/llm.gguf"  # For Phase 3+
  vector_store_path: "./data/memory_vectors"
  sqlite_path: "./data/agent_memory.db"
```

## Getting an Embedding Model

For production use, download a GGUF embedding model:

```bash
# Example: Using Nomic Embed
# Download from HuggingFace and convert to GGUF if needed

# Or use any embedding model supported by llama.cpp
```

Common choices:
- **all-MiniLM-L6-v2**: Small, fast (384 dims)
- **bge-small-en**: Good balance (384 dims)
- **nomic-embed-text**: High quality (768 dims)

## Development

### Adding New Features

1. Update schema in `src/schema.sql` if needed
2. Add functionality to relevant module
3. Write tests in `tests/`
4. Update documentation

### Running in Development

```python
# Use mock embeddings for fast iteration
from memory_store import MemoryStore

class MockGenerator:
    def generate_embedding(self, text, use_cache=True):
        import numpy as np
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(384).astype(np.float32)
        return emb / np.linalg.norm(emb)

store = MemoryStore(embedding_generator=MockGenerator())
```

## Troubleshooting

### "Model not loaded" error
- Make sure you call `load_model()` or pass `model_path` to `EmbeddingGenerator`
- For testing, use the mock generator (see example_usage.py)

### ChromaDB warnings
- Telemetry disabled by default in Settings
- Vector store created automatically in `data/memory_vectors/`

### Performance issues
- Ensure you're using appropriate embedding model size
- Check retrieval latency with performance tests
- Consider adjusting `n_ctx` parameter for embeddings

## Contributing

This is a learning project! Feel free to:
- Experiment with different designs
- Add new features from later phases
- Improve performance or code quality
- Add more comprehensive tests

## License

MIT (or whatever you prefer)

## Resources

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **ChromaDB**: https://www.trychroma.com/
- **Roadmap**: See `CLAUDE.md` for complete project plan
