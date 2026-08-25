# Dependency Injection Refactoring

## Problem: Boolean Flags in Production Code

**Before (anti-pattern):**
```python
def create_memory_store(use_mock: bool = True):
    if use_mock:
        # Mock code
    else:
        # Production code
```

**Issues:**
- 🚫 Production code contains test logic
- 🚫 Boolean flag to switch between modes
- 🚫 Hard to extend (what about different mock types?)
- 🚫 Not functional/pure

## Solution: Pure Dependency Injection

### Architecture

```
┌─────────────────────────────────────────────┐
│  Production: memory_cli.py                  │
│                                             │
│  main()                                     │
│    ├─> read config/env                      │
│    ├─> create_memory_store_from_config()   │
│    └─> pass store to command functions     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Demo: demo_cli.py                          │
│                                             │
│  interactive_demo()                         │
│    ├─> create MockEmbeddingGenerator        │
│    ├─> create MemoryStore(mock_gen)         │
│    └─> pass store to command functions     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Tests: test_cli.py                         │
│                                             │
│  test_command_stats(populated_store)        │
│    ├─> pytest fixture creates test store   │
│    └─> pass store to command functions     │
└─────────────────────────────────────────────┘
```

### Key Principles

1. **Command functions are pure** - they only depend on their parameters
2. **No global state** - everything passed as arguments
3. **No boolean flags** - behavior determined by what you pass in
4. **Separation of concerns** - config reading separate from business logic

## Implementation

### Production Code (`src/memory_cli.py`)

```python
def create_memory_store_from_config(config_path: Optional[str] = None) -> MemoryStore:
    """
    Create memory store from configuration sources:
    1. Environment variable EMBEDDING_MODEL_PATH
    2. Config file (--config flag)
    3. Default config.yaml
    """
    model_path = os.getenv('EMBEDDING_MODEL_PATH')

    if not model_path and config_path:
        # Load from YAML config
        pass

    if not model_path:
        raise RuntimeError("No embedding model configured")

    embedding_gen = EmbeddingGenerator(model_path=model_path)
    return MemoryStore(embedding_generator=embedding_gen)

def main():
    args = parser.parse_args()

    # Create store from config
    with create_memory_store_from_config(args.config) as store:
        # Pass store to command
        command_func(args, store)
```

### Demo Code (`demo_cli.py`)

```python
def create_demo_store() -> MemoryStore:
    """Explicitly create mock store for demos"""
    mock_gen = MockEmbeddingGenerator()
    store = MemoryStore(embedding_generator=mock_gen)

    # Add sample data
    store.store_episode(...)

    return store

def interactive_demo():
    store = create_demo_store()

    # Use the same command functions
    command_stats(args, store)
```

### Test Code (`tests/test_cli.py`)

```python
@pytest.fixture
def populated_store(temp_dir):
    """Create test store with sample data"""
    mock_gen = MockEmbeddingGenerator()
    store = MemoryStore(
        db_path=f"{temp_dir}/test.db",
        embedding_generator=mock_gen
    )

    # Add test data
    store.store_episode(...)

    return store

def test_command_stats(populated_store, capsys):
    """Test stats command with injected store"""
    args = Namespace()

    # Pass store directly - no mocking needed!
    command_stats(args, populated_store)

    # Verify output
    captured = capsys.readouterr()
    assert "Total Episodes" in captured.out
```

## Benefits

### For Testing
✅ **No mocking needed** - just create test stores
✅ **Fast tests** - no patching overhead
✅ **Clear what's being tested** - explicit dependencies
✅ **Easy to debug** - no magic mocking behavior

### For Production
✅ **Clean separation** - config reading vs business logic
✅ **Flexible configuration** - env vars, files, or code
✅ **No test code in production** - demo_cli.py is separate
✅ **Type-safe** - all dependencies explicit

### For Maintenance
✅ **Easy to extend** - just add new ways to create stores
✅ **Easy to understand** - no hidden behavior
✅ **Testable at all levels** - unit, integration, e2e
✅ **Functional style** - pure functions, explicit dependencies

## Configuration Sources

### 1. Environment Variable (Simplest)
```bash
export EMBEDDING_MODEL_PATH=/path/to/model.gguf
python src/memory_cli.py stats
```

### 2. Config File
```yaml
# config.yaml
memory:
  embedding_model: /path/to/model.gguf
  vector_store_path: ./data/vectors
  sqlite_path: ./data/memory.db
```

```bash
python src/memory_cli.py --config config.yaml stats
```

### 3. Programmatic (Demos/Tests)
```python
# Create store however you want
store = MemoryStore(embedding_generator=MockGen())

# Pass to commands
command_stats(args, store)
```

## Comparison to Clojure's `with-redefs`

### Clojure Approach
```clojure
(defn get-data [] (real-api-call))

(deftest test-something
  (with-redefs [get-data (fn [] "mock data")]
    ;; test code here
    ))
```

### Python Approach (What We Did)
```python
# Instead of rebinding, we use dependency injection
def command_stats(args, store):  # store is injected
    stats = store.get_stats()
    # ...

# Test: inject test store
def test_command_stats():
    test_store = create_test_store()
    command_stats(args, test_store)

# Production: inject production store
def main():
    production_store = create_memory_store_from_config()
    command_stats(args, production_store)
```

**Why this is better than `with-redefs`:**
- More explicit (you see what's being passed)
- Type-safe (can't pass wrong type)
- No global rebinding (safer)
- Works with any caller (not just tests)

**When you might still use `patch()` (Python's closest to `with-redefs`):**
- Legacy code you can't change
- Testing external library calls
- Mocking filesystem/network operations

But for **our code**, pure dependency injection is cleaner!

## Summary

| Approach | Production Code | Test Code | Clarity | Safety |
|----------|----------------|-----------|---------|--------|
| Boolean flag | Contains test logic | Uses flag | 😐 | 🚫 |
| Global patching | Clean | Uses `patch()` | 😐 | 😐 |
| **Dependency injection** | **Clean** | **Inject test deps** | **✅** | **✅** |

We chose **pure dependency injection** because it's:
- Most explicit and clear
- Easiest to test
- Most functional
- Type-safe
- No magic behavior

---

*This architectural decision makes the entire codebase more maintainable and testable.*
