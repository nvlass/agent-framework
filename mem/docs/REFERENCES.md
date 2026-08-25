# References and Resources

## Python Mocking and Testing (2026-02-02)

### Python's Equivalent to Clojure's `with-redefs`

**Question:** How to temporarily rebind functions in Python for testing (like Clojure's `with-redefs`)?

**Answer:** Python provides two main approaches:

1. **`unittest.mock.patch()`** - Standard library context manager
2. **pytest's `monkeypatch` fixture** - pytest-native approach

### Key Resources

- [unittest.mock — mock object library](https://docs.python.org/3/library/unittest.mock.html)
  Official Python documentation for the mock library

- [Understanding the Python Mock Object Library – Real Python](https://realpython.com/python-mock-library/)
  Comprehensive tutorial on Python mocking patterns

- [Mocking Vs. Patching (A Quick Guide For Beginners) | Pytest with Eric](https://pytest-with-eric.com/mocking/mocking-vs-patching/)
  Clear explanation of the difference between mocking and patching

- [Understanding the Patch Function in Python's unittest.mock Library | Medium](https://medium.com/@tankoraphael/understanding-the-patch-function-in-pythons-unittest-mock-library-5d98952fc17a)
  Detailed guide to patch() usage

- [Mastering unittest.mock in Python | Better Stack Community](https://betterstack.com/community/guides/scaling-python/python-unittest-mock/)
  Advanced mocking techniques

### Key Takeaways

1. **Use `unittest.mock.patch()` as context manager** for temporary rebinding:
   ```python
   with patch('module.function') as mock_fn:
       mock_fn.return_value = 42
       # Test code here
   ```

2. **Use pytest's `monkeypatch` fixture** for cleaner pytest code:
   ```python
   def test_something(monkeypatch):
       monkeypatch.setattr('module.function', mock_function)
   ```

3. **Best practice:** Avoid boolean flags like `use_mock` in production code. Use dependency injection instead.

4. **Patch where object is looked up, not where it's defined** - important gotcha!

---

## Project Design Decisions

### Memory CLI Architecture (2026-02-02)

**Decision:** Use pure dependency injection (Option C) for the CLI

**Rationale:**
- Most testable approach
- No global state
- No mock/production boolean flags
- Functional programming style (functions are pure)

**Implementation:**
- All command functions take `store: MemoryStore` as parameter
- `main()` creates store from config/environment
- Tests pass in test fixtures directly
- Demo scripts create mock stores explicitly

---

## SQLite JSON Querying (2026-02-02)

**Decision:** Use SQLite's native JSON functions for tag storage

**Key learnings:**
- SQLite has powerful JSON support via JSON1 extension
- Can use `CHECK(json_valid(tags))` for validation
- `json_each()` allows querying JSON arrays in SQL
- Cannot use subqueries in index expressions (limitation)

**Resources:**
- SQLite JSON1 Extension documentation
- Performance is good for thousands of episodes
- For millions, might need FTS or normalized tags table

---

## Technology Stack

### Core Dependencies
- **Python:** 3.11+ (using modern type hints)
- **llama.cpp:** Local LLM inference
- **llama-cpp-python:** Python bindings
- **ChromaDB:** Vector store for embeddings
- **SQLite:** Structured storage with JSON support
- **pytest:** Testing framework
- **rich:** Beautiful terminal output

### Design Principles
1. **Local-first:** No external API dependencies
2. **Incremental:** Each phase builds on previous
3. **Educational:** Well-commented, clear decisions
4. **Functional:** Pure functions, dependency injection
5. **Testable:** All code easily testable

---

_This document tracks important resources and design decisions for the project._
