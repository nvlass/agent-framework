# Phase 2 Implementation Guide: Learning from Outcomes

## Step 1: Schema Migration ✅ COMPLETE

You've already:
- ✅ Created schema_v2.sql
- ✅ Created migration script
- ✅ Migrated existing database

## Step 2: Add Learning Methods to MemoryStore

Now let's extend `src/memory_store.py` with outcome evaluation and pattern learning.

### Part A: Update Episode Class

First, update the `Episode` class to include new Phase 2 fields.

**Location:** `src/memory_store.py`, around line 20

**Add these parameters to `__init__`:**
```python
class Episode:
    def __init__(
        self,
        id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        context: str = "",
        action: str = "",
        outcome: str = "",
        success_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        embedding_id: Optional[str] = None,
        # Phase 2: Add these
        outcome_category: Optional[str] = None,  # 'success', 'failure', 'partial', 'unknown'
        failure_reason: Optional[str] = None,
    ):
        # ... existing code ...
        self.outcome_category = outcome_category
        self.failure_reason = failure_reason
```

**Update `to_dict()` method:**
```python
def to_dict(self) -> Dict[str, Any]:
    return {
        "id": self.id,
        "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        "context": self.context,
        "action": self.action,
        "outcome": self.outcome,
        "success_score": self.success_score,
        "tags": self.tags,
        "embedding_id": self.embedding_id,
        # Phase 2
        "outcome_category": self.outcome_category,
        "failure_reason": self.failure_reason,
    }
```

**Update `from_db_row()` static method:**

The database row now has 2 more columns at the end (outcome_category, failure_reason).
Update to handle them:

```python
@staticmethod
def from_db_row(row: Tuple) -> "Episode":
    tags_raw = row[6] if row[6] else '[]'
    return Episode(
        id=row[0],
        timestamp=datetime.fromisoformat(row[1]) if row[1] else None,
        context=row[2],
        action=row[3],
        outcome=row[4],
        success_score=row[5],
        tags=json.loads(tags_raw),
        embedding_id=row[7],
        # Phase 2: row[8] and row[9]
        outcome_category=row[8] if len(row) > 8 else None,
        failure_reason=row[9] if len(row) > 9 else None,
    )
```

### Part B: Add Outcome Evaluation Method

**Add this method to the `MemoryStore` class:**

Location: After `get_episode_count()` method

```python
def evaluate_outcome(
    self,
    episode_id: int,
    category: str,
    failure_reason: Optional[str] = None
) -> bool:
    """
    Evaluate and classify an episode's outcome

    Args:
        episode_id: Episode to evaluate
        category: One of 'success', 'failure', 'partial', 'unknown'
        failure_reason: Optional explanation if category is 'failure'

    Returns:
        True if evaluation was successful, False if episode not found

    Example:
        store.evaluate_outcome(42, 'success')
        store.evaluate_outcome(43, 'failure', 'Missing null check caused crash')
    """
    # TODO: Your implementation here
    #
    # Hints:
    # 1. Validate category is one of: 'success', 'failure', 'partial', 'unknown'
    # 2. Update the episode in the database
    # 3. Return True if update successful, False if episode not found
    #
    # SQL: UPDATE episodes SET outcome_category = ?, failure_reason = ? WHERE id = ?
    pass
```

### Part C: Add Pattern Discovery Methods

**Method 1: Get episodes by outcome category**

```python
def get_episodes_by_category(
    self,
    category: str,
    limit: Optional[int] = None
) -> List[Episode]:
    """
    Get episodes by outcome category

    Args:
        category: 'success', 'failure', 'partial', or 'unknown'
        limit: Optional max number of results

    Returns:
        List of episodes in that category
    """
    # TODO: Your implementation here
    #
    # Hints:
    # 1. Query episodes WHERE outcome_category = ?
    # 2. ORDER BY timestamp DESC (most recent first)
    # 3. Apply LIMIT if provided
    pass
```

**Method 2: Get success rate for a tag combination**

```python
def get_success_rate_for_tags(
    self,
    tags: List[str],
    match_all: bool = True
) -> Dict[str, Any]:
    """
    Calculate success rate for episodes with specific tags

    Args:
        tags: List of tags to filter by
        match_all: If True, episode must have ALL tags

    Returns:
        Dict with:
        - success_rate: float (0.0 to 1.0)
        - total_episodes: int
        - successful: int (category='success')
        - failed: int (category='failure')
        - partial: int (category='partial')

    Example:
        stats = store.get_success_rate_for_tags(['python', 'debugging'])
        # Returns: {'success_rate': 0.85, 'total_episodes': 20, ...}
    """
    # TODO: Your implementation here
    #
    # Hints:
    # 1. Use get_episodes_by_tags() to get matching episodes
    # 2. Count how many are in each category
    # 3. Calculate success_rate = successful / total
    # 4. Handle edge case: total_episodes = 0
    pass
```

### Part D: Update get_stats() Method

Add Phase 2 statistics to the existing `get_stats()` method:

```python
def get_stats(self) -> Dict[str, Any]:
    """Get memory store statistics"""
    cursor = self.conn.cursor()

    # ... existing stats code ...

    # Phase 2: Add outcome category statistics
    cursor.execute("""
        SELECT outcome_category, COUNT(*)
        FROM episodes
        WHERE outcome_category IS NOT NULL
        GROUP BY outcome_category
    """)
    category_counts = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "total_episodes": total_episodes,
        "episodes_with_outcomes": episodes_with_outcomes,
        "scored_episodes": scored_episodes,
        "average_success_score": round(avg_success, 3) if avg_success else None,
        "vector_store_size": self.episodes_collection.count(),
        # Phase 2 stats
        "success_count": category_counts.get('success', 0),
        "failure_count": category_counts.get('failure', 0),
        "partial_count": category_counts.get('partial', 0),
        "unknown_count": category_counts.get('unknown', 0),
    }
```

---

## Your Task

Implement these methods in `src/memory_store.py`:

1. **Update Episode class** (Part A)
   - Add outcome_category and failure_reason parameters
   - Update to_dict()
   - Update from_db_row()

2. **Implement evaluate_outcome()** (Part B)
   - Validate category
   - Update database
   - Return success/failure

3. **Implement get_episodes_by_category()** (Part C)
   - Query by outcome_category
   - Apply limit

4. **Implement get_success_rate_for_tags()** (Part C)
   - Get episodes by tags
   - Count categories
   - Calculate rate

5. **Update get_stats()** (Part D)
   - Add category counts

---

## Testing Your Implementation

After implementing, test with:

```python
# In Python REPL or test script
from memory_store import MemoryStore

# Use mock generator
import numpy as np
class MockGen:
    def generate_embedding(self, text, use_cache=True):
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(384).astype(np.float32)
        return emb / np.linalg.norm(emb)

store = MemoryStore(embedding_generator=MockGen())

# Store an episode
ep_id = store.store_episode(
    context="User had a bug in Python code",
    action="Suggested adding type hints and using mypy",
    outcome="Bug was found and fixed",
    tags=["python", "debugging"]
)

# Evaluate it
store.evaluate_outcome(ep_id, 'success')

# Check it worked
episode = store.get_episode_by_id(ep_id)
print(f"Category: {episode.outcome_category}")  # Should print: success

# Get success rate for python debugging
stats = store.get_success_rate_for_tags(['python', 'debugging'])
print(stats)  # Should show success rate
```

---

## When You're Done

Let me know and I'll:
1. Review your implementation
2. Write comprehensive tests
3. Move to the next step: Pattern Learning

Take your time! This is the core of Phase 2's learning capability.
