# Hierarchical Memory Architecture Discussion

**Date:** 2026-02-02
**Status:** Planning Phase
**Decision:** Implement Option B (Proper Hierarchical System) as next phase

---

## Context

After completing Phase 1 (Basic Storage) and Phase 2 (Learning from Outcomes), we discussed adding hierarchical memory (working/short-term/long-term) to improve agent performance.

**Key Question:** "How would adding hierarchical long-term short-term memory change the scope of the project? I am under the impression that it is rather important for agents / agent performance"

---

## Current Architecture (Flat Memory)

```
┌─────────────────────────────────────┐
│  Agent Query                        │
│         ↓                           │
│  Search ALL Episodes                │
│         ↓                           │
│  Return Top K Similar               │
└─────────────────────────────────────┘
```

**Issues:**
- Every retrieval searches entire history (slow for large databases)
- No distinction between "just happened" and "long ago"
- Recent context often more relevant than distant past
- LLM context window limits what you can use anyway
- No concept of "what's currently relevant"

---

## Why Hierarchical Memory Matters

### 1. Performance ⚡
- **Working memory:** O(1) lookup - just load recent items
- **Short-term:** Cached/indexed separately, fast access
- **Long-term:** Full search only when needed
- Skip expensive vector search for recent context

### 2. Relevance 🎯
- "I just told you X" → Working memory (immediate)
- Recent failures → Short-term (learn quickly)
- Current conversation context → Always accessible
- Historical patterns → Long-term (when needed)

### 3. LLM Context Limits 📏
- Working memory fits in prompt (recent N turns)
- Long-term retrieved selectively
- Better token budget management
- Prioritize recent over distant

### 4. Cognitive Realism 🧠
- Mirrors human memory systems
- More natural interaction patterns
- Supports temporal reasoning
- Enables proper "forgetting" mechanisms

---

## Three Implementation Options

### Option A: Minimal Working Memory (1-2 days)

**Simple in-memory buffer for recent episodes**

```python
class WorkingMemory:
    """In-memory buffer for recent episodes"""
    def __init__(self, max_size=20):
        self.buffer = deque(maxlen=max_size)
        self.current_session_id = None

    def add(self, episode):
        self.buffer.append(episode)

    def get_recent(self, n=10):
        return list(self.buffer)[-n:]

    def clear_session(self):
        self.buffer.clear()
```

**Scope:**
- ✅ Add WorkingMemory class
- ✅ Modify retrieval to check working first
- ✅ Auto-add to working memory on store
- ✅ Session management

**Benefits:**
- Quick to implement
- Recent context always available
- Faster for "just happened" queries
- Minimal code change

**Limitations:**
- No persistence (lost on restart)
- No consolidation
- No short-term tier
- Limited scalability

---

### Option B: Proper Hierarchical System (1 week) ⭐ RECOMMENDED

**Full three-tier memory with consolidation**

```python
class MemoryHierarchy:
    def __init__(self):
        self.working = WorkingMemory(max_size=20)      # Last ~20 items
        self.short_term = ShortTermMemory(hours=24)    # Last 24 hours
        self.long_term = LongTermMemory()              # Everything else

    def retrieve(self, query, context_aware=True):
        # 1. Working memory (always included)
        working_items = self.working.get_all()

        # 2. Short-term if relevant to current session
        if context_aware:
            short_term_items = self.short_term.get_recent(hours=4)
        else:
            short_term_items = []

        # 3. Long-term semantic search
        long_term_items = self.long_term.semantic_search(query, limit=5)

        return self._merge_and_rank(working_items, short_term_items, long_term_items)

    def consolidate(self):
        """Move items from short-term to long-term with summarization"""
        old_episodes = self.short_term.get_older_than(hours=24)

        # Extract patterns
        patterns = self._extract_patterns(old_episodes)
        self.long_term.store_patterns(patterns)

        # Keep only important individual episodes
        important = [ep for ep in old_episodes
                    if ep.success_score > 0.8 or ep.outcome_category == 'failure']
        self.long_term.store_episodes(important)

        self.short_term.remove(old_episodes)
```

**Architecture:**

```
┌────────────────────────────────────────────────┐
│  Working Memory (In-Memory, 20 items)          │
│  - Current conversation turns                  │
│  - Active tasks/goals                          │
│  - Last few minutes of context                 │
│  ↓ Instant access, no DB query                 │
├────────────────────────────────────────────────┤
│  Short-Term Memory (Recent, 24 hours)          │
│  - Recent episodes (hours/days)                │
│  - Session-level patterns                      │
│  - Current learning context                    │
│  ↓ Cached or indexed separately                │
├────────────────────────────────────────────────┤
│  Long-Term Memory (Consolidated)               │
│  - Historical patterns                         │
│  - Consolidated knowledge                      │
│  - Important/high-utility episodes             │
│  ↓ Full search when needed                     │
└────────────────────────────────────────────────┘
```

**Schema Changes:**

```sql
-- Add to episodes table
ALTER TABLE episodes ADD COLUMN memory_tier TEXT; -- 'working', 'short', 'long'
ALTER TABLE episodes ADD COLUMN last_accessed DATETIME;
ALTER TABLE episodes ADD COLUMN access_count INTEGER DEFAULT 0;

-- Track sessions
CREATE TABLE memory_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME,
    episode_count INTEGER,
    summary TEXT
);
```

**Implementation Timeline (1 week):**

1. **Day 1: Working Memory**
   - In-memory buffer implementation
   - Session tracking
   - Auto-add on store

2. **Day 2: Short-Term Storage**
   - Recent episodes cache
   - Time-based queries
   - SQLite view or separate indexing

3. **Day 3: Tiered Retrieval**
   - Check working → short → long
   - Context-aware merging
   - Deduplication and ranking

4. **Days 4-5: Consolidation Pipeline**
   - Time-based triggers
   - Pattern extraction from short-term
   - Selective promotion to long-term
   - Automatic forgetting

5. **Days 6-7: CLI & Testing**
   - Show memory tiers in stats
   - Session management commands
   - Comprehensive testing
   - Documentation

**Benefits:**
- ✅ Much faster retrieval (tiered search)
- ✅ Context-aware memory access
- ✅ Automatic consolidation
- ✅ Scalable to millions of episodes
- ✅ Natural "forgetting" of unimportant details
- ✅ Foundation for advanced features

**This is production-grade architecture used by systems like:**
- MemGPT
- AutoGPT with memory
- LangChain agents with memory
- ChatGPT (working context + long-term retrieval)

---

### Option C: Full Cognitive Architecture (3-4 weeks)

**Research-grade memory system with cognitive mechanisms**

```python
class CognitiveMemorySystem:
    """
    Full cognitive architecture with attention, consolidation, and meta-cognition
    """
    def __init__(self):
        # Memory tiers (from Option B)
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

        # Cognitive mechanisms
        self.attention = AttentionMechanism()
        self.consolidation = ConsolidationEngine()
        self.meta_cognition = MetaCognitiveMonitor()

        # Memory types
        self.episodic = EpisodicMemory()      # Specific experiences
        self.semantic = SemanticMemory()       # General knowledge
        self.procedural = ProceduralMemory()   # Skills/procedures

        # Advanced features
        self.forgetting = ActiveForgetting()
        self.dreaming = OfflineConsolidation()
        self.curiosity = CuriosityDrivenExploration()
```

**Key Features:**

1. **Attention Mechanism**
   - What to store in working memory?
   - Priority-based selection
   - Surprise/novelty detection
   - Goal-relevance filtering

2. **Memory Type Separation**
   - **Episodic:** "I debugged this Python error yesterday"
   - **Semantic:** "Python errors often come from type mismatches"
   - **Procedural:** "When debugging, first check the stack trace"

3. **Active Forgetting**
   - Utility-based forgetting
   - Interference-based removal
   - Temporal decay functions
   - Strategic forgetting (clear outdated info)

4. **Offline Consolidation ("Sleep")**
   - Batch processing when idle
   - Pattern discovery without user interaction
   - Memory replay for strengthening
   - Connection building between memories

5. **Meta-Cognitive Monitoring**
   - Track memory performance
   - Detect retrieval failures
   - Adjust retrieval strategies
   - Learn what to remember

6. **Curiosity-Driven Exploration**
   - Identify knowledge gaps
   - Generate exploration goals
   - Active information seeking
   - Novelty detection

**This is research territory, used by:**
- Academic AI research labs
- Cognitive architecture systems (SOAR, ACT-R)
- Advanced reasoning systems
- Novel agent architectures

---

## Decision: Implement Option B Next

**Rationale:**

1. **Right Time:** Phase 1 & 2 foundation is solid
2. **High Impact:** Dramatically improves agent performance
3. **Architectural:** Affects all future features
4. **Natural Extension:** Fits cleanly into existing code
5. **Production-Ready:** Industry-standard approach
6. **Learning Value:** Core concept in modern agent systems

**Revised Phase Plan:**

- ✅ Phase 1: Basic Storage (COMPLETE)
- ✅ Phase 2: Learning from Outcomes (COMPLETE)
- **Phase 3: Hierarchical Memory** (Option B) ← NEXT
- Phase 4: Pattern Learning & Recognition
- Phase 5: Reflection & Meta-Learning
- Phase 6: Adaptation & Transfer Learning
- Phase 7: Advanced Features & Polish

**Future:** Option C features can be added incrementally in later phases or as research extensions.

---

## Why Not Option C Immediately?

**Complexity:**
- 3-4 weeks vs 1 week
- Research-level concepts
- Higher risk of over-engineering
- Many unknown unknowns

**Learning Curve:**
- Attention mechanisms require deep understanding
- Meta-cognition is complex to implement correctly
- Active forgetting needs careful design
- Offline consolidation has many edge cases

**Diminishing Returns:**
- Option B gets 80% of the benefits
- Option C is 20% more benefit for 4x more work
- Can be added incrementally later
- Better to learn progressively

**Option B → C Upgrade Path Exists:**
- See OPTION_B_TO_C_UPGRADE.md for details
- Can add Option C features one at a time
- Option B is not a dead end
- Progressive enhancement strategy

---

## References

### Academic Papers
- "Memory Augmented Neural Networks" (Graves et al.)
- "Neural Turing Machines" (Graves et al.)
- "Episodic Memory in Lifelong Language Learning"
- "Human-like Episodic Memory for Infinite Context LLMs" (MemGPT paper)

### Systems
- **MemGPT:** Hierarchical memory for LLMs
- **Voyager:** Minecraft agent with skill memory
- **AutoGPT:** Agent with memory persistence
- **ChatGPT:** Working context + conversation history

### Cognitive Science
- **Working Memory:** 7±2 items (Miller's Law)
- **Consolidation:** Sleep-based memory strengthening
- **Forgetting Curve:** Ebbinghaus forgetting function
- **Recognition vs Recall:** Different memory access patterns

---

## Next Steps

1. **Review Option B architecture** in detail
2. **Design schema changes** for memory tiers
3. **Implement working memory** (Day 1)
4. **Build short-term storage** (Day 2)
5. **Create tiered retrieval** (Day 3)
6. **Add consolidation pipeline** (Days 4-5)
7. **CLI and testing** (Days 6-7)

---

**Document maintained by:** Nikos & Claude
**Last Updated:** 2026-02-02
**Status:** Active Planning
