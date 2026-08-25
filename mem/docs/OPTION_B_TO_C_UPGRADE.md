# Upgrade Path: Option B → Option C

**Purpose:** Document how to upgrade from Option B (Hierarchical Memory) to Option C (Full Cognitive Architecture) after Phase 5.

**Key Insight:** Option B is NOT a dead end. It's a solid foundation that Option C builds upon incrementally.

---

## What Changes Between B and C?

### Option B Provides (Already Built)
- ✅ Three-tier memory structure (working/short/long)
- ✅ Tiered retrieval system
- ✅ Basic consolidation pipeline
- ✅ Session management
- ✅ Temporal decay (basic forgetting)

### Option C Adds (Incremental Additions)
- 🆕 Attention mechanism (what to store)
- 🆕 Memory type separation (episodic/semantic/procedural)
- 🆕 Active forgetting strategies
- 🆕 Offline consolidation ("sleep" mode)
- 🆕 Meta-cognitive monitoring
- 🆕 Curiosity-driven exploration

**Key Point:** Each of these can be added **one at a time** without breaking existing functionality!

---

## Upgrade Strategy: Progressive Enhancement

Instead of a "big bang" rewrite, add Option C features incrementally:

### Phase A: Attention Mechanism (Week 1)
**What it does:** Decides what deserves to be in working memory

**Changes Required:**
1. Add `AttentionMechanism` class
2. Modify `WorkingMemory.add()` to use attention scores
3. Add attention scoring to episodes

**Schema Changes:**
```sql
ALTER TABLE episodes ADD COLUMN attention_score REAL; -- 0.0 to 1.0
ALTER TABLE episodes ADD COLUMN surprise_level REAL;  -- How unexpected/novel
ALTER TABLE episodes ADD COLUMN goal_relevance REAL;  -- How relevant to current goals
```

**Code Changes:**
```python
class AttentionMechanism:
    """Decides what to store in working memory"""

    def compute_attention(self, episode, context) -> float:
        """
        Compute attention score based on:
        - Novelty (is this new/surprising?)
        - Relevance (related to current goals?)
        - Importance (success/failure marker?)
        - Recency (just happened?)
        """
        novelty = self._compute_novelty(episode)
        relevance = self._compute_relevance(episode, context)
        importance = episode.success_score or 0.5
        recency = self._compute_recency(episode)

        return 0.3 * novelty + 0.3 * relevance + 0.2 * importance + 0.2 * recency

    def should_store_in_working(self, episode, context) -> bool:
        attention = self.compute_attention(episode, context)
        return attention > 0.6  # Threshold

# Modify existing WorkingMemory
class WorkingMemory:
    def __init__(self, max_size=20):
        self.buffer = deque(maxlen=max_size)
        self.attention = AttentionMechanism()  # NEW

    def add(self, episode, context=None):
        # NEW: Only add if attention score is high enough
        if context and not self.attention.should_store_in_working(episode, context):
            return False  # Skip working memory, go straight to short-term

        self.buffer.append(episode)
        return True
```

**Benefits After This Phase:**
- Working memory contains only relevant items
- Less noise in recent context
- Better token usage
- Automatic filtering of unimportant details

**Backward Compatible?** YES - Default attention scores can maintain current behavior

---

### Phase B: Memory Type Separation (Week 2)
**What it does:** Separate episodic (experiences) from semantic (knowledge) from procedural (skills)

**Schema Changes:**
```sql
-- Modify existing tables
ALTER TABLE episodes ADD COLUMN memory_type TEXT DEFAULT 'episodic';
-- 'episodic', 'semantic', 'procedural'

-- Semantic memory already exists (learned_patterns table from Phase 2)
-- Add procedural memory table
CREATE TABLE procedural_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT NOT NULL,
    procedure_steps TEXT NOT NULL,  -- JSON array of steps
    context_triggers TEXT,           -- When to use this procedure
    success_rate REAL,
    times_used INTEGER DEFAULT 0,
    last_used DATETIME,
    embedding_id TEXT
);
```

**Code Changes:**
```python
class MemorySystem:
    def __init__(self):
        # Existing
        self.hierarchy = MemoryHierarchy()

        # NEW: Memory type managers
        self.episodic = EpisodicMemoryManager(self.hierarchy)
        self.semantic = SemanticMemoryManager(self.hierarchy)
        self.procedural = ProceduralMemoryManager(self.hierarchy)

    def store(self, content, memory_type='episodic'):
        if memory_type == 'episodic':
            return self.episodic.store_episode(content)
        elif memory_type == 'semantic':
            return self.semantic.store_knowledge(content)
        elif memory_type == 'procedural':
            return self.procedural.store_procedure(content)

    def retrieve(self, query, memory_types=['episodic', 'semantic']):
        results = []
        if 'episodic' in memory_types:
            results.extend(self.episodic.retrieve(query))
        if 'semantic' in memory_types:
            results.extend(self.semantic.retrieve(query))
        if 'procedural' in memory_types:
            results.extend(self.procedural.retrieve(query))
        return results
```

**Benefits After This Phase:**
- Can query specific memory types
- "What do I know about X?" vs "When did I learn X?"
- Better organization
- Foundation for different retrieval strategies per type

**Backward Compatible?** YES - Default to 'episodic' for all existing episodes

---

### Phase C: Active Forgetting (Week 3)
**What it does:** Strategically remove low-utility memories

**Schema Changes:**
```sql
ALTER TABLE episodes ADD COLUMN utility_score REAL;    -- How useful is this?
ALTER TABLE episodes ADD COLUMN decay_rate REAL;       -- How fast to forget
ALTER TABLE episodes ADD COLUMN interference REAL;     -- Conflicts with other memories?
```

**Code Changes:**
```python
class ActiveForgetting:
    """Strategic forgetting of low-utility memories"""

    def compute_utility(self, episode) -> float:
        """
        Utility based on:
        - Access frequency (how often retrieved)
        - Recency (when last accessed)
        - Success indicator (did it work?)
        - Uniqueness (is it redundant?)
        """
        access_factor = episode.access_count / 10.0
        recency_factor = self._time_since_access(episode)
        success_factor = episode.success_score or 0.5
        uniqueness = self._compute_uniqueness(episode)

        return 0.3 * access_factor + 0.2 * recency_factor + 0.3 * success_factor + 0.2 * uniqueness

    def forget_candidates(self, threshold=0.2) -> List[Episode]:
        """Find episodes eligible for forgetting"""
        episodes = self.store.get_all_episodes()
        return [ep for ep in episodes if self.compute_utility(ep) < threshold]

    def forget(self, episode_id, reason='low_utility'):
        """Move to forgotten_memories table (from Phase 2)"""
        episode = self.store.get_episode_by_id(episode_id)

        # Archive before deletion
        self.store.archive_forgotten(episode, reason)

        # Remove from active memory
        self.store.delete_episode(episode_id)
```

**Benefits After This Phase:**
- Memory stays bounded (doesn't grow forever)
- Focuses on important information
- Removes redundant/obsolete memories
- Implements decay function

**Backward Compatible?** YES - Forgetting is opt-in, default utility scores keep everything

---

### Phase D: Offline Consolidation (Week 4)
**What it does:** Process memories when agent is idle ("sleep")

**Schema Changes:**
```sql
CREATE TABLE consolidation_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    episodes_processed INTEGER,
    patterns_discovered INTEGER,
    status TEXT  -- 'running', 'completed', 'failed'
);
```

**Code Changes:**
```python
class OfflineConsolidation:
    """Process memories during idle time"""

    def run_consolidation_cycle(self):
        """
        Run when agent is idle:
        1. Extract patterns from recent episodes
        2. Find connections between memories
        3. Strengthen important memories
        4. Replay key episodes for learning
        """
        job_id = self._start_job()

        try:
            # 1. Pattern extraction
            recent = self.store.get_recent_episodes(hours=24)
            patterns = self._extract_patterns(recent)
            self.store.store_patterns(patterns)

            # 2. Connection building
            self._build_memory_connections(recent)

            # 3. Memory strengthening
            important = [ep for ep in recent if ep.success_score > 0.8]
            for ep in important:
                self._strengthen_memory(ep)

            # 4. Memory replay (like dreaming)
            self._replay_for_learning(important)

            self._complete_job(job_id)

        except Exception as e:
            self._fail_job(job_id, str(e))

    def _replay_for_learning(self, episodes):
        """
        Replay episodes to strengthen learning
        Similar to how humans consolidate during sleep
        """
        for episode in episodes:
            # Retrieve similar episodes
            similar = self.store.retrieve_episodes(episode.context, limit=5)

            # Find common patterns
            pattern = self._find_common_strategy(similar)

            # Update learned patterns
            if pattern:
                self.store.update_pattern_confidence(pattern)
```

**Trigger Mechanisms:**
```python
# Option 1: Time-based (run at night)
schedule.every().day.at("02:00").do(consolidation.run_cycle)

# Option 2: Idle detection
if agent_idle_for(minutes=30):
    consolidation.run_cycle()

# Option 3: Episode count
if episodes_since_last_consolidation > 100:
    consolidation.run_cycle()
```

**Benefits After This Phase:**
- Continuous learning without user interaction
- Pattern discovery happens in background
- Memory optimization during downtime
- Better knowledge extraction

**Backward Compatible?** YES - Consolidation runs separately, doesn't affect queries

---

### Phase E: Meta-Cognitive Monitoring (Week 5)
**What it does:** Monitor and improve memory system performance

**Schema Changes:**
```sql
CREATE TABLE memory_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    query TEXT,
    retrieval_time_ms INTEGER,
    results_found INTEGER,
    results_used INTEGER,
    user_satisfaction REAL  -- Inferred or explicit
);

CREATE TABLE retrieval_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT,
    context_pattern TEXT,
    success_rate REAL,
    avg_retrieval_time_ms INTEGER,
    times_used INTEGER
);
```

**Code Changes:**
```python
class MetaCognitiveMonitor:
    """Monitor and improve memory performance"""

    def track_retrieval(self, query, results, execution_time):
        """Track each memory retrieval"""
        self.perf_db.insert({
            'query': query,
            'retrieval_time_ms': execution_time,
            'results_found': len(results),
            'timestamp': datetime.now()
        })

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze memory system performance"""
        recent = self.perf_db.get_recent(days=7)

        return {
            'avg_retrieval_time': np.mean([r['retrieval_time_ms'] for r in recent]),
            'retrieval_success_rate': self._compute_success_rate(recent),
            'most_common_queries': self._find_common_patterns(recent),
            'slow_queries': [r for r in recent if r['retrieval_time_ms'] > 1000]
        }

    def suggest_optimizations(self) -> List[str]:
        """Suggest improvements based on performance"""
        perf = self.analyze_performance()
        suggestions = []

        if perf['avg_retrieval_time'] > 500:
            suggestions.append("Consider adding more indexes")

        if perf['retrieval_success_rate'] < 0.7:
            suggestions.append("Improve embedding quality or query processing")

        return suggestions

    def adapt_strategy(self, context):
        """Choose best retrieval strategy for context"""
        # Learn which strategies work best for which contexts
        strategies = self.strategy_db.get_by_context(context)

        if strategies:
            # Pick best performing strategy
            best = max(strategies, key=lambda s: s.success_rate)
            return best.strategy_name
        else:
            return 'default'
```

**Benefits After This Phase:**
- Self-improving memory system
- Detect and fix performance issues
- Adaptive retrieval strategies
- Data-driven optimization

**Backward Compatible?** YES - Monitoring layer, doesn't change core functionality

---

### Phase F: Curiosity-Driven Exploration (Week 6)
**What it does:** Identify knowledge gaps and seek information

**Schema Changes:**
```sql
CREATE TABLE knowledge_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    gap_description TEXT,
    context TEXT,
    priority REAL,
    filled_at DATETIME,
    resolution TEXT
);

CREATE TABLE exploration_goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT,
    motivation TEXT,  -- 'curiosity', 'failure', 'uncertainty'
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    outcome TEXT
);
```

**Code Changes:**
```python
class CuriosityEngine:
    """Identify knowledge gaps and generate exploration goals"""

    def detect_knowledge_gap(self, query, retrieval_results):
        """Detect when agent doesn't have relevant knowledge"""
        if len(retrieval_results) == 0:
            # No memories match - knowledge gap
            return {
                'type': 'missing_knowledge',
                'query': query,
                'priority': 'high'
            }

        if all(r.success_score < 0.5 for r in retrieval_results):
            # All past attempts failed - knowledge gap
            return {
                'type': 'unsuccessful_strategy',
                'query': query,
                'priority': 'medium'
            }

        return None

    def generate_exploration_goal(self, gap):
        """Create an exploration goal to fill the gap"""
        if gap['type'] == 'missing_knowledge':
            return {
                'goal': f"Learn about: {gap['query']}",
                'motivation': 'curiosity',
                'suggested_actions': [
                    'Ask user for examples',
                    'Search documentation',
                    'Try exploratory approach'
                ]
            }

    def compute_information_gain(self, new_episode):
        """How much did this episode reduce uncertainty?"""
        similar_before = self.store.retrieve_episodes(new_episode.context, limit=5)

        # Compare uncertainty before and after
        uncertainty_before = self._compute_uncertainty(similar_before)
        uncertainty_after = self._compute_uncertainty(similar_before + [new_episode])

        return uncertainty_before - uncertainty_after
```

**Benefits After This Phase:**
- Agent aware of what it doesn't know
- Active learning behavior
- Better handling of novel situations
- Meta-cognitive awareness

**Backward Compatible?** YES - Operates as separate layer

---

## Summary: What Changes Are Required?

### Database Schema
- ✅ **Minimal:** 6 new columns to existing tables
- ✅ **Moderate:** 5 new tables (all non-breaking additions)
- ✅ **No breaking changes** to existing schema

### Code Architecture
- ✅ **Wraps existing code:** Option B classes remain, new classes wrap them
- ✅ **Progressive enhancement:** Each feature is independent
- ✅ **Backward compatible:** All defaults maintain current behavior

### API Changes
- ✅ **Non-breaking:** New methods added, old methods work as before
- ✅ **Optional features:** All Option C features are opt-in
- ✅ **Graceful degradation:** System works without Option C features enabled

---

## Timeline: Option B → Option C

**Conservative Estimate:** 6 weeks (one week per phase)

**Aggressive Estimate:** 3-4 weeks (parallel development)

**Incremental Approach:** Add one phase at a time as needed

**Key Point:** You don't need to do all 6 phases at once! Each adds independent value.

---

## Recommended Sequence

After completing Option B and original Phase 5:

1. **Phase A: Attention** (Week 1) - Immediate UX improvement
2. **Phase C: Forgetting** (Week 2) - Keeps system bounded
3. **Phase D: Consolidation** (Week 3) - Background learning
4. **Phase B: Memory Types** (Week 4) - Better organization
5. **Phase E: Meta-Cognition** (Week 5) - Self-improvement
6. **Phase F: Curiosity** (Week 6) - Advanced behavior

**Or pick and choose based on needs!**

---

## Decision Points: When to Upgrade

**Consider Option C features when:**
- ✅ Option B is working well
- ✅ System handles 1000+ episodes smoothly
- ✅ You want research-level capabilities
- ✅ You have time for 3-6 weeks development
- ✅ You're interested in cognitive architectures

**Stick with Option B when:**
- ❌ Still building core functionality
- ❌ System performance is good enough
- ❌ Focus is on other features
- ❌ Time is limited
- ❌ Production stability is priority

---

## Key Insight

**Option B is NOT a dead end.** It's a solid foundation that allows incremental upgrades to Option C.

Every Option C feature:
- ✅ Can be added independently
- ✅ Doesn't break existing code
- ✅ Brings incremental value
- ✅ Can be enabled/disabled

This means you can:
1. Build Option B (1 week)
2. Use it in production
3. Add Option C features one at a time
4. Pick only the features you need

**Progressive enhancement is the key!**

---

**Document maintained by:** Nikos & Claude
**Last Updated:** 2026-02-02
**Status:** Planning / Future Reference
