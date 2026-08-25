# Agent Long-Term Memory System - Development Roadmap

## Project Overview

Build a complete agent memory system with learning, reflection, and adaptation capabilities for local llama.cpp installation. This is a progressive, learn-by-doing project where Nikos will participate in development to understand core concepts.

## Architecture Principles

- **Local-first**: Everything runs on llama.cpp (no external API dependencies)
- **Incremental**: Each phase builds on the previous, all functional at every step
- **Educational**: Code is well-commented, design decisions explained
- **Practical**: Focus on working implementations over theoretical perfection

## Technology Stack

- **LLM Runtime**: llama.cpp (local inference)
- **Embeddings**: llama.cpp embedding models
- **Vector Store**: ChromaDB (simple, local, Python-native)
- **Structured Storage**: SQLite (lightweight, zero-config)
- **Language**: Python 3.10+
- **Agent Framework**: Custom (learn fundamentals before using frameworks)

---

## Phase 1: Foundation - Basic Episodic Memory

**Goal**: Store and retrieve simple interaction traces

**What You'll Learn**:
- How to structure agent memory entries
- Basic embedding generation with llama.cpp
- Simple vector similarity search
- SQLite schema design for temporal data

**Deliverables**:
1. `memory_store.py` - Core storage abstraction
2. `embeddings.py` - llama.cpp embedding interface
3. `schema.sql` - Database schema
4. `test_basic_memory.py` - Verification script

**Database Schema**:
```sql
-- Episodic memory: raw interaction traces
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    context TEXT NOT NULL,           -- What was the situation?
    action TEXT NOT NULL,             -- What did the agent do?
    outcome TEXT,                     -- What happened?
    success_score REAL,               -- 0.0 to 1.0, nullable initially
    tags TEXT,                        -- JSON array of tags
    embedding_id TEXT                 -- Reference to vector store
);

CREATE INDEX idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX idx_episodes_tags ON episodes(tags);
```

**Key Functions**:
```python
def store_episode(context, action, outcome, tags=None) -> int
def retrieve_episodes(query, limit=5) -> List[Episode]
def get_recent_episodes(hours=24, limit=10) -> List[Episode]
```

**Success Criteria**:
- Can store 100+ episodes
- Can retrieve relevant episodes via semantic search
- Retrieval latency < 1 second for 1000 episodes

---

## Phase 2: Learning from Outcomes

**Goal**: Classify what worked and what didn't, start building intuition

**What You'll Learn**:
- Outcome evaluation strategies
- Simple pattern recognition in agent behavior
- Statistical aggregation over memory
- How agents learn from experience

**Deliverables**:
1. `outcome_tracker.py` - Success/failure classification
2. `pattern_analyzer.py` - Find successful strategies
3. `test_learning.py` - Learning verification

**Enhanced Schema**:
```sql
-- Add outcome classification
ALTER TABLE episodes ADD COLUMN outcome_category TEXT; -- 'success', 'failure', 'partial', 'unknown'
ALTER TABLE episodes ADD COLUMN failure_reason TEXT;

-- Track patterns that work
CREATE TABLE learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_description TEXT NOT NULL,
    context_signature TEXT,          -- Simplified context representation
    recommended_action TEXT,
    success_rate REAL,
    sample_count INTEGER,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_id TEXT
);
```

**Key Functions**:
```python
def evaluate_outcome(episode_id, success_score, category, reason=None)
def find_successful_patterns(context, min_success_rate=0.7) -> List[Pattern]
def get_success_rate_for_action(context_type, action_type) -> float
```

**Success Criteria**:
- Can identify that action X works well in context Y
- Can recommend actions based on historical success rates
- Learns from at least 50 episodes to show pattern emergence

---

## Phase 3: Semantic Memory & Consolidation

**Goal**: Distill general knowledge from episodes

**What You'll Learn**:
- Memory consolidation strategies
- Clustering and summarization techniques
- How to compress episodic memories into semantic knowledge
- When to forget and when to remember

**Deliverables**:
1. `semantic_memory.py` - Knowledge extraction and storage
2. `consolidator.py` - Periodic memory processing
3. `test_consolidation.py` - Consolidation verification

**Enhanced Schema**:
```sql
-- Semantic memory: distilled knowledge
CREATE TABLE semantic_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_type TEXT,              -- 'fact', 'strategy', 'heuristic', 'caution'
    content TEXT NOT NULL,
    confidence REAL,                  -- 0.0 to 1.0
    source_episodes TEXT,             -- JSON array of episode IDs
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_id TEXT
);

-- Track what we've forgotten and why
CREATE TABLE forgotten_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_id INTEGER,
    original_type TEXT,               -- 'episode' or 'semantic'
    reason TEXT,                      -- 'low_utility', 'redundant', 'outdated'
    forgotten_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    summary TEXT                      -- Brief summary of what was forgotten
);
```

**Key Functions**:
```python
def consolidate_episodes(time_range, min_cluster_size=3) -> List[SemanticKnowledge]
def extract_strategy(episode_cluster) -> str
def prune_redundant_memories(similarity_threshold=0.95)
def forget_low_utility_memories(access_threshold=2, age_days=30)
```

**Consolidation Strategy**:
1. Every N episodes (e.g., 20), trigger consolidation
2. Cluster similar episodes by embedding similarity
3. For each cluster, ask LLM: "What general lesson emerges from these experiences?"
4. Store as semantic knowledge, link to source episodes
5. Mark low-value episodes for eventual pruning

**Success Criteria**:
- 100 episodes consolidate into ~10 semantic knowledge entries
- Semantic knowledge is retrievable and actionable
- System maintains <1000 episodes through intelligent forgetting

---

## Phase 4: Reflection & Meta-Learning

**Goal**: Agent understands *why* things work or fail

**What You'll Learn**:
- Causal reasoning in agent systems
- Reflection prompting strategies
- Meta-cognitive loops
- How agents develop intuition

**Deliverables**:
1. `reflector.py` - Reflection engine
2. `causal_analyzer.py` - Why did X lead to Y?
3. `test_reflection.py` - Reflection verification

**Enhanced Schema**:
```sql
-- Reflections: deeper analysis of experience
CREATE TABLE reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reflection_type TEXT,             -- 'success_analysis', 'failure_analysis', 'pattern_discovery'
    trigger_episode_id INTEGER,
    insight TEXT NOT NULL,
    causal_chain TEXT,                -- JSON: [{factor, contribution, confidence}]
    actionable_takeaway TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_id TEXT,
    FOREIGN KEY (trigger_episode_id) REFERENCES episodes(id)
);
```

**Key Functions**:
```python
def reflect_on_failure(episode_id) -> Reflection
def reflect_on_success(episode_id) -> Reflection
def discover_patterns(time_window='7d') -> List[Reflection]
def generate_causal_explanation(outcome, context, action) -> str
```

**Reflection Triggers**:
- After every significant failure (success_score < 0.3)
- After exceptional success (success_score > 0.9)
- Periodic (daily/weekly) pattern discovery
- On-demand user request

**Reflection Prompts** (examples):
```python
FAILURE_REFLECTION = """
You attempted: {action}
In context: {context}
The outcome was: {outcome} (score: {score})

Analyze what went wrong:
1. What assumptions were incorrect?
2. What information was missing?
3. What alternative approach might have worked?
4. What should you remember for similar situations?
"""

SUCCESS_REFLECTION = """
You attempted: {action}
In context: {context}
The outcome was: {outcome} (score: {score})

Analyze why this worked:
1. What key factors contributed to success?
2. Was this approach novel or refinement of known pattern?
3. What aspects are generalizable to other contexts?
4. What should you remember for similar situations?
"""
```

**Success Criteria**:
- Agent generates insightful reflections (human-evaluated)
- Reflections lead to behavior change in subsequent episodes
- Can explain reasoning chains for decisions

---

## Phase 5: Adaptation & Transfer Learning

**Goal**: Apply knowledge to new problem types

**What You'll Learn**:
- Cross-domain knowledge transfer
- Analogical reasoning in agents
- Dynamic strategy selection
- How agents handle novelty

**Deliverables**:
1. `adapter.py` - Adaptation engine
2. `analogy_finder.py` - Find similar-but-different situations
3. `strategy_selector.py` - Choose approach for new problems
4. `test_adaptation.py` - Adaptation verification

**Enhanced Schema**:
```sql
-- Track problem types and their relationships
CREATE TABLE problem_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    characteristic_features TEXT,     -- JSON array
    successful_strategies TEXT,       -- JSON array of strategy IDs
    similar_problem_types TEXT,       -- JSON array of related type IDs
    embedding_id TEXT
);

-- Track adaptations made
CREATE TABLE adaptations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_context TEXT,
    target_context TEXT,
    strategy_adapted TEXT,
    adaptation_reasoning TEXT,
    outcome TEXT,
    success_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Key Functions**:
```python
def identify_problem_type(context) -> ProblemType
def find_analogous_situations(new_context, threshold=0.6) -> List[Episode]
def adapt_strategy(original_strategy, new_context) -> str
def select_strategy(context, available_strategies) -> Strategy
```

**Adaptation Process**:
1. Encounter new problem type
2. Search semantic memory for analogous situations
3. Retrieve strategies that worked in similar contexts
4. Ask LLM to adapt strategy to new context
5. Execute adapted strategy
6. Store adaptation and outcome
7. Update problem type knowledge

**Transfer Learning Examples**:
- Code debugging → Hardware debugging (both require isolation, testing, hypothesis)
- Chess tactics → Business strategy (both involve pattern recognition, trade-offs)
- Text parsing → Log analysis (both require pattern matching, structure extraction)

**Success Criteria**:
- Successfully applies strategies to new problem domains
- Shows measurable improvement on second encounter with problem type
- Can explain analogical reasoning used

---

## Phase 6: Advanced Features & Polish

**Goal**: Production-ready memory system

**What You'll Learn**:
- Memory system optimization
- User interaction patterns
- System monitoring and debugging
- Production considerations

**Deliverables**:
1. `memory_tools.py` - Agent-facing tool interface
2. `memory_explorer.py` - CLI for inspecting memory
3. `memory_metrics.py` - System health monitoring
4. `config.yaml` - Configuration management
5. Complete documentation

**Agent Tools Interface**:
```python
# What the agent sees and uses
tools = [
    {
        "name": "store_memory",
        "description": "Store an important experience",
        "parameters": {
            "context": "Current situation",
            "action": "What you did",
            "outcome": "What happened",
            "importance": "1-10 scale"
        }
    },
    {
        "name": "recall_similar",
        "description": "Find similar past experiences",
        "parameters": {
            "query": "Description of current situation",
            "limit": "Max results to return"
        }
    },
    {
        "name": "reflect_on_recent",
        "description": "Analyze recent experiences for patterns",
        "parameters": {
            "timeframe": "Last N hours/days",
            "focus": "Optional: specific aspect to analyze"
        }
    },
    {
        "name": "learn_from_outcome",
        "description": "Explicitly mark an outcome for learning",
        "parameters": {
            "episode_id": "Memory to learn from",
            "success": "true/false",
            "reasoning": "Why it succeeded/failed"
        }
    },
    {
        "name": "get_strategy_advice",
        "description": "Get recommended approach for current situation",
        "parameters": {
            "context": "Current situation",
            "goal": "What you're trying to achieve"
        }
    }
]
```

**Configuration**:
```yaml
memory:
  embedding_model: "path/to/model.gguf"
  llm_model: "path/to/llm.gguf"
  vector_store_path: "./memory_vectors"
  sqlite_path: "./agent_memory.db"
  
consolidation:
  trigger_after_episodes: 20
  min_cluster_size: 3
  similarity_threshold: 0.75
  
reflection:
  auto_reflect_on_failure: true
  failure_threshold: 0.3
  auto_reflect_on_success: true
  success_threshold: 0.9
  periodic_reflection: "daily"
  
forgetting:
  enable_forgetting: true
  min_access_count: 2
  age_threshold_days: 30
  redundancy_threshold: 0.95
  
performance:
  max_episodes_in_memory: 1000
  max_semantic_knowledge: 500
  embedding_batch_size: 10
  retrieval_timeout_seconds: 2
```

**Memory Explorer CLI**:
```bash
# Inspect memory state
python memory_explorer.py stats
python memory_explorer.py episodes --recent 10
python memory_explorer.py semantic --query "debugging strategies"
python memory_explorer.py reflections --type failure_analysis

# Memory operations
python memory_explorer.py consolidate --force
python memory_explorer.py prune --dry-run
python memory_explorer.py export --format json

# Debugging
python memory_explorer.py trace-episode 42
python memory_explorer.py visualize-clusters
```

**Success Criteria**:
- Complete, documented codebase
- All tests passing
- Performance metrics within targets
- Easy to integrate with any llama.cpp-based agent

---

## Implementation Guidelines

### For Nikos & Claude Code Collaboration

**Nikos's Role**:
- Provide domain expertise and design feedback
- Test each phase thoroughly
- Suggest improvements based on your agent needs
- Write some core functions to learn the patterns
- Review and understand all code before merging

**Claude Code's Role**:
- Implement core functionality for each phase
- Write comprehensive tests
- Provide detailed comments explaining design decisions
- Suggest best practices and alternatives
- Ask clarifying questions when requirements are ambiguous

### Development Workflow

1. **Start of Phase**: Review phase goals together, clarify requirements
2. **Design**: Sketch out schema/API before coding
3. **Implementation**: Claude Code implements, Nikos reviews and contributes
4. **Testing**: Write tests together, verify success criteria
5. **Reflection**: Document what worked, what didn't, lessons learned
6. **Commit**: Only move to next phase when current phase is solid

### Code Quality Standards

- **Type hints**: Full type annotations for all functions
- **Documentation**: Docstrings for all public functions, inline comments for complex logic
- **Testing**: Unit tests for core functions, integration tests for workflows
- **Error handling**: Graceful degradation, informative error messages
- **Logging**: Structured logging for debugging and monitoring

### Learning Checkpoints

After each phase, Nikos should be able to:
- Explain the design decisions made
- Modify or extend the code independently
- Understand the tradeoffs involved
- Apply similar patterns to other projects

---

## Timeline & Milestones

**Phase 1**: 2-3 days (Foundation)
**Phase 2**: 2-3 days (Learning)
**Phase 3**: 3-4 days (Consolidation)
**Phase 4**: 3-4 days (Reflection)
**Phase 5**: 4-5 days (Adaptation)
**Phase 6**: 3-4 days (Polish)

**Total**: ~3-4 weeks with active participation

---

## Success Metrics for Complete System

### Functional Metrics
- Stores 1000+ episodes efficiently
- Retrieves relevant memories in <1s
- Generates useful reflections (human-evaluated)
- Shows learning (improving success rate over time)
- Successfully adapts to new problem types

### Learning Metrics
- Success rate improves by 20%+ after 100 episodes
- Average success score increases over time
- Fewer repeated mistakes (same failure pattern)
- Faster problem-solving on familiar problem types

### System Health Metrics
- Memory database size stays bounded (<100MB for 1000 episodes)
- No memory leaks over extended runs
- Graceful handling of llama.cpp failures
- Recovery from corrupted database states

---

## Extension Ideas (Post-Phase 6)

Once core system is working, consider:

1. **Multi-agent memory sharing**: Multiple agents learn from shared experiences
2. **Hierarchical memory**: Short-term working memory + long-term storage
3. **Curiosity-driven exploration**: Agent actively seeks novel experiences
4. **Memory visualization**: Web UI for exploring memory structures
5. **A/B testing framework**: Compare memory strategies empirically
6. **Dream/replay mechanism**: Offline learning from stored memories
7. **Episodic memory compression**: More aggressive forgetting strategies
8. **Cross-domain transfer**: Explicit analogical reasoning engine

---

## Resources & References

### Papers to Read
- "Memory Augmented Neural Networks" (Graves et al.)
- "Neural Turing Machines" (Graves et al.)
- "Learning to Learn" / Meta-learning surveys
- "Episodic Memory in Lifelong Language Learning" (recent ACL papers)

### Code References
- LangChain memory implementations (for patterns, not dependencies)
- MemGPT architecture
- Voyager (Minecraft agent with skill learning)

### llama.cpp Resources
- Embedding generation API
- Model recommendations for embeddings
- Performance tuning guides

---

## Questions to Address During Development

**Phase 1**:
- What embedding model works best for our use case?
- How to structure episode context for optimal retrieval?
- SQLite vs other lightweight databases?

**Phase 2**:
- How to automatically evaluate success without human feedback?
- What makes a pattern "stable enough" to learn from?
- How to handle ambiguous outcomes?

**Phase 3**:
- When to consolidate (time-based, count-based, or semantic triggers)?
- How aggressively to forget?
- How to balance specificity vs generality in semantic knowledge?

**Phase 4**:
- What makes a reflection "good"?
- How to prevent hallucinated causal explanations?
- Balance between reflection time and action time?

**Phase 5**:
- How to measure similarity across problem domains?
- When to use analogical reasoning vs direct memory?
- How to avoid negative transfer?

**Phase 6**:
- What configuration knobs are most important?
- How to make system debuggable?
- What metrics to expose to users?

---

## Getting Started

1. Set up Python environment with dependencies
2. Install and configure llama.cpp
3. Choose embedding model and LLM
4. Create project structure
5. Begin Phase 1 implementation

**Next Step**: Review this roadmap together, adjust based on Nikos's preferences and constraints, then begin Phase 1!

---

## Notes

- This is a living document - update as we learn
- Phases may blend or reorder based on discoveries
- Focus on learning and understanding over speed
- It's okay to iterate and revise earlier phases
- Document all significant decisions and their rationale

**Last Updated**: 2026-02-02
**Status**: Planning Complete, Ready to Begin Implementation
