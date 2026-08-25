-- Agent Memory System - Database Schema
-- Version: 0.4.0 (Phases 1-4)

-- =============================================================================
-- EPISODES: Core episodic memory
-- =============================================================================
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    context TEXT NOT NULL,              -- What was the situation?
    action TEXT NOT NULL,               -- What did the agent do?
    outcome TEXT,                       -- What happened?
    success_score REAL,                 -- 0.0 to 1.0
    tags TEXT DEFAULT '[]',             -- JSON array of tags
    embedding_id TEXT,                  -- Reference to vector store

    -- Outcome classification
    outcome_category TEXT CHECK(outcome_category IN ('success', 'failure', 'partial', 'unknown', NULL)),
    failure_reason TEXT,

    -- Reinforcement (dedup): how often this memory was re-observed
    occurrence_count INTEGER DEFAULT 1,
    last_confirmed DATETIME,

    CHECK(json_valid(tags))
);

CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_success_score ON episodes(success_score);
CREATE INDEX IF NOT EXISTS idx_episodes_tags ON episodes(tags);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome_category ON episodes(outcome_category);

-- =============================================================================
-- LEARNED PATTERNS: Consolidated knowledge from episode clusters
-- =============================================================================
CREATE TABLE IF NOT EXISTS learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_description TEXT NOT NULL,
    context_signature TEXT NOT NULL,    -- Simplified context (e.g., "python debugging")
    recommended_action TEXT NOT NULL,
    success_rate REAL NOT NULL,         -- 0.0 to 1.0
    sample_count INTEGER NOT NULL,      -- Number of source episodes
    confidence REAL DEFAULT 0.5,        -- 0.0 to 1.0
    source_episode_ids TEXT DEFAULT '[]', -- JSON array of episode IDs
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_id TEXT,

    CHECK(success_rate >= 0.0 AND success_rate <= 1.0),
    CHECK(confidence >= 0.0 AND confidence <= 1.0),
    CHECK(sample_count > 0),
    CHECK(json_valid(source_episode_ids))
);

CREATE INDEX IF NOT EXISTS idx_patterns_context ON learned_patterns(context_signature);
CREATE INDEX IF NOT EXISTS idx_patterns_success_rate ON learned_patterns(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_updated ON learned_patterns(last_updated DESC);

-- =============================================================================
-- REFLECTIONS: Deeper analysis of experiences
-- =============================================================================
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reflection_type TEXT NOT NULL,      -- 'success_analysis', 'failure_analysis', 'pattern_discovery'
    trigger_episode_id INTEGER,         -- Episode that triggered this reflection
    insight TEXT NOT NULL,              -- The main insight/lesson learned
    causal_chain TEXT DEFAULT '[]',     -- JSON: [{"factor": "...", "contribution": "positive|negative", "confidence": 0.8}]
    actionable_takeaway TEXT,           -- Specific action for similar situations
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_id TEXT,

    -- Reinforcement (dedup): how often this insight was rediscovered
    occurrence_count INTEGER DEFAULT 1,
    last_confirmed DATETIME,

    CHECK(reflection_type IN ('success_analysis', 'failure_analysis', 'pattern_discovery')),
    CHECK(json_valid(causal_chain)),
    FOREIGN KEY (trigger_episode_id) REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(reflection_type);
CREATE INDEX IF NOT EXISTS idx_reflections_episode ON reflections(trigger_episode_id);
CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at DESC);

-- =============================================================================
-- FORGOTTEN MEMORIES: Archived low-utility episodes
-- =============================================================================
CREATE TABLE IF NOT EXISTS forgotten_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_id INTEGER NOT NULL,
    original_type TEXT DEFAULT 'episode', -- 'episode' or 'pattern'
    reason TEXT NOT NULL,               -- 'low_utility', 'redundant', 'outdated', 'consolidated'
    summary TEXT,
    forgotten_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Original data for potential recovery
    original_context TEXT,
    original_action TEXT,
    original_outcome TEXT,
    original_success_score REAL,
    original_tags TEXT
);

CREATE INDEX IF NOT EXISTS idx_forgotten_reason ON forgotten_memories(reason);
CREATE INDEX IF NOT EXISTS idx_forgotten_at ON forgotten_memories(forgotten_at);

-- =============================================================================
-- METADATA: System tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS memory_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- PROBLEM TYPES: Categorization of problem domains for transfer learning
-- =============================================================================
CREATE TABLE IF NOT EXISTS problem_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,           -- e.g., "python_debugging", "git_workflow"
    description TEXT,                     -- Human-readable description
    characteristic_features TEXT DEFAULT '[]', -- JSON: ["error messages", "stack traces", ...]
    successful_strategies TEXT DEFAULT '[]',   -- JSON: [strategy_id, ...]
    similar_problem_types TEXT DEFAULT '[]',   -- JSON: [type_id, ...] for related domains
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_id TEXT,                    -- Reference to vector store

    CHECK(json_valid(characteristic_features)),
    CHECK(json_valid(successful_strategies)),
    CHECK(json_valid(similar_problem_types))
);

CREATE INDEX IF NOT EXISTS idx_problem_types_name ON problem_types(name);
CREATE INDEX IF NOT EXISTS idx_problem_types_updated ON problem_types(updated_at DESC);

-- =============================================================================
-- DOMAIN KEYWORDS: Learnable markers for domain detection
-- =============================================================================
CREATE TABLE IF NOT EXISTS domain_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_name TEXT NOT NULL,           -- e.g., 'python', 'docker', 'git'
    keyword TEXT NOT NULL,               -- e.g., 'typeerror', 'container', 'commit'
    weight REAL DEFAULT 1.0,             -- How strongly this keyword indicates the domain (0.0-1.0)
    source TEXT DEFAULT 'seed',          -- 'seed' (initial), 'learned' (from episodes), 'llm' (LLM-generated)
    occurrence_count INTEGER DEFAULT 0,  -- How many times seen in episodes with this domain
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(domain_name, keyword),
    CHECK(weight >= 0.0 AND weight <= 1.0),
    CHECK(source IN ('seed', 'learned', 'llm'))
);

CREATE INDEX IF NOT EXISTS idx_domain_keywords_domain ON domain_keywords(domain_name);
CREATE INDEX IF NOT EXISTS idx_domain_keywords_keyword ON domain_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_domain_keywords_weight ON domain_keywords(weight DESC);

-- =============================================================================
-- ADAPTATIONS: Track strategy transfers between domains
-- =============================================================================
CREATE TABLE IF NOT EXISTS adaptations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_problem_type_id INTEGER,       -- Original problem type (nullable if unknown)
    target_problem_type_id INTEGER,       -- New problem type (nullable if unknown)
    source_context TEXT NOT NULL,         -- Original context where strategy worked
    target_context TEXT NOT NULL,         -- New context where strategy was applied
    original_strategy TEXT NOT NULL,      -- Strategy from source domain
    adapted_strategy TEXT NOT NULL,       -- Modified strategy for target domain
    adaptation_reasoning TEXT,            -- LLM explanation of how/why adapted
    outcome TEXT,                         -- What happened after adaptation
    success_score REAL,                   -- 0.0 to 1.0
    source_episode_ids TEXT DEFAULT '[]', -- JSON: Episodes that informed the adaptation
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    CHECK(success_score IS NULL OR (success_score >= 0.0 AND success_score <= 1.0)),
    CHECK(json_valid(source_episode_ids)),
    FOREIGN KEY (source_problem_type_id) REFERENCES problem_types(id),
    FOREIGN KEY (target_problem_type_id) REFERENCES problem_types(id)
);

CREATE INDEX IF NOT EXISTS idx_adaptations_source_type ON adaptations(source_problem_type_id);
CREATE INDEX IF NOT EXISTS idx_adaptations_target_type ON adaptations(target_problem_type_id);
CREATE INDEX IF NOT EXISTS idx_adaptations_success ON adaptations(success_score DESC);
CREATE INDEX IF NOT EXISTS idx_adaptations_created ON adaptations(created_at DESC);

-- =============================================================================
-- METADATA: System tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS memory_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Initialize metadata
INSERT OR IGNORE INTO memory_metadata (key, value) VALUES
    ('schema_version', '0.5.0'),
    ('created_at', datetime('now')),
    ('total_episodes', '0'),
    ('total_patterns', '0'),
    ('total_reflections', '0'),
    ('total_problem_types', '0'),
    ('total_adaptations', '0');
