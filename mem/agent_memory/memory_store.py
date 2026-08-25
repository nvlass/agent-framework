"""
Core memory storage system

This module implements the episodic memory store, combining:
- SQLite for structured data and metadata
- ChromaDB for vector similarity search
- EmbeddingGenerator for semantic encoding
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import Counter

from .embeddings import EmbeddingGenerator
from .vector_backend import VectorBackend, build_backend
from .working_memory import WorkingMemory
from .short_term_memory import ShortTermMemory
from .consolidation import ConsolidationEngine, LearnedPattern, ConsolidationReport


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between -1 and 1 (1 = most similar)
    """
    # Normalize vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    # Cosine similarity = dot product of normalized vectors
    return float(np.dot(a, b) / (a_norm * b_norm))


def _episode_search_text(context: str, action: str, outcome: str = "") -> str:
    """Text an episode is embedded under — must match at store and query time."""
    text = f"Context: {context}\nAction: {action}"
    if outcome:
        text += f"\nOutcome: {outcome}"
    return text


class Episode:
    """Represents a single episodic memory"""

    categories = ['success', 'failure', 'partial', 'unknown']

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
        outcome_category: Optional[str] = None, # 'success', 'failure', 'partial', 'unknown'
        failure_reason: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,  # NEW: Store actual embedding vector
    ):
        self.id = id
        self.timestamp = timestamp or datetime.now()
        self.context = context
        self.action = action
        self.outcome = outcome
        self.success_score = success_score
        self.tags = tags or []
        self.embedding_id = embedding_id
        self.outcome_category = outcome_category
        self.failure_reason = failure_reason
        self.embedding = embedding  # NEW: Store for working memory

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "success_score": self.success_score,
            "tags": self.tags,
            "embedding_id": self.embedding_id,
            "outcome_category": self.outcome_category,
            "failure_reason": self.failure_reason,
        }

    @staticmethod
    def from_db_row(row: Tuple) -> "Episode":
        """Create Episode from database row"""
        # SQLite stores JSON as TEXT, so we still need to parse it
        # But SQLite's CHECK constraint ensures it's always valid JSON
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
            outcome_category=row[8] if len(row) > 8 else None,
            failure_reason=row[9] if len(row) > 9 else None,
        )

    def __repr__(self) -> str:
        return f"Episode(id={self.id}, timestamp={self.timestamp}, context={self.context[:50]}...)"


class CausalFactor:
    """Represents a single factor in a causal chain"""

    def __init__(
        self,
        factor: str,
        contribution: str = "positive",  # 'positive', 'negative', 'neutral'
        confidence: float = 0.5,
    ):
        self.factor = factor
        self.contribution = contribution
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "contribution": self.contribution,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CausalFactor":
        return CausalFactor(
            factor=d.get("factor", ""),
            contribution=d.get("contribution", "positive"),
            confidence=d.get("confidence", 0.5),
        )


class Reflection:
    """
    Represents a reflection on an episode or pattern.

    Reflections capture insights about why something worked or failed,
    with causal analysis and actionable takeaways.
    """

    reflection_types = ['success_analysis', 'failure_analysis', 'pattern_discovery']

    def __init__(
        self,
        id: Optional[int] = None,
        reflection_type: str = "success_analysis",
        trigger_episode_id: Optional[int] = None,
        insight: str = "",
        causal_chain: Optional[List[CausalFactor]] = None,
        actionable_takeaway: Optional[str] = None,
        created_at: Optional[datetime] = None,
        embedding_id: Optional[str] = None,
    ):
        if reflection_type not in self.reflection_types:
            raise ValueError(f"reflection_type must be one of {self.reflection_types}")

        self.id = id
        self.reflection_type = reflection_type
        self.trigger_episode_id = trigger_episode_id
        self.insight = insight
        self.causal_chain = causal_chain or []
        self.actionable_takeaway = actionable_takeaway
        self.created_at = created_at or datetime.now()
        self.embedding_id = embedding_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert reflection to dictionary"""
        return {
            "id": self.id,
            "reflection_type": self.reflection_type,
            "trigger_episode_id": self.trigger_episode_id,
            "insight": self.insight,
            "causal_chain": [cf.to_dict() for cf in self.causal_chain],
            "actionable_takeaway": self.actionable_takeaway,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "embedding_id": self.embedding_id,
        }

    @staticmethod
    def from_db_row(row: Tuple) -> "Reflection":
        """Create Reflection from database row"""
        # Row: (id, reflection_type, trigger_episode_id, insight, causal_chain, actionable_takeaway, created_at, embedding_id)
        causal_chain_raw = row[4] if row[4] else '[]'
        causal_chain_data = json.loads(causal_chain_raw)

        return Reflection(
            id=row[0],
            reflection_type=row[1],
            trigger_episode_id=row[2],
            insight=row[3],
            causal_chain=[CausalFactor.from_dict(cf) for cf in causal_chain_data],
            actionable_takeaway=row[5],
            created_at=datetime.fromisoformat(row[6]) if row[6] else None,
            embedding_id=row[7] if len(row) > 7 else None,
        )

    def __repr__(self) -> str:
        return f"Reflection(id={self.id}, type={self.reflection_type}, insight={self.insight[:50]}...)"


class MemoryStore:
    """
    Main memory storage system

    Manages both structured (SQLite) and vector (ChromaDB) storage
    for agent episodic memories.
    """

    def __init__(
        self,
        db_path: str = "data/agent_memory.db",
        vector_store_path: str = "data/memory_vectors",
        embedding_generator: Optional[EmbeddingGenerator] = None,
        working_memory_size: int = 20,
        short_term_ttl_seconds: int = 300,
        short_term_window_hours: int = 24,
        backend: str = "chromadb",
    ):
        """
        Initialize memory store

        Args:
            db_path: Path to SQLite database
            vector_store_path: Directory for ChromaDB data (ignored when backend='sqlite-vec')
            embedding_generator: Optional pre-initialized embedding generator
            working_memory_size: Size of working memory hot cache (default: 20)
            short_term_ttl_seconds: TTL for short-term cache in seconds (default: 300 = 5 min)
            short_term_window_hours: Default time window for short-term memory (default: 24 hours)
            backend: Vector store backend — 'chromadb' (default) or 'sqlite-vec'
        """
        self.db_path = Path(db_path)
        self.vector_store_path = Path(vector_store_path)
        self.embedding_generator = embedding_generator

        # Create data directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_sqlite()

        # Initialize vector backend (ChromaDB or sqlite-vec)
        self._vec: VectorBackend = build_backend(backend, self.conn, str(self.vector_store_path))

        # Initialize memory hierarchy
        self.working_memory = WorkingMemory(max_size=working_memory_size)
        self.short_term = ShortTermMemory(
            ttl_seconds=short_term_ttl_seconds,
            default_time_window_hours=short_term_window_hours
        )

        # Initialize consolidation engine
        self.consolidation = ConsolidationEngine(
            min_cluster_size=3,
            min_samples=1,
            metric='euclidean'
        )

        # Track consolidation metadata
        self._ensure_consolidation_metadata()

    def _init_sqlite(self):
        """Initialize SQLite database with schema"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r") as f:
            schema = f.read()

        self.conn.executescript(schema)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self):
        """Add columns introduced after a table was created (existing DBs).

        schema.sql only runs CREATE TABLE IF NOT EXISTS, so databases created
        before a column existed need an explicit ALTER TABLE.
        """
        cursor = self.conn.cursor()
        for table in ("episodes", "reflections"):
            columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
            if "occurrence_count" not in columns:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN occurrence_count INTEGER DEFAULT 1"
                )
            if "last_confirmed" not in columns:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN last_confirmed DATETIME"
                )
        self.conn.commit()

    def start_session(self, session_id: str) -> None:
        self.working_memory.start_session(session_id)

    def clear_working_memory(self) -> None:
        self.working_memory.clear()

    def get_working_memory_episodes(self, n: Optional[int]) -> List[Episode]:
        if n is None:
            return self.working_memory.get_all()
        return self.working_memory.get_recent(n)

    def clear_short_term_cache(self) -> None:
        """Clear short-term memory cache"""
        self.short_term.clear()

    def get_short_term_cache_stats(self) -> Dict[str, int]:
        """Get short-term cache statistics"""
        return self.short_term.cache_stats()

    def store_episode(
        self,
        context: str,
        action: str,
        outcome: str = "",
        success_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Store a new episode

        Args:
            context: Description of the situation
            action: What the agent did
            outcome: What happened as a result
            success_score: Optional success rating (0.0 to 1.0)
            tags: Optional list of tags for categorization

        Returns:
            Episode ID

        Raises:
            RuntimeError: If embedding generator is not set
        """
        if self.embedding_generator is None:
            raise RuntimeError("EmbeddingGenerator must be set to store episodes")

        # Create episode
        episode = Episode(
            context=context,
            action=action,
            outcome=outcome,
            success_score=success_score,
            tags=tags or [],
        )

        # Generate embedding for semantic search.
        # The outcome must be included: callers like save_note and L1 extraction
        # use fixed context/action strings and carry all content in outcome —
        # without it, those episodes are indistinguishable in vector space.
        search_text = _episode_search_text(context, action, outcome)
        embedding = self.embedding_generator.generate_embedding(search_text)

        # Store in SQLite
        # Use SQLite's json() function to store tags as validated JSON
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO episodes (context, action, outcome, success_score, tags)
            VALUES (?, ?, ?, ?, json(?))
            """,
            (context, action, outcome, success_score, json.dumps(tags or [])),
        )
        episode_id = cursor.lastrowid
        self.conn.commit()

        # Store embedding in vector backend
        self._vec.add(episode_id, embedding)
        embedding_id = f"episode_{episode_id}"

        # Update embedding_id in SQLite
        cursor.execute(
            "UPDATE episodes SET embedding_id = ? WHERE id = ?",
            (embedding_id, episode_id),
        )
        self.conn.commit()

        # Update metadata
        self._update_metadata("total_episodes", episode_id)

        # Add to working memory with full data
        episode.id = episode_id
        episode.embedding_id = embedding_id
        episode.embedding = embedding  # Store embedding vector for similarity computation
        self.working_memory.add(episode)

        # Invalidate short-term cache (new episode available)
        self.short_term.invalidate_on_new_episode()

        # Track for consolidation
        self._increment_episodes_since_consolidation()

        return episode_id

    def retrieve_episodes(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.0,
        check_working_memory: bool = True,
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve episodes similar to query

        Args:
            query: Search query (natural language)
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (Episode, similarity_score) tuples, sorted by similarity

        Raises:
            RuntimeError: If embedding generator is not set
        """
        if self.embedding_generator is None:
            raise RuntimeError("EmbeddingGenerator must be set to retrieve episodes")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Tier 1: Check working memory first for relevant episodes
        working_results = []
        if check_working_memory:
            for episode in self.working_memory.get_all():
                if episode.embedding is not None:
                    similarity = cosine_similarity(query_embedding, episode.embedding)
                    if similarity >= min_similarity:
                        working_results.append((episode, similarity))

        # Tier 2: Check short-term memory (recent episodes with embeddings)
        shortterm_results = []
        recent_episodes = self.short_term.get_recent_episodes(
            hours=24,
            loader=lambda: self._fetch_recent_with_embeddings(24)
        )
        for episode in recent_episodes:
            if episode.embedding is not None:
                similarity = cosine_similarity(query_embedding, episode.embedding)
                if similarity >= min_similarity:
                    shortterm_results.append((episode, similarity))

        # Tier 3: Search in vector store (long-term memory)
        vec_results = self._vec.query(query_embedding, limit)
        if not vec_results:
            return []

        episode_ids = [ep_id for ep_id, _ in vec_results]
        distances = [dist for _, dist in vec_results]

        # Convert distances to similarity scores (1 - normalized_distance)
        # ChromaDB returns L2 distances, we convert to similarity
        max_distance = max(distances) if distances else 1.0
        similarities = [1.0 - (d / max_distance) if max_distance > 0 else 1.0 for d in distances]

        # Fetch full episodes from SQLite (long-term memory)
        cursor = self.conn.cursor()
        longterm_results = []

        for episode_id, similarity in zip(episode_ids, similarities):
            if similarity < min_similarity:
                continue

            cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                episode = Episode.from_db_row(row)
                longterm_results.append((episode, similarity))

        # Merge three-tier results: working → short-term → long-term
        # Remove duplicates (earlier tiers take precedence)
        working_ids = {ep.id for ep, _ in working_results}
        shortterm_ids = {ep.id for ep, _ in shortterm_results}

        all_results = (
            working_results +  # Tier 1: Working memory (highest priority)
            [(ep, sim) for ep, sim in shortterm_results if ep.id not in working_ids] +  # Tier 2
            [(ep, sim) for ep, sim in longterm_results if ep.id not in working_ids and ep.id not in shortterm_ids]  # Tier 3
        )

        # Sort by similarity (highest first) and limit
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:limit]

    def get_recent_episodes(self, hours: int = 24, limit: int = 10) -> List[Episode]:
        """
        Get recent episodes within time window

        Args:
            hours: Time window in hours
            limit: Maximum number of results

        Returns:
            List of recent episodes, newest first
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM episodes
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (cutoff_time.strftime('%Y-%m-%d %H:%M:%S'), limit),
        )

        return [Episode.from_db_row(row) for row in cursor.fetchall()]

    def _fetch_recent_with_embeddings(self, hours: int) -> List[Episode]:
        """
        Fetch recent episodes from DB with embeddings loaded

        This is used by short-term memory cache loader.
        Includes embedding vectors for similarity search.

        Args:
            hours: Time window in hours

        Returns:
            List of episodes with embeddings loaded
        """
        # Get recent episodes from DB
        episodes = self.get_recent_episodes(hours=hours, limit=1000)

        # Load embeddings from vector backend for each episode
        if self.embedding_generator and episodes:
            ep_ids = [ep.id for ep in episodes if ep.id is not None]
            if ep_ids:
                embedding_map = self._vec.get_embeddings(ep_ids)
                for episode in episodes:
                    if episode.id in embedding_map:
                        episode.embedding = embedding_map[episode.id]

        return episodes

    def get_episode_by_id(self, episode_id: int) -> Optional[Episode]:
        """Get specific episode by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = cursor.fetchone()
        return Episode.from_db_row(row) if row else None

    def get_tag_counts(
        self,
        skip_tags: Optional[set] = None,
    ) -> List[Tuple[str, int]]:
        """
        Aggregate tags across all episodes, weighted by occurrence_count.

        Reinforced memories push their tags up the ranking, but the weight is
        log-damped (1 + floor(log2(count))): a memory reinforced 1800 times by
        a repetitive background loop contributes 11, not 1800 — otherwise one
        runaway topic drowns the whole index.

        Args:
            skip_tags: Tags to exclude (type markers like "note", "atom").

        Returns:
            List of (tag, weighted_count), most frequent first.
        """
        import math
        skip = {str(t).lower() for t in (skip_tags or set())}
        counts: Dict[str, int] = {}
        cursor = self.conn.cursor()
        for tags_json, occurrence in cursor.execute(
            "SELECT tags, COALESCE(occurrence_count, 1) FROM episodes"
        ):
            weight = 1 + int(math.log2(max(occurrence, 1)))
            try:
                tags = json.loads(tags_json or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            for tag in tags:
                t = str(tag).strip().lower()
                if t and t not in skip:
                    counts[t] = counts.get(t, 0) + weight
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def get_all_episodes(self, limit: Optional[int] = None) -> List[Episode]:
        """Get all episodes, optionally limited"""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM episodes ORDER BY timestamp DESC")

        return [Episode.from_db_row(row) for row in cursor.fetchall()]

    def get_episode_count(self) -> int:
        """Get total number of stored episodes"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]

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
        """
        if category not in Episode.categories:
            return False

        cursor = self.conn.cursor()
        cursor.execute("UPDATE episodes SET outcome_category = ?, failure_reason = ? WHERE id = ?",
                       (category, failure_reason, episode_id))
        if cursor.rowcount <= 0:
            return False

        self.conn.commit()
        return True
        
    def get_episodes_by_category(
            self,
            category: str,
            limit: Optional[int] = None
    ) -> List[Episode]:
        """
        Get episodes by outcome category

        Returns empty list for unknown categories (lenient behavior).

        Args:
            category: 'success', 'failure', 'partial', or 'unknown'
            limit: Optional max number of results

        Returns:
            List of episodes in that category, newest first
        """
        # we could add a check for category validity, but perhaps,
        # it's ok to get an empty list when asking for an unknown
        # category
        cursor = self.conn.cursor()
        query = "SELECT * FROM episodes WHERE outcome_category = ? ORDER BY timestamp DESC"
        params = (category,)

        if limit is not None:
            query = query + " LIMIT ?"
            params = params + (limit,)
        cursor.execute(query, params)

        return [Episode.from_db_row(row) for row in cursor.fetchall()]

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
            - unknown: int (category='unknown')
        """
        episodes = self.get_episodes_by_tags(tags, match_all)
        categories = [ep.outcome_category for ep in episodes]
        counts = Counter(categories)

        success = counts['success']
        failure = counts['failure']
        partial = counts['partial']
        unknown = counts['unknown']

        # Sum the counts (alternative: total = len(episodes))
        total = success + failure + partial + unknown
        success_rate = success / total if total > 0 else 0.0

        return {
            'success_rate': success_rate,
            'total_episodes': total,
            'successful': success,
            'failed': failure,
            'partial': partial,
            'unknown': unknown
        }

        

    def get_episodes_by_tag(self, tag: str, limit: Optional[int] = None) -> List[Episode]:
        """
        Get episodes containing a specific tag

        Uses SQLite's JSON functions to efficiently query the tags array.

        Args:
            tag: Tag to search for
            limit: Optional maximum number of results

        Returns:
            List of episodes containing the tag
        """
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM episodes
            WHERE EXISTS (
                SELECT 1 FROM json_each(episodes.tags)
                WHERE json_each.value = ?
            )
            ORDER BY timestamp DESC
        """

        if limit:
            query += " LIMIT ?"
            cursor.execute(query, (tag, limit))
        else:
            cursor.execute(query, (tag,))

        return [Episode.from_db_row(row) for row in cursor.fetchall()]

    def get_episodes_by_tags(
        self, tags: List[str], match_all: bool = False, limit: Optional[int] = None
    ) -> List[Episode]:
        """
        Get episodes containing specific tags

        Args:
            tags: List of tags to search for
            match_all: If True, episode must contain ALL tags. If False, ANY tag matches
            limit: Optional maximum number of results

        Returns:
            List of matching episodes
        """
        if not tags:
            return []

        cursor = self.conn.cursor()

        if match_all:
            # Episode must contain ALL specified tags
            # Build a query that checks for each tag
            conditions = []
            for _ in tags:
                conditions.append(
                    "EXISTS (SELECT 1 FROM json_each(episodes.tags) WHERE json_each.value = ?)"
                )

            query = f"""
                SELECT * FROM episodes
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC
            """
        else:
            # Episode must contain ANY of the specified tags
            query = """
                SELECT * FROM episodes
                WHERE EXISTS (
                    SELECT 1 FROM json_each(episodes.tags)
                    WHERE json_each.value IN ({})
                )
                ORDER BY timestamp DESC
            """.format(','.join('?' * len(tags)))

        if limit:
            query += " LIMIT ?"
            params = (*tags, limit)
        else:
            params = tags

        cursor.execute(query, params)
        return [Episode.from_db_row(row) for row in cursor.fetchall()]

    def get_all_tags(self) -> List[Tuple[str, int]]:
        """
        Get all unique tags with their usage counts

        Returns:
            List of (tag, count) tuples, sorted by count descending
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT json_each.value as tag, COUNT(*) as count
            FROM episodes, json_each(episodes.tags)
            GROUP BY tag
            ORDER BY count DESC
        """)

        return [(row[0], row[1]) for row in cursor.fetchall()]

    def _update_metadata(self, key: str, value: Any):
        """Update metadata value"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO memory_metadata (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            """,
            (key, str(value)),
        )
        self.conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM memory_metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        cursor = self.conn.cursor()

        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM episodes")
        total_episodes = cursor.fetchone()[0]

        # Episodes with outcomes
        cursor.execute("SELECT COUNT(*) FROM episodes WHERE outcome IS NOT NULL AND outcome != ''")
        episodes_with_outcomes = cursor.fetchone()[0]

        # Episodes with success scores
        cursor.execute("SELECT COUNT(*) FROM episodes WHERE success_score IS NOT NULL")
        scored_episodes = cursor.fetchone()[0]

        # Average success score
        cursor.execute("SELECT AVG(success_score) FROM episodes WHERE success_score IS NOT NULL")
        avg_success = cursor.fetchone()[0]

        # Phase 2: Outcome category statistics (if column exists)
        # Check if outcome_category column exists (Phase 2 schema)
        cursor.execute("PRAGMA table_info(episodes)")
        columns = [row[1] for row in cursor.fetchall()]
        has_outcome_category = 'outcome_category' in columns

        category_counts = {}
        if has_outcome_category:
            cursor.execute("""
                SELECT outcome_category, COUNT(*)
                FROM episodes
                WHERE outcome_category IS NOT NULL
                GROUP BY outcome_category
            """)
            category_counts = {row[0]: row[1] for row in cursor.fetchall()}

        stats = {
            "total_episodes": total_episodes,
            "episodes_with_outcomes": episodes_with_outcomes,
            "scored_episodes": scored_episodes,
            "average_success_score": round(avg_success, 3) if avg_success else None,
            "vector_store_size": self._vec.count(),
        }

        # Add Phase 2 stats only if column exists
        if has_outcome_category:
            stats.update({
                "success_count": category_counts.get('success', 0),
                "failure_count": category_counts.get('failure', 0),
                "partial_count": category_counts.get('partial', 0),
                "unknown_count": category_counts.get('unknown', 0),
            })

        return stats

    def _ensure_consolidation_metadata(self):
        """Ensure consolidation metadata exists"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO memory_metadata (key, value) VALUES (?, ?)",
            ('last_consolidation', '')
        )
        cursor.execute(
            "INSERT OR IGNORE INTO memory_metadata (key, value) VALUES (?, ?)",
            ('episodes_since_consolidation', '0')
        )
        self.conn.commit()

    def _get_consolidation_metadata(self) -> Tuple[Optional[datetime], int]:
        """Get consolidation tracking metadata"""
        cursor = self.conn.cursor()

        # Last consolidation time
        cursor.execute("SELECT value FROM memory_metadata WHERE key = 'last_consolidation'")
        row = cursor.fetchone()
        last_consolidation = None
        if row and row[0]:
            try:
                last_consolidation = datetime.fromisoformat(row[0])
            except ValueError:
                pass

        # Episodes since last consolidation
        cursor.execute("SELECT value FROM memory_metadata WHERE key = 'episodes_since_consolidation'")
        row = cursor.fetchone()
        episodes_since = int(row[0]) if row and row[0] else 0

        return last_consolidation, episodes_since

    def _update_consolidation_metadata(self):
        """Update consolidation metadata after running consolidation"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memory_metadata SET value = ?, updated_at = ? WHERE key = 'last_consolidation'",
            (datetime.now().isoformat(), datetime.now())
        )
        cursor.execute(
            "UPDATE memory_metadata SET value = '0', updated_at = ? WHERE key = 'episodes_since_consolidation'",
            (datetime.now(),)
        )
        self.conn.commit()

    def _increment_episodes_since_consolidation(self):
        """Increment episode counter for consolidation tracking"""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE memory_metadata
               SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                   updated_at = ?
               WHERE key = 'episodes_since_consolidation'""",
            (datetime.now(),)
        )
        self.conn.commit()

    def run_consolidation(
        self,
        hours_back: int = 168,  # 1 week
        auto_trigger: bool = False,
        episode_threshold: int = 100,
        time_threshold_hours: float = 24.0
    ) -> ConsolidationReport:
        """
        Run memory consolidation pipeline

        Clusters similar episodes and extracts learned patterns

        Args:
            hours_back: How far back to consolidate (default: 1 week)
            auto_trigger: If True, only run if thresholds are met
            episode_threshold: Episodes needed to trigger (if auto_trigger)
            time_threshold_hours: Hours needed to trigger (if auto_trigger)

        Returns:
            ConsolidationReport with results
        """
        # Check if consolidation should run
        if auto_trigger:
            last_consolidation, episodes_since = self._get_consolidation_metadata()

            if last_consolidation:
                hours_since = (datetime.now() - last_consolidation).total_seconds() / 3600.0
            else:
                hours_since = float('inf')  # Never run before

            should_run = self.consolidation.should_consolidate(
                episodes_since_last=episodes_since,
                hours_since_last=hours_since,
                episode_threshold=episode_threshold,
                time_threshold_hours=time_threshold_hours
            )

            if not should_run:
                # Return empty report
                return ConsolidationReport()

        # Get recent episodes for consolidation
        cutoff = datetime.now() - timedelta(hours=hours_back)
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT * FROM episodes
               WHERE timestamp >= ?
               ORDER BY timestamp DESC""",
            (cutoff.strftime('%Y-%m-%d %H:%M:%S'),)
        )

        episodes = [Episode.from_db_row(row) for row in cursor.fetchall()]

        # Load embeddings for episodes
        if self.embedding_generator and episodes:
            ep_ids = [ep.id for ep in episodes if ep.id is not None]
            if ep_ids:
                embedding_map = self._vec.get_embeddings(ep_ids)
                for episode in episodes:
                    if episode.id in embedding_map:
                        episode.embedding = embedding_map[episode.id]

        # Run consolidation
        report = self.consolidation.run_consolidation(episodes)

        # Store learned patterns
        for pattern in report.patterns:
            self._store_learned_pattern(pattern)

        # Update metadata
        self._update_consolidation_metadata()

        return report

    def _store_learned_pattern(self, pattern: LearnedPattern) -> int:
        """
        Store learned pattern in database

        Args:
            pattern: LearnedPattern to store

        Returns:
            Pattern ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO learned_patterns
               (pattern_description, context_signature, recommended_action,
                success_rate, sample_count, confidence, source_episode_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, json(?), ?)""",
            (
                pattern.pattern_description,
                pattern.context_signature,
                pattern.recommended_action,
                pattern.success_rate,
                pattern.sample_count,
                pattern.confidence,
                json.dumps(pattern.source_episode_ids),
                pattern.created_at.isoformat() if pattern.created_at else datetime.now().isoformat()
            )
        )
        pattern_id = cursor.lastrowid
        self.conn.commit()

        # Update metadata
        self._update_metadata('total_patterns', pattern_id)

        return pattern_id

    def get_learned_patterns(
        self,
        min_confidence: float = 0.0,
        min_success_rate: float = 0.0,
        limit: Optional[int] = None
    ) -> List[LearnedPattern]:
        """
        Retrieve learned patterns

        Args:
            min_confidence: Minimum confidence threshold
            min_success_rate: Minimum success rate threshold
            limit: Maximum patterns to return

        Returns:
            List of LearnedPattern objects
        """
        cursor = self.conn.cursor()

        query = """
            SELECT id, pattern_description, context_signature, recommended_action,
                   success_rate, sample_count, confidence, source_episode_ids, created_at
            FROM learned_patterns
            WHERE confidence >= ? AND success_rate >= ?
            ORDER BY confidence DESC, success_rate DESC
        """
        params = [min_confidence, min_success_rate]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)

        patterns = []
        for row in cursor.fetchall():
            pattern = LearnedPattern(
                pattern_id=row[0],
                pattern_description=row[1],
                context_signature=row[2],
                recommended_action=row[3],
                success_rate=row[4],
                sample_count=row[5],
                confidence=row[6],
                source_episode_ids=json.loads(row[7]) if row[7] else [],
                created_at=datetime.fromisoformat(row[8]) if row[8] else None
            )
            patterns.append(pattern)

        return patterns

    def recommend_actions(
        self,
        context: str,
        limit: int = 5,
        min_confidence: float = 0.5,
        min_success_rate: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Recommend actions based on learned patterns matching the context

        Uses keyword matching between context and pattern signatures.
        Returns ranked recommendations based on match quality, confidence, and success rate.

        Args:
            context: Current situation/context to match
            limit: Maximum recommendations to return
            min_confidence: Minimum pattern confidence
            min_success_rate: Minimum pattern success rate

        Returns:
            List of recommendation dicts with:
            - pattern: LearnedPattern
            - match_score: How well context matches pattern (0.0-1.0)
            - reason: Why this pattern was matched
        """
        # Get all qualifying patterns
        patterns = self.get_learned_patterns(
            min_confidence=min_confidence,
            min_success_rate=min_success_rate
        )

        if not patterns:
            return []

        # Extract keywords from context
        context_words = set(
            w.lower().strip('.,!?;:()[]{}')
            for w in context.split()
            if len(w) > 3
        )

        # Score each pattern by keyword overlap
        scored_patterns = []
        for pattern in patterns:
            # Extract keywords from pattern signature
            signature_words = set(
                w.lower().strip('.,!?;:()[]{}[]')
                for w in pattern.context_signature.replace(',', ' ').replace('[', ' ').replace(']', ' ').split()
                if len(w) > 2
            )

            # Calculate keyword overlap
            if signature_words:
                overlap = context_words & signature_words
                match_score = len(overlap) / len(signature_words)
            else:
                match_score = 0.0

            if match_score > 0:
                # Combine match score with pattern quality
                combined_score = (
                    match_score * 0.4 +
                    pattern.confidence * 0.3 +
                    pattern.success_rate * 0.3
                )

                scored_patterns.append({
                    'pattern': pattern,
                    'match_score': match_score,
                    'combined_score': combined_score,
                    'matched_keywords': list(overlap) if match_score > 0 else [],
                    'reason': f"Matched keywords: {', '.join(overlap)}" if overlap else "Low match"
                })

        # Sort by combined score
        scored_patterns.sort(key=lambda x: x['combined_score'], reverse=True)

        return scored_patterns[:limit]

    def get_action_advice(
        self,
        context: str,
        goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get advice for current situation based on learned patterns

        Higher-level API that provides actionable recommendations.

        Args:
            context: Current situation description
            goal: Optional goal to achieve

        Returns:
            Dictionary with:
            - recommendations: List of action recommendations
            - confidence: Overall confidence in recommendations
            - patterns_matched: Number of matching patterns
        """
        full_context = context
        if goal:
            full_context = f"{context} Goal: {goal}"

        recommendations = self.recommend_actions(full_context, limit=5)

        if not recommendations:
            return {
                'recommendations': [],
                'confidence': 0.0,
                'patterns_matched': 0,
                'message': "No matching patterns found. Consider similar past experiences."
            }

        # Build response
        advice = []
        for rec in recommendations:
            pattern = rec['pattern']
            advice.append({
                'action': pattern.recommended_action,
                'success_rate': pattern.success_rate,
                'confidence': pattern.confidence,
                'reason': rec['reason'],
                'based_on': pattern.sample_count
            })

        avg_confidence = sum(r['pattern'].confidence for r in recommendations) / len(recommendations)

        return {
            'recommendations': advice,
            'confidence': avg_confidence,
            'patterns_matched': len(recommendations),
            'message': f"Found {len(recommendations)} matching patterns"
        }

    # =========================================================================
    # Phase 4: Reflection Methods
    # =========================================================================

    def store_reflection(self, reflection: Reflection) -> int:
        """
        Store a reflection in the database.

        Args:
            reflection: Reflection to store

        Returns:
            ID of the stored reflection
        """
        cursor = self.conn.cursor()

        # Serialize causal chain to JSON
        causal_chain_json = json.dumps([cf.to_dict() for cf in reflection.causal_chain])

        cursor.execute("""
            INSERT INTO reflections
            (reflection_type, trigger_episode_id, insight, causal_chain,
             actionable_takeaway, created_at, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            reflection.reflection_type,
            reflection.trigger_episode_id,
            reflection.insight,
            causal_chain_json,
            reflection.actionable_takeaway,
            reflection.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            reflection.embedding_id,
        ))

        reflection_id = cursor.lastrowid
        self.conn.commit()

        return reflection_id

    # =========================================================================
    # Deduplication: novelty gate + reinforcement
    # =========================================================================

    def find_similar_episode(
        self,
        context: str,
        action: str,
        outcome: str = "",
        threshold: float = 0.9,
    ) -> Optional[Tuple[int, float]]:
        """
        Find an existing episode nearly identical to the candidate.

        Uses true cosine similarity against the raw stored embeddings (the
        relative scores from retrieve_episodes are normalized per-query and
        can't be compared against an absolute threshold).

        Returns:
            (episode_id, similarity) of the best match at or above threshold,
            or None. Also None when no embedding generator is configured —
            dedup is skipped rather than guessed.
        """
        if self.embedding_generator is None:
            return None
        candidate = self.embedding_generator.generate_embedding(
            _episode_search_text(context, action, outcome)
        )
        neighbors = self._vec.query(candidate, n_results=3)
        if not neighbors:
            return None
        stored = self._vec.get_embeddings([ep_id for ep_id, _ in neighbors])
        best_id, best_sim = None, -1.0
        for ep_id, emb in stored.items():
            sim = cosine_similarity(candidate, emb)
            if sim > best_sim:
                best_id, best_sim = ep_id, sim
        if best_id is not None and best_sim >= threshold:
            return best_id, best_sim
        return None

    def reinforce_episode(self, episode_id: int) -> int:
        """
        Record a repeated observation of an existing episode.

        Bumps occurrence_count and last_confirmed instead of storing a
        duplicate. Repetition is signal, not noise: the count is an
        importance weight for future retrieval ranking.

        Returns:
            The new occurrence count (0 if the episode doesn't exist).
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE episodes
            SET occurrence_count = COALESCE(occurrence_count, 1) + 1,
                last_confirmed = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (episode_id,),
        )
        self.conn.commit()
        row = cursor.execute(
            "SELECT occurrence_count FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return row[0] if row else 0

    def find_similar_reflection(
        self,
        insight: str,
        reflection_type: Optional[str] = None,
        threshold: float = 0.85,
        limit: int = 50,
    ) -> Optional[Tuple[int, float]]:
        """
        Find an existing reflection whose insight nearly matches the candidate.

        Reflections are not stored in the vector backend, so the recent ones
        are embedded on the fly (generators cache, and reflections number in
        the dozens, so this stays cheap).

        Returns:
            (reflection_id, similarity) of the best match at or above
            threshold, or None. None when no embedding generator is set.
        """
        if self.embedding_generator is None or not insight:
            return None
        cursor = self.conn.cursor()
        if reflection_type:
            rows = cursor.execute(
                "SELECT id, insight FROM reflections WHERE reflection_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (reflection_type, limit),
            ).fetchall()
        else:
            rows = cursor.execute(
                "SELECT id, insight FROM reflections ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        rows = [(rid, text) for rid, text in rows if text]
        if not rows:
            return None
        candidate = self.embedding_generator.generate_embedding(insight)
        existing = self.embedding_generator.generate_embeddings_batch(
            [text for _, text in rows]
        )
        best_id, best_sim = None, -1.0
        for (rid, _), emb in zip(rows, existing):
            sim = cosine_similarity(candidate, emb)
            if sim > best_sim:
                best_id, best_sim = rid, sim
        if best_id is not None and best_sim >= threshold:
            return best_id, best_sim
        return None

    def reinforce_reflection(self, reflection_id: int) -> int:
        """Bump occurrence_count/last_confirmed on an existing reflection.

        Returns the new occurrence count (0 if the reflection doesn't exist).
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE reflections
            SET occurrence_count = COALESCE(occurrence_count, 1) + 1,
                last_confirmed = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (reflection_id,),
        )
        self.conn.commit()
        row = cursor.execute(
            "SELECT occurrence_count FROM reflections WHERE id = ?",
            (reflection_id,),
        ).fetchone()
        return row[0] if row else 0

    def store_reflection_if_novel(
        self,
        reflection: Reflection,
        threshold: float = 0.85,
    ) -> Tuple[int, bool]:
        """
        Store a reflection unless a near-duplicate already exists.

        On a match, the existing reflection is reinforced instead of
        appending another copy.

        Returns:
            (reflection_id, is_novel) — the existing id with is_novel=False
            when a duplicate was found.
        """
        match = self.find_similar_reflection(
            reflection.insight,
            reflection_type=reflection.reflection_type,
            threshold=threshold,
        )
        if match:
            reflection_id, _ = match
            self.reinforce_reflection(reflection_id)
            return reflection_id, False
        return self.store_reflection(reflection), True

    def get_reflection_by_id(self, reflection_id: int) -> Optional[Reflection]:
        """Get a reflection by its ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, reflection_type, trigger_episode_id, insight, causal_chain,
                   actionable_takeaway, created_at, embedding_id
            FROM reflections
            WHERE id = ?
        """, (reflection_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return Reflection.from_db_row(row)

    def get_reflections(
        self,
        reflection_type: Optional[str] = None,
        episode_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Reflection]:
        """
        Get reflections with optional filtering.

        Args:
            reflection_type: Filter by type ('success_analysis', 'failure_analysis', 'pattern_discovery')
            episode_id: Filter by trigger episode
            limit: Maximum reflections to return

        Returns:
            List of matching reflections
        """
        cursor = self.conn.cursor()

        query = """
            SELECT id, reflection_type, trigger_episode_id, insight, causal_chain,
                   actionable_takeaway, created_at, embedding_id
            FROM reflections
            WHERE 1=1
        """
        params = []

        if reflection_type:
            query += " AND reflection_type = ?"
            params.append(reflection_type)

        if episode_id:
            query += " AND trigger_episode_id = ?"
            params.append(episode_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [Reflection.from_db_row(row) for row in rows]

    def get_reflections_for_episode(self, episode_id: int) -> List[Reflection]:
        """Get all reflections for a specific episode"""
        return self.get_reflections(episode_id=episode_id)

    def get_recent_reflections(self, hours: int = 24, limit: int = 10) -> List[Reflection]:
        """Get recent reflections within a time window"""
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(hours=hours)

        cursor.execute("""
            SELECT id, reflection_type, trigger_episode_id, insight, causal_chain,
                   actionable_takeaway, created_at, embedding_id
            FROM reflections
            WHERE created_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (cutoff.strftime('%Y-%m-%d %H:%M:%S'), limit))

        rows = cursor.fetchall()
        return [Reflection.from_db_row(row) for row in rows]

    def count_reflections(self, reflection_type: Optional[str] = None) -> int:
        """Count reflections, optionally filtered by type"""
        cursor = self.conn.cursor()

        if reflection_type:
            cursor.execute(
                "SELECT COUNT(*) FROM reflections WHERE reflection_type = ?",
                (reflection_type,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM reflections")

        return cursor.fetchone()[0]

    def forget_episode(
        self,
        episode_id: int,
        reason: str,
        summary: Optional[str] = None
    ) -> bool:
        """
        Forget (archive) an episode

        Moves episode to forgotten_memories table and deletes from active storage.

        Args:
            episode_id: Episode to forget
            reason: Why forgetting ('low_utility', 'redundant', 'outdated', 'consolidated')
            summary: Optional summary of what was forgotten

        Returns:
            True if episode was forgotten, False if not found
        """
        # Get the episode
        episode = self.get_episode_by_id(episode_id)
        if episode is None:
            return False

        cursor = self.conn.cursor()

        # Ensure forgotten_memories table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forgotten_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER NOT NULL,
                original_type TEXT DEFAULT 'episode',
                reason TEXT NOT NULL,
                summary TEXT,
                forgotten_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_context TEXT,
                original_action TEXT,
                original_outcome TEXT,
                original_success_score REAL,
                original_tags TEXT
            )
        """)

        # Auto-generate summary if not provided
        if summary is None:
            summary = f"{episode.context[:50]}... → {episode.action[:30]}..."

        # Archive to forgotten_memories
        cursor.execute("""
            INSERT INTO forgotten_memories
            (original_id, original_type, reason, summary, original_context,
             original_action, original_outcome, original_success_score, original_tags)
            VALUES (?, 'episode', ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode_id,
            reason,
            summary,
            episode.context,
            episode.action,
            episode.outcome,
            episode.success_score,
            json.dumps(episode.tags)
        ))

        # Delete from episodes
        cursor.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

        # Delete from vector store
        self._vec.delete(episode_id)

        self.conn.commit()
        return True

    def apply_forgetting_policy(
        self,
        age_threshold_days: int = 30,
        min_success_for_keep: float = 0.8,
        max_failure_for_keep: float = 0.3,
        max_forget: int = 50,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply forgetting policy to archive low-utility episodes

        Policy:
        - KEEP: High success (>0.8) episodes
        - KEEP: Low success (<0.3) episodes (learn from failures)
        - KEEP: Episodes younger than age_threshold
        - FORGET: Medium success, old episodes

        Args:
            age_threshold_days: Don't forget episodes newer than this
            min_success_for_keep: Keep episodes with success >= this
            max_failure_for_keep: Keep episodes with success <= this (failures to learn from)
            max_forget: Maximum episodes to forget in one run
            dry_run: If True, only report what would be forgotten

        Returns:
            Report with forgotten/kept counts
        """
        cursor = self.conn.cursor()

        # Find candidates for forgetting
        cutoff_date = datetime.now() - timedelta(days=age_threshold_days)

        cursor.execute("""
            SELECT * FROM episodes
            WHERE timestamp < ?
              AND (success_score IS NULL
                   OR (success_score < ? AND success_score > ?))
            ORDER BY timestamp ASC
            LIMIT ?
        """, (
            cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),
            min_success_for_keep,
            max_failure_for_keep,
            max_forget
        ))

        candidates = [Episode.from_db_row(row) for row in cursor.fetchall()]

        report = {
            'candidates_found': len(candidates),
            'forgotten': 0,
            'dry_run': dry_run,
            'episodes': []
        }

        for episode in candidates:
            episode_info = {
                'id': episode.id,
                'context': episode.context[:50] + "...",
                'success_score': episode.success_score,
                'age_days': (datetime.now() - episode.timestamp).days if episode.timestamp else None
            }
            report['episodes'].append(episode_info)

            if not dry_run:
                self.forget_episode(
                    episode.id,
                    reason='low_utility',
                    summary=f"Score: {episode.success_score}, Age: {episode_info['age_days']} days"
                )
                report['forgotten'] += 1

        return report

    def get_forgotten_memories(
        self,
        reason: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve archived forgotten memories

        Args:
            reason: Filter by forget reason (optional)
            limit: Maximum to return

        Returns:
            List of forgotten memory records
        """
        cursor = self.conn.cursor()

        if reason:
            cursor.execute("""
                SELECT * FROM forgotten_memories
                WHERE reason = ?
                ORDER BY forgotten_at DESC
                LIMIT ?
            """, (reason, limit))
        else:
            cursor.execute("""
                SELECT * FROM forgotten_memories
                ORDER BY forgotten_at DESC
                LIMIT ?
            """, (limit,))

        memories = []
        for row in cursor.fetchall():
            memories.append({
                'id': row[0],
                'original_id': row[1],
                'original_type': row[2],
                'reason': row[3],
                'summary': row[4],
                'forgotten_at': row[5],
                'original_context': row[6],
                'original_action': row[7],
                'original_outcome': row[8],
                'original_success_score': row[9],
                'original_tags': json.loads(row[10]) if row[10] else []
            })

        return memories

    # =========================================================================
    # Phase 5: Problem Types and Adaptations
    # =========================================================================

    def store_problem_type(
        self,
        name: str,
        description: str = "",
        characteristics: Optional[List[str]] = None,
        domain: str = "",
    ) -> int:
        """
        Store or update a problem type.

        Args:
            name: Unique name for the problem type (e.g., "python_debugging")
            description: Human-readable description
            characteristics: List of characteristic features
            domain: Broader domain category

        Returns:
            Problem type ID
        """
        cursor = self.conn.cursor()

        # Check if exists
        cursor.execute("SELECT id FROM problem_types WHERE name = ?", (name,))
        existing = cursor.fetchone()

        characteristics_json = json.dumps(characteristics or [])

        if existing:
            # Update existing
            cursor.execute("""
                UPDATE problem_types
                SET description = ?, characteristic_features = ?, updated_at = datetime('now')
                WHERE name = ?
            """, (description, characteristics_json, name))
            problem_type_id = existing[0]
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO problem_types (name, description, characteristic_features)
                VALUES (?, ?, ?)
            """, (name, description, characteristics_json))
            problem_type_id = cursor.lastrowid

        self.conn.commit()
        return problem_type_id

    def get_problem_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a problem type by name"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, description, characteristic_features,
                   successful_strategies, similar_problem_types, created_at, updated_at
            FROM problem_types WHERE name = ?
        """, (name,))

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'characteristics': json.loads(row[3]) if row[3] else [],
            'successful_strategies': json.loads(row[4]) if row[4] else [],
            'similar_problem_types': json.loads(row[5]) if row[5] else [],
            'created_at': row[6],
            'updated_at': row[7],
        }

    def get_problem_type_by_id(self, problem_type_id: int) -> Optional[Dict[str, Any]]:
        """Get a problem type by ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, description, characteristic_features,
                   successful_strategies, similar_problem_types, created_at, updated_at
            FROM problem_types WHERE id = ?
        """, (problem_type_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'characteristics': json.loads(row[3]) if row[3] else [],
            'successful_strategies': json.loads(row[4]) if row[4] else [],
            'similar_problem_types': json.loads(row[5]) if row[5] else [],
            'created_at': row[6],
            'updated_at': row[7],
        }

    def get_all_problem_types(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all problem types"""
        cursor = self.conn.cursor()

        query = """
            SELECT id, name, description, characteristic_features,
                   successful_strategies, similar_problem_types, created_at, updated_at
            FROM problem_types ORDER BY name
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)

        return [
            {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'characteristics': json.loads(row[3]) if row[3] else [],
                'successful_strategies': json.loads(row[4]) if row[4] else [],
                'similar_problem_types': json.loads(row[5]) if row[5] else [],
                'created_at': row[6],
                'updated_at': row[7],
            }
            for row in cursor.fetchall()
        ]

    def link_similar_problem_types(
        self,
        type_id_1: int,
        type_id_2: int,
    ) -> bool:
        """
        Link two problem types as similar.

        Creates bidirectional link for transfer learning.

        Args:
            type_id_1: First problem type ID
            type_id_2: Second problem type ID

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()

        # Get current similar types for both
        for type_id, other_id in [(type_id_1, type_id_2), (type_id_2, type_id_1)]:
            cursor.execute(
                "SELECT similar_problem_types FROM problem_types WHERE id = ?",
                (type_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return False

            similar = json.loads(row[0]) if row[0] else []
            if other_id not in similar:
                similar.append(other_id)
                cursor.execute(
                    "UPDATE problem_types SET similar_problem_types = ? WHERE id = ?",
                    (json.dumps(similar), type_id)
                )

        self.conn.commit()
        return True

    def store_adaptation(
        self,
        source_context: str,
        target_context: str,
        original_strategy: str,
        adapted_strategy: str,
        adaptation_reasoning: Optional[str] = None,
        outcome: Optional[str] = None,
        success_score: Optional[float] = None,
        source_episode_ids: Optional[List[int]] = None,
        source_problem_type_id: Optional[int] = None,
        target_problem_type_id: Optional[int] = None,
    ) -> int:
        """
        Store an adaptation record.

        Args:
            source_context: Original context where strategy worked
            target_context: New context where it was applied
            original_strategy: The original action
            adapted_strategy: The modified strategy
            adaptation_reasoning: How/why the adaptation was made
            outcome: What happened (set later after trying)
            success_score: How well it worked (0.0-1.0)
            source_episode_ids: Episodes that informed this adaptation
            source_problem_type_id: Source problem type (optional)
            target_problem_type_id: Target problem type (optional)

        Returns:
            Adaptation ID
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO adaptations
            (source_problem_type_id, target_problem_type_id, source_context,
             target_context, original_strategy, adapted_strategy,
             adaptation_reasoning, outcome, success_score, source_episode_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            source_problem_type_id,
            target_problem_type_id,
            source_context,
            target_context,
            original_strategy,
            adapted_strategy,
            adaptation_reasoning,
            outcome,
            success_score,
            json.dumps(source_episode_ids or []),
        ))

        adaptation_id = cursor.lastrowid
        self.conn.commit()

        return adaptation_id

    def update_adaptation_outcome(
        self,
        adaptation_id: int,
        outcome: str,
        success_score: float,
    ) -> bool:
        """
        Update an adaptation with its outcome after trying it.

        Args:
            adaptation_id: Adaptation to update
            outcome: What happened
            success_score: How well it worked (0.0-1.0)

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE adaptations
            SET outcome = ?, success_score = ?
            WHERE id = ?
        """, (outcome, success_score, adaptation_id))

        success = cursor.rowcount > 0
        self.conn.commit()
        return success

    def get_adaptations(
        self,
        source_type_id: Optional[int] = None,
        target_type_id: Optional[int] = None,
        min_success: Optional[float] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get adaptations with optional filtering.

        Args:
            source_type_id: Filter by source problem type
            target_type_id: Filter by target problem type
            min_success: Minimum success score
            limit: Maximum results

        Returns:
            List of adaptation records
        """
        cursor = self.conn.cursor()

        query = """
            SELECT id, source_problem_type_id, target_problem_type_id,
                   source_context, target_context, original_strategy,
                   adapted_strategy, adaptation_reasoning, outcome,
                   success_score, source_episode_ids, created_at
            FROM adaptations WHERE 1=1
        """
        params = []

        if source_type_id is not None:
            query += " AND source_problem_type_id = ?"
            params.append(source_type_id)

        if target_type_id is not None:
            query += " AND target_problem_type_id = ?"
            params.append(target_type_id)

        if min_success is not None:
            query += " AND success_score >= ?"
            params.append(min_success)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        return [
            {
                'id': row[0],
                'source_problem_type_id': row[1],
                'target_problem_type_id': row[2],
                'source_context': row[3],
                'target_context': row[4],
                'original_strategy': row[5],
                'adapted_strategy': row[6],
                'adaptation_reasoning': row[7],
                'outcome': row[8],
                'success_score': row[9],
                'source_episode_ids': json.loads(row[10]) if row[10] else [],
                'created_at': row[11],
            }
            for row in cursor.fetchall()
        ]

    def get_successful_adaptations_for_type(
        self,
        problem_type_id: int,
        as_target: bool = True,
        min_success: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Get successful adaptations for a problem type.

        Args:
            problem_type_id: Problem type to search for
            as_target: If True, find adaptations TO this type; if False, FROM this type
            min_success: Minimum success score

        Returns:
            List of successful adaptation records
        """
        if as_target:
            return self.get_adaptations(
                target_type_id=problem_type_id,
                min_success=min_success,
            )
        else:
            return self.get_adaptations(
                source_type_id=problem_type_id,
                min_success=min_success,
            )

    # =========================================================================
    # Domain Keywords (Learnable Domain Markers)
    # =========================================================================

    def add_domain_keyword(
        self,
        domain_name: str,
        keyword: str,
        weight: float = 1.0,
        source: str = 'seed',
    ) -> bool:
        """
        Add or update a domain keyword.

        Args:
            domain_name: Domain this keyword indicates (e.g., 'python')
            keyword: The keyword (e.g., 'typeerror')
            weight: How strongly it indicates the domain (0.0-1.0)
            source: Origin of keyword ('seed', 'learned', 'llm')

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()

        # Use INSERT OR REPLACE to handle updates
        cursor.execute("""
            INSERT INTO domain_keywords (domain_name, keyword, weight, source, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(domain_name, keyword) DO UPDATE SET
                weight = excluded.weight,
                source = excluded.source,
                updated_at = datetime('now')
        """, (domain_name.lower(), keyword.lower(), weight, source))

        self.conn.commit()
        return True

    def add_domain_keywords_bulk(
        self,
        domain_name: str,
        keywords: List[str],
        weight: float = 1.0,
        source: str = 'seed',
    ) -> int:
        """
        Add multiple keywords for a domain at once.

        Args:
            domain_name: Domain name
            keywords: List of keywords
            weight: Weight for all keywords
            source: Source for all keywords

        Returns:
            Number of keywords added/updated
        """
        cursor = self.conn.cursor()
        count = 0

        for keyword in keywords:
            cursor.execute("""
                INSERT INTO domain_keywords (domain_name, keyword, weight, source, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(domain_name, keyword) DO UPDATE SET
                    weight = excluded.weight,
                    source = excluded.source,
                    updated_at = datetime('now')
            """, (domain_name.lower(), keyword.lower(), weight, source))
            count += 1

        self.conn.commit()
        return count

    def get_domain_keywords(
        self,
        domain_name: Optional[str] = None,
        min_weight: float = 0.0,
        source: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Get domain keywords, optionally filtered.

        Args:
            domain_name: Filter by specific domain (None = all domains)
            min_weight: Minimum weight threshold
            source: Filter by source ('seed', 'learned', 'llm')

        Returns:
            Dict mapping domain_name -> list of keywords
        """
        cursor = self.conn.cursor()

        query = """
            SELECT domain_name, keyword FROM domain_keywords
            WHERE weight >= ?
        """
        params = [min_weight]

        if domain_name:
            query += " AND domain_name = ?"
            params.append(domain_name.lower())

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY domain_name, weight DESC"

        cursor.execute(query, params)

        result = {}
        for row in cursor.fetchall():
            domain = row[0]
            keyword = row[1]
            if domain not in result:
                result[domain] = []
            result[domain].append(keyword)

        return result

    def get_domain_keywords_with_weights(
        self,
        domain_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get domain keywords with their weights.

        Returns:
            Dict mapping domain_name -> {keyword: weight}
        """
        cursor = self.conn.cursor()

        if domain_name:
            cursor.execute("""
                SELECT domain_name, keyword, weight FROM domain_keywords
                WHERE domain_name = ?
                ORDER BY weight DESC
            """, (domain_name.lower(),))
        else:
            cursor.execute("""
                SELECT domain_name, keyword, weight FROM domain_keywords
                ORDER BY domain_name, weight DESC
            """)

        result = {}
        for row in cursor.fetchall():
            domain = row[0]
            keyword = row[1]
            weight = row[2]
            if domain not in result:
                result[domain] = {}
            result[domain][keyword] = weight

        return result

    def increment_keyword_occurrence(
        self,
        domain_name: str,
        keyword: str,
    ) -> bool:
        """
        Increment the occurrence count for a keyword.

        Used when learning keywords from episodes.

        Args:
            domain_name: Domain name
            keyword: Keyword to increment

        Returns:
            True if keyword exists and was incremented
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE domain_keywords
            SET occurrence_count = occurrence_count + 1,
                updated_at = datetime('now')
            WHERE domain_name = ? AND keyword = ?
        """, (domain_name.lower(), keyword.lower()))

        success = cursor.rowcount > 0
        self.conn.commit()
        return success

    def get_all_domains(self) -> List[str]:
        """Get list of all known domains"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT domain_name FROM domain_keywords ORDER BY domain_name")
        return [row[0] for row in cursor.fetchall()]

    def delete_domain_keywords(
        self,
        domain_name: Optional[str] = None,
        source: Optional[str] = None,
        min_weight_to_keep: Optional[float] = None,
    ) -> int:
        """
        Delete domain keywords with optional filtering.

        Useful for cleanup or resetting learned keywords.

        Args:
            domain_name: Delete only for this domain (None = all)
            source: Delete only from this source
            min_weight_to_keep: Delete keywords with weight below this

        Returns:
            Number of keywords deleted
        """
        cursor = self.conn.cursor()

        conditions = []
        params = []

        if domain_name:
            conditions.append("domain_name = ?")
            params.append(domain_name.lower())

        if source:
            conditions.append("source = ?")
            params.append(source)

        if min_weight_to_keep is not None:
            conditions.append("weight < ?")
            params.append(min_weight_to_keep)

        if not conditions:
            # Safety: require at least one filter
            return 0

        query = f"DELETE FROM domain_keywords WHERE {' AND '.join(conditions)}"
        cursor.execute(query, params)

        count = cursor.rowcount
        self.conn.commit()
        return count

    def close(self):
        """Close database connections"""
        if hasattr(self, "conn"):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"MemoryStore(episodes={self.get_episode_count()}, db={self.db_path})"
