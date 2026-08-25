"""
Agent-Facing Memory Tools

This module provides a clean, standardized interface for AI agents
to interact with the memory system. It's designed to be:

1. Simple - Easy for agents to understand and use
2. Safe - Handles errors gracefully, returns structured responses
3. Complete - Covers all common memory operations

Usage:
    from agent_memory import MemoryTools

    tools = MemoryTools(store, llm=my_llm)

    # Store an experience
    result = tools.store_memory(
        context="Debugging Python TypeError",
        action="Added null check",
        outcome="Bug fixed",
        importance=8,
        tags=["python", "debugging"]
    )

    # Recall similar experiences
    memories = tools.recall_similar("How to fix TypeErrors?", limit=5)

    # Get strategy advice
    advice = tools.get_strategy_advice(
        context="Docker container won't start",
        goal="Get the service running"
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .memory_store import MemoryStore, Episode, Reflection
from .llm_interface import LLMInterface
from .reflector import Reflector, ReflectionConfig
from .strategy_selector import StrategySelector
from .config import MemoryConfig


@dataclass
class ToolResult:
    """
    Standardized result from memory tools.

    All tool methods return this structure for consistency.
    """
    success: bool
    data: Any = None
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'message': self.message,
            'error': self.error,
        }


@dataclass
class MemoryEntry:
    """Simplified memory representation for agents"""
    id: int
    context: str
    action: str
    outcome: str
    success_score: Optional[float]
    tags: List[str]
    timestamp: str
    similarity: Optional[float] = None  # When retrieved by similarity

    @classmethod
    def from_episode(cls, episode: Episode, similarity: Optional[float] = None) -> "MemoryEntry":
        return cls(
            id=episode.id,
            context=episode.context,
            action=episode.action,
            outcome=episode.outcome or "",
            success_score=episode.success_score,
            tags=episode.tags or [],
            timestamp=episode.timestamp.isoformat() if episode.timestamp else "",
            similarity=similarity,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'context': self.context,
            'action': self.action,
            'outcome': self.outcome,
            'success_score': self.success_score,
            'tags': self.tags,
            'timestamp': self.timestamp,
            'similarity': self.similarity,
        }


@dataclass
class StrategyAdvice:
    """Strategy recommendation for agents"""
    recommended_action: str
    confidence: float
    reasoning: str
    source_type: str  # 'direct', 'pattern', 'adapted'
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'recommended_action': self.recommended_action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'source_type': self.source_type,
            'alternatives': self.alternatives,
        }


class MemoryTools:
    """
    Agent-facing interface for memory operations.

    This class provides a clean, error-handled API that agents
    can use to store, recall, and learn from experiences.

    All methods return ToolResult for consistent handling.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: Optional[LLMInterface] = None,
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize memory tools.

        Args:
            store: Configured MemoryStore instance
            llm: Optional LLM for reflection and adaptation
            config: Optional configuration (uses defaults if not provided)
        """
        self.store = store
        self.llm = llm
        self.config = config or MemoryConfig()

        # Initialize sub-components
        if llm:
            self.reflector = Reflector(
                llm=llm,
                config=ReflectionConfig(
                    failure_threshold=self.config.reflection.failure_threshold,
                    success_threshold=self.config.reflection.success_threshold,
                    auto_reflect=self.config.reflection.enabled,
                )
            )
            self.strategy_selector = StrategySelector(
                memory_store=store,
                llm=llm,
            )
        else:
            self.reflector = None
            self.strategy_selector = StrategySelector(
                memory_store=store,
                llm=None,
            )

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    def store_memory(
        self,
        context: str,
        action: str,
        outcome: str = "",
        importance: int = 5,
        tags: Optional[List[str]] = None,
        auto_reflect: bool = True,
        dedup: bool = False,
        dedup_threshold: float = 0.9,
    ) -> ToolResult:
        """
        Store an important experience in memory.

        Use this after completing a task to remember what worked
        or didn't work for future reference.

        Args:
            context: What was the situation? (the problem or task)
            action: What did you do? (the approach taken)
            outcome: What happened? (the result)
            importance: How important is this? (1-10 scale)
                1-3: Minor, routine tasks
                4-6: Normal tasks worth remembering
                7-9: Important lessons or significant outcomes
                10: Critical experiences to never forget
            tags: Categories for this experience (e.g., ["python", "debugging"])
            auto_reflect: Whether to automatically reflect on significant outcomes
            dedup: When True, a near-duplicate existing episode is reinforced
                (occurrence count bumped) instead of storing a new copy.
            dedup_threshold: Cosine similarity above which an episode counts
                as a duplicate (only used when dedup=True).

        Returns:
            ToolResult with episode_id in data. When a duplicate was
            reinforced, data also has reinforced=True and occurrence_count.
        """
        try:
            # Convert importance to success_score (normalize to 0-1)
            # Higher importance = more likely to be remembered
            success_score = max(0.0, min(1.0, importance / 10.0))

            if dedup:
                match = self.store.find_similar_episode(
                    context, action, outcome, threshold=dedup_threshold
                )
                if match:
                    episode_id, similarity = match
                    count = self.store.reinforce_episode(episode_id)
                    return ToolResult(
                        success=True,
                        data={
                            'episode_id': episode_id,
                            'stored': False,
                            'reinforced': True,
                            'occurrence_count': count,
                            'similarity': round(similarity, 3),
                        },
                        message=(
                            f"Reinforced existing memory #{episode_id} "
                            f"(seen {count} times, similarity {similarity:.2f})"
                        ),
                    )

            # Store the episode
            episode_id = self.store.store_episode(
                context=context,
                action=action,
                outcome=outcome,
                success_score=success_score,
                tags=tags,
            )

            result_data = {
                'episode_id': episode_id,
                'stored': True,
            }

            # Auto-reflect on significant outcomes
            if auto_reflect and self.reflector and self.config.reflection.enabled:
                episode = self.store.get_episode_by_id(episode_id)
                if episode:
                    reflection_type = self.reflector.should_reflect(episode)
                    if reflection_type:
                        reflection = self.reflector.auto_reflect(episode)
                        if reflection:
                            reflection_id = self.store.store_reflection(reflection)
                            result_data['reflection_id'] = reflection_id
                            result_data['reflected'] = True

            return ToolResult(
                success=True,
                data=result_data,
                message=f"Memory stored (id={episode_id})",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to store memory",
            )

    def recall_similar(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> ToolResult:
        """
        Find similar past experiences.

        Use this when facing a new situation to see how you
        handled similar situations before.

        Args:
            query: Description of the current situation
            limit: Maximum memories to return (default: 5)
            min_similarity: Minimum relevance threshold (0.0-1.0)

        Returns:
            ToolResult with list of MemoryEntry in data
        """
        try:
            results = self.store.retrieve_episodes(
                query=query,
                limit=limit,
                min_similarity=min_similarity,
            )

            memories = [
                MemoryEntry.from_episode(episode, similarity).to_dict()
                for episode, similarity in results
            ]

            return ToolResult(
                success=True,
                data={'memories': memories, 'count': len(memories)},
                message=f"Found {len(memories)} similar memories",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to recall memories",
            )

    def recall_analogies(
        self,
        query: str,
        limit: int = 3,
        min_domain_distance: float = 0.3,
        pool_hours: int = 24 * 365,
    ) -> ToolResult:
        """
        Find past experiences from DIFFERENT domains that are structurally
        similar to the query — associative recall for transfer learning.

        Where recall_similar finds "the same kind of thing", this finds
        "a different kind of thing that works the same way": high semantic
        similarity, different surface domain (tags/keywords).

        Args:
            query: The current situation or problem
            limit: Maximum analogies to return
            min_domain_distance: 0.0 allows same-domain matches, 1.0 demands
                completely different domains (default 0.3)
            pool_hours: How far back to search (default: one year)

        Returns:
            ToolResult with list of analogies in data
        """
        try:
            if self.store.embedding_generator is None:
                return ToolResult(
                    success=False,
                    error="No embedding generator configured",
                    message="Analogy search requires embeddings",
                )
            from .analogy_finder import AnalogyFinder

            episodes = self.store._fetch_recent_with_embeddings(hours=pool_hours)
            episodes = [e for e in episodes if e.embedding is not None]
            if not episodes:
                return ToolResult(
                    success=True,
                    data={'analogies': [], 'count': 0},
                    message="No memories with embeddings yet",
                )
            finder = AnalogyFinder(
                embedding_generator=self.store.embedding_generator,
                memory_store=self.store,
            )
            matches = finder.find_analogies(
                query, episodes, limit=limit,
                min_domain_distance=min_domain_distance,
            )
            analogies = [
                {
                    'episode_id': m.episode.id,
                    'memory': (m.episode.outcome or m.episode.context or "")[:400],
                    'context': (m.episode.context or "")[:200],
                    'tags': m.episode.tags,
                    'similarity': round(m.similarity_score, 2),
                    'domain_distance': round(m.domain_distance, 2),
                    'shared_features': m.shared_features[:5],
                    'different_features': m.different_features[:5],
                }
                for m in matches
            ]
            return ToolResult(
                success=True,
                data={'analogies': analogies, 'count': len(analogies)},
                message=f"Found {len(analogies)} cross-domain analogies",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to find analogies",
            )

    def recall_recent(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> ToolResult:
        """
        Get recent memories from the specified time window.

        Useful for reviewing what you've done recently.

        Args:
            hours: Time window in hours (default: 24)
            limit: Maximum memories to return

        Returns:
            ToolResult with list of MemoryEntry in data
        """
        try:
            episodes = self.store.get_recent_episodes(hours=hours, limit=limit)

            memories = [
                MemoryEntry.from_episode(episode).to_dict()
                for episode in episodes
            ]

            return ToolResult(
                success=True,
                data={'memories': memories, 'count': len(memories)},
                message=f"Retrieved {len(memories)} recent memories",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to recall recent memories",
            )

    def recall_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 10,
    ) -> ToolResult:
        """
        Find memories with specific tags.

        Args:
            tags: Tags to search for
            match_all: If True, memory must have ALL tags; if False, ANY tag
            limit: Maximum memories to return

        Returns:
            ToolResult with list of MemoryEntry in data
        """
        try:
            episodes = self.store.get_episodes_by_tags(
                tags=tags,
                match_all=match_all,
                limit=limit,
            )

            memories = [
                MemoryEntry.from_episode(episode).to_dict()
                for episode in episodes
            ]

            return ToolResult(
                success=True,
                data={'memories': memories, 'count': len(memories)},
                message=f"Found {len(memories)} memories with tags {tags}",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to recall memories by tags",
            )

    # =========================================================================
    # Learning Operations
    # =========================================================================

    def learn_from_outcome(
        self,
        episode_id: int,
        success: bool,
        reasoning: str = "",
    ) -> ToolResult:
        """
        Explicitly mark an outcome for learning.

        Use this to reinforce what worked or learn from what didn't.

        Args:
            episode_id: Memory to learn from
            success: Whether the approach worked (True/False)
            reasoning: Why it succeeded or failed

        Returns:
            ToolResult with reflection_id if reflection was created
        """
        try:
            # Get the episode
            episode = self.store.get_episode_by_id(episode_id)
            if not episode:
                return ToolResult(
                    success=False,
                    error=f"Episode {episode_id} not found",
                    message="Memory not found",
                )

            # Update outcome category
            category = 'success' if success else 'failure'
            self.store.evaluate_outcome(
                episode_id=episode_id,
                category=category,
                failure_reason=reasoning if not success else None,
            )

            result_data = {
                'episode_id': episode_id,
                'category': category,
                'updated': True,
            }

            # Create reflection if LLM available
            if self.reflector:
                if success:
                    reflection = self.reflector.reflect_on_success(episode)
                else:
                    reflection = self.reflector.reflect_on_failure(episode)

                if reflection:
                    reflection_id = self.store.store_reflection(reflection)
                    result_data['reflection_id'] = reflection_id
                    result_data['insight'] = reflection.insight

            return ToolResult(
                success=True,
                data=result_data,
                message=f"Learned from {'success' if success else 'failure'}",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to learn from outcome",
            )

    def reflect_on_recent(
        self,
        hours: int = 24,
        focus: Optional[str] = None,
    ) -> ToolResult:
        """
        Analyze recent experiences for patterns.

        Use periodically to consolidate learning and discover patterns.

        Args:
            hours: Time window to analyze (default: 24 hours)
            focus: Optional focus area (e.g., "failures", "python")

        Returns:
            ToolResult with reflection insights
        """
        try:
            if not self.reflector:
                return ToolResult(
                    success=False,
                    error="LLM required for reflection",
                    message="Reflection not available (no LLM configured)",
                )

            # Get recent episodes
            episodes = self.store.get_recent_episodes(hours=hours, limit=50)

            if not episodes:
                return ToolResult(
                    success=True,
                    data={'patterns': [], 'count': 0},
                    message="No recent memories to analyze",
                )

            # Filter by focus if specified
            if focus:
                if focus.lower() == "failures":
                    episodes = [e for e in episodes if e.success_score and e.success_score < 0.5]
                elif focus.lower() == "successes":
                    episodes = [e for e in episodes if e.success_score and e.success_score > 0.7]
                else:
                    # Treat focus as a tag
                    episodes = [e for e in episodes if focus.lower() in [t.lower() for t in (e.tags or [])]]

            if not episodes:
                return ToolResult(
                    success=True,
                    data={'patterns': [], 'count': 0},
                    message=f"No matching memories for focus: {focus}",
                )

            # Discover patterns — novelty-gated: rediscovering the same
            # pattern reinforces the existing reflection instead of
            # appending a near-duplicate.
            reflection = self.reflector.discover_patterns(episodes)
            if reflection:
                reflection_id, novel = self.store.store_reflection_if_novel(reflection)

                return ToolResult(
                    success=True,
                    data={
                        'reflection_id': reflection_id,
                        'novel': novel,
                        'insight': reflection.insight,
                        'actionable_takeaway': reflection.actionable_takeaway,
                        'episodes_analyzed': len(episodes),
                    },
                    message=(
                        "Pattern analysis complete" if novel
                        else f"Pattern already known — reinforced reflection #{reflection_id}"
                    ),
                )
            else:
                return ToolResult(
                    success=True,
                    data={'patterns': [], 'count': 0},
                    message="No clear patterns found",
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to reflect on recent memories",
            )

    # =========================================================================
    # Strategy and Advice
    # =========================================================================

    def get_strategy_advice(
        self,
        context: str,
        goal: Optional[str] = None,
    ) -> ToolResult:
        """
        Get recommended approach for the current situation.

        Uses past experiences, learned patterns, and cross-domain
        analogies to suggest how to handle the situation.

        Args:
            context: Current situation/problem description
            goal: What you're trying to achieve (optional)

        Returns:
            ToolResult with StrategyAdvice in data
        """
        try:
            advice = self.strategy_selector.get_advice(context=context, goal=goal)

            if advice.get('has_recommendation'):
                strategy_advice = StrategyAdvice(
                    recommended_action=advice['recommended_strategy'],
                    confidence=advice['confidence'],
                    reasoning=advice['reasoning'],
                    source_type=advice['source_type'],
                    alternatives=advice.get('alternatives', []),
                )

                return ToolResult(
                    success=True,
                    data=strategy_advice.to_dict(),
                    message=advice.get('selection_reasoning', 'Strategy found'),
                )
            else:
                return ToolResult(
                    success=True,
                    data=None,
                    message=advice.get('message', 'No matching strategy found'),
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to get strategy advice",
            )

    def get_success_rate(
        self,
        tags: List[str],
    ) -> ToolResult:
        """
        Get success rate for a category of tasks.

        Useful for understanding how well you've done with
        certain types of problems.

        Args:
            tags: Tags to analyze (e.g., ["python", "debugging"])

        Returns:
            ToolResult with success statistics
        """
        try:
            stats = self.store.get_success_rate_for_tags(tags, match_all=True)

            return ToolResult(
                success=True,
                data=stats,
                message=f"Success rate: {stats['success_rate']:.0%} ({stats['total_episodes']} episodes)",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to calculate success rate",
            )

    # =========================================================================
    # Memory Management
    # =========================================================================

    def get_memory_stats(self) -> ToolResult:
        """
        Get statistics about your memory.

        Returns counts, averages, and health metrics.
        """
        try:
            stats = self.store.get_stats()

            # Add reflection count
            stats['total_reflections'] = self.store.count_reflections()

            # Add pattern count
            patterns = self.store.get_learned_patterns()
            stats['total_patterns'] = len(patterns)

            return ToolResult(
                success=True,
                data=stats,
                message=f"Memory: {stats['total_episodes']} episodes, {stats['total_patterns']} patterns",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to get memory stats",
            )

    def consolidate_memories(self) -> ToolResult:
        """
        Run memory consolidation.

        Clusters similar experiences and extracts patterns.
        Run periodically to improve memory organization.
        """
        try:
            report = self.store.run_consolidation()

            return ToolResult(
                success=True,
                data={
                    'episodes_processed': report.episodes_processed,
                    'clusters_found': report.clusters_found,
                    'patterns_created': report.patterns_created,
                    'duration_seconds': report.duration_seconds,
                },
                message=f"Consolidated: {report.clusters_found} clusters, {report.patterns_created} patterns",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to consolidate memories",
            )

    # =========================================================================
    # Tool Descriptions (for agent tool registration)
    # =========================================================================

    @classmethod
    def get_tool_definitions(cls) -> List[Dict[str, Any]]:
        """
        Get tool definitions for agent registration.

        Returns a list of tool definitions in a standard format
        that can be used to register these tools with an agent.
        """
        return [
            {
                "name": "store_memory",
                "description": "Store an important experience for future reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "What was the situation or problem?"
                        },
                        "action": {
                            "type": "string",
                            "description": "What approach did you take?"
                        },
                        "outcome": {
                            "type": "string",
                            "description": "What was the result?"
                        },
                        "importance": {
                            "type": "integer",
                            "description": "How important is this? (1-10)",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories for this experience"
                        }
                    },
                    "required": ["context", "action"]
                }
            },
            {
                "name": "recall_similar",
                "description": "Find similar past experiences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Description of current situation"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_strategy_advice",
                "description": "Get recommended approach for current situation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Current situation or problem"
                        },
                        "goal": {
                            "type": "string",
                            "description": "What you're trying to achieve"
                        }
                    },
                    "required": ["context"]
                }
            },
            {
                "name": "learn_from_outcome",
                "description": "Mark an experience as success or failure for learning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "episode_id": {
                            "type": "integer",
                            "description": "ID of the memory to learn from"
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether the approach worked"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why it succeeded or failed"
                        }
                    },
                    "required": ["episode_id", "success"]
                }
            },
            {
                "name": "reflect_on_recent",
                "description": "Analyze recent experiences for patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "Time window in hours",
                            "default": 24
                        },
                        "focus": {
                            "type": "string",
                            "description": "Optional focus: 'failures', 'successes', or a tag"
                        }
                    }
                }
            },
            {
                "name": "get_memory_stats",
                "description": "Get statistics about your memory",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
