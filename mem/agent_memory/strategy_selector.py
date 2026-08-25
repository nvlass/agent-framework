"""
Strategy Selector - Choose the best approach for new problems

This module orchestrates the adaptation system by:
1. Finding analogous past experiences
2. Adapting strategies from those experiences
3. Ranking and selecting the best approach

It's the high-level API for transfer learning in the agent memory system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from .memory_store import Episode, MemoryStore, cosine_similarity
from .embeddings import EmbeddingGenerator
from .analogy_finder import AnalogyFinder, AnalogousMatch
from .adapter import StrategyAdapter, AdaptationResult, ProblemType
from .llm_interface import LLMInterface
from .consolidation import LearnedPattern


@dataclass
class StrategyCandidate:
    """A candidate strategy for the current problem"""
    strategy: str                        # The recommended action
    source_type: str                     # 'direct', 'adapted', 'pattern'
    confidence: float                    # 0.0-1.0
    reasoning: str                       # Why this strategy might work
    source_episode: Optional[Episode] = None
    source_pattern: Optional[LearnedPattern] = None
    adaptation: Optional[AdaptationResult] = None
    analogy_match: Optional[AnalogousMatch] = None

    def __repr__(self) -> str:
        return (
            f"StrategyCandidate(type={self.source_type}, "
            f"confidence={self.confidence:.2f}, "
            f"strategy={self.strategy[:50]}...)"
        )


@dataclass
class StrategySelection:
    """Result of strategy selection process"""
    context: str
    problem_type: Optional[ProblemType] = None
    candidates: List[StrategyCandidate] = field(default_factory=list)
    selected: Optional[StrategyCandidate] = None
    selection_reasoning: str = ""

    def has_recommendation(self) -> bool:
        return self.selected is not None

    def get_recommendation(self) -> Optional[str]:
        return self.selected.strategy if self.selected else None


class StrategySelector:
    """
    Selects the best strategy for a given problem context.

    Combines multiple sources of knowledge:
    1. Direct matches: Past episodes with similar context
    2. Learned patterns: Consolidated knowledge from many episodes
    3. Adapted strategies: Strategies from different domains, adapted

    The selector ranks these candidates and picks the best approach.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        llm: Optional[LLMInterface] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the strategy selector.

        Args:
            memory_store: Access to episodic memory and patterns
            llm: LLM for strategy adaptation (optional, disables adaptation if None)
            embedding_generator: For semantic similarity (uses store's if not provided)
        """
        self.store = memory_store
        self.llm = llm
        self.embedding_generator = embedding_generator or memory_store.embedding_generator

        # Initialize sub-components
        self.analogy_finder = AnalogyFinder(
            embedding_generator=self.embedding_generator,
            similarity_threshold=0.4,
            domain_distance_weight=0.3,
        )

        self.adapter = StrategyAdapter(llm) if llm else None

    def select_strategy(
        self,
        context: str,
        goal: Optional[str] = None,
        include_adaptations: bool = True,
        max_candidates: int = 5,
        min_confidence: float = 0.3,
    ) -> StrategySelection:
        """
        Select the best strategy for a given context.

        Searches memory for relevant experiences and patterns,
        optionally adapts strategies from other domains, and
        ranks all candidates to select the best approach.

        Args:
            context: Current situation/problem description
            goal: Optional goal to achieve
            include_adaptations: Whether to include adapted strategies
            max_candidates: Maximum candidates to consider
            min_confidence: Minimum confidence threshold

        Returns:
            StrategySelection with ranked candidates and recommendation
        """
        full_context = f"{context} Goal: {goal}" if goal else context

        selection = StrategySelection(context=full_context)

        # Step 1: Identify problem type (if LLM available)
        if self.adapter:
            selection.problem_type = self.adapter.identify_problem_type(context)

        # Step 2: Gather candidates from different sources
        candidates = []

        # 2a: Direct matches from similar episodes
        direct_candidates = self._find_direct_matches(full_context, max_candidates)
        candidates.extend(direct_candidates)

        # 2b: Learned patterns
        pattern_candidates = self._find_pattern_matches(full_context, max_candidates)
        candidates.extend(pattern_candidates)

        # 2c: Adapted strategies from analogous situations
        if include_adaptations and self.adapter:
            adapted_candidates = self._find_adapted_strategies(
                full_context, max_candidates
            )
            candidates.extend(adapted_candidates)

        # Step 3: Filter by confidence
        candidates = [c for c in candidates if c.confidence >= min_confidence]

        # Step 4: Rank candidates
        candidates = self._rank_candidates(candidates)

        # Step 5: Select best
        selection.candidates = candidates[:max_candidates]
        if selection.candidates:
            selection.selected = selection.candidates[0]
            selection.selection_reasoning = self._explain_selection(selection)

        return selection

    def _find_direct_matches(
        self,
        context: str,
        limit: int,
    ) -> List[StrategyCandidate]:
        """Find strategies from directly similar episodes"""
        candidates = []

        try:
            results = self.store.retrieve_episodes(context, limit=limit, min_similarity=0.5)

            for episode, similarity in results:
                # Only include successful episodes
                if episode.success_score is not None and episode.success_score >= 0.6:
                    confidence = similarity * episode.success_score

                    candidates.append(StrategyCandidate(
                        strategy=episode.action,
                        source_type='direct',
                        confidence=confidence,
                        reasoning=f"Similar situation with {episode.success_score:.0%} success rate",
                        source_episode=episode,
                    ))
        except RuntimeError:
            # No embedding generator available
            pass

        return candidates

    def _find_pattern_matches(
        self,
        context: str,
        limit: int,
    ) -> List[StrategyCandidate]:
        """Find strategies from learned patterns"""
        candidates = []

        recommendations = self.store.recommend_actions(
            context=context,
            limit=limit,
            min_confidence=0.5,
            min_success_rate=0.6,
        )

        for rec in recommendations:
            pattern = rec['pattern']
            match_score = rec['match_score']

            # Confidence combines pattern quality and match quality
            confidence = (
                pattern.confidence * 0.4 +
                pattern.success_rate * 0.3 +
                match_score * 0.3
            )

            candidates.append(StrategyCandidate(
                strategy=pattern.recommended_action,
                source_type='pattern',
                confidence=confidence,
                reasoning=f"Learned pattern with {pattern.success_rate:.0%} success "
                          f"from {pattern.sample_count} episodes",
                source_pattern=pattern,
            ))

        return candidates

    def _find_adapted_strategies(
        self,
        context: str,
        limit: int,
    ) -> List[StrategyCandidate]:
        """Find strategies adapted from analogous situations"""
        if not self.adapter or not self.embedding_generator:
            return []

        candidates = []

        # Get recent episodes with embeddings for analogy search
        episodes = self.store._fetch_recent_with_embeddings(hours=168)  # 1 week

        # Filter to successful episodes only
        successful_episodes = [
            ep for ep in episodes
            if ep.success_score is not None and ep.success_score >= 0.7
            and ep.embedding is not None
        ]

        if not successful_episodes:
            return []

        # Find analogies (similar structure, different domain)
        analogies = self.analogy_finder.find_cross_domain_analogies(
            query_context=context,
            episodes=successful_episodes,
            limit=limit,
        )

        # Adapt each analogy
        for match in analogies:
            try:
                adaptation = self.adapter.adapt_strategy(
                    source_episode=match.episode,
                    target_context=context,
                )

                # Confidence based on analogy quality and adaptation confidence
                confidence = (
                    match.analogy_score * 0.5 +
                    adaptation.confidence_score() * 0.3 +
                    (match.episode.success_score or 0.5) * 0.2
                )

                candidates.append(StrategyCandidate(
                    strategy=adaptation.adapted_strategy,
                    source_type='adapted',
                    confidence=confidence,
                    reasoning=f"Adapted from {match.episode.context[:50]}... "
                              f"(analogy score: {match.analogy_score:.2f})",
                    source_episode=match.episode,
                    adaptation=adaptation,
                    analogy_match=match,
                ))
            except Exception:
                # Skip if adaptation fails
                continue

        return candidates

    def _rank_candidates(
        self,
        candidates: List[StrategyCandidate]
    ) -> List[StrategyCandidate]:
        """
        Rank candidates by quality.

        Ranking considers:
        - Confidence score
        - Source type (direct > pattern > adapted, all else equal)
        - Diversity (avoid too-similar strategies)
        """
        if not candidates:
            return []

        # Sort by confidence, with source type as tiebreaker
        source_priority = {'direct': 0.02, 'pattern': 0.01, 'adapted': 0.0}

        def score(c: StrategyCandidate) -> float:
            return c.confidence + source_priority.get(c.source_type, 0)

        candidates.sort(key=score, reverse=True)

        return candidates

    def _explain_selection(self, selection: StrategySelection) -> str:
        """Generate explanation for why the selected strategy was chosen"""
        if not selection.selected:
            return "No suitable strategy found."

        selected = selection.selected
        parts = [f"Selected '{selected.source_type}' strategy"]

        if selected.source_type == 'direct':
            parts.append(f"from similar past experience")
        elif selected.source_type == 'pattern':
            if selected.source_pattern:
                parts.append(f"based on learned pattern from {selected.source_pattern.sample_count} episodes")
        elif selected.source_type == 'adapted':
            if selected.analogy_match:
                parts.append(f"adapted from analogous situation "
                            f"(domain distance: {selected.analogy_match.domain_distance:.2f})")

        parts.append(f"with {selected.confidence:.0%} confidence")

        return ". ".join(parts) + "."

    def get_advice(
        self,
        context: str,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level API for getting strategy advice.

        Simpler interface that returns a dictionary suitable for
        presenting to users or agents.

        Args:
            context: Current situation
            goal: What you're trying to achieve

        Returns:
            Dictionary with recommendation and alternatives
        """
        selection = self.select_strategy(context, goal)

        if not selection.has_recommendation():
            return {
                'has_recommendation': False,
                'message': "No relevant past experience found for this situation.",
                'suggestion': "Try a new approach and record the outcome for future learning.",
            }

        result = {
            'has_recommendation': True,
            'recommended_strategy': selection.selected.strategy,
            'confidence': selection.selected.confidence,
            'reasoning': selection.selected.reasoning,
            'source_type': selection.selected.source_type,
            'selection_reasoning': selection.selection_reasoning,
        }

        # Add problem type if identified
        if selection.problem_type:
            result['problem_type'] = selection.problem_type.name
            result['problem_domain'] = selection.problem_type.domain

        # Add alternatives
        if len(selection.candidates) > 1:
            result['alternatives'] = [
                {
                    'strategy': c.strategy,
                    'confidence': c.confidence,
                    'source_type': c.source_type,
                }
                for c in selection.candidates[1:4]  # Top 3 alternatives
            ]

        return result


def quick_select(
    store: MemoryStore,
    context: str,
    llm: Optional[LLMInterface] = None,
) -> Optional[str]:
    """
    Quick convenience function to get a strategy recommendation.

    Args:
        store: Memory store with past experiences
        context: Current situation
        llm: Optional LLM for adaptations

    Returns:
        Recommended strategy string, or None if no recommendation
    """
    selector = StrategySelector(store, llm)
    selection = selector.select_strategy(context)
    return selection.get_recommendation()
