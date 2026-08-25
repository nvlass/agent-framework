"""
Reflection Engine for Agent Memory System

This module provides the Reflector class that generates insights
from agent experiences using an LLM. It supports:
- Failure analysis: Understanding why things went wrong
- Success analysis: Understanding why things worked
- Pattern discovery: Finding general principles from clusters of episodes
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .llm_interface import (
    LLMInterface,
    LLMResponse,
    FAILURE_REFLECTION_PROMPT,
    SUCCESS_REFLECTION_PROMPT,
    PATTERN_DISCOVERY_PROMPT,
    parse_reflection_response,
)
from .memory_store import Episode, Reflection, CausalFactor


class ReflectionConfig:
    """Configuration for the reflection engine"""

    def __init__(
        self,
        failure_threshold: float = 0.3,     # Reflect on episodes with score < this
        success_threshold: float = 0.9,     # Reflect on episodes with score > this
        max_tokens: int = 512,              # Max tokens for LLM generation
        temperature: float = 0.7,           # LLM temperature
        auto_reflect: bool = True,          # Automatically reflect on extreme outcomes
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.auto_reflect = auto_reflect


class Reflector:
    """
    Generates reflections on agent experiences.

    Uses an LLM to analyze episodes and extract insights,
    causal factors, and actionable takeaways.
    """

    def __init__(
        self,
        llm: LLMInterface,
        config: Optional[ReflectionConfig] = None,
    ):
        """
        Initialize the reflector.

        Args:
            llm: LLM interface for generating reflections
            config: Reflection configuration (uses defaults if None)
        """
        self.llm = llm
        self.config = config or ReflectionConfig()

    def reflect_on_failure(self, episode: Episode) -> Reflection:
        """
        Generate a reflection analyzing why an episode failed.

        Args:
            episode: The failed episode to analyze

        Returns:
            Reflection with insights about the failure
        """
        if not self.llm.is_available():
            raise RuntimeError("LLM is not available for reflection generation")

        prompt = FAILURE_REFLECTION_PROMPT.format(
            context=episode.context,
            action=episode.action,
            outcome=episode.outcome or "No outcome recorded",
            score=episode.success_score if episode.success_score is not None else "Unknown",
        )

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return self._parse_response_to_reflection(
            response=response,
            reflection_type="failure_analysis",
            trigger_episode_id=episode.id,
        )

    def reflect_on_success(self, episode: Episode) -> Reflection:
        """
        Generate a reflection analyzing why an episode succeeded.

        Args:
            episode: The successful episode to analyze

        Returns:
            Reflection with insights about the success
        """
        if not self.llm.is_available():
            raise RuntimeError("LLM is not available for reflection generation")

        prompt = SUCCESS_REFLECTION_PROMPT.format(
            context=episode.context,
            action=episode.action,
            outcome=episode.outcome or "No outcome recorded",
            score=episode.success_score if episode.success_score is not None else "Unknown",
        )

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return self._parse_response_to_reflection(
            response=response,
            reflection_type="success_analysis",
            trigger_episode_id=episode.id,
        )

    def discover_patterns(
        self,
        episodes: List[Episode],
        common_tags: Optional[List[str]] = None,
    ) -> Reflection:
        """
        Generate a reflection analyzing patterns across multiple episodes.

        Args:
            episodes: List of similar episodes to analyze
            common_tags: Tags shared by these episodes

        Returns:
            Reflection with pattern insights
        """
        if not self.llm.is_available():
            raise RuntimeError("LLM is not available for reflection generation")

        if not episodes:
            raise ValueError("Need at least one episode for pattern discovery")

        # Format episodes for the prompt
        episodes_text = "\n".join([
            f"- Context: {ep.context}\n  Action: {ep.action}\n  Score: {ep.success_score}"
            for ep in episodes[:10]  # Limit to 10 to fit context
        ])

        # Calculate average success rate
        scores = [ep.success_score for ep in episodes if ep.success_score is not None]
        avg_success = sum(scores) / len(scores) if scores else 0.0

        # Gather common tags
        if common_tags is None:
            common_tags = self._find_common_tags(episodes)

        prompt = PATTERN_DISCOVERY_PROMPT.format(
            episodes=episodes_text,
            tags=", ".join(common_tags) if common_tags else "None",
            success_rate=f"{avg_success:.0%}",
        )

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return self._parse_response_to_reflection(
            response=response,
            reflection_type="pattern_discovery",
            trigger_episode_id=episodes[0].id if episodes else None,
        )

    def should_reflect(self, episode: Episode) -> Optional[str]:
        """
        Determine if an episode should trigger automatic reflection.

        Args:
            episode: Episode to check

        Returns:
            'failure' or 'success' if reflection needed, None otherwise
        """
        if not self.config.auto_reflect:
            return None

        if episode.success_score is None:
            return None

        if episode.success_score < self.config.failure_threshold:
            return "failure"

        if episode.success_score > self.config.success_threshold:
            return "success"

        return None

    def auto_reflect(self, episode: Episode) -> Optional[Reflection]:
        """
        Automatically reflect on an episode if it meets the threshold.

        Args:
            episode: Episode to potentially reflect on

        Returns:
            Reflection if generated, None if not needed
        """
        reflection_type = self.should_reflect(episode)

        if reflection_type == "failure":
            return self.reflect_on_failure(episode)
        elif reflection_type == "success":
            return self.reflect_on_success(episode)

        return None

    def _parse_response_to_reflection(
        self,
        response: LLMResponse,
        reflection_type: str,
        trigger_episode_id: Optional[int],
    ) -> Reflection:
        """Parse LLM response into a Reflection object"""
        parsed = parse_reflection_response(response.text)

        causal_chain = [
            CausalFactor(
                factor=cf["factor"],
                contribution=cf["contribution"],
                confidence=cf["confidence"],
            )
            for cf in parsed.get("causal_factors", [])
        ]

        return Reflection(
            reflection_type=reflection_type,
            trigger_episode_id=trigger_episode_id,
            insight=parsed.get("insight", response.text[:200]),
            causal_chain=causal_chain,
            actionable_takeaway=parsed.get("actionable_takeaway"),
            created_at=datetime.now(),
        )

    def _find_common_tags(self, episodes: List[Episode]) -> List[str]:
        """Find tags that appear in multiple episodes"""
        if not episodes:
            return []

        # Count tag occurrences
        tag_counts: Dict[str, int] = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Return tags that appear in at least half the episodes
        threshold = len(episodes) / 2
        common = [tag for tag, count in tag_counts.items() if count >= threshold]

        return sorted(common, key=lambda t: tag_counts[t], reverse=True)
