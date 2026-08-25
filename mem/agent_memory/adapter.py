"""
Strategy Adapter - Adapt strategies from one domain to another

This module uses LLM to intelligently adapt strategies that worked
in one context to work in a new, different context.

Example: A debugging strategy from Python ("add logging to trace execution")
might be adapted to Docker ("add verbose output to container logs").
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .llm_interface import LLMInterface, LLMResponse
from .memory_store import Episode


# Prompt templates for strategy adaptation

ADAPT_STRATEGY_PROMPT = """You are helping an AI agent adapt a successful strategy from one domain to another.

ORIGINAL SITUATION (where the strategy worked):
Context: {source_context}
Action taken: {source_action}
Outcome: {source_outcome}

NEW SITUATION (where we want to apply similar thinking):
Context: {target_context}

The strategy worked well in the original situation. How should we adapt it for the new situation?

Please provide:
1. ADAPTED_STRATEGY: The modified strategy for the new context (1-2 sentences)
2. KEY_SIMILARITIES: What makes these situations analogous (bullet points)
3. KEY_DIFFERENCES: Important differences to account for (bullet points)
4. CONFIDENCE: How confident are you this will work? (low/medium/high)
5. REASONING: Brief explanation of your adaptation logic

Format your response exactly like this:
ADAPTED_STRATEGY: [your adapted strategy here]

KEY_SIMILARITIES:
- [similarity 1]
- [similarity 2]

KEY_DIFFERENCES:
- [difference 1]
- [difference 2]

CONFIDENCE: [low/medium/high]

REASONING: [your explanation]
"""

IDENTIFY_PROBLEM_TYPE_PROMPT = """Analyze the following problem context and identify what type of problem it is.

CONTEXT: {context}

Consider these aspects:
- What domain is this? (e.g., programming, database, networking, testing)
- What category of problem? (e.g., debugging, performance, configuration, integration)
- What are the key characteristics that define this problem type?

Respond in this format:
PROBLEM_TYPE: [short name, e.g., "python_debugging" or "docker_networking"]
DOMAIN: [broader domain category]
DESCRIPTION: [one sentence description]
CHARACTERISTICS:
- [characteristic 1]
- [characteristic 2]
- [characteristic 3]
"""


@dataclass
class AdaptationResult:
    """Result of adapting a strategy to a new context"""
    source_episode: Episode
    target_context: str
    original_strategy: str
    adapted_strategy: str
    similarities: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    confidence: str = "medium"  # low, medium, high
    reasoning: str = ""
    adaptation_id: Optional[int] = None  # Set when stored to DB

    def confidence_score(self) -> float:
        """Convert confidence to numeric score"""
        return {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(self.confidence.lower(), 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_episode_id': self.source_episode.id if self.source_episode else None,
            'target_context': self.target_context,
            'original_strategy': self.original_strategy,
            'adapted_strategy': self.adapted_strategy,
            'similarities': self.similarities,
            'differences': self.differences,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
        }


@dataclass
class ProblemType:
    """Represents a category of problem"""
    id: Optional[int] = None
    name: str = ""
    domain: str = ""
    description: str = ""
    characteristics: List[str] = field(default_factory=list)
    successful_strategies: List[int] = field(default_factory=list)  # Pattern IDs
    similar_problem_types: List[int] = field(default_factory=list)  # Problem type IDs
    created_at: Optional[datetime] = None
    embedding_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'domain': self.domain,
            'description': self.description,
            'characteristics': self.characteristics,
            'successful_strategies': self.successful_strategies,
            'similar_problem_types': self.similar_problem_types,
        }


class StrategyAdapter:
    """
    Adapts strategies from one domain to another using LLM.

    The adapter takes a strategy that worked in context A and
    modifies it to work in context B, explaining the reasoning.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initialize the strategy adapter.

        Args:
            llm: LLM interface for generating adaptations
        """
        self.llm = llm

    def adapt_strategy(
        self,
        source_episode: Episode,
        target_context: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AdaptationResult:
        """
        Adapt a strategy from one context to another.

        Args:
            source_episode: Episode with successful strategy to adapt
            target_context: New context to apply strategy to
            max_tokens: Maximum response length
            temperature: LLM temperature (higher = more creative)

        Returns:
            AdaptationResult with adapted strategy and reasoning
        """
        # Build the prompt
        prompt = ADAPT_STRATEGY_PROMPT.format(
            source_context=source_episode.context,
            source_action=source_episode.action,
            source_outcome=source_episode.outcome or "Success",
            target_context=target_context,
        )

        # Generate adaptation
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Parse response
        return self._parse_adaptation_response(
            response_text=response.text,
            source_episode=source_episode,
            target_context=target_context,
        )

    def identify_problem_type(
        self,
        context: str,
        max_tokens: int = 256,
    ) -> ProblemType:
        """
        Identify the type of problem from context.

        Args:
            context: Problem description/situation
            max_tokens: Maximum response length

        Returns:
            ProblemType with classification
        """
        prompt = IDENTIFY_PROBLEM_TYPE_PROMPT.format(context=context)

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for classification
        )

        return self._parse_problem_type_response(response.text)

    def _parse_adaptation_response(
        self,
        response_text: str,
        source_episode: Episode,
        target_context: str,
    ) -> AdaptationResult:
        """Parse LLM response into AdaptationResult"""
        lines = response_text.strip().split('\n')

        adapted_strategy = ""
        similarities = []
        differences = []
        confidence = "medium"
        reasoning = ""

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith('ADAPTED_STRATEGY:'):
                adapted_strategy = line.replace('ADAPTED_STRATEGY:', '').strip()
                current_section = None
            elif line.startswith('KEY_SIMILARITIES:'):
                current_section = 'similarities'
            elif line.startswith('KEY_DIFFERENCES:'):
                current_section = 'differences'
            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip().lower()
                current_section = None
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
                current_section = 'reasoning'
            elif line.startswith('- '):
                # Bullet point in current section
                item = line[2:].strip()
                if current_section == 'similarities':
                    similarities.append(item)
                elif current_section == 'differences':
                    differences.append(item)
            elif current_section == 'reasoning':
                # Continue reasoning on next line
                reasoning += ' ' + line

        # Fallback if parsing failed
        if not adapted_strategy:
            # Try to extract something useful
            adapted_strategy = response_text[:200] if len(response_text) > 200 else response_text

        return AdaptationResult(
            source_episode=source_episode,
            target_context=target_context,
            original_strategy=source_episode.action,
            adapted_strategy=adapted_strategy,
            similarities=similarities,
            differences=differences,
            confidence=confidence if confidence in ['low', 'medium', 'high'] else 'medium',
            reasoning=reasoning.strip(),
        )

    def _parse_problem_type_response(self, response_text: str) -> ProblemType:
        """Parse LLM response into ProblemType"""
        lines = response_text.strip().split('\n')

        name = "unknown"
        domain = "general"
        description = ""
        characteristics = []

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('PROBLEM_TYPE:'):
                name = line.replace('PROBLEM_TYPE:', '').strip().lower().replace(' ', '_')
            elif line.startswith('DOMAIN:'):
                domain = line.replace('DOMAIN:', '').strip().lower()
            elif line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('CHARACTERISTICS:'):
                current_section = 'characteristics'
            elif line.startswith('- ') and current_section == 'characteristics':
                characteristics.append(line[2:].strip())

        return ProblemType(
            name=name,
            domain=domain,
            description=description,
            characteristics=characteristics,
        )

    def batch_adapt(
        self,
        source_episodes: List[Episode],
        target_context: str,
        max_adaptations: int = 3,
    ) -> List[AdaptationResult]:
        """
        Adapt multiple strategies to the same target context.

        Useful when you have several potentially relevant episodes
        and want to see how each might be adapted.

        Args:
            source_episodes: Episodes with strategies to adapt
            target_context: New context
            max_adaptations: Maximum number to generate

        Returns:
            List of adaptation results
        """
        results = []
        for episode in source_episodes[:max_adaptations]:
            result = self.adapt_strategy(episode, target_context)
            results.append(result)

        return results


def parse_adaptation_response(response_text: str) -> Dict[str, Any]:
    """
    Standalone parser for adaptation responses.

    Useful for testing or when you have raw LLM output.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed dictionary with adaptation components
    """
    lines = response_text.strip().split('\n')

    result = {
        'adapted_strategy': '',
        'similarities': [],
        'differences': [],
        'confidence': 'medium',
        'reasoning': '',
    }

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('ADAPTED_STRATEGY:'):
            result['adapted_strategy'] = line.replace('ADAPTED_STRATEGY:', '').strip()
            current_section = None
        elif line.startswith('KEY_SIMILARITIES:'):
            current_section = 'similarities'
        elif line.startswith('KEY_DIFFERENCES:'):
            current_section = 'differences'
        elif line.startswith('CONFIDENCE:'):
            result['confidence'] = line.replace('CONFIDENCE:', '').strip().lower()
            current_section = None
        elif line.startswith('REASONING:'):
            result['reasoning'] = line.replace('REASONING:', '').strip()
            current_section = 'reasoning'
        elif line.startswith('- '):
            item = line[2:].strip()
            if current_section in ('similarities', 'differences'):
                result[current_section].append(item)
        elif current_section == 'reasoning':
            result['reasoning'] += ' ' + line

    result['reasoning'] = result['reasoning'].strip()

    # Fallback: if no structured content found, use full text as strategy
    if not result['adapted_strategy'] and response_text.strip():
        # Use first 200 chars of response as fallback strategy
        result['adapted_strategy'] = response_text.strip()[:200]

    return result
