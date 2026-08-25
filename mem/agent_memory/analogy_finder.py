"""
Analogy Finder - Find similar-but-different situations across domains

This module enables transfer learning by finding episodes that are
semantically similar (same underlying structure) but from different
problem domains.

Key insight: "Python debugging" and "hardware troubleshooting" are different
domains but share similar strategies (isolate, test hypothesis, iterate).

Domain markers can be:
1. Loaded from database (learnable, preferred)
2. Fallback to hardcoded defaults if DB is empty
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict, TYPE_CHECKING
import numpy as np

from .memory_store import Episode, cosine_similarity
from .embeddings import EmbeddingGenerator

if TYPE_CHECKING:
    from .memory_store import MemoryStore


@dataclass
class AnalogousMatch:
    """Represents a match between current context and a past episode"""
    episode: Episode
    similarity_score: float      # Embedding similarity (0.0-1.0)
    domain_distance: float       # How different the domains are (0.0-1.0, higher = more different)
    analogy_score: float         # Combined score: similar content + different domain = good analogy
    shared_features: List[str]   # What makes them similar
    different_features: List[str]  # What makes them different (domain markers)

    def __repr__(self) -> str:
        return (
            f"AnalogousMatch(episode_id={self.episode.id}, "
            f"similarity={self.similarity_score:.2f}, "
            f"domain_dist={self.domain_distance:.2f}, "
            f"analogy={self.analogy_score:.2f})"
        )


class AnalogyFinder:
    """
    Find analogous situations from different domains.

    The key insight is that good analogies are:
    1. Semantically similar (similar underlying structure/problem)
    2. From different surface domains (different tags, keywords)

    This enables transfer learning: "How I solved X in domain A
    might help me solve Y in domain B."

    Domain markers can be loaded from database (learnable) or use
    hardcoded defaults as fallback.
    """

    # Default domain markers (used as fallback if DB is empty)
    DEFAULT_DOMAIN_MARKERS = {
        'python': {'python', 'typeerror', 'valueerror', 'exception', 'traceback', 'pip', 'import'},
        'javascript': {'javascript', 'js', 'node', 'npm', 'typescript', 'react', 'async', 'promise'},
        'git': {'git', 'commit', 'branch', 'merge', 'rebase', 'push', 'pull', 'clone'},
        'docker': {'docker', 'container', 'image', 'dockerfile', 'compose', 'kubernetes', 'k8s'},
        'database': {'sql', 'database', 'query', 'table', 'index', 'sqlite', 'postgres', 'mysql'},
        'api': {'api', 'rest', 'http', 'endpoint', 'request', 'response', 'cors', 'json'},
        'testing': {'test', 'unittest', 'pytest', 'mock', 'fixture', 'assertion', 'coverage'},
        'networking': {'network', 'port', 'socket', 'tcp', 'udp', 'dns', 'http', 'connection'},
        'performance': {'slow', 'performance', 'optimize', 'memory', 'cpu', 'cache', 'latency'},
        'debugging': {'debug', 'error', 'bug', 'fix', 'issue', 'crash', 'fail'},
    }

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        similarity_threshold: float = 0.5,
        domain_distance_weight: float = 0.3,
        memory_store: Optional["MemoryStore"] = None,
        domain_markers: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Initialize analogy finder.

        Args:
            embedding_generator: For computing semantic similarity
            similarity_threshold: Minimum embedding similarity to consider (0.0-1.0)
            domain_distance_weight: How much to reward different domains (0.0-1.0)
                Higher = prefer analogies from more different domains
            memory_store: Optional store to load domain markers from DB
            domain_markers: Optional explicit domain markers (overrides DB and defaults)
        """
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.domain_distance_weight = domain_distance_weight
        self.memory_store = memory_store
        self._custom_markers = domain_markers
        self._cached_markers: Optional[Dict[str, Set[str]]] = None

    @property
    def domain_markers(self) -> Dict[str, Set[str]]:
        """
        Get domain markers, loading from DB if available.

        Priority:
        1. Custom markers passed to __init__
        2. Markers from database (if memory_store provided)
        3. Default hardcoded markers
        """
        # Return custom markers if provided
        if self._custom_markers is not None:
            return self._custom_markers

        # Return cached markers if available
        if self._cached_markers is not None:
            return self._cached_markers

        # Try to load from database
        if self.memory_store is not None:
            db_markers = self._load_markers_from_db()
            if db_markers:
                self._cached_markers = db_markers
                return self._cached_markers

        # Fallback to defaults
        return self.DEFAULT_DOMAIN_MARKERS

    def _load_markers_from_db(self) -> Dict[str, Set[str]]:
        """Load domain markers from database"""
        try:
            keywords_dict = self.memory_store.get_domain_keywords(min_weight=0.3)
            # Convert lists to sets
            return {
                domain: set(keywords)
                for domain, keywords in keywords_dict.items()
            }
        except Exception:
            return {}

    def refresh_markers(self) -> None:
        """
        Refresh domain markers from database.

        Call this after learning new keywords to update the cache.
        """
        self._cached_markers = None

    def get_marker_stats(self) -> Dict[str, int]:
        """Get statistics about loaded domain markers"""
        markers = self.domain_markers
        return {
            'num_domains': len(markers),
            'total_keywords': sum(len(kw) for kw in markers.values()),
            'domains': list(markers.keys()),
        }

    def find_analogies(
        self,
        query_context: str,
        episodes: List[Episode],
        limit: int = 5,
        min_domain_distance: float = 0.0,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[AnalogousMatch]:
        """
        Find episodes that are analogous to the query context.

        An analogy is a past experience that:
        1. Has similar underlying structure (high embedding similarity)
        2. Is from a different surface domain (different keywords/tags)

        Args:
            query_context: Current situation to find analogies for
            episodes: Pool of episodes to search
            limit: Maximum analogies to return
            min_domain_distance: Minimum domain difference (0=same domain OK, 1=must be different)
            query_embedding: Pre-computed embedding (optional, saves computation)

        Returns:
            List of AnalogousMatch sorted by analogy_score (best first)
        """
        if not episodes:
            return []

        # Compute query embedding if not provided
        if query_embedding is None:
            if self.embedding_generator is None:
                raise RuntimeError("EmbeddingGenerator required for analogy finding")
            query_embedding = self.embedding_generator.generate_embedding(query_context)

        # Extract domain markers from query
        query_domains = self._extract_domains(query_context)

        matches = []
        for episode in episodes:
            # Skip episodes without embeddings
            if episode.embedding is None:
                continue

            # Compute semantic similarity
            similarity = cosine_similarity(query_embedding, episode.embedding)

            # Skip if below threshold
            if similarity < self.similarity_threshold:
                continue

            # Compute domain distance
            episode_text = f"{episode.context} {episode.action} {episode.outcome}"
            episode_domains = self._extract_domains(episode_text)
            episode_domains.update(set(episode.tags) if episode.tags else set())

            domain_distance = self._compute_domain_distance(query_domains, episode_domains)

            # Skip if domains are too similar (if min_domain_distance is set)
            if domain_distance < min_domain_distance:
                continue

            # Compute analogy score
            # Good analogy = high similarity + high domain distance
            analogy_score = self._compute_analogy_score(similarity, domain_distance)

            # Extract shared and different features
            shared = self._extract_shared_features(query_context, episode)
            different = self._extract_different_features(query_domains, episode_domains)

            matches.append(AnalogousMatch(
                episode=episode,
                similarity_score=similarity,
                domain_distance=domain_distance,
                analogy_score=analogy_score,
                shared_features=shared,
                different_features=different,
            ))

        # Sort by analogy score (best analogies first)
        matches.sort(key=lambda m: m.analogy_score, reverse=True)

        return matches[:limit]

    def find_cross_domain_analogies(
        self,
        query_context: str,
        episodes: List[Episode],
        limit: int = 5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[AnalogousMatch]:
        """
        Find analogies specifically from different domains.

        This is useful when you want to apply strategies from
        completely different problem areas.

        Args:
            query_context: Current situation
            episodes: Pool of episodes
            limit: Maximum results
            query_embedding: Pre-computed embedding (optional)

        Returns:
            Analogies from different domains only
        """
        return self.find_analogies(
            query_context=query_context,
            episodes=episodes,
            limit=limit,
            min_domain_distance=0.3,  # Must be at least 30% different
            query_embedding=query_embedding,
        )

    def find_same_domain_similar(
        self,
        query_context: str,
        episodes: List[Episode],
        limit: int = 5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[AnalogousMatch]:
        """
        Find similar episodes from the same domain.

        This is traditional semantic search - finding past experiences
        that are directly relevant to the current situation.

        Args:
            query_context: Current situation
            episodes: Pool of episodes
            limit: Maximum results
            query_embedding: Pre-computed embedding (optional)

        Returns:
            Similar episodes from the same domain
        """
        # Use lower domain distance weight for same-domain search
        original_weight = self.domain_distance_weight
        self.domain_distance_weight = 0.0  # Don't reward domain distance

        try:
            results = self.find_analogies(
                query_context=query_context,
                episodes=episodes,
                limit=limit,
                min_domain_distance=0.0,  # Same domain is fine
                query_embedding=query_embedding,
            )
            # Re-sort by similarity only
            results.sort(key=lambda m: m.similarity_score, reverse=True)
            return results
        finally:
            self.domain_distance_weight = original_weight

    def _extract_domains(self, text: str) -> Set[str]:
        """
        Extract domain markers from text.

        Returns set of domain names that appear in the text.
        Uses learnable domain markers from DB if available.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        detected_domains = set()
        for domain, markers in self.domain_markers.items():
            # Check if any marker words appear in the text
            if words & markers:  # Set intersection
                detected_domains.add(domain)

        return detected_domains

    def _compute_domain_distance(
        self,
        domains_a: Set[str],
        domains_b: Set[str]
    ) -> float:
        """
        Compute how different two sets of domains are.

        Returns 0.0 if identical, 1.0 if completely different.
        """
        if not domains_a and not domains_b:
            return 0.0  # Both empty = same domain

        if not domains_a or not domains_b:
            return 1.0  # One empty = maximum difference

        # Jaccard distance = 1 - (intersection / union)
        intersection = len(domains_a & domains_b)
        union = len(domains_a | domains_b)

        jaccard_similarity = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard_similarity

    def _compute_analogy_score(
        self,
        similarity: float,
        domain_distance: float
    ) -> float:
        """
        Compute overall analogy score.

        Good analogies have:
        - High semantic similarity (same underlying problem structure)
        - High domain distance (different surface domain)

        The formula balances these two factors.
        """
        # Base score is semantic similarity
        base_score = similarity

        # Bonus for domain distance (transfer learning potential)
        domain_bonus = domain_distance * self.domain_distance_weight

        # Combined score (capped at 1.0)
        return min(1.0, base_score + domain_bonus)

    def _extract_shared_features(
        self,
        query_context: str,
        episode: Episode
    ) -> List[str]:
        """
        Extract features shared between query and episode.

        These are the structural similarities that make them analogous.
        """
        shared = []

        query_lower = query_context.lower()
        episode_text = f"{episode.context} {episode.action}".lower()

        # Check for shared problem patterns
        patterns = [
            ('error handling', ['error', 'exception', 'fail', 'crash']),
            ('debugging', ['debug', 'fix', 'bug', 'issue']),
            ('performance', ['slow', 'optimize', 'fast', 'performance']),
            ('configuration', ['config', 'setting', 'parameter', 'option']),
            ('connection', ['connect', 'connection', 'network', 'timeout']),
            ('validation', ['valid', 'check', 'verify', 'assert']),
            ('data processing', ['parse', 'process', 'transform', 'convert']),
        ]

        for pattern_name, keywords in patterns:
            query_has = any(kw in query_lower for kw in keywords)
            episode_has = any(kw in episode_text for kw in keywords)
            if query_has and episode_has:
                shared.append(pattern_name)

        return shared

    def _extract_different_features(
        self,
        query_domains: Set[str],
        episode_domains: Set[str]
    ) -> List[str]:
        """
        Extract features that differ between query and episode.

        These are the surface-level differences despite structural similarity.
        """
        # Domains unique to each
        query_only = query_domains - episode_domains
        episode_only = episode_domains - query_domains

        different = []
        if query_only:
            different.append(f"query: {', '.join(sorted(query_only))}")
        if episode_only:
            different.append(f"episode: {', '.join(sorted(episode_only))}")

        return different


def find_structural_analogies(
    query: str,
    episodes: List[Episode],
    embedding_generator: EmbeddingGenerator,
    limit: int = 5,
) -> List[AnalogousMatch]:
    """
    Convenience function to find structural analogies.

    Args:
        query: Current situation/problem
        episodes: Past episodes to search
        embedding_generator: For computing similarities
        limit: Maximum results

    Returns:
        Best analogies (similar structure, different domain)
    """
    finder = AnalogyFinder(
        embedding_generator=embedding_generator,
        similarity_threshold=0.4,
        domain_distance_weight=0.3,
    )

    return finder.find_cross_domain_analogies(
        query_context=query,
        episodes=episodes,
        limit=limit,
    )
