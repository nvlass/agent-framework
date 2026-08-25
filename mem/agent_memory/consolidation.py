"""
Memory Consolidation Pipeline

Consolidates episodic memories into learned patterns:
1. Cluster similar episodes (HDBSCAN)
2. Extract patterns from clusters (heuristic analysis)
3. Selective archival (keep high-value, forget redundant)

This implements Phase 3 consolidation from the roadmap.
"""

from typing import List, Dict, Optional, Set, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter
import numpy as np

# Clustering
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

if TYPE_CHECKING:
    from .memory_store import Episode

# Common English stop words to filter out
STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
    'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'and', 'but', 'or', 'if', 'for', 'with', 'about',
    'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'of', 'in', 'by'
}


@dataclass
class LearnedPattern:
    """
    Represents a learned pattern extracted from similar episodes
    """
    pattern_id: Optional[int] = None
    context_signature: str = ""
    recommended_action: str = ""
    success_rate: float = 0.0
    sample_count: int = 0
    pattern_description: str = ""
    confidence: float = 0.0
    source_episode_ids: List[int] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.source_episode_ids is None:
            self.source_episode_ids = []
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'pattern_id': self.pattern_id,
            'context_signature': self.context_signature,
            'recommended_action': self.recommended_action,
            'success_rate': self.success_rate,
            'sample_count': self.sample_count,
            'pattern_description': self.pattern_description,
            'confidence': self.confidence,
            'source_episode_ids': self.source_episode_ids,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class ConsolidationReport:
    """
    Report of consolidation results
    """
    patterns_created: int = 0
    episodes_processed: int = 0
    episodes_archived: int = 0
    noise_episodes: int = 0
    clusters_found: int = 0
    patterns: List[LearnedPattern] = None
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []


class EpisodeClusterer:
    """
    Cluster episodes by embedding similarity using HDBSCAN

    HDBSCAN advantages:
    - No need to specify number of clusters
    - Handles noise/outliers (unique episodes)
    - Finds arbitrary-shaped clusters
    - Robust to varying density
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 1,
        metric: str = 'euclidean'
    ):
        """
        Initialize clusterer

        Args:
            min_cluster_size: Minimum episodes for a pattern (default: 3)
            min_samples: Conservative factor (1 = less conservative)
            metric: Distance metric ('euclidean' or 'cosine')
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def cluster(self, episodes: List["Episode"]) -> Tuple[List[List["Episode"]], List["Episode"]]:
        """
        Cluster episodes by embedding similarity

        Args:
            episodes: Episodes to cluster (must have embeddings)

        Returns:
            Tuple of (clusters, noise_episodes)
            - clusters: List of episode clusters (each cluster is a list of episodes)
            - noise_episodes: Episodes that don't fit any pattern
        """
        # Filter episodes with embeddings
        valid_episodes = [ep for ep in episodes if ep.embedding is not None]

        if len(valid_episodes) < self.min_cluster_size:
            return [], valid_episodes  # Not enough for clustering

        # Extract embeddings as numpy array
        embeddings = np.array([ep.embedding for ep in valid_episodes])

        # Normalize embeddings for cosine similarity
        if self.metric == 'cosine':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=0.0
        )

        labels = clusterer.fit_predict(embeddings)

        # Group episodes by cluster label
        clusters_dict: Dict[int, List["Episode"]] = {}
        noise_episodes = []

        for episode, label in zip(valid_episodes, labels):
            if label == -1:
                # Noise - unique episode
                noise_episodes.append(episode)
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(episode)

        # Convert to list of clusters
        clusters = list(clusters_dict.values())

        return clusters, noise_episodes


class PatternExtractor:
    """
    Extract learned patterns from episode clusters using heuristic analysis

    Uses statistical analysis (no LLM):
    - Word frequency for context signatures
    - Success rate analysis for action recommendations
    - Confidence based on sample size and consistency
    """

    def extract_context_signature(self, cluster: List["Episode"]) -> str:
        """
        Create context signature from cluster

        Uses word frequency to identify common themes

        Args:
            cluster: Episodes in the cluster

        Returns:
            Context signature string
        """
        # Collect all context texts
        contexts = [ep.context for ep in cluster]

        # Extract keywords via word frequency
        word_counts = Counter()
        for context in contexts:
            # Simple tokenization: lowercase, filter length and stop words
            words = [
                w.lower().strip('.,!?;:()[]{}')
                for w in context.split()
                if len(w) > 3 and w.lower() not in STOP_WORDS
            ]
            word_counts.update(words)

        # Top 5 most common words
        keywords = [word for word, _ in word_counts.most_common(5)]

        # Most common tags across cluster
        tag_counts = Counter()
        for ep in cluster:
            tag_counts.update(ep.tags)
        common_tags = [tag for tag, _ in tag_counts.most_common(3)]

        # Build signature
        if keywords and common_tags:
            signature = f"{', '.join(keywords)} [{', '.join(common_tags)}]"
        elif keywords:
            signature = ', '.join(keywords)
        elif common_tags:
            signature = f"[{', '.join(common_tags)}]"
        else:
            signature = f"Pattern with {len(cluster)} episodes"

        return signature

    def analyze_actions(self, cluster: List["Episode"]) -> List[Dict[str, Any]]:
        """
        Analyze action success rates in cluster

        Args:
            cluster: Episodes in the cluster

        Returns:
            List of action analysis dicts, sorted by success rate
        """
        action_data: Dict[str, List[float]] = {}

        # Collect scores per action
        for episode in cluster:
            action = episode.action
            if action not in action_data:
                action_data[action] = []

            if episode.success_score is not None:
                action_data[action].append(episode.success_score)

        # Calculate statistics per action
        results = []
        for action, scores in action_data.items():
            if scores:  # Only if we have data
                avg_score = sum(scores) / len(scores)
                # Success = score > 0.7
                success_count = sum(1 for s in scores if s > 0.7)
                success_rate = success_count / len(scores)

                results.append({
                    'action': action,
                    'success_rate': success_rate,
                    'sample_count': len(scores),
                    'avg_score': avg_score
                })

        # Sort by success rate (descending)
        results.sort(key=lambda x: x['success_rate'], reverse=True)
        return results

    def calculate_confidence(
        self,
        cluster: List["Episode"],
        actions: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence in the pattern

        Based on:
        - Sample size (more episodes = more confident)
        - Consistency (similar outcomes = more confident)
        - Best action success rate

        Args:
            cluster: Episodes in the cluster
            actions: Analyzed actions

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Factor 1: Sample size (diminishing returns after 20)
        size_factor = min(len(cluster) / 20.0, 1.0)

        # Factor 2: Consistency (variance of success scores)
        scored_episodes = [ep for ep in cluster if ep.success_score is not None]
        if scored_episodes and len(scored_episodes) > 1:
            scores = [ep.success_score for ep in scored_episodes]
            std_dev = np.std(scores)
            # Lower std dev = higher consistency
            # Normalize: std of 0.5 is considered low confidence
            consistency = max(0.0, 1.0 - (std_dev / 0.5))
        else:
            consistency = 0.5

        # Factor 3: Best action success rate
        if actions:
            best_action_rate = actions[0]['success_rate']
        else:
            best_action_rate = 0.5

        # Weighted combination
        confidence = (
            size_factor * 0.3 +
            consistency * 0.3 +
            best_action_rate * 0.4
        )

        return min(confidence, 1.0)

    def build_pattern_description(
        self,
        signature: str,
        actions: List[Dict[str, Any]]
    ) -> str:
        """
        Build human-readable pattern description

        Args:
            signature: Context signature
            actions: Analyzed actions

        Returns:
            Formatted pattern description
        """
        desc = f"Pattern: {signature}\n\n"

        if actions:
            desc += "Recommended actions (by success rate):\n"
            for i, action_data in enumerate(actions[:5], 1):  # Top 5
                action = action_data['action']
                rate = action_data['success_rate']
                count = action_data['sample_count']
                avg = action_data['avg_score']

                desc += f"{i}. {action}\n"
                desc += f"   Success: {rate:.0%} | Avg Score: {avg:.2f} | Samples: {count}\n"
        else:
            desc += "No action data available.\n"

        return desc

    def extract_pattern(self, cluster: List["Episode"]) -> LearnedPattern:
        """
        Extract complete pattern from cluster

        Args:
            cluster: Episodes in the cluster

        Returns:
            LearnedPattern object
        """
        # Context signature
        signature = self.extract_context_signature(cluster)

        # Action analysis
        actions = self.analyze_actions(cluster)

        # Best action
        if actions:
            recommended_action = actions[0]['action']
            success_rate = actions[0]['success_rate']
        else:
            recommended_action = "No clear action pattern"
            success_rate = 0.0

        # Confidence
        confidence = self.calculate_confidence(cluster, actions)

        # Description
        description = self.build_pattern_description(signature, actions)

        # Source episodes
        source_ids = [ep.id for ep in cluster if ep.id is not None]

        return LearnedPattern(
            context_signature=signature,
            recommended_action=recommended_action,
            success_rate=success_rate,
            sample_count=len(cluster),
            pattern_description=description,
            confidence=confidence,
            source_episode_ids=source_ids
        )


class ConsolidationEngine:
    """
    Orchestrates the consolidation pipeline

    Pipeline:
    1. Cluster similar episodes (HDBSCAN)
    2. Extract patterns from clusters
    3. (Future) Selective archival of redundant episodes
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 1,
        metric: str = 'euclidean'
    ):
        """
        Initialize consolidation engine

        Args:
            min_cluster_size: Minimum episodes for a pattern
            min_samples: HDBSCAN conservative factor
            metric: Distance metric for clustering
        """
        self.clusterer = EpisodeClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
        self.extractor = PatternExtractor()

    def run_consolidation(
        self,
        episodes: List["Episode"]
    ) -> ConsolidationReport:
        """
        Run full consolidation pipeline

        Args:
            episodes: Episodes to consolidate

        Returns:
            ConsolidationReport with results
        """
        start_time = datetime.now()

        # Step 1: Cluster episodes
        clusters, noise_episodes = self.clusterer.cluster(episodes)

        # Step 2: Extract patterns from clusters
        patterns = []
        for cluster in clusters:
            pattern = self.extractor.extract_pattern(cluster)
            patterns.append(pattern)

        # Build report
        duration = (datetime.now() - start_time).total_seconds()

        report = ConsolidationReport(
            patterns_created=len(patterns),
            episodes_processed=len(episodes),
            episodes_archived=0,  # Not implemented yet
            noise_episodes=len(noise_episodes),
            clusters_found=len(clusters),
            patterns=patterns,
            duration_seconds=duration
        )

        return report

    def should_consolidate(
        self,
        episodes_since_last: int = 0,
        hours_since_last: float = 0.0,
        episode_threshold: int = 100,
        time_threshold_hours: float = 24.0
    ) -> bool:
        """
        Determine if consolidation should run

        Triggers:
        - Count-based: After N episodes
        - Time-based: After X hours

        Args:
            episodes_since_last: Episodes stored since last consolidation
            hours_since_last: Hours since last consolidation
            episode_threshold: Episode count trigger
            time_threshold_hours: Time trigger

        Returns:
            True if consolidation should run
        """
        return (
            episodes_since_last >= episode_threshold or
            hours_since_last >= time_threshold_hours
        )
