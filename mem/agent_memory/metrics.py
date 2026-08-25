"""
Memory System Metrics and Health Monitoring

Provides metrics for:
- Memory usage and growth
- Learning effectiveness
- System performance
- Health alerts

Usage:
    from agent_memory import MemoryMetrics

    metrics = MemoryMetrics(store)

    # Get comprehensive health report
    health = metrics.get_health_report()

    # Get specific metrics
    learning = metrics.get_learning_metrics()
    performance = metrics.get_performance_metrics()
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import time

from .memory_store import MemoryStore
from .config import MemoryConfig


@dataclass
class HealthStatus:
    """Overall health status"""
    healthy: bool
    score: float  # 0.0 to 1.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'healthy': self.healthy,
            'score': self.score,
            'warnings': self.warnings,
            'errors': self.errors,
        }


@dataclass
class UsageMetrics:
    """Memory storage usage metrics"""
    total_episodes: int = 0
    total_patterns: int = 0
    total_reflections: int = 0
    total_adaptations: int = 0
    total_problem_types: int = 0
    total_domain_keywords: int = 0
    vector_store_size: int = 0
    episodes_last_24h: int = 0
    episodes_last_7d: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_episodes': self.total_episodes,
            'total_patterns': self.total_patterns,
            'total_reflections': self.total_reflections,
            'total_adaptations': self.total_adaptations,
            'total_problem_types': self.total_problem_types,
            'total_domain_keywords': self.total_domain_keywords,
            'vector_store_size': self.vector_store_size,
            'episodes_last_24h': self.episodes_last_24h,
            'episodes_last_7d': self.episodes_last_7d,
        }


@dataclass
class LearningMetrics:
    """Metrics about learning effectiveness"""
    avg_success_score: Optional[float] = None
    success_rate: float = 0.0
    failure_rate: float = 0.0
    patterns_per_100_episodes: float = 0.0
    reflections_per_100_episodes: float = 0.0
    success_trend: str = "stable"  # "improving", "declining", "stable"
    top_failure_tags: List[Tuple[str, int]] = field(default_factory=list)
    top_success_tags: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'avg_success_score': self.avg_success_score,
            'success_rate': self.success_rate,
            'failure_rate': self.failure_rate,
            'patterns_per_100_episodes': self.patterns_per_100_episodes,
            'reflections_per_100_episodes': self.reflections_per_100_episodes,
            'success_trend': self.success_trend,
            'top_failure_tags': self.top_failure_tags,
            'top_success_tags': self.top_success_tags,
        }


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    retrieval_latency_ms: Optional[float] = None
    consolidation_last_run: Optional[str] = None
    episodes_since_consolidation: int = 0
    working_memory_size: int = 0
    short_term_cache_hits: int = 0
    short_term_cache_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'retrieval_latency_ms': self.retrieval_latency_ms,
            'consolidation_last_run': self.consolidation_last_run,
            'episodes_since_consolidation': self.episodes_since_consolidation,
            'working_memory_size': self.working_memory_size,
            'short_term_cache_hits': self.short_term_cache_hits,
            'short_term_cache_misses': self.short_term_cache_misses,
        }


class MemoryMetrics:
    """
    Collects and reports metrics about the memory system.

    Use this to monitor system health, learning progress,
    and identify potential issues.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize metrics collector.

        Args:
            store: MemoryStore to monitor
            config: Optional configuration for thresholds
        """
        self.store = store
        self.config = config or MemoryConfig()

    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.

        Returns:
            Dictionary with health status and all metrics
        """
        usage = self.get_usage_metrics()
        learning = self.get_learning_metrics()
        performance = self.get_performance_metrics()
        health = self._assess_health(usage, learning, performance)

        return {
            'timestamp': datetime.now().isoformat(),
            'health': health.to_dict(),
            'usage': usage.to_dict(),
            'learning': learning.to_dict(),
            'performance': performance.to_dict(),
        }

    def get_usage_metrics(self) -> UsageMetrics:
        """Get memory usage statistics"""
        stats = self.store.get_stats()
        metrics = UsageMetrics()

        metrics.total_episodes = stats.get('total_episodes', 0)
        metrics.vector_store_size = stats.get('vector_store_size', 0)

        # Count patterns
        patterns = self.store.get_learned_patterns()
        metrics.total_patterns = len(patterns)

        # Count reflections
        metrics.total_reflections = self.store.count_reflections()

        # Count adaptations and problem types
        try:
            adaptations = self.store.get_adaptations(limit=10000)
            metrics.total_adaptations = len(adaptations)
            problem_types = self.store.get_all_problem_types()
            metrics.total_problem_types = len(problem_types)
        except Exception:
            pass  # Tables might not exist

        # Count domain keywords
        try:
            keywords = self.store.get_domain_keywords()
            metrics.total_domain_keywords = sum(len(kw) for kw in keywords.values())
        except Exception:
            pass

        # Recent activity
        episodes_24h = self.store.get_recent_episodes(hours=24, limit=10000)
        episodes_7d = self.store.get_recent_episodes(hours=168, limit=10000)
        metrics.episodes_last_24h = len(episodes_24h)
        metrics.episodes_last_7d = len(episodes_7d)

        return metrics

    def get_learning_metrics(self) -> LearningMetrics:
        """Get metrics about learning effectiveness"""
        metrics = LearningMetrics()
        stats = self.store.get_stats()

        # Average success score
        metrics.avg_success_score = stats.get('average_success_score')

        # Success/failure rates
        total = stats.get('total_episodes', 0)
        success_count = stats.get('success_count', 0)
        failure_count = stats.get('failure_count', 0)

        if total > 0:
            metrics.success_rate = success_count / total
            metrics.failure_rate = failure_count / total

        # Patterns and reflections per 100 episodes
        patterns = self.store.get_learned_patterns()
        reflections = self.store.count_reflections()

        if total > 0:
            metrics.patterns_per_100_episodes = (len(patterns) / total) * 100
            metrics.reflections_per_100_episodes = (reflections / total) * 100

        # Success trend (compare last 7 days to previous 7 days)
        metrics.success_trend = self._calculate_success_trend()

        # Top tags for failures and successes
        metrics.top_failure_tags = self._get_top_tags_for_category('failure', limit=5)
        metrics.top_success_tags = self._get_top_tags_for_category('success', limit=5)

        return metrics

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get system performance metrics"""
        metrics = PerformanceMetrics()

        # Measure retrieval latency
        if self.store.embedding_generator:
            metrics.retrieval_latency_ms = self._measure_retrieval_latency()

        # Consolidation status
        last_consolidation = self.store.get_metadata('last_consolidation')
        episodes_since = self.store.get_metadata('episodes_since_consolidation')

        metrics.consolidation_last_run = last_consolidation
        metrics.episodes_since_consolidation = int(episodes_since) if episodes_since else 0

        # Working memory
        metrics.working_memory_size = len(self.store.working_memory.get_all())

        # Short-term cache
        cache_stats = self.store.get_short_term_cache_stats()
        metrics.short_term_cache_hits = cache_stats.get('hits', 0)
        metrics.short_term_cache_misses = cache_stats.get('misses', 0)

        return metrics

    def _assess_health(
        self,
        usage: UsageMetrics,
        learning: LearningMetrics,
        performance: PerformanceMetrics,
    ) -> HealthStatus:
        """Assess overall system health"""
        warnings = []
        errors = []
        score = 1.0

        # Check episode limits
        max_episodes = self.config.performance.max_episodes_in_memory
        if usage.total_episodes > max_episodes * 0.9:
            warnings.append(f"Approaching episode limit ({usage.total_episodes}/{max_episodes})")
            score -= 0.1
        if usage.total_episodes > max_episodes:
            errors.append(f"Exceeded episode limit ({usage.total_episodes}/{max_episodes})")
            score -= 0.2

        # Check consolidation status
        threshold = self.config.consolidation.trigger_after_episodes
        if performance.episodes_since_consolidation > threshold * 2:
            warnings.append(f"Consolidation overdue ({performance.episodes_since_consolidation} episodes)")
            score -= 0.1

        # Check learning effectiveness
        if learning.avg_success_score is not None and learning.avg_success_score < 0.5:
            warnings.append(f"Low average success score ({learning.avg_success_score:.2f})")
            score -= 0.1

        if learning.success_trend == "declining":
            warnings.append("Success rate is declining")
            score -= 0.1

        # Check retrieval performance
        if performance.retrieval_latency_ms:
            max_latency = self.config.performance.retrieval_timeout_seconds * 1000
            if performance.retrieval_latency_ms > max_latency * 0.8:
                warnings.append(f"Retrieval latency high ({performance.retrieval_latency_ms:.0f}ms)")
                score -= 0.1

        # Check for stagnation
        if usage.total_episodes > 100 and usage.episodes_last_7d == 0:
            warnings.append("No new episodes in the last 7 days")
            score -= 0.05

        # Determine health status
        healthy = len(errors) == 0 and score >= 0.7

        return HealthStatus(
            healthy=healthy,
            score=max(0.0, min(1.0, score)),
            warnings=warnings,
            errors=errors,
        )

    def _calculate_success_trend(self) -> str:
        """Calculate whether success rate is improving, declining, or stable"""
        try:
            # Get episodes from last 7 days
            recent = self.store.get_recent_episodes(hours=168, limit=1000)
            # Get episodes from 7-14 days ago
            older = [
                ep for ep in self.store.get_recent_episodes(hours=336, limit=2000)
                if ep.timestamp and ep.timestamp < datetime.now() - timedelta(days=7)
            ]

            if len(recent) < 10 or len(older) < 10:
                return "stable"  # Not enough data

            # Calculate success rates
            recent_successes = sum(1 for ep in recent if ep.success_score and ep.success_score >= 0.7)
            older_successes = sum(1 for ep in older if ep.success_score and ep.success_score >= 0.7)

            recent_rate = recent_successes / len(recent)
            older_rate = older_successes / len(older)

            diff = recent_rate - older_rate

            if diff > 0.1:
                return "improving"
            elif diff < -0.1:
                return "declining"
            else:
                return "stable"

        except Exception:
            return "stable"

    def _get_top_tags_for_category(
        self,
        category: str,
        limit: int = 5
    ) -> List[Tuple[str, int]]:
        """Get most common tags for a success/failure category"""
        try:
            episodes = self.store.get_episodes_by_category(category, limit=500)

            tag_counts: Counter = Counter()
            for ep in episodes:
                for tag in (ep.tags or []):
                    tag_counts[tag] += 1

            return tag_counts.most_common(limit)

        except Exception:
            return []

    def _measure_retrieval_latency(self) -> Optional[float]:
        """Measure retrieval latency with a test query"""
        try:
            start = time.perf_counter()
            self.store.retrieve_episodes("test query for latency measurement", limit=5)
            end = time.perf_counter()

            return (end - start) * 1000  # Convert to milliseconds

        except Exception:
            return None

    def get_summary(self) -> str:
        """
        Get a human-readable summary of system status.

        Returns:
            Formatted string with key metrics
        """
        report = self.get_health_report()

        lines = [
            "=" * 50,
            "MEMORY SYSTEM HEALTH REPORT",
            "=" * 50,
            "",
            f"Status: {'HEALTHY' if report['health']['healthy'] else 'NEEDS ATTENTION'}",
            f"Health Score: {report['health']['score']:.0%}",
            "",
            "USAGE:",
            f"  Episodes: {report['usage']['total_episodes']}",
            f"  Patterns: {report['usage']['total_patterns']}",
            f"  Reflections: {report['usage']['total_reflections']}",
            f"  Last 24h: {report['usage']['episodes_last_24h']} episodes",
            "",
            "LEARNING:",
            f"  Avg Success: {report['learning']['avg_success_score']:.2f}" if report['learning']['avg_success_score'] else "  Avg Success: N/A",
            f"  Success Rate: {report['learning']['success_rate']:.0%}",
            f"  Trend: {report['learning']['success_trend']}",
            "",
            "PERFORMANCE:",
            f"  Retrieval Latency: {report['performance']['retrieval_latency_ms']:.0f}ms" if report['performance']['retrieval_latency_ms'] else "  Retrieval Latency: N/A",
            f"  Episodes Since Consolidation: {report['performance']['episodes_since_consolidation']}",
        ]

        if report['health']['warnings']:
            lines.extend(["", "WARNINGS:"])
            for warning in report['health']['warnings']:
                lines.append(f"  - {warning}")

        if report['health']['errors']:
            lines.extend(["", "ERRORS:"])
            for error in report['health']['errors']:
                lines.append(f"  - {error}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)
