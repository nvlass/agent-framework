"""
Domain Learner - Learn domain keywords from episodes

This module enables the memory system to learn which keywords
indicate which problem domains, based on actual usage patterns.

Learning happens in several ways:
1. Statistical: Extract keywords that frequently co-occur with tags
2. LLM-based: Ask LLM to identify domain-specific terminology
3. Hybrid: Combine both approaches for best results
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime

from .memory_store import MemoryStore, Episode
from .llm_interface import LLMInterface


# Prompt for LLM-based keyword extraction
EXTRACT_KEYWORDS_PROMPT = """Analyze the following text from a software development context and extract keywords that indicate what domain/technology this relates to.

TEXT:
{text}

KNOWN DOMAINS: {domains}

Extract keywords that would help identify which domain this text belongs to.
Focus on:
- Technology names (languages, frameworks, tools)
- Error types or patterns
- Domain-specific terminology
- Action verbs common to this domain

Return ONLY a comma-separated list of lowercase keywords (3-10 keywords).
Example output: python, typeerror, exception, debugging, traceback
"""

VALIDATE_KEYWORDS_PROMPT = """Review these candidate keywords for the domain "{domain}":

KEYWORDS: {keywords}

CONTEXT: These keywords were extracted from software development episodes tagged with "{domain}".

Rate each keyword:
- KEEP: Strongly indicates this domain
- WEAK: Somewhat indicates this domain
- REMOVE: Too generic or incorrect

Respond in this format:
KEEP: keyword1, keyword2, keyword3
WEAK: keyword4, keyword5
REMOVE: keyword6, keyword7
"""

EXPAND_DOMAIN_PROMPT = """For the software development domain "{domain}", suggest additional keywords that would indicate this domain.

EXISTING KEYWORDS: {existing}

Suggest 5-10 NEW keywords (not in the existing list) that are:
- Specific to this domain
- Commonly used in error messages, logs, or documentation
- Technical terms that professionals in this area would recognize

Return ONLY a comma-separated list of lowercase keywords.
"""


# Common stop words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'this', 'that', 'these', 'those', 'what',
    'which', 'who', 'whom', 'i', 'me', 'my', 'myself', 'we', 'our', 'it',
    'its', 'they', 'them', 'their', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'hers', 'try', 'tried', 'using', 'used', 'use', 'get',
    'got', 'make', 'made', 'work', 'worked', 'works', 'working', 'code',
    'file', 'files', 'error', 'issue', 'problem', 'fix', 'fixed', 'run',
    'running', 'ran', 'add', 'added', 'adding', 'change', 'changed',
}


@dataclass
class LearningReport:
    """Report from a keyword learning run"""
    episodes_processed: int = 0
    keywords_discovered: int = 0
    keywords_updated: int = 0
    domains_affected: List[str] = field(default_factory=list)
    new_keywords: Dict[str, List[str]] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class KeywordCandidate:
    """A candidate keyword with its statistics"""
    keyword: str
    domain: str
    occurrence_count: int
    domain_frequency: float  # How often it appears in this domain vs others
    weight: float = 0.5


class DomainLearner:
    """
    Learns domain keywords from episode tags and content.

    The learner extracts keywords from episodes that have tags,
    then determines which keywords are good indicators of each domain.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: Optional[LLMInterface] = None,
        min_keyword_length: int = 3,
        min_occurrences: int = 2,
    ):
        """
        Initialize the domain learner.

        Args:
            store: Memory store to learn from and update
            llm: Optional LLM for enhanced keyword extraction
            min_keyword_length: Minimum keyword length to consider
            min_occurrences: Minimum times a keyword must appear
        """
        self.store = store
        self.llm = llm
        self.min_keyword_length = min_keyword_length
        self.min_occurrences = min_occurrences

    def learn_from_episodes(
        self,
        episodes: Optional[List[Episode]] = None,
        limit: int = 500,
    ) -> LearningReport:
        """
        Learn domain keywords from tagged episodes.

        Extracts keywords from episode text and associates them
        with the episode's tags, building a statistical model
        of which keywords indicate which domains.

        Args:
            episodes: Episodes to learn from (None = fetch from store)
            limit: Maximum episodes to process

        Returns:
            LearningReport with results
        """
        start_time = datetime.now()
        report = LearningReport()

        # Get episodes with tags
        if episodes is None:
            episodes = self.store.get_all_episodes(limit=limit)

        # Filter to episodes with tags
        tagged_episodes = [ep for ep in episodes if ep.tags]
        report.episodes_processed = len(tagged_episodes)

        if not tagged_episodes:
            return report

        # Count keyword occurrences per domain (tag)
        domain_keywords: Dict[str, Counter] = defaultdict(Counter)
        total_keywords: Counter = Counter()

        for episode in tagged_episodes:
            # Extract keywords from episode text
            text = f"{episode.context} {episode.action} {episode.outcome or ''}"
            keywords = self._extract_keywords(text)

            # Associate with each tag
            for tag in episode.tags:
                domain = tag.lower()
                for keyword in keywords:
                    domain_keywords[domain][keyword] += 1
                    total_keywords[keyword] += 1

        # Calculate discriminative keywords for each domain
        candidates = self._calculate_discriminative_keywords(
            domain_keywords, total_keywords
        )

        # Store learned keywords
        for candidate in candidates:
            self.store.add_domain_keyword(
                domain_name=candidate.domain,
                keyword=candidate.keyword,
                weight=candidate.weight,
                source='learned',
            )
            report.keywords_discovered += 1

            if candidate.domain not in report.new_keywords:
                report.new_keywords[candidate.domain] = []
            report.new_keywords[candidate.domain].append(candidate.keyword)

        report.domains_affected = list(report.new_keywords.keys())
        report.duration_seconds = (datetime.now() - start_time).total_seconds()

        return report

    def expand_domain_with_llm(
        self,
        domain_name: str,
        max_new_keywords: int = 10,
    ) -> List[str]:
        """
        Use LLM to suggest new keywords for a domain.

        Args:
            domain_name: Domain to expand
            max_new_keywords: Maximum keywords to add

        Returns:
            List of new keywords added
        """
        if not self.llm:
            return []

        # Get existing keywords
        existing = self.store.get_domain_keywords(domain_name=domain_name)
        existing_keywords = existing.get(domain_name, [])

        # Ask LLM for suggestions
        prompt = EXPAND_DOMAIN_PROMPT.format(
            domain=domain_name,
            existing=", ".join(existing_keywords) if existing_keywords else "(none)",
        )

        response = self.llm.generate(prompt, max_tokens=256, temperature=0.7)

        # Parse response
        new_keywords = self._parse_keyword_list(response.text)

        # Filter out existing
        new_keywords = [k for k in new_keywords if k not in existing_keywords]

        # Add to store
        added = []
        for keyword in new_keywords[:max_new_keywords]:
            self.store.add_domain_keyword(
                domain_name=domain_name,
                keyword=keyword,
                weight=0.7,  # LLM-suggested keywords get moderate weight
                source='llm',
            )
            added.append(keyword)

        return added

    def validate_keywords_with_llm(
        self,
        domain_name: str,
    ) -> Dict[str, List[str]]:
        """
        Use LLM to validate and filter keywords for a domain.

        Args:
            domain_name: Domain to validate

        Returns:
            Dict with 'keep', 'weak', 'remove' lists
        """
        if not self.llm:
            return {'keep': [], 'weak': [], 'remove': []}

        # Get current keywords
        keywords_dict = self.store.get_domain_keywords(domain_name=domain_name)
        keywords = keywords_dict.get(domain_name, [])

        if not keywords:
            return {'keep': [], 'weak': [], 'remove': []}

        # Ask LLM to validate
        prompt = VALIDATE_KEYWORDS_PROMPT.format(
            domain=domain_name,
            keywords=", ".join(keywords),
        )

        response = self.llm.generate(prompt, max_tokens=256, temperature=0.3)

        # Parse response
        result = self._parse_validation_response(response.text)

        # Update weights based on validation
        for keyword in result.get('keep', []):
            self.store.add_domain_keyword(
                domain_name=domain_name,
                keyword=keyword,
                weight=0.9,
                source='llm',
            )

        for keyword in result.get('weak', []):
            self.store.add_domain_keyword(
                domain_name=domain_name,
                keyword=keyword,
                weight=0.5,
                source='llm',
            )

        for keyword in result.get('remove', []):
            # Don't delete, just set very low weight
            self.store.add_domain_keyword(
                domain_name=domain_name,
                keyword=keyword,
                weight=0.1,
                source='llm',
            )

        return result

    def extract_keywords_with_llm(
        self,
        text: str,
        known_domains: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Use LLM to extract domain keywords from text.

        Args:
            text: Text to extract keywords from
            known_domains: List of known domains for context

        Returns:
            List of extracted keywords
        """
        if not self.llm:
            return self._extract_keywords(text)

        if known_domains is None:
            known_domains = self.store.get_all_domains()

        prompt = EXTRACT_KEYWORDS_PROMPT.format(
            text=text[:1000],  # Limit text length
            domains=", ".join(known_domains) if known_domains else "(discovering new domains)",
        )

        response = self.llm.generate(prompt, max_tokens=128, temperature=0.3)

        return self._parse_keyword_list(response.text)

    def seed_default_domains(self) -> int:
        """
        Seed the database with default domain keywords.

        Uses the same keywords that were previously hardcoded
        in AnalogyFinder.

        Returns:
            Number of keywords seeded
        """
        default_domains = {
            'python': ['python', 'typeerror', 'valueerror', 'exception', 'traceback', 'pip', 'import', 'def', 'class'],
            'javascript': ['javascript', 'js', 'node', 'npm', 'typescript', 'react', 'async', 'promise', 'undefined'],
            'git': ['git', 'commit', 'branch', 'merge', 'rebase', 'push', 'pull', 'clone', 'checkout', 'stash'],
            'docker': ['docker', 'container', 'image', 'dockerfile', 'compose', 'kubernetes', 'k8s', 'port', 'volume'],
            'database': ['sql', 'database', 'query', 'table', 'index', 'sqlite', 'postgres', 'mysql', 'select', 'insert'],
            'api': ['api', 'rest', 'http', 'endpoint', 'request', 'response', 'cors', 'json', 'get', 'post'],
            'testing': ['test', 'unittest', 'pytest', 'mock', 'fixture', 'assertion', 'coverage', 'spec', 'expect'],
            'networking': ['network', 'port', 'socket', 'tcp', 'udp', 'dns', 'connection', 'timeout', 'ping'],
            'performance': ['slow', 'performance', 'optimize', 'memory', 'cpu', 'cache', 'latency', 'bottleneck', 'profile'],
            'debugging': ['debug', 'error', 'bug', 'fix', 'issue', 'crash', 'fail', 'breakpoint', 'log'],
        }

        total = 0
        for domain, keywords in default_domains.items():
            count = self.store.add_domain_keywords_bulk(
                domain_name=domain,
                keywords=keywords,
                weight=0.8,  # Seed keywords get high but not max weight
                source='seed',
            )
            total += count

        return total

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using simple heuristics.

        Args:
            text: Text to extract from

        Returns:
            List of lowercase keywords
        """
        # Normalize and tokenize
        text = text.lower()
        # Keep alphanumeric and some special chars
        words = re.findall(r'[a-z][a-z0-9_-]*[a-z0-9]|[a-z]', text)

        # Filter
        keywords = []
        for word in words:
            if len(word) >= self.min_keyword_length:
                if word not in STOP_WORDS:
                    keywords.append(word)

        return keywords

    def _calculate_discriminative_keywords(
        self,
        domain_keywords: Dict[str, Counter],
        total_keywords: Counter,
    ) -> List[KeywordCandidate]:
        """
        Calculate which keywords are discriminative for each domain.

        Uses a TF-IDF-like approach: keywords that appear frequently
        in one domain but rarely in others are good indicators.

        Args:
            domain_keywords: Counter per domain
            total_keywords: Total counts across all domains

        Returns:
            List of keyword candidates with weights
        """
        candidates = []
        num_domains = len(domain_keywords)

        for domain, keyword_counts in domain_keywords.items():
            for keyword, count in keyword_counts.items():
                # Skip rare keywords
                if count < self.min_occurrences:
                    continue

                # Calculate domain frequency (TF-IDF-like)
                total_count = total_keywords[keyword]
                domain_ratio = count / total_count if total_count > 0 else 0

                # Bonus if keyword appears mostly in this domain
                # (discriminative power)
                if domain_ratio > 0.5:  # Majority in this domain
                    weight = 0.5 + (domain_ratio * 0.4)  # 0.5-0.9 range
                else:
                    weight = domain_ratio * 0.5  # 0.0-0.25 range

                # Only keep reasonably discriminative keywords
                if weight >= 0.3:
                    candidates.append(KeywordCandidate(
                        keyword=keyword,
                        domain=domain,
                        occurrence_count=count,
                        domain_frequency=domain_ratio,
                        weight=min(1.0, weight),
                    ))

        return candidates

    def _parse_keyword_list(self, text: str) -> List[str]:
        """Parse a comma-separated keyword list from LLM response"""
        # Clean up the text
        text = text.strip().lower()
        # Remove any explanation text (keep only the list)
        if ':' in text:
            text = text.split(':')[-1]

        # Split by comma and clean
        keywords = []
        for part in text.split(','):
            keyword = part.strip().strip('.-•*')
            # Remove any remaining special chars
            keyword = re.sub(r'[^a-z0-9_-]', '', keyword)
            if keyword and len(keyword) >= self.min_keyword_length:
                keywords.append(keyword)

        return keywords

    def _parse_validation_response(self, text: str) -> Dict[str, List[str]]:
        """Parse LLM validation response into keep/weak/remove lists"""
        result = {'keep': [], 'weak': [], 'remove': []}

        lines = text.strip().split('\n')
        current_category = None

        for line in lines:
            line = line.strip().upper()

            if line.startswith('KEEP'):
                current_category = 'keep'
                # Extract keywords from same line
                if ':' in line:
                    keywords_part = line.split(':', 1)[1]
                    result['keep'].extend(self._parse_keyword_list(keywords_part))
            elif line.startswith('WEAK'):
                current_category = 'weak'
                if ':' in line:
                    keywords_part = line.split(':', 1)[1]
                    result['weak'].extend(self._parse_keyword_list(keywords_part))
            elif line.startswith('REMOVE'):
                current_category = 'remove'
                if ':' in line:
                    keywords_part = line.split(':', 1)[1]
                    result['remove'].extend(self._parse_keyword_list(keywords_part))

        return result


def seed_domains(store: MemoryStore) -> int:
    """
    Convenience function to seed default domain keywords.

    Args:
        store: Memory store to seed

    Returns:
        Number of keywords seeded
    """
    learner = DomainLearner(store)
    return learner.seed_default_domains()
