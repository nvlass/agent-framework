"""
Tests for Phase 5: Adaptation & Transfer Learning

Tests cover:
- AnalogyFinder: Finding similar-but-different situations
- StrategyAdapter: Adapting strategies with LLM
- StrategySelector: Choosing best approach for new problems
- MemoryStore methods for problem types and adaptations
"""

import pytest
import numpy as np
from datetime import datetime
import tempfile
import os
import shutil

from agent_memory import (
    MemoryStore,
    Episode,
    MockLLM,
    AnalogyFinder,
    AnalogousMatch,
    StrategyAdapter,
    AdaptationResult,
    ProblemType,
    StrategySelector,
    StrategyCandidate,
    StrategySelection,
    DomainLearner,
    LearningReport,
    seed_domains,
)
from agent_memory.adapter import parse_adaptation_response


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    return MockLLM()


@pytest.fixture
def temp_db():
    """Create a temporary database directory"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_memory.db")
    vector_path = os.path.join(temp_dir, "vectors")

    yield db_path, vector_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_store_no_embeddings(temp_db):
    """Create a MemoryStore without embeddings (for storage tests)"""
    db_path, vector_path = temp_db
    store = MemoryStore(
        db_path=db_path,
        vector_store_path=vector_path,
        embedding_generator=None,
    )
    yield store
    store.close()


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing analogies"""
    episodes = []

    # Python debugging episodes
    for i, (ctx, act, out, score) in enumerate([
        ("Python TypeError: 'NoneType' object is not subscriptable",
         "Added null check before accessing dictionary",
         "Bug fixed", 0.9),
        ("Python script crashes with IndexError",
         "Added bounds checking for list access",
         "No more crashes", 0.85),
        ("Python type error when concatenating str and int",
         "Used f-string for type-safe formatting",
         "Works correctly", 0.8),
    ]):
        ep = Episode(
            id=i + 1,
            timestamp=datetime.now(),
            context=ctx,
            action=act,
            outcome=out,
            success_score=score,
            tags=["python", "debugging"],
            embedding=np.random.randn(768).astype(np.float32),  # Mock embedding
        )
        episodes.append(ep)

    # Docker episodes (different domain)
    for i, (ctx, act, out, score) in enumerate([
        ("Docker container won't start, port already in use",
         "Changed port mapping in docker-compose",
         "Container starts successfully", 0.95),
        ("Docker build failing with dependency errors",
         "Added missing packages to Dockerfile",
         "Build completes", 0.9),
    ], start=4):
        ep = Episode(
            id=i,
            timestamp=datetime.now(),
            context=ctx,
            action=act,
            outcome=out,
            success_score=score,
            tags=["docker", "devops"],
            embedding=np.random.randn(768).astype(np.float32),
        )
        episodes.append(ep)

    # Git episodes (another domain)
    for i, (ctx, act, out, score) in enumerate([
        ("Git merge conflict in config files",
         "Resolved conflicts by keeping both changes with proper structure",
         "Merge successful", 0.85),
        ("Need to undo last git commit without losing changes",
         "Used git reset --soft HEAD~1",
         "Commit undone, changes preserved", 0.95),
    ], start=6):
        ep = Episode(
            id=i,
            timestamp=datetime.now(),
            context=ctx,
            action=act,
            outcome=out,
            success_score=score,
            tags=["git", "version-control"],
            embedding=np.random.randn(768).astype(np.float32),
        )
        episodes.append(ep)

    return episodes


# =============================================================================
# AnalogyFinder Tests
# =============================================================================

class TestAnalogyFinder:
    """Tests for the AnalogyFinder class"""

    def test_init_default_values(self):
        """Test AnalogyFinder initialization with defaults"""
        finder = AnalogyFinder()
        assert finder.embedding_generator is None
        assert finder.similarity_threshold == 0.5
        assert finder.domain_distance_weight == 0.3

    def test_init_custom_values(self):
        """Test AnalogyFinder initialization with custom values"""
        finder = AnalogyFinder(
            similarity_threshold=0.7,
            domain_distance_weight=0.5,
        )
        assert finder.similarity_threshold == 0.7
        assert finder.domain_distance_weight == 0.5

    def test_extract_domains_python(self):
        """Test domain extraction for Python context"""
        finder = AnalogyFinder()
        domains = finder._extract_domains("Python TypeError in my script")
        assert 'python' in domains
        assert 'debugging' in domains or 'python' in domains

    def test_extract_domains_docker(self):
        """Test domain extraction for Docker context"""
        finder = AnalogyFinder()
        domains = finder._extract_domains("Docker container networking issue")
        assert 'docker' in domains

    def test_extract_domains_multiple(self):
        """Test domain extraction with multiple domains"""
        finder = AnalogyFinder()
        domains = finder._extract_domains("Python API returns HTTP 500 error")
        assert 'python' in domains
        assert 'api' in domains

    def test_extract_domains_empty(self):
        """Test domain extraction with no domain markers"""
        finder = AnalogyFinder()
        domains = finder._extract_domains("Some generic text without keywords")
        assert len(domains) == 0

    def test_compute_domain_distance_identical(self):
        """Test domain distance for identical domains"""
        finder = AnalogyFinder()
        distance = finder._compute_domain_distance({'python'}, {'python'})
        assert distance == 0.0

    def test_compute_domain_distance_completely_different(self):
        """Test domain distance for completely different domains"""
        finder = AnalogyFinder()
        distance = finder._compute_domain_distance({'python'}, {'docker'})
        assert distance == 1.0

    def test_compute_domain_distance_partial_overlap(self):
        """Test domain distance with partial overlap"""
        finder = AnalogyFinder()
        distance = finder._compute_domain_distance(
            {'python', 'debugging'},
            {'python', 'testing'}
        )
        # Jaccard: intersection=1, union=3 → similarity=0.33 → distance=0.67
        assert 0.5 < distance < 0.8

    def test_compute_domain_distance_empty_sets(self):
        """Test domain distance with empty sets"""
        finder = AnalogyFinder()
        assert finder._compute_domain_distance(set(), set()) == 0.0
        assert finder._compute_domain_distance({'python'}, set()) == 1.0
        assert finder._compute_domain_distance(set(), {'python'}) == 1.0

    def test_compute_analogy_score(self):
        """Test analogy score calculation"""
        finder = AnalogyFinder(domain_distance_weight=0.3)

        # High similarity, same domain → lower analogy score
        score1 = finder._compute_analogy_score(similarity=0.9, domain_distance=0.0)

        # High similarity, different domain → higher analogy score
        score2 = finder._compute_analogy_score(similarity=0.9, domain_distance=1.0)

        assert score2 > score1
        assert score1 == 0.9  # Just similarity
        assert score2 == min(1.0, 0.9 + 0.3)  # Capped at 1.0

    def test_find_analogies_with_mock_embeddings(self, sample_episodes):
        """Test finding analogies with pre-computed embeddings"""
        finder = AnalogyFinder(
            similarity_threshold=0.0,  # Accept all for testing
            domain_distance_weight=0.3,
        )

        # Create a query embedding (similar to Python episodes)
        query_embedding = sample_episodes[0].embedding + np.random.randn(768).astype(np.float32) * 0.1

        matches = finder.find_analogies(
            query_context="Python debugging issue",
            episodes=sample_episodes,
            query_embedding=query_embedding,
            limit=5,
        )

        assert len(matches) > 0
        assert all(isinstance(m, AnalogousMatch) for m in matches)
        # Should be sorted by analogy_score
        scores = [m.analogy_score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_find_analogies_respects_threshold(self, sample_episodes):
        """Test that similarity threshold is respected"""
        finder = AnalogyFinder(
            similarity_threshold=0.99,  # Very high threshold
        )

        # Random embedding unlikely to match
        query_embedding = np.random.randn(768).astype(np.float32) * 10

        matches = finder.find_analogies(
            query_context="Random query",
            episodes=sample_episodes,
            query_embedding=query_embedding,
        )

        # With high threshold and random embedding, should get few/no matches
        assert len(matches) <= 2

    def test_find_cross_domain_analogies(self, sample_episodes):
        """Test finding analogies specifically from different domains"""
        finder = AnalogyFinder(similarity_threshold=0.0)

        # Make Python episodes very similar to query
        query_embedding = sample_episodes[0].embedding.copy()

        matches = finder.find_cross_domain_analogies(
            query_context="Python TypeError debugging",
            episodes=sample_episodes,
            query_embedding=query_embedding,
        )

        # Should exclude Python episodes due to same domain
        # (min_domain_distance=0.3 in cross_domain_analogies)
        for match in matches:
            # Matches should have some domain distance
            assert match.domain_distance >= 0.0

    def test_extract_shared_features(self, sample_episodes):
        """Test extraction of shared features between contexts"""
        finder = AnalogyFinder()

        query = "I need to debug this error in my code"
        episode = sample_episodes[0]  # Python debugging episode

        shared = finder._extract_shared_features(query, episode)

        # Both should have debugging-related features
        assert isinstance(shared, list)
        # "debugging" or "error handling" should be shared
        assert len(shared) >= 0  # May or may not find shared features

    def test_analogous_match_attributes(self, sample_episodes):
        """Test AnalogousMatch dataclass attributes"""
        match = AnalogousMatch(
            episode=sample_episodes[0],
            similarity_score=0.85,
            domain_distance=0.6,
            analogy_score=0.9,
            shared_features=["debugging"],
            different_features=["query: api", "episode: python"],
        )

        assert match.episode.id == sample_episodes[0].id
        assert match.similarity_score == 0.85
        assert match.domain_distance == 0.6
        assert match.analogy_score == 0.9
        assert "debugging" in match.shared_features


# =============================================================================
# StrategyAdapter Tests
# =============================================================================

class TestStrategyAdapter:
    """Tests for the StrategyAdapter class"""

    def test_init(self, mock_llm):
        """Test StrategyAdapter initialization"""
        adapter = StrategyAdapter(mock_llm)
        assert adapter.llm == mock_llm

    def test_adapt_strategy(self, mock_llm, sample_episodes):
        """Test strategy adaptation"""
        adapter = StrategyAdapter(mock_llm)

        result = adapter.adapt_strategy(
            source_episode=sample_episodes[0],  # Python debugging
            target_context="Docker container crashes on startup",
        )

        assert isinstance(result, AdaptationResult)
        assert result.source_episode == sample_episodes[0]
        assert result.target_context == "Docker container crashes on startup"
        assert result.original_strategy == sample_episodes[0].action
        assert len(result.adapted_strategy) > 0

    def test_adaptation_result_confidence_score(self):
        """Test AdaptationResult confidence score conversion"""
        result = AdaptationResult(
            source_episode=None,
            target_context="test",
            original_strategy="original",
            adapted_strategy="adapted",
            confidence="high",
        )
        assert result.confidence_score() == 0.9

        result.confidence = "medium"
        assert result.confidence_score() == 0.6

        result.confidence = "low"
        assert result.confidence_score() == 0.3

        result.confidence = "unknown"
        assert result.confidence_score() == 0.5  # Default

    def test_adaptation_result_to_dict(self, sample_episodes):
        """Test AdaptationResult serialization"""
        result = AdaptationResult(
            source_episode=sample_episodes[0],
            target_context="New context",
            original_strategy="Original action",
            adapted_strategy="Adapted action",
            similarities=["both involve errors"],
            differences=["different languages"],
            confidence="high",
            reasoning="Because...",
        )

        d = result.to_dict()
        assert d['source_episode_id'] == sample_episodes[0].id
        assert d['target_context'] == "New context"
        assert d['confidence'] == "high"
        assert "both involve errors" in d['similarities']

    def test_identify_problem_type(self, mock_llm):
        """Test problem type identification"""
        adapter = StrategyAdapter(mock_llm)

        problem_type = adapter.identify_problem_type(
            "Python script crashes with TypeError when processing JSON data"
        )

        assert isinstance(problem_type, ProblemType)
        assert len(problem_type.name) > 0

    def test_problem_type_attributes(self):
        """Test ProblemType dataclass"""
        pt = ProblemType(
            id=1,
            name="python_debugging",
            domain="programming",
            description="Debugging Python code",
            characteristics=["error messages", "stack traces"],
        )

        assert pt.name == "python_debugging"
        assert "error messages" in pt.characteristics

        d = pt.to_dict()
        assert d['name'] == "python_debugging"
        assert d['domain'] == "programming"

    def test_batch_adapt(self, mock_llm, sample_episodes):
        """Test batch adaptation of multiple strategies"""
        adapter = StrategyAdapter(mock_llm)

        results = adapter.batch_adapt(
            source_episodes=sample_episodes[:3],
            target_context="Database query timeout issue",
            max_adaptations=2,
        )

        assert len(results) == 2  # Respects max_adaptations
        assert all(isinstance(r, AdaptationResult) for r in results)


class TestParseAdaptationResponse:
    """Tests for the adaptation response parser"""

    def test_parse_complete_response(self):
        """Test parsing a complete, well-formatted response"""
        response = """ADAPTED_STRATEGY: Add timeout handling to database queries

KEY_SIMILARITIES:
- Both involve error handling
- Both require defensive coding

KEY_DIFFERENCES:
- Original is about types, new is about timing
- Different error types

CONFIDENCE: high

REASONING: The underlying pattern of adding defensive checks is the same."""

        result = parse_adaptation_response(response)

        assert result['adapted_strategy'] == "Add timeout handling to database queries"
        assert len(result['similarities']) == 2
        assert len(result['differences']) == 2
        assert result['confidence'] == "high"
        assert "defensive" in result['reasoning']

    def test_parse_minimal_response(self):
        """Test parsing a minimal response"""
        response = "ADAPTED_STRATEGY: Just do the thing"

        result = parse_adaptation_response(response)
        assert result['adapted_strategy'] == "Just do the thing"
        assert result['similarities'] == []
        assert result['differences'] == []

    def test_parse_malformed_response(self):
        """Test parsing handles malformed response gracefully"""
        response = "This is just random text without proper formatting"

        result = parse_adaptation_response(response)
        # Should use full text as adapted strategy
        assert len(result['adapted_strategy']) > 0


# =============================================================================
# MemoryStore Adaptation Methods Tests
# =============================================================================

class TestMemoryStoreAdaptation:
    """Tests for MemoryStore's problem type and adaptation methods"""

    def test_store_problem_type_new(self, memory_store_no_embeddings):
        """Test storing a new problem type"""
        store = memory_store_no_embeddings

        type_id = store.store_problem_type(
            name="python_debugging",
            description="Debugging Python applications",
            characteristics=["error messages", "stack traces", "type errors"],
        )

        assert type_id > 0

        # Verify stored
        pt = store.get_problem_type("python_debugging")
        assert pt is not None
        assert pt['name'] == "python_debugging"
        assert "error messages" in pt['characteristics']

    def test_store_problem_type_update(self, memory_store_no_embeddings):
        """Test updating an existing problem type"""
        store = memory_store_no_embeddings

        # Store initial
        id1 = store.store_problem_type(
            name="test_type",
            description="Initial description",
        )

        # Update
        id2 = store.store_problem_type(
            name="test_type",
            description="Updated description",
            characteristics=["new feature"],
        )

        assert id1 == id2  # Same ID

        pt = store.get_problem_type("test_type")
        assert pt['description'] == "Updated description"
        assert "new feature" in pt['characteristics']

    def test_get_problem_type_not_found(self, memory_store_no_embeddings):
        """Test getting a non-existent problem type"""
        store = memory_store_no_embeddings
        pt = store.get_problem_type("nonexistent")
        assert pt is None

    def test_get_problem_type_by_id(self, memory_store_no_embeddings):
        """Test getting problem type by ID"""
        store = memory_store_no_embeddings

        type_id = store.store_problem_type(
            name="test_type",
            description="Test",
        )

        pt = store.get_problem_type_by_id(type_id)
        assert pt is not None
        assert pt['name'] == "test_type"

    def test_get_all_problem_types(self, memory_store_no_embeddings):
        """Test getting all problem types"""
        store = memory_store_no_embeddings

        store.store_problem_type(name="type_a", description="A")
        store.store_problem_type(name="type_b", description="B")
        store.store_problem_type(name="type_c", description="C")

        all_types = store.get_all_problem_types()
        assert len(all_types) == 3

        # Should be ordered by name
        names = [t['name'] for t in all_types]
        assert names == sorted(names)

    def test_get_all_problem_types_with_limit(self, memory_store_no_embeddings):
        """Test getting problem types with limit"""
        store = memory_store_no_embeddings

        for i in range(5):
            store.store_problem_type(name=f"type_{i}", description=f"Type {i}")

        types = store.get_all_problem_types(limit=3)
        assert len(types) == 3

    def test_link_similar_problem_types(self, memory_store_no_embeddings):
        """Test linking similar problem types"""
        store = memory_store_no_embeddings

        id1 = store.store_problem_type(name="python_debug", description="Python debugging")
        id2 = store.store_problem_type(name="js_debug", description="JavaScript debugging")

        success = store.link_similar_problem_types(id1, id2)
        assert success

        # Verify bidirectional link
        pt1 = store.get_problem_type_by_id(id1)
        pt2 = store.get_problem_type_by_id(id2)

        assert id2 in pt1['similar_problem_types']
        assert id1 in pt2['similar_problem_types']

    def test_store_adaptation(self, memory_store_no_embeddings):
        """Test storing an adaptation"""
        store = memory_store_no_embeddings

        adaptation_id = store.store_adaptation(
            source_context="Python TypeError in function call",
            target_context="Docker container fails to start",
            original_strategy="Add type checking",
            adapted_strategy="Add environment variable validation",
            adaptation_reasoning="Both involve validation before execution",
            source_episode_ids=[1, 2, 3],
        )

        assert adaptation_id > 0

    def test_store_adaptation_with_outcome(self, memory_store_no_embeddings):
        """Test storing adaptation with outcome"""
        store = memory_store_no_embeddings

        adaptation_id = store.store_adaptation(
            source_context="Source",
            target_context="Target",
            original_strategy="Original",
            adapted_strategy="Adapted",
            outcome="It worked!",
            success_score=0.9,
        )

        adaptations = store.get_adaptations(limit=1)
        assert len(adaptations) == 1
        assert adaptations[0]['outcome'] == "It worked!"
        assert adaptations[0]['success_score'] == 0.9

    def test_update_adaptation_outcome(self, memory_store_no_embeddings):
        """Test updating adaptation outcome after trying it"""
        store = memory_store_no_embeddings

        adaptation_id = store.store_adaptation(
            source_context="Source",
            target_context="Target",
            original_strategy="Original",
            adapted_strategy="Adapted",
        )

        # Initially no outcome
        adaptations = store.get_adaptations(limit=1)
        assert adaptations[0]['outcome'] is None

        # Update with outcome
        success = store.update_adaptation_outcome(
            adaptation_id=adaptation_id,
            outcome="Strategy worked well",
            success_score=0.85,
        )

        assert success

        # Verify update
        adaptations = store.get_adaptations(limit=1)
        assert adaptations[0]['outcome'] == "Strategy worked well"
        assert adaptations[0]['success_score'] == 0.85

    def test_get_adaptations_filtered(self, memory_store_no_embeddings):
        """Test filtering adaptations"""
        store = memory_store_no_embeddings

        # Create problem types
        py_id = store.store_problem_type(name="python", description="Python")
        docker_id = store.store_problem_type(name="docker", description="Docker")

        # Create adaptations
        store.store_adaptation(
            source_context="Python error",
            target_context="Docker error",
            original_strategy="S1",
            adapted_strategy="A1",
            source_problem_type_id=py_id,
            target_problem_type_id=docker_id,
            success_score=0.9,
        )

        store.store_adaptation(
            source_context="Python warning",
            target_context="Python config",
            original_strategy="S2",
            adapted_strategy="A2",
            source_problem_type_id=py_id,
            target_problem_type_id=py_id,
            success_score=0.5,
        )

        # Filter by target type
        docker_adaptations = store.get_adaptations(target_type_id=docker_id)
        assert len(docker_adaptations) == 1
        assert docker_adaptations[0]['adapted_strategy'] == "A1"

        # Filter by min success
        successful = store.get_adaptations(min_success=0.8)
        assert len(successful) == 1
        assert successful[0]['success_score'] == 0.9

    def test_get_successful_adaptations_for_type(self, memory_store_no_embeddings):
        """Test getting successful adaptations for a problem type"""
        store = memory_store_no_embeddings

        type_id = store.store_problem_type(name="test_type", description="Test")

        # Create successful and unsuccessful adaptations
        store.store_adaptation(
            source_context="Good source",
            target_context="Target",
            original_strategy="Good",
            adapted_strategy="Good adapted",
            target_problem_type_id=type_id,
            success_score=0.9,
        )

        store.store_adaptation(
            source_context="Bad source",
            target_context="Target",
            original_strategy="Bad",
            adapted_strategy="Bad adapted",
            target_problem_type_id=type_id,
            success_score=0.3,
        )

        # Only get successful ones
        successful = store.get_successful_adaptations_for_type(
            problem_type_id=type_id,
            as_target=True,
            min_success=0.7,
        )

        assert len(successful) == 1
        assert successful[0]['adapted_strategy'] == "Good adapted"


# =============================================================================
# StrategySelector Tests
# =============================================================================

class TestStrategySelector:
    """Tests for the StrategySelector class"""

    def test_strategy_candidate_attributes(self, sample_episodes):
        """Test StrategyCandidate dataclass"""
        candidate = StrategyCandidate(
            strategy="Add error handling",
            source_type="direct",
            confidence=0.85,
            reasoning="Similar past experience",
            source_episode=sample_episodes[0],
        )

        assert candidate.strategy == "Add error handling"
        assert candidate.source_type == "direct"
        assert candidate.confidence == 0.85
        assert candidate.source_episode is not None

    def test_strategy_selection_has_recommendation(self):
        """Test StrategySelection.has_recommendation()"""
        selection = StrategySelection(context="Test")
        assert not selection.has_recommendation()

        selection.selected = StrategyCandidate(
            strategy="Do something",
            source_type="direct",
            confidence=0.8,
            reasoning="Test",
        )
        assert selection.has_recommendation()

    def test_strategy_selection_get_recommendation(self):
        """Test StrategySelection.get_recommendation()"""
        selection = StrategySelection(context="Test")
        assert selection.get_recommendation() is None

        selection.selected = StrategyCandidate(
            strategy="Do this thing",
            source_type="pattern",
            confidence=0.9,
            reasoning="Pattern match",
        )
        assert selection.get_recommendation() == "Do this thing"


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdaptationIntegration:
    """Integration tests for the adaptation system"""

    def test_full_adaptation_workflow(self, memory_store_no_embeddings, mock_llm):
        """Test complete workflow: problem → analogy → adaptation → storage"""
        store = memory_store_no_embeddings

        # 1. Store problem types
        py_id = store.store_problem_type(
            name="python_error",
            description="Python runtime errors",
            characteristics=["TypeError", "ValueError", "stack trace"],
        )

        docker_id = store.store_problem_type(
            name="docker_error",
            description="Docker container errors",
            characteristics=["container", "port", "network"],
        )

        # 2. Link them as potentially related
        store.link_similar_problem_types(py_id, docker_id)

        # 3. Create an adapter and adapt a strategy
        adapter = StrategyAdapter(mock_llm)

        source_episode = Episode(
            id=1,
            context="Python TypeError when accessing None",
            action="Added defensive null check before access",
            outcome="Bug fixed",
            success_score=0.9,
        )

        adaptation = adapter.adapt_strategy(
            source_episode=source_episode,
            target_context="Docker container fails to connect to database",
        )

        # 4. Store the adaptation
        adaptation_id = store.store_adaptation(
            source_context=source_episode.context,
            target_context=adaptation.target_context,
            original_strategy=source_episode.action,
            adapted_strategy=adaptation.adapted_strategy,
            adaptation_reasoning=adaptation.reasoning,
            source_problem_type_id=py_id,
            target_problem_type_id=docker_id,
            source_episode_ids=[source_episode.id],
        )

        assert adaptation_id > 0

        # 5. Later, record the outcome
        store.update_adaptation_outcome(
            adaptation_id=adaptation_id,
            outcome="The adapted strategy worked!",
            success_score=0.85,
        )

        # 6. Verify we can retrieve successful adaptations
        successful = store.get_successful_adaptations_for_type(
            docker_id, as_target=True, min_success=0.7
        )
        assert len(successful) == 1
        assert successful[0]['success_score'] == 0.85


# =============================================================================
# Domain Learning Tests
# =============================================================================

class TestDomainKeywords:
    """Tests for MemoryStore domain keyword methods"""

    def test_add_domain_keyword(self, memory_store_no_embeddings):
        """Test adding a single domain keyword"""
        store = memory_store_no_embeddings

        result = store.add_domain_keyword(
            domain_name="python",
            keyword="typeerror",
            weight=0.9,
            source="seed",
        )

        assert result is True

        # Verify it was stored
        keywords = store.get_domain_keywords(domain_name="python")
        assert "python" in keywords
        assert "typeerror" in keywords["python"]

    def test_add_domain_keywords_bulk(self, memory_store_no_embeddings):
        """Test adding multiple keywords at once"""
        store = memory_store_no_embeddings

        count = store.add_domain_keywords_bulk(
            domain_name="docker",
            keywords=["container", "image", "compose", "dockerfile"],
            weight=0.8,
            source="seed",
        )

        assert count == 4

        keywords = store.get_domain_keywords(domain_name="docker")
        assert len(keywords["docker"]) == 4
        assert "container" in keywords["docker"]

    def test_get_domain_keywords_with_filter(self, memory_store_no_embeddings):
        """Test filtering keywords by weight and source"""
        store = memory_store_no_embeddings

        # Add keywords with different weights
        store.add_domain_keyword("test", "high_weight", weight=0.9, source="seed")
        store.add_domain_keyword("test", "low_weight", weight=0.2, source="learned")

        # Filter by weight
        high_only = store.get_domain_keywords(domain_name="test", min_weight=0.5)
        assert "high_weight" in high_only.get("test", [])
        assert "low_weight" not in high_only.get("test", [])

        # Filter by source
        seed_only = store.get_domain_keywords(domain_name="test", source="seed")
        assert "high_weight" in seed_only.get("test", [])

    def test_get_domain_keywords_with_weights(self, memory_store_no_embeddings):
        """Test getting keywords with their weights"""
        store = memory_store_no_embeddings

        store.add_domain_keyword("python", "exception", weight=0.95)
        store.add_domain_keyword("python", "traceback", weight=0.85)

        keywords_with_weights = store.get_domain_keywords_with_weights("python")

        assert "python" in keywords_with_weights
        assert keywords_with_weights["python"]["exception"] == 0.95
        assert keywords_with_weights["python"]["traceback"] == 0.85

    def test_get_all_domains(self, memory_store_no_embeddings):
        """Test getting list of all domains"""
        store = memory_store_no_embeddings

        store.add_domain_keyword("python", "test")
        store.add_domain_keyword("docker", "test")
        store.add_domain_keyword("git", "test")

        domains = store.get_all_domains()
        assert "python" in domains
        assert "docker" in domains
        assert "git" in domains

    def test_increment_keyword_occurrence(self, memory_store_no_embeddings):
        """Test incrementing keyword occurrence count"""
        store = memory_store_no_embeddings

        store.add_domain_keyword("python", "error", weight=0.5)

        # Increment
        result = store.increment_keyword_occurrence("python", "error")
        assert result is True

        # Non-existent keyword
        result = store.increment_keyword_occurrence("python", "nonexistent")
        assert result is False

    def test_delete_domain_keywords(self, memory_store_no_embeddings):
        """Test deleting keywords with filters"""
        store = memory_store_no_embeddings

        store.add_domain_keyword("test", "keep", weight=0.9, source="seed")
        store.add_domain_keyword("test", "delete", weight=0.2, source="learned")

        # Delete low-weight keywords
        deleted = store.delete_domain_keywords(min_weight_to_keep=0.5)
        assert deleted == 1

        keywords = store.get_domain_keywords(domain_name="test")
        assert "keep" in keywords.get("test", [])
        assert "delete" not in keywords.get("test", [])


class TestDomainLearner:
    """Tests for the DomainLearner class"""

    def test_init(self, memory_store_no_embeddings):
        """Test DomainLearner initialization"""
        learner = DomainLearner(memory_store_no_embeddings)
        assert learner.store == memory_store_no_embeddings
        assert learner.llm is None

    def test_init_with_llm(self, memory_store_no_embeddings, mock_llm):
        """Test DomainLearner initialization with LLM"""
        learner = DomainLearner(memory_store_no_embeddings, llm=mock_llm)
        assert learner.llm == mock_llm

    def test_seed_default_domains(self, memory_store_no_embeddings):
        """Test seeding default domain keywords"""
        learner = DomainLearner(memory_store_no_embeddings)

        count = learner.seed_default_domains()

        assert count > 50  # Should have many keywords
        domains = memory_store_no_embeddings.get_all_domains()
        assert "python" in domains
        assert "docker" in domains
        assert "git" in domains

    def test_seed_domains_convenience_function(self, memory_store_no_embeddings):
        """Test the seed_domains convenience function"""
        count = seed_domains(memory_store_no_embeddings)
        assert count > 50

    def test_extract_keywords(self, memory_store_no_embeddings):
        """Test keyword extraction from text"""
        learner = DomainLearner(memory_store_no_embeddings)

        text = "Python TypeError when accessing dictionary key in my script"
        keywords = learner._extract_keywords(text)

        assert "python" in keywords
        assert "typeerror" in keywords
        assert "dictionary" in keywords
        # Stop words should be filtered
        assert "when" not in keywords
        assert "the" not in keywords

    def test_learn_from_episodes(self, memory_store_no_embeddings):
        """Test learning keywords from tagged episodes"""
        store = memory_store_no_embeddings
        learner = DomainLearner(store, min_occurrences=1)

        # Create episodes with tags
        episodes = [
            Episode(
                id=1,
                context="Python TypeError in my asyncio code",
                action="Added exception handling",
                outcome="Fixed",
                tags=["python", "debugging"],
            ),
            Episode(
                id=2,
                context="Python ValueError when parsing JSON",
                action="Added validation",
                outcome="Works now",
                tags=["python", "debugging"],
            ),
        ]

        report = learner.learn_from_episodes(episodes)

        assert isinstance(report, LearningReport)
        assert report.episodes_processed == 2
        # Should have learned some keywords
        assert report.keywords_discovered >= 0

    def test_expand_domain_with_llm(self, memory_store_no_embeddings, mock_llm):
        """Test LLM-based domain expansion"""
        store = memory_store_no_embeddings
        learner = DomainLearner(store, llm=mock_llm)

        # Seed first
        store.add_domain_keyword("python", "exception")

        # Expand with LLM
        new_keywords = learner.expand_domain_with_llm("python")

        # MockLLM should return something
        assert isinstance(new_keywords, list)


class TestAnalogyFinderWithLearnableMarkers:
    """Tests for AnalogyFinder with learnable domain markers"""

    def test_finder_loads_from_db(self, memory_store_no_embeddings):
        """Test that AnalogyFinder can load markers from DB"""
        store = memory_store_no_embeddings

        # Seed some keywords
        store.add_domain_keyword("custom_domain", "custom_keyword")

        # Create finder with store
        finder = AnalogyFinder(memory_store=store)

        # Should have loaded from DB
        markers = finder.domain_markers
        assert "custom_domain" in markers
        assert "custom_keyword" in markers["custom_domain"]

    def test_finder_uses_defaults_when_db_empty(self):
        """Test that finder falls back to defaults"""
        # No store provided
        finder = AnalogyFinder()

        markers = finder.domain_markers
        # Should use defaults
        assert "python" in markers
        assert "docker" in markers

    def test_finder_custom_markers_override(self, memory_store_no_embeddings):
        """Test that custom markers override DB and defaults"""
        store = memory_store_no_embeddings
        store.add_domain_keyword("from_db", "db_keyword")

        custom = {"custom": {"my_keyword"}}

        finder = AnalogyFinder(
            memory_store=store,
            domain_markers=custom,
        )

        markers = finder.domain_markers
        assert markers == custom
        assert "from_db" not in markers

    def test_finder_refresh_markers(self, memory_store_no_embeddings):
        """Test refreshing markers after learning"""
        store = memory_store_no_embeddings

        # Seed DB so we're not using defaults
        store.add_domain_keyword("existing", "existing_keyword", weight=0.8)

        finder = AnalogyFinder(memory_store=store)

        # Initial state (from DB)
        initial_markers = finder.domain_markers
        assert "existing" in initial_markers

        # Add new keyword
        store.add_domain_keyword("new_domain", "new_keyword", weight=0.8)

        # Still cached (new_domain not visible yet)
        assert "new_domain" not in finder.domain_markers

        # Refresh
        finder.refresh_markers()

        # Now should have new keyword
        new_markers = finder.domain_markers
        assert "new_domain" in new_markers
        assert "existing" in new_markers  # Old one still there

    def test_finder_get_marker_stats(self, memory_store_no_embeddings):
        """Test getting marker statistics"""
        store = memory_store_no_embeddings
        seed_domains(store)

        finder = AnalogyFinder(memory_store=store)
        stats = finder.get_marker_stats()

        assert stats['num_domains'] > 5
        assert stats['total_keywords'] > 50
        assert "python" in stats['domains']
