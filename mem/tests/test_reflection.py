"""
Tests for the reflection system (Phase 4)

Tests cover:
- Reflection and CausalFactor data models
- LLM interface (MockLLM)
- Reflector class
- MemoryStore reflection methods
"""

import pytest
from datetime import datetime, timedelta

from agent_memory.memory_store import Episode, Reflection, CausalFactor, MemoryStore
from agent_memory.llm_interface import (
    MockLLM,
    LLMResponse,
    parse_reflection_response,
    FAILURE_REFLECTION_PROMPT,
    SUCCESS_REFLECTION_PROMPT,
)
from agent_memory.reflector import Reflector, ReflectionConfig


# =============================================================================
# CausalFactor Tests
# =============================================================================

class TestCausalFactor:
    """Test CausalFactor data model"""

    def test_creation(self):
        """Test basic creation"""
        cf = CausalFactor(
            factor="Missing null check",
            contribution="negative",
            confidence=0.8
        )
        assert cf.factor == "Missing null check"
        assert cf.contribution == "negative"
        assert cf.confidence == 0.8

    def test_to_dict(self):
        """Test serialization to dict"""
        cf = CausalFactor(
            factor="Good error handling",
            contribution="positive",
            confidence=0.9
        )
        d = cf.to_dict()
        assert d["factor"] == "Good error handling"
        assert d["contribution"] == "positive"
        assert d["confidence"] == 0.9

    def test_from_dict(self):
        """Test deserialization from dict"""
        d = {
            "factor": "Type hints",
            "contribution": "positive",
            "confidence": 0.75
        }
        cf = CausalFactor.from_dict(d)
        assert cf.factor == "Type hints"
        assert cf.contribution == "positive"
        assert cf.confidence == 0.75

    def test_default_values(self):
        """Test default values"""
        cf = CausalFactor(factor="Test")
        assert cf.contribution == "positive"
        assert cf.confidence == 0.5


# =============================================================================
# Reflection Tests
# =============================================================================

class TestReflection:
    """Test Reflection data model"""

    def test_creation(self):
        """Test basic creation"""
        ref = Reflection(
            reflection_type="failure_analysis",
            trigger_episode_id=42,
            insight="Should have checked for None",
            causal_chain=[
                CausalFactor("Missing null check", "negative", 0.9)
            ],
            actionable_takeaway="Always validate input"
        )
        assert ref.reflection_type == "failure_analysis"
        assert ref.trigger_episode_id == 42
        assert ref.insight == "Should have checked for None"
        assert len(ref.causal_chain) == 1
        assert ref.actionable_takeaway == "Always validate input"

    def test_invalid_reflection_type(self):
        """Test that invalid reflection type raises error"""
        with pytest.raises(ValueError):
            Reflection(
                reflection_type="invalid_type",
                insight="Test"
            )

    def test_valid_reflection_types(self):
        """Test all valid reflection types"""
        for rtype in ['success_analysis', 'failure_analysis', 'pattern_discovery']:
            ref = Reflection(reflection_type=rtype, insight="Test")
            assert ref.reflection_type == rtype

    def test_to_dict(self):
        """Test serialization to dict"""
        ref = Reflection(
            id=1,
            reflection_type="success_analysis",
            trigger_episode_id=10,
            insight="Great approach",
            causal_chain=[
                CausalFactor("Clear code", "positive", 0.8)
            ],
            actionable_takeaway="Keep it simple"
        )
        d = ref.to_dict()
        assert d["id"] == 1
        assert d["reflection_type"] == "success_analysis"
        assert d["trigger_episode_id"] == 10
        assert d["insight"] == "Great approach"
        assert len(d["causal_chain"]) == 1
        assert d["actionable_takeaway"] == "Keep it simple"

    def test_default_values(self):
        """Test default values"""
        ref = Reflection(reflection_type="success_analysis", insight="Test")
        assert ref.id is None
        assert ref.trigger_episode_id is None
        assert ref.causal_chain == []
        assert ref.actionable_takeaway is None
        assert ref.created_at is not None


# =============================================================================
# MockLLM Tests
# =============================================================================

class TestMockLLM:
    """Test MockLLM interface"""

    def test_default_response(self):
        """Test default response when no keywords match"""
        llm = MockLLM()
        response = llm.generate("Test prompt")
        assert response.text == "This is a mock LLM response for testing."
        assert response.model == "mock"

    def test_keyword_matching(self):
        """Test keyword-based responses"""
        llm = MockLLM()
        llm.set_response("failure", "This is a failure response")
        llm.set_response("success", "This is a success response")

        resp1 = llm.generate("Analyze this failure")
        assert resp1.text == "This is a failure response"

        resp2 = llm.generate("Why was this a success?")
        assert resp2.text == "This is a success response"

    def test_custom_default_response(self):
        """Test setting custom default response"""
        llm = MockLLM()
        llm.set_default_response("Custom default")

        response = llm.generate("No matching keywords here")
        assert response.text == "Custom default"

    def test_is_available(self):
        """Test availability check"""
        llm = MockLLM()
        assert llm.is_available() is True

    def test_model_name(self):
        """Test model name property"""
        llm = MockLLM()
        assert llm.model_name == "mock"


# =============================================================================
# Response Parsing Tests
# =============================================================================

class TestResponseParsing:
    """Test LLM response parsing"""

    def test_parse_full_response(self):
        """Test parsing a complete reflection response"""
        response_text = """
INSIGHT: The function failed because it didn't handle None values

CAUSAL FACTORS:
- Missing null check: negative (confidence: 0.9)
- No input validation: negative (confidence: 0.7)

ACTIONABLE TAKEAWAY: Always add defensive null checks before accessing object properties
"""
        parsed = parse_reflection_response(response_text)

        assert parsed["insight"] == "The function failed because it didn't handle None values"
        assert len(parsed["causal_factors"]) == 2
        assert parsed["causal_factors"][0]["factor"] == "Missing null check"
        assert parsed["causal_factors"][0]["contribution"] == "negative"
        assert parsed["causal_factors"][0]["confidence"] == 0.9
        assert "null checks" in parsed["actionable_takeaway"]

    def test_parse_minimal_response(self):
        """Test parsing response with minimal content"""
        response_text = "INSIGHT: Something went wrong"
        parsed = parse_reflection_response(response_text)

        assert parsed["insight"] == "Something went wrong"
        assert parsed["causal_factors"] == []
        assert parsed["actionable_takeaway"] == ""

    def test_parse_positive_factors(self):
        """Test parsing positive causal factors"""
        response_text = """
INSIGHT: Success!

CAUSAL FACTORS:
- Good error handling: positive (confidence: 0.85)

ACTIONABLE TAKEAWAY: Keep doing this
"""
        parsed = parse_reflection_response(response_text)
        assert parsed["causal_factors"][0]["contribution"] == "positive"


# =============================================================================
# Reflector Tests
# =============================================================================

class TestReflector:
    """Test Reflector class"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with useful responses"""
        llm = MockLLM()

        failure_response = """
INSIGHT: The code crashed due to unhandled null value

CAUSAL FACTORS:
- Missing input validation: negative (confidence: 0.9)
- No defensive coding: negative (confidence: 0.7)

ACTIONABLE TAKEAWAY: Add null checks before accessing properties
"""
        success_response = """
INSIGHT: The approach worked because of thorough validation

CAUSAL FACTORS:
- Type checking: positive (confidence: 0.85)
- Clear error messages: positive (confidence: 0.75)

ACTIONABLE TAKEAWAY: Continue using type hints and validation
"""
        pattern_response = """
INSIGHT: Python debugging patterns show that type hints prevent errors

CAUSAL FACTORS:
- Consistent use of type hints: positive (confidence: 0.8)
- Early validation: positive (confidence: 0.7)

ACTIONABLE TAKEAWAY: Always use type hints in Python code
"""

        llm.set_response("failure", failure_response)
        llm.set_response("success", success_response)
        llm.set_response("pattern", pattern_response)

        return llm

    @pytest.fixture
    def reflector(self, mock_llm):
        """Create a Reflector with mock LLM"""
        return Reflector(llm=mock_llm)

    def test_reflect_on_failure(self, reflector):
        """Test failure reflection generation"""
        episode = Episode(
            id=1,
            context="Python script crashed with TypeError",
            action="Tried to access dict key without checking",
            outcome="Script crashed",
            success_score=0.2
        )

        reflection = reflector.reflect_on_failure(episode)

        assert reflection.reflection_type == "failure_analysis"
        assert reflection.trigger_episode_id == 1
        assert "null" in reflection.insight.lower() or "crashed" in reflection.insight.lower()
        assert len(reflection.causal_chain) >= 1

    def test_reflect_on_success(self, reflector):
        """Test success reflection generation"""
        episode = Episode(
            id=2,
            context="Implemented input validation",
            action="Added type hints and validation",
            outcome="Code works reliably",
            success_score=0.95
        )

        reflection = reflector.reflect_on_success(episode)

        assert reflection.reflection_type == "success_analysis"
        assert reflection.trigger_episode_id == 2
        assert len(reflection.insight) > 0

    def test_discover_patterns(self, reflector):
        """Test pattern discovery"""
        episodes = [
            Episode(id=i, context=f"Python debugging {i}", action="Added validation",
                    success_score=0.85, tags=["python"])
            for i in range(5)
        ]

        reflection = reflector.discover_patterns(episodes, common_tags=["python"])

        assert reflection.reflection_type == "pattern_discovery"
        assert len(reflection.insight) > 0

    def test_should_reflect_failure(self, reflector):
        """Test automatic failure detection"""
        episode = Episode(success_score=0.2)
        assert reflector.should_reflect(episode) == "failure"

    def test_should_reflect_success(self, reflector):
        """Test automatic success detection"""
        episode = Episode(success_score=0.95)
        assert reflector.should_reflect(episode) == "success"

    def test_should_reflect_none(self, reflector):
        """Test no reflection for medium scores"""
        episode = Episode(success_score=0.5)
        assert reflector.should_reflect(episode) is None

    def test_auto_reflect(self, reflector):
        """Test automatic reflection"""
        failure_episode = Episode(
            id=1,
            context="Failed task",
            action="Did something wrong",
            success_score=0.1
        )

        reflection = reflector.auto_reflect(failure_episode)
        assert reflection is not None
        assert reflection.reflection_type == "failure_analysis"

    def test_auto_reflect_skips_medium_score(self, reflector):
        """Test that auto_reflect skips medium scores"""
        episode = Episode(success_score=0.5)
        reflection = reflector.auto_reflect(episode)
        assert reflection is None


class TestReflectionConfig:
    """Test ReflectionConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = ReflectionConfig()
        assert config.failure_threshold == 0.3
        assert config.success_threshold == 0.9
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.auto_reflect is True

    def test_custom_values(self):
        """Test custom configuration"""
        config = ReflectionConfig(
            failure_threshold=0.2,
            success_threshold=0.85,
            auto_reflect=False
        )
        assert config.failure_threshold == 0.2
        assert config.success_threshold == 0.85
        assert config.auto_reflect is False


# =============================================================================
# MemoryStore Reflection Tests
# =============================================================================

class TestMemoryStoreReflections:
    """Test MemoryStore reflection methods"""

    @pytest.fixture
    def memory_store(self, tmp_path):
        """Create a temporary memory store"""
        db_path = tmp_path / "test_memory.db"
        vector_path = tmp_path / "test_vectors"
        store = MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(vector_path),
            embedding_generator=None
        )
        yield store
        store.close()

    def test_store_and_retrieve_reflection(self, memory_store):
        """Test storing and retrieving a reflection"""
        reflection = Reflection(
            reflection_type="failure_analysis",
            trigger_episode_id=1,
            insight="Test insight",
            causal_chain=[
                CausalFactor("Test factor", "negative", 0.8)
            ],
            actionable_takeaway="Test takeaway"
        )

        ref_id = memory_store.store_reflection(reflection)
        assert ref_id > 0

        retrieved = memory_store.get_reflection_by_id(ref_id)
        assert retrieved is not None
        assert retrieved.reflection_type == "failure_analysis"
        assert retrieved.insight == "Test insight"
        assert len(retrieved.causal_chain) == 1

    def test_get_reflections_filtered(self, memory_store):
        """Test filtering reflections by type"""
        # Store different types
        memory_store.store_reflection(Reflection(
            reflection_type="failure_analysis",
            insight="Failure 1"
        ))
        memory_store.store_reflection(Reflection(
            reflection_type="success_analysis",
            insight="Success 1"
        ))
        memory_store.store_reflection(Reflection(
            reflection_type="failure_analysis",
            insight="Failure 2"
        ))

        failures = memory_store.get_reflections(reflection_type="failure_analysis")
        assert len(failures) == 2

        successes = memory_store.get_reflections(reflection_type="success_analysis")
        assert len(successes) == 1

    def test_count_reflections(self, memory_store):
        """Test counting reflections"""
        assert memory_store.count_reflections() == 0

        memory_store.store_reflection(Reflection(
            reflection_type="failure_analysis",
            insight="Test 1"
        ))
        memory_store.store_reflection(Reflection(
            reflection_type="success_analysis",
            insight="Test 2"
        ))

        assert memory_store.count_reflections() == 2
        assert memory_store.count_reflections("failure_analysis") == 1

    def test_get_reflections_for_episode(self, memory_store):
        """Test getting reflections for a specific episode"""
        memory_store.store_reflection(Reflection(
            reflection_type="failure_analysis",
            trigger_episode_id=42,
            insight="Episode 42 reflection"
        ))
        memory_store.store_reflection(Reflection(
            reflection_type="success_analysis",
            trigger_episode_id=99,
            insight="Episode 99 reflection"
        ))

        refs = memory_store.get_reflections_for_episode(42)
        assert len(refs) == 1
        assert refs[0].insight == "Episode 42 reflection"

    def test_reflection_not_found(self, memory_store):
        """Test getting non-existent reflection"""
        result = memory_store.get_reflection_by_id(9999)
        assert result is None
