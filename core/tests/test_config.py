"""Tests for AgentConfig — log_level validation and serialization."""

from agent_core.config import AgentConfig


class TestLogLevel:
    def test_default_is_warning(self):
        config = AgentConfig()
        assert config.log_level == "WARNING"

    def test_valid_levels_pass_validation(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            config = AgentConfig(log_level=level)
            assert config.validate() == []

    def test_invalid_level_fails_validation(self):
        config = AgentConfig(log_level="BOGUS")
        errors = config.validate()
        assert any("log_level" in e for e in errors)

    def test_to_dict_includes_log_level(self):
        config = AgentConfig(log_level="DEBUG")
        d = config.to_dict()
        assert d["log_level"] == "DEBUG"

    def test_from_dict_roundtrip(self):
        original = AgentConfig(log_level="INFO")
        restored = AgentConfig.from_dict(original.to_dict())
        assert restored.log_level == "INFO"

    def test_from_dict_default(self):
        config = AgentConfig.from_dict({})
        assert config.log_level == "WARNING"


class TestPatternValidation:
    def test_react_passes(self):
        config = AgentConfig(pattern="react")
        assert config.validate() == []

    def test_plan_and_execute_passes(self):
        config = AgentConfig(pattern="plan_and_execute")
        assert config.validate() == []

    def test_unknown_pattern_fails(self):
        config = AgentConfig(pattern="bogus")
        errors = config.validate()
        assert any("pattern" in e.lower() or "bogus" in e for e in errors)
