# Generalizing the Memory System for Different Agent Types

**Status**: Design document for future implementation
**Created**: 2026-02-05
**Branch**: Implement on a separate branch after Phase 6

## Problem Statement

The current memory system has prompts and seed data biased towards software development agents. To reuse it for other agent types (personal assistant, research agent, customer support, etc.), we need to make the domain-specific parts configurable.

## What's Already Generic

The core **data structures** work for any agent:

```python
Episode(context, action, outcome, success_score)  # Universal experience format
Reflection(insight, causal_chain, actionable_takeaway)  # Universal learning
Adaptation(source_context, target_context, ...)  # Universal transfer learning
```

The **mechanisms** are also generic:
- Embedding-based similarity search
- HDBSCAN clustering for pattern discovery
- Jaccard distance for domain differentiation
- LLM-based reflection and adaptation

## What Needs Generalization

### 1. Prompts in `adapter.py`

**Current** (software-dev biased):
```python
IDENTIFY_PROBLEM_TYPE_PROMPT = """...
Consider these aspects:
- What domain is this? (e.g., programming, database, networking, testing)
..."""
```

**Generalized**:
```python
IDENTIFY_PROBLEM_TYPE_PROMPT = """...
Consider these aspects:
- What domain is this? (e.g., {example_domains})
..."""
```

### 2. Prompts in `domain_learner.py`

**Current**:
```python
EXTRACT_KEYWORDS_PROMPT = """Analyze the following text from a software development context..."""
```

**Generalized**:
```python
EXTRACT_KEYWORDS_PROMPT = """Analyze the following text from a {domain_context} context..."""
```

### 3. Seed Keywords in `DomainLearner.seed_default_domains()`

**Current**: Hardcoded software domains (python, docker, git, etc.)

**Generalized**: Accept custom seed data or load from config.

---

## Implementation Options

### Option 1: Parameterized Prompts (Recommended)

Add a `domain_context` parameter that shapes prompts.

**Changes to `adapter.py`**:

```python
class StrategyAdapter:
    def __init__(
        self,
        llm: LLMInterface,
        domain_context: str = "general problem-solving",
        example_domains: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.domain_context = domain_context
        self.example_domains = example_domains or ["task management", "communication", "research"]

    def _get_problem_type_prompt(self, context: str) -> str:
        return IDENTIFY_PROBLEM_TYPE_PROMPT.format(
            context=context,
            domain_context=self.domain_context,
            example_domains=", ".join(self.example_domains),
        )
```

**Changes to `domain_learner.py`**:

```python
class DomainLearner:
    def __init__(
        self,
        store: MemoryStore,
        llm: Optional[LLMInterface] = None,
        domain_context: str = "general",
        custom_seed_domains: Optional[Dict[str, List[str]]] = None,
    ):
        self.domain_context = domain_context
        self.custom_seed_domains = custom_seed_domains

    def seed_domains(self) -> int:
        """Seed with custom domains or defaults"""
        domains = self.custom_seed_domains or self._get_default_domains()
        # ... seed logic

    def _get_default_domains(self) -> Dict[str, List[str]]:
        """Return domain-appropriate defaults"""
        if self.domain_context == "software":
            return {...}  # Current defaults
        else:
            return {}  # Empty, let the system learn
```

**Usage for different agents**:

```python
# Software development agent (current behavior)
adapter = StrategyAdapter(llm, domain_context="software development")

# Personal assistant
adapter = StrategyAdapter(
    llm,
    domain_context="personal task management and life organization",
    example_domains=["scheduling", "reminders", "email", "shopping", "health"],
)

# Customer support agent
adapter = StrategyAdapter(
    llm,
    domain_context="customer support and issue resolution",
    example_domains=["billing", "technical issues", "refunds", "account management"],
)
```

---

### Option 2: Prompt Provider Interface

Create a pluggable prompt system.

**New file `prompts.py`**:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class PromptProvider(ABC):
    """Interface for domain-specific prompts"""

    @abstractmethod
    def get_adaptation_prompt(self, source_episode, target_context) -> str:
        """Prompt for adapting a strategy"""
        pass

    @abstractmethod
    def get_problem_type_prompt(self, context: str) -> str:
        """Prompt for identifying problem type"""
        pass

    @abstractmethod
    def get_reflection_prompt(self, episode, reflection_type: str) -> str:
        """Prompt for generating reflections"""
        pass

    @abstractmethod
    def get_keyword_extraction_prompt(self, text: str) -> str:
        """Prompt for extracting domain keywords"""
        pass


class SoftwareDevPrompts(PromptProvider):
    """Prompts optimized for software development agents"""

    def get_adaptation_prompt(self, source_episode, target_context) -> str:
        return f"""You are helping a software development AI agent...
        ..."""


class PersonalAssistantPrompts(PromptProvider):
    """Prompts for personal assistant agents"""

    def get_adaptation_prompt(self, source_episode, target_context) -> str:
        return f"""You are helping a personal assistant adapt a successful approach...

        ORIGINAL SITUATION (what worked before):
        Task: {source_episode.context}
        Approach: {source_episode.action}
        Result: {source_episode.outcome}

        NEW SITUATION:
        Task: {target_context}

        How should the assistant adapt the approach?
        ..."""


class GenericPrompts(PromptProvider):
    """Domain-agnostic prompts"""
    # ... minimal, generic versions
```

**Usage**:

```python
adapter = StrategyAdapter(llm, prompts=PersonalAssistantPrompts())
```

---

### Option 3: Configuration Files

Store prompts in YAML files that can be swapped.

**File `config/prompts/personal_assistant.yaml`**:

```yaml
domain_name: "personal_assistant"
domain_description: "Personal task management and life organization"

example_domains:
  - scheduling
  - reminders
  - email management
  - shopping
  - health tracking
  - finance

prompts:
  adaptation: |
    You are helping a personal assistant adapt a successful approach.

    ORIGINAL SITUATION:
    Task: {source_context}
    Approach: {source_action}
    Result: {source_outcome}

    NEW SITUATION:
    Task: {target_context}

    Provide:
    1. ADAPTED_APPROACH: How to modify the approach
    2. KEY_SIMILARITIES: What makes these situations similar
    3. KEY_DIFFERENCES: Important differences to consider
    4. CONFIDENCE: low/medium/high

  problem_type: |
    Identify what type of task this is:
    TASK: {context}

    Categories to consider: {example_domains}

    Respond with:
    TASK_TYPE: [type]
    CATEGORY: [broader category]
    CHARACTERISTICS: [bullet points]

seed_domains:
  scheduling:
    - calendar
    - appointment
    - meeting
    - reminder
    - deadline
  communication:
    - email
    - message
    - call
    - reply
    - follow-up
  # ... etc
```

**Loading**:

```python
import yaml

class ConfigurableAdapter:
    def __init__(self, llm, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _get_prompt(self, prompt_name: str, **kwargs) -> str:
        template = self.config['prompts'][prompt_name]
        return template.format(**kwargs)
```

---

## Recommended Implementation Path

### Phase 1: Quick Win (Option 1)

1. Add `domain_context` parameter to `StrategyAdapter` and `DomainLearner`
2. Update prompts to use `{domain_context}` placeholder
3. Make `seed_default_domains()` accept optional custom data
4. Update `__init__.py` exports

**Estimated effort**: 1-2 hours

### Phase 2: Full Flexibility (Option 2 or 3)

If you need multiple agent types with very different prompts:
1. Implement `PromptProvider` interface
2. Create concrete implementations for each agent type
3. Add configuration loading if needed

**Estimated effort**: 3-4 hours

---

## Files to Modify

| File | Changes |
|------|---------|
| `agent_memory/adapter.py` | Add `domain_context` param, update prompts |
| `agent_memory/domain_learner.py` | Add `domain_context`, custom seed support |
| `agent_memory/reflector.py` | Add `domain_context` to reflection prompts |
| `agent_memory/strategy_selector.py` | Pass through `domain_context` |
| `agent_memory/__init__.py` | Export new config classes if added |

---

## Testing the Generalization

Create a simple test to verify the system works for a non-software domain:

```python
def test_personal_assistant_domain():
    """Test memory system with personal assistant context"""
    store = MemoryStore(...)

    # Custom domain setup
    learner = DomainLearner(
        store,
        domain_context="personal assistance",
        custom_seed_domains={
            "scheduling": ["calendar", "appointment", "meeting", "reminder"],
            "shopping": ["buy", "purchase", "order", "grocery", "list"],
        }
    )
    learner.seed_domains()

    # Store a personal assistant episode
    store.store_episode(
        context="User asked to remind them about dentist appointment tomorrow",
        action="Set calendar reminder for 9 AM with notification",
        outcome="User thanked me, said the reminder was helpful",
        success_score=0.95,
        tags=["scheduling", "reminders"],
    )

    # Adapter with personal assistant context
    adapter = StrategyAdapter(
        llm,
        domain_context="personal task management",
        example_domains=["scheduling", "shopping", "communication"],
    )

    # This should work and produce sensible results
    adaptation = adapter.adapt_strategy(
        source_episode=episode,
        target_context="User wants to remember to buy groceries this weekend",
    )

    assert "reminder" in adaptation.adapted_strategy.lower() or \
           "calendar" in adaptation.adapted_strategy.lower()
```

---

## Quick Start After Phase 6

1. Create a new branch: `git checkout -b feature/domain-generalization`
2. Start with Option 1 (parameterized prompts)
3. Test with a simple non-software use case
4. Iterate based on what works

---

## Notes

- The embedding model (nomic-embed-text) is general-purpose and should work fine for any domain
- HDBSCAN clustering is also domain-agnostic
- The main work is in making prompts and seed data configurable
- Consider: should we ship with multiple prompt presets, or start minimal and let users configure?
