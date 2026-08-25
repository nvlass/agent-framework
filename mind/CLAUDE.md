# agent-mind

Self-direction, introspection, and metacognition for agents.

## Status

**Phase:** Initial development
**Location:** `./mind/`

## Overview

agent-mind is the "thinking about thinking" layer. It gives an agent the ability to:
- **Set and manage goals** — what should I be doing?
- **Plan** — how do I break this down into steps?
- **Reflect** — what worked, what didn't, what should I change?
- **Self-prompt** — what questions should I be asking?

agent-mind owns the *cognitive* side. agent-patterns owns the *execution* side.
The planner thinks, the executor acts.

## Architecture

```
agent_mind/
├── goals/          # Goal tree — what to do
│   ├── model.py    # Goal, GoalState, GoalTree
│   └── manager.py  # GoalManager — operations on the tree
├── planning/       # Planner — how to do it
│   ├── model.py    # Plan, PlanStep
│   └── planner.py  # Planner — generates plans from goals
└── introspection/  # Self-awareness — how am I doing
    ├── triggers.py # When to reflect (event, idle, curiosity)
    └── monitor.py  # Track progress, detect stuck states
```

---

## Core: Goal Tree

### Design (from master roadmap)

Goals form a **tree with priority**:
- Root: top-level objective
- Children: sub-goals needed to achieve parent
- Leaves: actionable tasks

### Goal States

```
pending → active → completed
              ↓
           blocked → active (when unblocked)
              ↓
           abandoned
```

- **pending**: Not yet started
- **active**: Currently being worked on
- **blocked**: Waiting on something (tracks why and what unblocks)
- **completed**: Done
- **abandoned**: No longer relevant

### Goal Model

```python
@dataclass
class Goal:
    id: str
    description: str
    state: GoalState = GoalState.PENDING
    priority: int = 5              # 1 (low) to 10 (urgent)
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    unblocked_by: Optional[str] = None  # what needs to happen
    created_at: datetime
    completed_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
```

### Operations

- **push(goal)** — add a goal (from user, self-generated, or decomposition)
- **complete(goal_id)** — mark done, propagate up if all siblings done
- **block(goal_id, reason)** — pause, record why
- **unblock(goal_id)** — resume
- **abandon(goal_id)** — no longer relevant
- **decompose(goal_id, sub_goals)** — create children
- **reprioritize(goal_id, new_priority)** — shift focus
- **get_next()** — return highest-priority active leaf goal

---

## Core: Planner

### Design (from master roadmap)

agent-mind owns the Planner — the cognitive capability that generates plans:
- Breaking down goals into steps
- Sequencing, anticipating obstacles
- Revising plans when context changes
- Deciding *what* to do before doing it

agent-patterns owns the Plan Executor — the loop that runs plans.

### Plan Model

```python
@dataclass
class PlanStep:
    id: str
    description: str
    tool_name: Optional[str] = None   # which tool to use
    tool_args: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None

@dataclass
class Plan:
    goal_id: str
    steps: list[PlanStep]
    created_at: datetime
    revised_at: Optional[datetime] = None
    revision_reason: Optional[str] = None
```

### Planner Interface

```python
class Planner:
    def create_plan(self, goal: Goal, context: dict) -> Plan:
        """Generate a plan to achieve the goal."""

    def revise_plan(self, plan: Plan, reason: str, context: dict) -> Plan:
        """Revise an existing plan based on new information."""
```

The planner needs an LLM to reason about decomposition. Use dependency injection
(same pattern as agent-memory's Reflector).

---

## Introspection

### Triggers (from master roadmap)

Three types of triggers for self-reflection:

1. **Event-triggered (automatic):**
   - Significant success (score > threshold): "Why did this work?"
   - Significant failure (score < threshold): "What caused this to fail?"

2. **Idle-triggered (opportunistic):**
   - "I have downtime, let me consolidate experiences"
   - Batch reflection on clusters, not individual episodes

3. **Self-prompted (curiosity-driven):**
   - "I keep failing at X, let me analyze why"
   - More deliberate, goal-directed reflection

### Progress Monitor

Tracks recent actions and detects patterns:
- Consecutive failures → trigger reflection
- Actions with no progress → trigger reflection
- Agent says "I'm uncertain" → trigger reflection

Initial heuristics (fixed, designed to become learnable):
- 3+ consecutive failures → reflect
- 5+ actions with "no change" → reflect
- Immutable guardrails: min 3 actions between reflections, max 10 failures before forced reflect

### Exploration vs Exploitation

Familiarity + stakes gating:
```
familiar AND low_stakes   → exploit immediately
familiar AND high_stakes  → exploit with verification
unfamiliar AND low_stakes → quick exploration
unfamiliar AND high_stakes → full parallel evaluation
```

---

## Interface with agent-patterns

Event-based communication with shared context:

**Events from patterns → mind:**
- `ActionCompleted(action, result)`
- `StepFailed(step, reason)`
- `Stuck(attempts, last_actions)`
- `PatternComplete(outcome)`

**Events from mind → patterns:**
- `Reflect(trigger)`
- `Replan(reason)`
- `Reprioritize(new_goal)`
- `Abort(reason)`

---

## Development Phases

### Phase 1: Goal Tree ✅
- [x] GoalState enum, Goal dataclass
- [x] GoalTree — the tree data structure
- [x] GoalManager — operations (push, complete, block, decompose, get_next)
- [x] Tests

### Phase 2: Planner ✅
- [x] PlanStep, Plan dataclasses
- [x] Planner interface + simple implementation
- [x] Plan revision
- [x] Tests

### Phase 3: Introspection ✅ (partial)
- [x] Progress monitor (track actions, detect stuck)
- [x] Reflection triggers (event, idle, curiosity)
- [ ] Integration with agent-memory's Reflector
- [x] Tests

### Phase 4: Integration
- [ ] Event types for agent-patterns interface
- [ ] Shared context object
- [ ] Exploration/exploitation gating

---

## Project Structure

```
mind/
├── CLAUDE.md                      # This file
├── pyproject.toml
├── agent_mind/
│   ├── __init__.py
│   ├── goals/
│   │   ├── __init__.py
│   │   ├── model.py               # Goal, GoalState
│   │   └── manager.py             # GoalTree, GoalManager
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── model.py               # Plan, PlanStep, StepStatus
│   │   └── planner.py             # Planner
│   └── introspection/
│       ├── __init__.py
│       ├── triggers.py            # Reflection triggers
│       └── monitor.py             # Progress monitor
└── tests/
    ├── __init__.py
    ├── test_goal_model.py
    ├── test_goal_manager.py
    ├── test_plan_model.py
    └── test_planner.py
```

---

**Last Updated:** 2026-02-08
**Status:** Initial skeleton
**Developers:** Nikos & Claude
