# agent-patterns

Reusable agent execution patterns — the loop layer that wires thinking (agent-mind) to acting (agent-tools).

## Status

**Phase:** Phase 2 (composite patterns)
**Location:** `./patterns/`

## Overview

agent-patterns owns the *execution* side. agent-mind owns the *cognitive* side.
The planner thinks, the executor acts.

**Core components:**
- **EventBus** — sync pub/sub for mind↔patterns communication
- **SharedContext** — mutable state shared between mind and pattern layers
- **PlanExecutor** — runs Plan steps via ToolExecutor with DAG scheduling
- **ReactLoop** — Reason → Act → Observe → Repeat
- **PlanAndExecute** — creates plan via PlannerInterface, executes via PlanExecutor, replans on failure

## Architecture

```
agent_patterns/
├── events/
│   ├── __init__.py
│   └── types.py        # Event dataclasses + EventBus
├── context.py           # SharedContext
├── executor.py          # PlanExecutor
├── plan_and_execute.py  # PlanAndExecute (plan → execute → replan)
└── react.py             # ReAct loop + ReasonerInterface
```

## Event Types

**Patterns → Mind (observation):**
- `ActionCompleted(action, result, classification)`
- `StepFailed(step_id, reason)`
- `Stuck(attempts, last_actions)`
- `PatternComplete(goal_id, success, summary)`

**Mind → Patterns (control):**
- `Reflect(trigger)` — pause for reflection
- `Replan(reason)` — request new plan
- `Abort(reason)` — stop execution

## Development Phases

### Phase 1: Foundation ✅
- [x] Event system (types + EventBus)
- [x] SharedContext
- [x] PlanExecutor (DAG scheduling, monitor, events)
- [x] ReactLoop (reason→act→observe, MockReasoner)

### Phase 2: Composite Patterns 🔨
- [x] Plan-and-Execute (planner + executor + replan on failure)
- [ ] Reflexion (needs memory integration)
- [ ] Goal-Oriented pattern

**Deferred for Plan-and-Execute:**
- Replan-tail: keep completed steps, revise only remaining from failure point
- Retry-step: retry a single failed step with backoff or different args
- Step-as-Pattern: failing step gets expanded into a sub-pattern (needs `pattern: Pattern` on PlanStep)

### Phase 3: Pattern Selection
- [ ] Heuristic-based pattern selection
- [ ] Learned pattern selection

### Future: Composability
- [ ] Pattern nesting
- [ ] Pattern combinators

---

## Running Tests

```bash
cd /Users/nvlass/work/agents/patterns && python -m pytest tests/ -v
```

---

**Last Updated:** 2026-02-14
**Status:** Phase 2 in progress — PlanAndExecute done, 71 tests
**Developers:** Nikos & Claude
