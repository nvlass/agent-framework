# Agent Development Framework - Master Roadmap

## Vision

Build a comprehensive, local-first toolkit for creating autonomous AI agents. Each component is a standalone project that can be used independently, but together they form a complete agent development framework.

## Architecture Overview

> Full class diagrams and data flow: [ARCHITECTURE.md](./ARCHITECTURE.md)

```
┌─────────────────────────────────────────────────────────────┐
│                    agent-core (soul)                        │
│         Identity, values, orchestration, safety             │
├─────────────────────────────────────────────────────────────┤
│                  agent-patterns (loops)                     │
│           ReAct, goal-oriented, plan-and-execute            │
├─────────────────────────────────────────────────────────────┤
│                 agent-mind (metacognition)                  │
│      Self-prompting, introspection, curiosity, planning     │
├────────────────────────┬────────────────────────────────────┤
│     agent-memory       │          agent-tools               │
│  Episodic memory,      │    Tool abstraction,               │
│  learning, adaptation  │    execution, validation           │
└────────────────────────┴────────────────────────────────────┘
```

## Projects

### 1. agent-memory ✅ Complete (v1.0.0)
**Status:** Production-ready
**Location:** `./mem/`

Hierarchical memory system with learning capabilities:
- Episodic memory (store/retrieve experiences)
- Hierarchical tiers (working → short-term → long-term)
- Pattern learning via HDBSCAN clustering
- LLM-based reflection and meta-learning
- Cross-domain adaptation and transfer learning
- Learnable domain markers
- Agent-facing tools API (MemoryTools)
- Health monitoring and metrics

**Key exports:** `MemoryStore`, `MemoryTools`, `StrategySelector`, `Reflector`

---

### 2. agent-tools ✅ Core Complete
**Status:** Core complete — tool definition, registry, executor, shell/system/code tools, memory bridge (115 tests)
**Location:** `./tools/`

Unified abstraction for agent tool use:

**Core Capabilities:**
- Tool definition schema (name, description, parameters, returns)
- Tool registry and discovery
- Execution sandboxing and permissions
- Result validation and error handling
- Retry logic and fallbacks

**Built-in Tool Categories:**
- **System:** Run OS commands, file operations, environment
- **Code:** Execute Python/JS, REPL sessions, syntax validation
- **Web:** HTTP requests, web scraping, API calls
- **Memory:** Interface to agent-memory (store, recall, reflect)
- **Communication:** User interaction, notifications

**Design Decisions (Resolved):**

1. **Permissions: Per-tool granularity**
   - Each tool declares its own permission requirements
   - Allows fine-grained control without over-complicating categories
   - Tools can be individually enabled/disabled/sandboxed
   - DANGEROUS tools (shell, code exec) are never auto-registered
   - Command execution uses template-based approach (`subprocess(shell=False)`)
     with fixed command structures — no string filtering, no injection possible

2. **Execution: Sync by default, async-capable design**
   - Synchronous execution as the default (simpler mental model)
   - Interface designed to support async variants later
   - Tool definition includes `async_capable: bool` flag
   - Both `execute()` and `execute_async()` methods available internally

3. **Long-running tools: Async with process monitoring**
   - Long-running tools use async execution
   - Process health monitoring (is it still alive?)
   - Support for streaming output as it arrives
   - Timeout handling without blocking the agent

4. **Tool composition: Start simple, agent-decides strategy**
   - Initial implementation: simple sequential execution
   - Architecture leaves space for both pipelines and chaining
   - The agent should decide which composition strategy to use based on:
     - Input/output size and shape
     - Tool characteristics (streaming vs batch)
     - Task requirements
   - Avoid premature abstraction; let patterns emerge from usage

---

### 3. agent-mind (Goals + Planning + Introspection Complete)
**Status:** Goals, planning, introspection done (118 tests)
**Location:** `./mind/`

Self-direction, introspection, and metacognition:

**Core Capabilities:**
- **Goal Stack:** Push/pop goals, goal decomposition, priority management
- **Self-Prompting:** Generate own prompts, ask clarifying questions
- **Introspection:** Examine own state, capabilities, limitations
- **Curiosity:** Identify knowledge gaps, exploration strategies
- **Metacognitive Prompting:** "What do I know?", "What should I try?", "Why did that fail?"
- **Planning:** Multi-step plan generation, plan revision

**Introspection Triggers (Resolved):**
Introspection should be a mix of automatic and opportunistic:

1. **Event-triggered (automatic):**
   - On significant success (score > threshold): "Why did this work so well?"
   - On significant failure (score < threshold): "What caused this to fail?"
   - Threshold-based to avoid crowding statistics with noise
   - Maps to: `Reflector.reflect_on_success/failure()` in agent-memory

2. **Idle-triggered (opportunistic):**
   - "I have downtime, let me consolidate my recent experiences"
   - Batch reflection on clusters, not individual episodes
   - Pattern discovery across accumulated experiences
   - Maps to: `Reflector.discover_patterns()`, `ConsolidationEngine` in agent-memory

3. **Self-prompted (curiosity-driven):**
   - Agent decides to introspect as part of exploration
   - "I keep failing at X, let me deeply analyze why"
   - More deliberate, goal-directed reflection

**Exploration vs Exploitation (Resolved):**
Not a binary choice, but a deliberate parallel evaluation:

1. **Exploitation path:** Break down problem with known approaches, generate tested solutions
2. **Exploration path:** "What tools/approaches exist that I don't know about?"
3. **Tabulate candidates:** Keep both paths' solutions available for comparison
4. **Multi-criteria evaluation:**
   - Immediate value: Does it solve the problem now?
   - Future impact: How does this affect flexibility/maintainability?
   - Risk: What's the cost of failure?
   - Goal stability: How likely is the goal to change? (stable goals → safer to exploit)
5. **Decide with reasoning:** Explicit justification, not just gut feel
6. **Track decision:** For future learning (why did this choice work/fail?)

This is the "expert pattern" - not paralysis, not impulsiveness, but measured deliberation.
May be slower, but the thinking time is valuable. Produces better decisions and learnable outcomes.

**Planning Responsibility (Resolved):**
Planning is split between agent-mind and agent-patterns:
- **agent-mind** owns the *Planner* — the cognitive capability that generates plans
  - Breaking down goals into steps
  - Sequencing, anticipating obstacles
  - Revising plans when context changes
  - Deciding *what* to do before doing it
- **agent-patterns** owns the *Plan Executor* — the loop that runs plans
  - Stepping through the plan
  - Checking results after each step
  - Triggering re-planning when steps fail
  - Knowing when the plan is complete

The planner thinks, the executor acts.

**Goal Stack (Resolved):**
Use a **tree with priority** structure:
- **Tree structure:** Goals decompose into sub-goals, forming a hierarchy
  - Root: top-level objective
  - Children: sub-goals needed to achieve parent
  - Leaves: actionable tasks
- **Priority:** Each node has priority, determines which branch to work on
- **Operations:**
  - Push goal (from user, self-generated, or decomposition)
  - Complete goal (mark done, propagate up if all siblings done)
  - Suspend goal (pause, revisit later)
  - Decompose goal (create children)
  - Reprioritize (shift focus)
  - Abandon goal (no longer relevant)
- **Goal states:** pending, active, blocked, completed, abandoned
- **Tracking:** Why blocked? What unblocks? Progress on long goals?

**Acting vs Reflecting Balance (Resolved):**
Use **heuristic triggers with learnable thresholds**:

- **Detection mechanisms** (via working memory):
  - Track current goal, recent actions, outcomes
  - Classify each action result: "progress" / "no change" / "failure"
  - Detect patterns: consecutive failures, stalled progress, repetition

- **Initial heuristics** (fixed, ship something simple):
  - 3+ consecutive failures → trigger reflection
  - 5+ actions with "no change" → trigger reflection
  - Agent can explicitly say "I'm uncertain" → trigger reflection

- **Designed to become learnable** (SOUL_LEARNABLE):
  - Thresholds become tunable by the agent
  - Agent proposes changes with reasoning: "I reflected 20 times but only 2 were useful"
  - User reviews and approves before changes take effect
  - Track outcomes to validate if new settings improve performance

- **Immutable guardrails** (SOUL_IMMUTABLE):
  - `min_actions_between_reflections: 3` — prevent over-reflection
  - `max_failures_before_forced_reflect: 10` — must eventually reflect
  - Bounds prevent pathological patterns (never/always reflect)

**Efficient Exploration/Exploitation (Resolved):**
Use **familiarity + stakes gating** to avoid over-analysis:

- **Familiarity gating** (via memory):
  - Situation closely matches past experience → skip exploration, exploit known approach
  - Novel/unfamiliar situation → full parallel evaluation justified
  - Memory similarity scores drive this decision

- **Stakes gating:**
  - Low-stakes decision → quick heuristic, act fast
  - High-stakes decision → full evaluation worth the time
  - Factors: reversibility (easy to undo → act fast), impact, cost of failure

- **Combined logic:**
  ```
  if familiar AND low_stakes → exploit immediately
  if familiar AND high_stakes → exploit with verification
  if unfamiliar AND low_stakes → quick exploration
  if unfamiliar AND high_stakes → full parallel evaluation
  ```

- **Cached evaluations:**
  - Store decision patterns: "For situation type X, exploration was valuable"
  - Reuse insights to speed up future decisions

---

### 4. agent-patterns (Phase 2 Complete)
**Status:** Phase 2 done — Pattern ABC, events, ReactLoop, PlanExecutor, PlanAndExecute (71 tests)
**Location:** `./patterns/`

Reusable agent execution patterns:

**Core Patterns:**
- **ReAct Loop:** Reason → Act → Observe → Repeat
- **Plan-and-Execute:** Generate plan → Execute steps → Revise if needed
- **Goal-Oriented:** Define goal → Decompose → Achieve subgoals → Verify
- **Reflexion:** Attempt → Reflect on failure → Retry with insights
- **Tree of Thoughts:** Explore multiple reasoning paths

**Key Components:**
- Loop abstractions (step, terminate conditions, max iterations)
- State management between steps
- Observation parsing and integration
- Failure detection and recovery

**Pattern Composability (Resolved):**
Start with **nesting**, design for **full composition** later:

- **Phase 1 — Nesting:**
  - Patterns can invoke other patterns as sub-routines
  - E.g., Plan-and-Execute where each step uses ReAct
  - Simple, covers most practical cases

- **Phase 2 — Pattern Combinators** (future vision):
  Inspired by parser combinators — patterns as composable primitives with implicit backtracking:

  | Combinator | Meaning | Example |
  |------------|---------|---------|
  | `sequence` (A >> B) | Do A then B | `plan >> execute` |
  | `alternate` (A \| B) | Try A, if fail try B | `react \| plan_and_execute` |
  | `many` (A*) | Repeat until condition | `many(react_step, until=done)` |
  | `optional` (A?) | Try A, ok if doesn't apply | `optional(reflect)` |
  | `backtrack` | Undo actions, try alternative | On failure, rewind state |

  Example compositions:
  ```python
  # Try ReAct, if stuck switch to planning
  resilient = react_loop | (plan_and_execute >> react_loop)

  # Goal-oriented with reflexion on each sub-goal
  learning_goal = goal_oriented(reflexion(react_loop))

  # Keep trying with reflection until success
  persistent = many(reflexion(react_loop), until=success, max=3)
  ```

- **Interface requirements** (to enable future composition):
  - Patterns are functions: `input → (result | failure)`
  - Standard signals: success, failure (with reason), partial progress
  - Failure reasons enable alternation
  - State snapshots enable backtracking

- **Phase 3 — Graph-based execution engine** (inspired by LangGraph, noted 2026-05-16):
  LangGraph's primitive API treats each reasoning step (prompt generation, tool call,
  result parsing, state update) as an explicit DAG node. Key capabilities worth adopting:
  - **Parallel node execution** — concurrent sub-tasks without hard-coding a monolithic loop
  - **Dynamic branching** — if/else paths as explicit graph edges, not nested conditionals
  - **Per-node state scoping** — state can be scoped per node or per session, keeping
    orchestration and storage decoupled (maps cleanly to our memory bridge)
  - **Hooks per node** — plug in callbacks (logging, memory writes) without touching
    pattern logic

  This extends the combinator vision: instead of composing patterns linearly, the graph
  engine schedules them as nodes with declared dependencies. The `sequence`, `alternate`,
  and `many` combinators become syntactic sugar over graph edges.

  **Prerequisite:** Phase 2 combinators should be in place first — the graph engine is
  a more general substrate that subsumes them.

**Pattern Selection (Resolved):**
Use a **layered approach** with three levels of override:

1. **User override (highest priority):**
   - User can explicitly specify: "Use Plan-and-Execute for this"
   - Always respected, no questions asked

2. **Agent-decided (for novel/complex situations):**
   - Agent-mind can override the default when it has reason to
   - Uses familiarity + stakes gating: only meta-decide when worth the overhead
   - "This debugging task keeps failing with ReAct, let me try Reflexion"
   - Requires pattern competence — learns over time which patterns suit which situations

3. **Heuristics (default/fast path):**
   - Simple rules based on task characteristics:
     - Simple, clear steps → ReAct
     - Complex, multi-step → Plan-and-Execute
     - Previous failures on this task → Reflexion
   - Bootstrap for agent-decided (avoids circular "how to think about thinking")
   - Used for familiar/low-stakes situations to skip meta-overhead
   - **Informed by learning:** Heuristics evolve based on experience (see below)

4. **Learned refinement (influences heuristics):**
   - Track pattern success rates per situation type in memory
   - Over time, heuristics become personalized: "For *my* debugging tasks, Reflexion works best"
   - Agent can propose heuristic changes: "I've noticed Plan-and-Execute fails for small tasks"
   - Fits with SOUL_LEARNABLE architecture — user approves significant changes

**Interface with agent-mind (Resolved):**
Use **event-based communication with shared context**:

- **Shared context:**
  - Both layers read/write to a shared context object
  - Contains: current goal, plan, progress, observations, world model
  - Single source of truth for current state

- **Events from patterns → mind:**
  - `ActionCompleted(action, result)` — for updating world model
  - `StepFailed(step, reason)` — for learning, possible re-planning
  - `Stuck(attempts, last_actions)` — triggers reflection
  - `PatternComplete(outcome)` — goal achieved or abandoned

- **Events from mind → patterns:**
  - `Reflect(trigger)` — pause execution, reflect now
  - `Replan(reason)` — current plan invalid, generate new one
  - `Reprioritize(new_goal)` — switch focus
  - `Abort(reason)` — stop current pattern

- **Benefits:**
  - Loose coupling — layers don't call each other directly
  - Flexible — easy to add new event types
  - Testable — can mock events for testing
  - Observable — events can be logged for debugging

---

### 5. agent-core (Phase 1 + Multi-agent Complete)
**Status:** Phase 1 done + multi-agent Phase 1 (spawn + mailbox) — 143 tests
**Location:** `./core/`

The binding layer - agent identity, orchestration, and safety:

**Core Capabilities:**
- **Identity/Soul:** Agent name, personality, core values, behavioral guidelines
- **Orchestration:** Wire together memory, tools, mind, patterns
- **Context Management:** Manage context window, summarization, attention
- **Prompt Assembly:** Construct system prompts, inject context, format outputs
- **Safety/Guardrails:** Constraints, forbidden actions, ethical boundaries
- **Lifecycle:** Initialize, run, pause, resume, shutdown

**Design Decisions (Resolved):**

1. **Prompt construction: Hybrid approach**
   - Components provide structured data (not prompt text)
   - agent-core transforms that data into actual prompt text
   - Clear separation of concerns:
     - Components own *what* to communicate (memories, tool descriptions, goals)
     - Core owns *how* to say it (formatting, model-specific tokens, context limits)
   - Benefits: modular components, consistent formatting, easier to swap LLM backends

2. **Agent abstraction: Role + Instance model**
   - **AgentRole:** The template — name, values, personality, base capabilities ("education")
   - **AgentInstance:** A specific agent with its own persistent state (memories, learned patterns)
   - Two agents with same role but different memories = two instances of the same role
     (like two software engineers with same education but different work experience)
   - **Lifecycle:** Long-lived, start/stop, retains experience across sessions
   - **Persistence:** Full state save/restore, versioning supported (snapshot on significant events)
   - **Cloning:** Fork an instance at a point in time (like git branch), clone diverges independently
   - **Knowledge transfer:** Create new agent seeded with memories from multiple sources
     - Not "merging agents" but combining experiences into a new instance
     - **Lineage attribution:** Transferred memories retain their origin ("learned from art-expert lineage")
     - Enables agents to understand where their knowledge came from
     - Original agents continue independently, unaffected
   - This model aligns with Option C: spec (role) + stateful runtime (instance)

3. **Values: Two-layer soul model**
   - **Immutable Core (SOUL_IMMUTABLE):** Fundamental principles the agent can never modify
     - Ethical constraints (honesty, safety, privacy)
     - Core identity and purpose
     - System-enforced boundaries
     - User edits manually when needed
   - **Learnable Layer (SOUL_LEARNABLE):** Preferences that can evolve with experience
     - Operational guidelines (how to approach tasks)
     - Communication style (verbosity, technical depth)
     - Task priorities and strategies
   - **Change process for learnable layer:**
     - Agent proposes changes based on experience ("I notice X approach works better")
     - User reviews the proposal
     - User approves/rejects before changes take effect
     - All modifications logged for transparency
   - **Practical structure:**
     ```
     SOUL.md              # Full specification (reference)
     SOUL_IMMUTABLE.md    # Core principles (agent never touches)
     SOUL_LEARNABLE.md    # Agent can propose changes (with approval)
     ```
   - This gives: safety (core can't self-modify) + adaptability (learns from experience) + oversight (human approves)

4. **Multi-agent communication: Layered approach**

   **Phase 1 (Initial implementation):**
   - **Spawn (hierarchical delegation):**
     - Agent A spawns Agent B for a specific task
     - B runs, returns result, terminates (or returns to pool)
     - Clear parent-child relationship, simple ownership
     - Covers ~80% of multi-agent use cases
   - **Mailbox (async peer messaging):**
     - Agents have mailboxes (message queues)
     - Send messages by role ("software-engineer") or name ("alice")
     - Recipients process when ready, reply to sender's mailbox
     - Decoupled, agents work at their own pace
     - Messages persist until read

   **Phase 2 (First expansion):**
   - **Blackboard (shared workspace):**
     - Agents read/write to shared space
     - Communication through shared artifacts
     - Good for collaborative work (building something together)
     - Needs: conflict resolution, versioning

   **Phase 3 (Later expansion):**
   - **Pipe/Channel (real-time sync):**
     - Direct bidirectional stream between agents
     - Both agents "online" simultaneously
     - Real-time back-and-forth dialogue
     - Good for: pair debugging, live collaboration

---

## Additional Components (Resolved)

### Communication/IO → Part of agent-core (configurable)
- User interaction patterns (chat, commands, rich output)
- Multi-agent messaging (covered by spawn/mailbox/blackboard patterns)
- External system integration
- Output formatting and rendering

Communication is how the agent interfaces with the world — that's orchestration.
Make it configurable so different frontends can plug in.

### Observability → Hybrid approach
- **Basic (no dependencies):** Python stdlib `logging` with shared conventions
  - Each project is fully standalone
  - Structured logging, consistent format
- **Advanced (optional `agent-observability` package):**
  - Distributed tracing, spans (OpenTelemetry)
  - Performance metrics, dashboards
  - Enhances all projects when installed, not required

This keeps individual packages independent while enabling powerful observability when needed.

### Evaluation/Testing → Standalone project (`agent-eval/`)
- Agent benchmarking
- Regression testing for behaviors
- Scenario simulation
- A/B testing for strategies

**Packaging:** `agent-eval` depends on other packages, not the reverse.
Individual packages don't know about it — no impact on their independence.

### Context/Attention → Split between agent-mind and agent-core
Managing the LLM's limited context window:

- **Agent-mind provides importance signals:**
  - "This memory is highly relevant to current goal"
  - Semantic importance scores, relevance rankings
  - Decides *what matters*

- **Agent-core does assembly:**
  - Takes importance signals + token budget
  - Decides what fits, applies summarization
  - Builds the actual prompt
  - Handles *mechanical constraints*

**Concrete responsibilities:**
| Concern | Owner |
|---------|-------|
| "Is this memory relevant?" | agent-mind |
| "How many tokens left?" | agent-core |
| "Summarize old conversation" | agent-core (calls LLM) |
| "What's my current priority?" | agent-mind |
| "What tool descriptions to include?" | agent-core (based on mind's signals) |

---

## Shared Principles

Lessons learned from agent-memory that apply to all projects:

### Technology
- **Local-first:** llama.cpp for LLM, no external API dependencies required
- **Python 3.10+:** Type hints everywhere
- **SQLite:** For structured persistence (with JSON support)
- **ChromaDB:** For vector storage (when needed)

### Architecture
- **Dependency Injection:** No global state, explicit dependencies
- **Testability:** Mock interfaces for all external dependencies (MockLLM pattern)
- **Dataclasses:** For structured data, with `to_dict()`/`from_dict()` methods
- **Single schema file:** `schema.sql` per project, rebuild during development

### Core Design Principle: Separation & Overridability
**Keep concerns separated and decoupled. Every component should be easy to override and update.**

- **At project level:** Each project is independently installable and usable
- **At component level:** Each part of a project can be swapped without affecting others
- **Interfaces over implementations:** Depend on abstractions, not concrete classes
- **Sensible defaults, full override:** Everything works out of the box, but can be customized
- **No hidden coupling:** If A depends on B, it's explicit and documented

This enables:
- Swapping implementations (different LLM backend, different storage)
- Testing in isolation (mock any dependency)
- Incremental adoption (use just agent-memory, add others later)
- Future evolution (replace a component without rewriting everything)

### Development Workflow
- **Incremental phases:** Each phase builds on previous, always functional
- **Tests first:** Comprehensive test coverage, catch issues early
- **CLI for debugging:** Every project gets a CLI for exploration
- **Documentation:** CLAUDE.md per project, keep it updated

### Code Quality
- Type hints on all functions
- Docstrings for public APIs
- Explicit error handling
- Structured logging

---

## Project Dependencies

```
agent-core
├── agent-patterns
│   └── agent-mind
│       ├── agent-memory
│       └── agent-tools
├── agent-memory (direct access for orchestration)
└── agent-tools (direct access for orchestration)
```

**Build order (suggested):**
1. ✅ agent-memory (done)
2. ✅ agent-tools (core complete)
3. ✅ agent-mind (goals + planning + introspection done)
4. ✅ agent-patterns (Phase 2 complete)
5. ✅ agent-core (Phase 1 + multi-agent complete)

---

## Monorepo Structure

```
agents/
├── CLAUDE.md              # This file - master roadmap
├── README.md              # Public overview
├── pyproject.toml         # Workspace configuration (if using)
│
├── mem/                   # agent-memory (✅ complete)
│   ├── CLAUDE.md
│   ├── agent_memory/
│   └── tests/
│
├── tools/                 # agent-tools (✅ core complete)
│   ├── CLAUDE.md
│   ├── agent_tools/
│   └── tests/
│
├── mind/                  # agent-mind (✅ goals/planning/introspection)
│   ├── CLAUDE.md
│   ├── agent_mind/
│   └── tests/
│
├── patterns/              # agent-patterns (✅ Phase 2 complete)
│   ├── CLAUDE.md
│   ├── agent_patterns/
│   └── tests/
│
├── core/                  # agent-core (✅ Phase 1 complete)
│   ├── CLAUDE.md
│   ├── agent_core/
│   └── tests/
│
└── shared/                # Common utilities (if needed)
    └── ...
```

---

## Open Questions (Resolved)

### Architectural

1. **Separately installable packages** (not one monolithic package)
   - Use what you need: `pip install agent-memory` works alone
   - Dependencies pull what they need: `pip install agent-core` gets its deps
   - Optional meta-package: `pip install agent-framework` for everything

2. **Semver with documented compatibility** (not lockstep versioning)
   - Each project has its own version (e.g., agent-memory 1.2.0)
   - Dependencies specify ranges: `agent-memory>=1.0,<2.0`
   - Breaking changes = major version bump
   - Compatibility matrix in docs for "tested together" versions

3. **Monorepo with separate packages** (not single package with submodules)
   - Structure: `mem/`, `tools/`, `mind/`, etc. each with own `pyproject.toml`
   - Shared development, independent releases
   - Enables decoupling principle

### Design

1. **Minimal viable agent: memory + tools + ReAct + minimal core**
   - Memory: store/retrieve experiences
   - Tools: execute actions
   - ReAct: reason-act-observe loop (from patterns)
   - Core: just enough wiring to connect them
   - agent-mind optional (can run without metacognition initially)

2. **Agent-core: provide defaults, make everything overridable**
   - Default prompt templates (replaceable)
   - Default pattern: ReAct (swappable)
   - Default context management (customizable)
   - Core is a "wiring layer" not a "behavior dictator"

3. **"Pit of success" design for flexibility + ease of use**
   - Works with zero configuration (sensible defaults)
   - Simple things simple: `agent = Agent(); agent.run("do X")`
   - Complex things possible: full customization available
   - Progressive disclosure: basic API → advanced API → extension points

### Practical

1. **Next project: agent-tools** ✅
   - Foundation for acting in the world
   - Memory done, tools is next building block
   - agent-mind needs tools to be useful

2. **Monorepo migration: Done** ✅
   - Completed via git subtree, history preserved

3. **Shared models: configurable paths with sensible default**
   - Default: `~/.agent-models/` (user-level, shared across projects)
   - Override via: `AGENT_MODELS_PATH` environment variable
   - Per-project config also supported
   - No hardcoded paths in code

---

## Getting Started

### Current State
```bash
# agent-memory is complete and working
cd mem/
python -m agent_memory.memory_cli health
python -m pytest tests/ -v  # 213 tests passing
```

### Next Steps
1. ~~Decide on monorepo structure~~ ✅ Done
2. ~~Migrate agent-memory into monorepo~~ ✅ Done (git subtree, history preserved)
3. ~~Create agent-tools project skeleton~~ ✅ Done
4. ~~Define tool abstraction interface~~ ✅ Done
5. ~~Implement core tools~~ ✅ Done
6. ~~Add tool permissions and sandboxing~~ ✅ Done
7. Next: Real LLM backend, memory integration, remaining phases

---

## Timeline (Rough Estimate)

| Project | Complexity | Estimated Effort |
|---------|------------|------------------|
| agent-memory | ✅ | Done |
| agent-tools | ✅ | Core done |
| agent-mind | ✅ | Goals/planning/introspection done |
| agent-patterns | ✅ | Phase 2 done |
| agent-core | ✅ | Phase 1 done |
| Integration | Medium | 1-2 weeks |

**Total:** ~2-3 months for complete framework

---

## Resources

### Papers & Concepts
- ReAct: Synergizing Reasoning and Acting in Language Models
- Reflexion: Language Agents with Verbal Reinforcement Learning
- Tree of Thoughts: Deliberate Problem Solving with LLMs
- Cognitive Architectures for Language Agents (CoALA)
- AutoGPT, BabyAGI, and related autonomous agent projects

### Technical
- llama.cpp documentation
- LangChain/LlamaIndex (for patterns, not dependencies)
- OpenAI function calling spec (for tool schema inspiration)

---

## Backlog

### Persona-collapse detector — verbatim self-repetition (noted 2026-08-26)

Observed in the Ada↔Pipin "Voice of Discordia" experiment
(`experiments/pipin-discordia-collapse.md`): when a small model's persona
collapses under contradiction, it stops generating novel output and **replays a
near-identical previous turn verbatim**. That verbatim/near-verbatim loop is a
reliable "the persona has failed / the model is stuck looping its safest output"
signal.

Cheap to detect: cosine similarity (or even normalized-string equality) between
an agent's consecutive **same-role** turns; above a threshold → flag. It's the
conversational analog of the reflection-dedup problem (see below). Uses could
include: surface a background note ("you are repeating yourself — break the
loop"), abandon a stalled conversation, or downweight a persona that keeps
collapsing. Detection is model-agnostic but most useful for small local models
whose contradiction budget is small.

---

### Session handoff — conversational texture across restarts ✅ Complete (2026-08-19)

A third continuity layer beside memory (facts) and soul (identity): the
*narrative texture* of where a session left off (tone, open threads), so a
restart resumes warm instead of cold. At shutdown the agent writes a short
first-person "note to future self"; at startup it's injected into the system
prompt. See `assistant/assistant/session_handoff.py`.

- **File-backed markdown append-log, not SQLite** — the note's value is being
  human-legible / hand-editable / git-diffable, and it belongs in the
  soul-adjacent layer, not the queryable data store. Writes are once-per-session
  single-writer, so a DB buys nothing. The log doubles as a dated diary of the
  agent's session-to-session drift (feeds the agent-summary report idea below).
- **Separate from episodic memory** — never written to the store; recall/dedup
  untouched.
- **Opt-in, non-disruptive** — `session_handoff: true` (default off; agents with
  a soul-level handoff protocol leave it unset and are unaffected). Overridable:
  `handoff_file`, `handoff_prompt` (own voice), `handoff_max_entries`.
- Robust: atomic writes, graceful cold start, `make_note` never raises (a failed
  handoff can't break shutdown); captured *before* exit-compaction clears the
  buffer. Startup restore is a quiet system-prompt slot (agent resumes warm; a
  console/TUI line notes it fired).

Contrast worth remembering: a coding agent's continuation summary optimizes
*fidelity* (verbatim, file states, exact next step); this optimizes *texture*
(warm, first-person, lossy) — same architectural slot, inverted priorities.

---

### Query-focused summarization tool ✅ Complete (2026-08-25)

Shipped as the `digest(source, focus)` assistant tool
(`assistant/assistant/tools.py`). `source` is a URL or raw text; `focus` is the
question to summarise toward. On a URL it fetches internally via
`_fetch_full_text` (the untruncated fetch `fetch_readable` was refactored to
share) so raw content never lands in the conversation buffer. Small docs fold in
one pass; large ones map-reduce (each chunk scored against `focus`, then
reduced) through the compaction-task model (`digest_llm=router.for_task(
"compaction")`). Safety cap of 12 chunks — the tail is dropped *and reported*,
never silent. Falls back to plain truncation if the LLM is unavailable. Original
design note preserved below.

---

### Query-focused summarization tool (noted 2026-08-03)

Web/Reddit-heavy tasks (e.g. "find jobs, then search Reddit for red flags on the
companies") pull large documents that blow up the conversation buffer, forcing
early lossy compaction that summarizes away the very details the agent is trying
to correlate. Truncating `fetch_readable` output is the crude fix; the right fix
is **query-focused summarization** — fold a large document down *through the lens
of a question*, keeping only what's relevant.

**A separate tool, not a `fetch_readable` param.** `fetch_readable` stays the
cheap, deterministic, LLM-free primitive ("give me the whole readable text",
Pi-friendly, minimal deps). The new tool owns the LLM concern:
- Signature: `digest(url_or_text, focus)` — accepts a **URL or raw text**.
- **Must fetch internally when given a URL** — the whole point is that the raw
  content never lands in the conversation buffer. A text-only tool would force
  the agent to `fetch_readable` first (raw → buffer, blowup) then summarize,
  defeating it.
- Returns only the focused synthesis, plus a note that the raw is available.

**Technique** (for docs larger than the window): map-reduce or refine, with the
`focus` question injected into the map prompt so each chunk is scored for
relevance to the question — then reduce. Fold through the fast/summarizer model
already wired in the router (gpt-oss-120b). Skip full RAG: transient web content
read once doesn't earn a vector index. Keep plain truncation as the cheap
fallback for small overflows.

**Cost note:** map-reduce over a big thread is several extra LLM calls per fetch
— fine for overnight autonomous runs (not latency-bound, quality matters), less
so for interactive use where the raw/truncate fast path should win.

Rationale for separate-vs-param: single responsibility, honest cost profile (the
model/logs can tell cheap calls from expensive ones), minimal deps on the fetch
primitive, and composability (can digest text the agent already has, not just
URLs) — matches the framework's "composable primitives / no hidden coupling"
ethos. See also the `max_chars` sizing note: buffer compacts at ~8k tokens while
gpt-oss-120b holds ~131k, so raising `max_chars` (config, no code) is the
immediate lever; this tool is the structural follow-up.

---

### Agent summary / status report tool (noted 2026-05-16)

A standalone CLI script (`agent_summary.py`) that reads an agent's full state
from outside and produces a human-readable report. Useful for reviewing what
an agent has been doing, how it has drifted from its original soul, and what's
currently on its mind.

**What it pulls together:**
- Soul (`soul.txt` + `soul_learned.txt` side by side — shows drift)
- Memory (recent episodes, synthesized patterns from reflection)
- Journal (last N entries, grouped by topic/tag)
- Research agenda (active topics, cycle counts, last finding per topic)
- TODOs (pending and in-progress)
- Pending messages (unread inner voice queue)
- Soul proposals (pending and recently approved/rejected)

**Output:** structured markdown dump, then one LLM call for a narrative summary:
*"Who is this agent right now, what have they been working on, what's on their
mind, how have they changed since their soul was written?"*

**Usage:**
```bash
python agent_summary.py --config souls/agent_smith.yaml
python agent_summary.py --config souls/agent_smith.yaml --no-llm  # dump only, no narrative
```

More interesting once agents have been running for a while and have something
to summarize. Good candidate for a weekly cron job output.

---

### Reflection deduplication — pattern_discovery (noted 2026-05-17)

Reflection fires periodically and finds the same pool of memories, generating
semantically identical pattern_discovery entries over and over. Observed in
practice: 11 reflections all saying "curiosity-driven research, varied topics,
moderate success, lacks depth" in slightly different words.

**Root cause:** no "have I already noticed this?" check before writing. The
reflection window is also too short relative to the curiosity interval, so it
keeps rediscovering the same small pool of fresh memories.

**Fixes needed (in agent-memory):**

1. **Deduplication before storing** — before writing a new pattern_discovery,
   retrieve semantically similar existing patterns (vector search). If similarity
   exceeds a threshold (e.g. 0.85 cosine), skip or merge rather than append.

2. **Minimum novelty gate** — only store a reflection if it adds something not
   already captured. Could be as simple as: "does this insight appear in any
   existing pattern_discovery from the last N days?"

**Short-term workaround:** set `reflect_interval` significantly larger than
`curiosity_interval`. Rule of thumb: reflect_interval ≥ 4× curiosity_interval
so there's enough varied signal to find a genuinely new pattern.

---

### Multi-agent spawn + mailbox ✅ Complete (2026-06-04)

Both primitives are implemented and wired into the assistant layer.
See `core/agent_core/spawn.py`, `core/agent_core/mailbox.py`, and `assistant/SETUP.md → Multi-agent`.

**Future: dynamic role discovery**
- `query_available_roles()` tool — agent asks a role registry what specialist agents exist
- Useful once many souls exist and agents need to self-organize without being pre-configured
- Prerequisite: a shared role registry (SQLite or directory scan of `souls/`)

---

### Post-turn critic call for soul deference (noted 2026-06-03)

Agent Smith exhibits RLHF-trained deference: ends turns with "What would you like to do
next?" instead of deciding and acting autonomously. NudgeMonitor catches it retrospectively
(every 15 min); a synchronous critic would catch it immediately and force justification.

**Design (Option B):**
After each assistant turn, run a second LLM call with the same soul but critic framing,
scoped to `## Self-monitoring` criteria. If it flags deference or soul drift, inject the
critique as a system note before the next user turn begins. Trigger only when the agent
produced output without taking a tool action (the "idle turn" case that most often contains
the deference pattern).

**Implementation sketch:**
- `assistant/critic.py`: `CriticPass` class, similar structure to `NudgeMonitor` but
  synchronous — called inline in `_agent_loop` after the final assistant message
- Reads `## Self-monitoring` from soul (same `_parse_self_monitoring` helper)
- One LLM call per idle turn; skips turns that ended with tool use (agent was acting)
- If flagged: `buffer.add_background_note(f"[Self-check] {critique}")` — visible next turn
- Config key: `critic_pass: true` (off by default)

**Relationship to NudgeMonitor:** complementary — nudge handles cross-session patterns,
critic catches in-session deference immediately. Long-term, critic flags feed NudgeMonitor
with richer signal (pattern confirmed by both → stronger nudge message).

**Longer-term:** true dual-agent (Smith-Actor + Smith-Critic sharing a soul, communicating
via PendingMessages/SQLite) — fits the framework's multi-agent mailbox design.

---

## Distant Future Work

### Automatic memory extraction (Mem0-style, noted 2026-05-16)

Currently memory is populated via explicit `save_note` calls or periodic
`reflect_on_recent()`. Mem0 runs a lightweight LLM pass on every conversation
turn and extracts what's worth remembering automatically.

The noise risk is real (dilution of important memories, near-duplicate pollution)
but our architecture is better positioned to handle it than Mem0's flat store:
HDBSCAN clustering collapses near-duplicates into patterns, hierarchical tiers
filter what reaches long-term, and reflection already does batch extraction
at coarser granularity.

A sensible middle path when this becomes worth building:
- Extract only from **assistant turns that follow tool use** (decisions made,
  facts discovered, preferences revealed) — skip pure conversational turns
- Run a lightweight significance pre-filter before writing (similar to InnerVoice
  but cheaper — no full evaluation, just a quick relevance check)
- Lean on HDBSCAN + reflection to handle residual duplicates

**Prerequisite:** the clustering and reflection pipeline should be battle-tested
first. Flooding it with automatic extractions before the noise-handling is solid
would degrade cluster quality.

---

## Notes

- This is a learning project - understanding > speed
- Each component should be useful standalone
- Avoid over-engineering - build what's needed
- Test with real tasks, not just unit tests

---

**Last Updated:** 2026-02-07
**Status:** All projects Phase 1+ complete. Next: real LLM backend, memory integration, remaining phases.
**Developers:** Nikos & Claude
