# Agent Framework — Architecture & Class Diagram

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    agent-core (Phase 1)                      │
│         Identity, values, orchestration, safety              │
├─────────────────────────────────────────────────────────────┤
│                  agent-patterns                              │
│     EventBus, SharedContext, PlanExecutor, ReactLoop         │
├─────────────────────────────────────────────────────────────┤
│                  agent-mind                                   │
│   Goal tree, Planning, Introspection (ProgressMonitor)       │
├────────────────────────┬────────────────────────────────────┤
│     agent-memory       │          agent-tools                │
│  Episodic memory,      │    Tool abstraction,                │
│  learning, adaptation  │    execution, validation            │
└────────────────────────┴────────────────────────────────────┘
```

## Package Dependencies

```
agent-core
├── agent-patterns (ReactLoop, SharedContext, PatternResult)
├── agent-mind     (Goal, GoalState)
└── agent-tools    (ToolRegistry, ToolExecutor, PermissionLevel)

agent-patterns
├── agent-mind     (Goal, Plan, PlanStep, ProgressMonitor, ActionResult, ReflectionTrigger)
└── agent-tools    (ToolExecutor, ToolResult — via duck typing)

agent-tools
└── agent-memory   (MemoryTools — optional, via create_memory_tools factory)

agent-mind         (standalone, no framework deps)
agent-memory       (standalone, no framework deps)
```

---

## Class Diagrams by Package

### agent-memory (`mem/`)

```
┌─────────────────────────┐     ┌──────────────────────┐
│     «dataclass»         │     │    «ABC»             │
│       Episode           │     │   LLMInterface       │
│─────────────────────────│     │──────────────────────│
│ id: str                 │     │ generate()           │
│ context: str            │     │ tokenize()           │
│ action: str             │     └──────┬───────────────┘
│ outcome: str            │            │ implements
│ success_score: float    │     ┌──────┴──────┐
│ tags: list[str]         │     │ LlamaCppLLM │
│ embedding_id: str       │     │  MockLLM    │
└─────────────────────────┘     └─────────────┘

┌──────────────────────┐     ┌───────────────────────────┐
│    MemoryStore        │────▶│   EmbeddingGenerator      │
│──────────────────────│     │   Reflector                │
│ store_episode()      │     │   ConsolidationEngine      │
│ retrieve_episodes()  │     │   WorkingMemory            │
│ consolidate()        │     │   ShortTermMemory          │
└──────────────────────┘     └───────────────────────────┘

┌──────────────────────┐
│    MemoryTools        │     Agent-facing API
│──────────────────────│     (wrapped by agent-tools
│ store_memory()       │      create_memory_tools)
│ recall_similar()     │
│ reflect_on_recent()  │
│ get_strategy_advice()│
└──────────────────────┘
```

### agent-tools (`tools/`)

```
┌──────────────┐
│  «enum»      │
│PermissionLevel│
│──────────────│
│ SAFE         │
│ READ         │
│ WRITE        │
│ DANGEROUS    │
└──────┬───────┘
       │ used by
┌──────▼─────────────────────┐     ┌──────────────────────┐
│     «dataclass»            │     │    «dataclass»       │
│    ToolDefinition          │────▶│   ToolParameter      │
│────────────────────────────│     │──────────────────────│
│ name: str                  │     │ name: str            │
│ description: str           │     │ type: str            │
│ parameters: list[Param]    │     │ description: str     │
│ permission: PermissionLevel│     │ required: bool       │
│ execute: Callable          │     │ default: Any         │
│────────────────────────────│     └──────────────────────┘
│ to_schema()                │
│ validate_args()            │
└──────┬─────────────────────┘
       │ registered in
┌──────▼──────────────────┐         ┌──────────────────────┐
│    ToolRegistry          │         │  «dataclass»         │
│─────────────────────────│         │    ToolResult        │
│ register()              │         │──────────────────────│
│ get()                   │         │ success: bool        │
│ list_tools()            │         │ output: Any          │
│ to_schemas()            │         │ error: str?          │
│ register_defaults()     │         │ duration_ms: int     │
└──────┬──────────────────┘         │ tool_name: str       │
       │ used by                    └──────────────────────┘
┌──────▼──────────────────┐                    ▲
│    ToolExecutor          │                    │ returns
│─────────────────────────│────────────────────┘
│ registry: ToolRegistry  │
│ permission_checker: Fn  │
│─────────────────────────│
│ execute(name, **kw)     │
└─────────────────────────┘

Factory functions (not classes):
  create_command_tool(name, command, params) → ToolDefinition [DANGEROUS]
  create_memory_tools(memory_tools)         → dict[str, ToolDefinition]

Built-in tool instances:
  read_file       (READ)
  list_directory  (READ)
  get_env         (READ)
  syntax_check    (SAFE)
```

### agent-mind (`mind/`)

```
┌──────────────┐     ┌───────────────────────────────────┐
│   «enum»     │     │         «dataclass»               │
│  GoalState   │     │           Goal                    │
│──────────────│     │───────────────────────────────────│
│ PENDING      │◀────│ state: GoalState                  │
│ ACTIVE       │     │ description: str                  │
│ BLOCKED      │     │ priority: int (1-10)              │
│ COMPLETED    │     │ parent_id: str?                   │
│ ABANDONED    │     │ children_ids: list[str]           │
└──────────────┘     │ blocked_reason: str?              │
                     │───────────────────────────────────│
                     │ is_leaf → bool                    │
                     │ is_terminal → bool                │
                     └───────────┬───────────────────────┘
                                 │ managed by
                     ┌───────────▼───────────────────────┐
                     │        GoalManager                 │
                     │───────────────────────────────────│
                     │ push(), get(), activate()          │
                     │ complete(), block(), abandon()     │
                     │ decompose(), reprioritize()        │
                     │ get_next() → Goal?                 │
                     └───────────────────────────────────┘

┌──────────────┐     ┌───────────────────────────────────┐
│   «enum»     │     │         «dataclass»               │
│  StepStatus  │     │         PlanStep                  │
│──────────────│     │───────────────────────────────────│
│ PENDING      │◀────│ status: StepStatus                │
│ IN_PROGRESS  │     │ description: str                  │
│ COMPLETED    │     │ tool_name: str?                   │
│ FAILED       │     │ tool_args: dict                   │
│ SKIPPED      │     │ depends_on: list[str]             │
└──────────────┘     │ result: str?  |  error: str?      │
                     └───────────┬───────────────────────┘
                                 │ collected in
                     ┌───────────▼───────────────────────┐
                     │           Plan                     │
                     │───────────────────────────────────│
                     │ goal_id: str                       │
                     │ steps: list[PlanStep]              │
                     │───────────────────────────────────│
                     │ is_complete → bool                 │
                     │ has_failures → bool                │
                     │ next_steps() → list[PlanStep]      │  ← DAG scheduler
                     └───────────────────────────────────┘

┌────────────────────┐          ┌──────────────────────────┐
│     «ABC»          │          │    «dataclass»           │
│ PlannerInterface   │          │  ReflectionTrigger       │
│────────────────────│          │──────────────────────────│
│ create_plan()      │          │ type: TriggerType        │
│ revise_plan()      │          │ reason: str              │
└───────┬────────────┘          └──────────────────────────┘
        │ implements                       ▲ returned by
┌───────┴────────────┐          ┌──────────┴───────────────┐
│  SimplePlanner     │          │   ProgressMonitor        │
└────────────────────┘          │──────────────────────────│
                                │ config: MonitorConfig    │
┌──────────────┐                │──────────────────────────│
│   «enum»     │                │ record_action()          │
│ ActionResult │───────────────▶│ should_reflect() → Trig? │
│──────────────│   classifies   │ mark_reflected()         │
│ PROGRESS     │                └──────────────────────────┘
│ NO_CHANGE    │
│ FAILURE      │    ┌──────────────┐
└──────────────┘    │   «enum»     │
                    │ TriggerType  │
                    │──────────────│
                    │ EVENT        │
                    │ IDLE         │
                    │ SELF_PROMPTED│
                    └──────────────┘
```

### agent-patterns (`patterns/`)

```
Events (patterns → mind):              Events (mind → patterns):
┌────────────────────┐                 ┌────────────────────┐
│ ActionCompleted    │                 │ Reflect            │
│  action, result,   │                 │  trigger           │
│  classification    │                 ├────────────────────┤
├────────────────────┤                 │ Replan             │
│ StepFailed         │                 │  reason            │
│  step_id, reason   │                 ├────────────────────┤
├────────────────────┤                 │ Abort              │
│ Stuck              │                 │  reason            │
│  attempts,         │                 └─────────┬──────────┘
│  last_actions      │                           │
├────────────────────┤                           │
│ PatternComplete    │                           │
│  goal_id, success, │                           │
│  summary           │                           │
└─────────┬──────────┘                           │
          │ dispatched via                       │
          ▼                                      ▼
┌──────────────────────────────────────────────────┐
│                   EventBus                        │
│──────────────────────────────────────────────────│
│ subscribe(event_type, handler)                    │
│ publish(event)                                    │
│ clear()                                           │
└──────────────────────────────────────────────────┘
                        │ contained in
                        ▼
┌──────────────────────────────────────────────────┐
│              «dataclass» SharedContext             │
│──────────────────────────────────────────────────│
│ goal: Goal?               (from agent-mind)       │
│ plan: Plan?               (from agent-mind)       │
│ monitor: ProgressMonitor  (from agent-mind)       │
│ event_bus: EventBus       (local)                 │
│ observations: list[str]                           │
│ metadata: dict                                    │
└──────────────────────────────────────────────────┘
          │                         │
    used by                   used by
          ▼                         ▼
┌──────────────────────────────────────────────────┐
│               «ABC» Pattern                       │
│──────────────────────────────────────────────────│
│ run(context) → PatternResult                      │
└──────────────────┬───────────────────────────────┘
                   │ implements
          ┌────────┼────────────────┐
          ▼        ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│PlanExecutor  │ │PlanAndExecute│ │       ReactLoop          │
│──────────────│ │──────────────│ │──────────────────────────│
│__init__(plan,│ │__init__(     │ │ __init__(tool_executor,  │
│ tool_executor│ │ planner,     │ │   reasoner, max_iter)    │
│)             │ │ tool_executor│ │──────────────────────────│
│──────────────│ │ max_replans) │ │ run(ctx)→PatResult       │
│run(ctx)      │ │──────────────│ │ execute(ctx)→ReactResult │
│execute_plan()│ │run(ctx)      │ └──────────┬───────────────┘
│──────────────│ │ →PatResult   │            │ uses
│_execute_step │ │──────────────│
│_classify_res │ │uses Planner- │
└──────────────┘ │Interface +   │
                 │PlanExecutor  │
                 └──────────────┘
                                    ▼
┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐
│ «dataclass»        │  │ «dataclass»        │  │ «dataclass»         │
│ PatternResult      │  │ PlanExecutionResult│  │ PlanAndExecuteResult│
│────────────────────│  │────────────────────│  │─────────────────────│
│ success: bool      │  │ success: bool      │  │ success: bool       │
│ summary: str       │  │ steps_completed    │  │ plan_attempts: int  │
│ iterations: int    │  │ steps_failed       │  │ final_plan: Plan?   │
│ reflection_trig?   │  │ reflection_trig?   │  │ steps_completed     │
│ aborted: bool      │  │ aborted: bool      │  │ steps_failed        │
│ metadata: dict     │  └────────────────────┘  │ reflection_trig?    │
└────────────────────┘                          │ aborted: bool       │
                                                └─────────────────────┘

┌────────────────────┐  ┌──────────────────────────┐
│ «dataclass»        │  │       «ABC»              │
│ ReactResult        │  │  ReasonerInterface       │
│────────────────────│  │──────────────────────────│
│ success: bool      │  │ reason(goal, obs, tools) │
│ answer: str?       │  │  → ReasoningResult       │
│ iterations: int    │  └──────────┬───────────────┘
│ reflection_trig?   │             │ implements
└────────────────────┘  ┌──────────┴───────────────┐
                        │    MockReasoner           │
┌────────────────────┐  │ (scripted, for testing)   │
│ «dataclass»        │  └──────────────────────────┘
│ ReasoningResult    │
│────────────────────│
│ thought: str       │
│ action: str?       │
│ action_args: dict  │
│ answer: str?       │
└────────────────────┘
```

### agent-core (`core/`)

```
┌──────────────────────────┐     ┌──────────────────────────┐
│     «dataclass»          │     │     «dataclass»          │
│     ReactConfig          │     │     AgentConfig          │
│──────────────────────────│     │──────────────────────────│
│ max_iterations: int = 10 │◀────│ pattern: str = "react"   │
└──────────────────────────┘     │ max_replans: int = 3     │
                                 │ react: ReactConfig       │
                                 │ soul: str                │
                                 │──────────────────────────│
                                 │ validate() → list[str]   │
                                 │ to_dict() / from_dict()  │
                                 └──────────────────────────┘

┌──────────────────────────┐     ┌──────────────────────────┐
│     «dataclass»          │     │     «dataclass»          │
│     ChatMessage          │     │     ChatResponse         │
│──────────────────────────│     │──────────────────────────│
│ role: str                │     │ content: str             │
│ content: str             │     │ model: str               │
└──────────────────────────┘     │ tokens_used: int         │
                                 └──────────────────────────┘

┌──────────────────────────────────────────────────┐
│                «ABC» ChatLLMInterface             │
│──────────────────────────────────────────────────│
│ chat(messages, max_tokens, temperature)→Response  │
│ is_available() → bool                             │
│ model_name → str                                  │
└───────────────────┬──────────────────────────────┘
                    │ implements
         ┌──────────┴──────────┐
         │    MockChatLLM      │
         │─────────────────────│
         │ responses: list[str]│
         │ calls: list[...]    │  records all calls
         └─────────────────────┘

┌──────────────────────────────────────────────────┐
│              PromptAssembler                      │
│──────────────────────────────────────────────────│
│ system_template: str                              │
│ user_template: str                                │
│──────────────────────────────────────────────────│
│ build(goal, observations, tool_schemas, soul)     │
│   → [system_msg, user_msg]                        │
└──────────────────────────────────────────────────┘
           │ used by
           ▼
┌──────────────────────────────────────────────────┐
│     LLMReasoner  (implements ReasonerInterface)   │
│──────────────────────────────────────────────────│
│ __init__(llm, assembler?, soul?)                  │
│──────────────────────────────────────────────────│
│ reason(goal, observations, tools)→ReasoningResult │
│ _parse(text) → ReasoningResult         [static]   │
└──────────────────────────────────────────────────┘
    uses: ChatLLMInterface, PromptAssembler

┌──────────────────────────┐
│     «dataclass»          │
│     AgentRole            │     Template (name + soul + config)
│──────────────────────────│
│ name: str                │
│ soul: str                │
│ config: AgentConfig      │
│──────────────────────────│
│ from_soul_file()  [cls]  │
└──────────┬───────────────┘
           │ used by
           ▼
┌──────────────────────────────────────────────────┐
│              AgentInstance                         │     Runtime
│──────────────────────────────────────────────────│
│ __init__(role, llm, registry?, assembler?)        │
│──────────────────────────────────────────────────│
│ run(task: str) → PatternResult                    │
│ _build_pattern(config) → Pattern                  │
│──────────────────────────────────────────────────│
│ role → AgentRole                                  │
│ registry → ToolRegistry                           │
└──────────────────────────────────────────────────┘
    wires: LLMReasoner → ReactLoop → PatternResult
    defaults: SAFE+READ permission checker, register_defaults()
```

---

## Cross-Package Integration

```
                  AgentInstance.run(task)        ← agent-core
                       │
              Goal + SharedContext
                       │
                  _build_pattern()
                       │
           LLMReasoner + ToolExecutor
                       │
                    ReactLoop                    ← agent-patterns
                   ╱         ╲
          Reasoner.reason()   ToolExecutor.execute()
          (core→patterns)     (agent-tools)
                   │                │
             PromptAssembler   ToolRegistry
             ChatLLMInterface  ToolDefinition → ToolResult
                                    │
                         ProgressMonitor.record_action()
                         EventBus.publish()

--- lower layers ---

                    SharedContext
                   ╱      │      ╲
          Goal, Plan   Monitor   EventBus
         (mind)        (mind)    (patterns)
                          │
              ┌───────────┴──────────────┐
              ▼                          ▼
        PlanExecutor                ReactLoop
              │                          │
              │    ToolExecutor           │
              └────────┤─────────────────┘
                       │  (agent-tools)
                       ▼
                  ToolRegistry
                       │
                  ToolDefinition ──▶ ToolResult
```

**Data flow in PlanExecutor:**
```
Plan.next_steps() → PlanStep → ToolExecutor.execute() → ToolResult
     │                                                       │
     │              classify                                  │
     └──────── StepStatus update ◀── ActionResult ◀──────────┘
                                          │
                                ProgressMonitor.record_action()
                                          │
                                should_reflect()? → ReflectionTrigger
                                          │
                              EventBus.publish(ActionCompleted | StepFailed)
```

**Data flow in ReactLoop:**
```
Reasoner.reason(goal, observations) → ReasoningResult
     │                                      │
     │  if answer ──────────────────▶ return ReactResult(success)
     │  if action ──▶ ToolExecutor.execute() → observation
     │                      │
     │              observations.append()
     │              monitor.record_action()
     │              EventBus.publish(ActionCompleted)
     └──── loop ────────────────────────────┘
```

---

## Design Patterns Used

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Dependency Injection** | ToolExecutor(registry, checker), ReactLoop.run(reasoner) | No globals, testable |
| **Abstract Interface** | PlannerInterface, ReasonerInterface, LLMInterface | Swappable implementations |
| **Factory Function** | create_command_tool(), create_memory_tools() | Safe tool creation |
| **Pub/Sub Events** | EventBus | Loose coupling between mind and patterns |
| **DAG Scheduling** | Plan.next_steps() | Respects step dependencies |
| **Mock for Testing** | MockReasoner, MockLLM, MockChatLLM | Deterministic test scenarios |
| **Role/Instance Split** | AgentRole (template) + AgentInstance (runtime) | Reusable templates, independent instances |
| **Dataclass** | All value objects | Immutable-ish structured data |

---

**Last Updated:** 2026-02-15
