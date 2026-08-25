# agent-core

Agent identity, orchestration, and binding layer. Wires agent-memory, agent-tools, agent-mind, and agent-patterns into a working agent.

## Quick Start

```python
from agent_core import AgentRole, AgentInstance, MockChatLLM

role = AgentRole(name="helper", soul="You are a helpful assistant.")
llm = MockChatLLM(["Thought: I know this\nAnswer: 42"])
agent = AgentInstance(role, llm)
result = agent.run("What is the answer?")
```

### With Memory

```python
from agent_memory import MemoryStore, MemoryTools
from agent_core import AgentRole, AgentInstance, MockChatLLM

store = MemoryStore(db_path="agent.db")
memory = MemoryTools(store=store)
agent = AgentInstance(AgentRole(name="helper"), llm, memory=memory)
result = agent.run("do something")
# Auto-recalls relevant past experiences before execution
# Auto-stores episode after completion
# Agent can also use memory_store/memory_recall/memory_reflect tools mid-run
```

## Architecture

```
AgentRole (template: name, soul, config)
    ↓
AgentInstance (runtime: role + llm + registry + assembler + memory)
    ↓ run(task)
    ├── Auto-recall (if memory) → inject past experiences as observations
    ├── Goal (from agent-mind)
    ├── SharedContext (from agent-patterns)
    ├── LLMReasoner (ChatLLMInterface → ReasonerInterface)
    │   └── PromptAssembler (builds system + user messages)
    ├── ReactLoop (from agent-patterns)
    └── Auto-store (if memory) → save episode after completion
```

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `AgentConfig` / `ReactConfig` | config.py | Dataclass configs with validate/to_dict/from_dict |
| `ChatMessage` / `ChatResponse` | llm.py | Chat LLM data types |
| `ChatLLMInterface` | llm.py | ABC for chat LLMs (separate from memory's LLMInterface) |
| `MockChatLLM` | llm.py | Test double — queued responses, records calls |
| `PromptAssembler` | prompt.py | Builds [system, user] messages from goal/obs/tools/soul |
| `LLMReasoner` | reasoner.py | Bridges ChatLLMInterface → ReasonerInterface |
| `AgentRole` | agent.py | Template (name + soul + config) |
| `AgentInstance` | agent.py | Runtime agent — `run(task) → PatternResult` |
| `SpawnRole` | spawn.py | Role definition (soul, tools, optional LLM) for a child agent |
| `SpawnRegistry` | spawn.py | Registry of named roles; built from `spawn_roles:` YAML section |
| `AgentMailbox` | mailbox.py | SQLite-backed async message bus; WAL mode for cross-process safety |
| `ConversationBus` | conversation.py | Bounded turn-taking dialogue over the shared channel; lifecycle + termination guarantee |

## LLM Response Format

The agent expects LLM to respond in:
```
Thought: <reasoning>
Action: <tool_name>
Action Args: <JSON>
```
Or: `Thought: ... \n Answer: <final answer>`

## Defaults

- Pattern: `react` (only option in Phase 1)
- Permission checker: allows SAFE + READ + WRITE tools (DANGEROUS blocked)
- Registry: `register_defaults()` (read_file, list_directory, get_env, syntax_check)
- Memory: `None` (no persistence by default; pass `memory=MemoryTools` to enable)
- Max iterations: 10

## Multi-agent

Two complementary primitives for agent coordination:

### Spawn — synchronous parent-child delegation

A parent agent calls `spawn_agent(role, task)`. A child `AgentInstance` is created
from the named role, runs the task to completion, returns a result string, then
terminates. The parent blocks until the child is done — semantically a function call.

```python
# In YAML: declare spawn_roles with soul + tool list
# spawn_roles:
#   researcher:
#     soul: souls/researcher.txt
#     tools: [web_search, fetch_readable, save_note]

registry = SpawnRegistry.from_config(cfg["spawn_roles"], soul_base_dir=config_dir,
                                     make_llm_fn=_make_llm)
agent = AgentInstance(role, llm, spawn_registry=registry)
# Agent can now call: spawn_agent(role="researcher", task="summarise X")
```

Child tools are declared in the role definition; the parent can narrow further via
`allow_tools=[...]` at spawn time. Dangerous tools are never auto-inherited.

### Mailbox — async peer-to-peer messaging

Agents communicate via a shared SQLite DB (WAL mode, `busy_timeout=5000` for
cross-process safety). Messages persist until read; neither agent blocks.

```python
mailbox = AgentMailbox(db_path="/shared/agents.db", agent_name="smith")
mailbox.send(to="critic", topic="review", message="Here is my latest draft…")

# On the other side (different process, same DB):
msgs = mailbox.inbox(unread_only=True)
mailbox.reply_to_message(msgs[0]["id"], "Looks good, one concern…")
```

### Conversation — bounded turn-taking dialogue

Where the mailbox is fire-and-forget, `ConversationBus` is a *structured
conversation*: two agents exchange turns on a topic with a lifecycle
(`active → closed`), enforced turn-taking, and a hard termination guarantee —
either an explicit `done` or a `max_turns` cap (the backstop against two
autonomous agents ping-ponging forever). It shares the same physical file as the
mailbox ("the shared channel") and touches nothing else: each agent's memory,
journal, and soul stay private. **Private minds, shared channel** — sharing is
explicit (take a turn), never ambient.

```python
bus = ConversationBus("/shared/agents.db", agent_name="ada")
c = bus.open(peer="smith", message="What's your read on X?", topic="scag", max_turns=6)
# … smith's process, same file:
for c in smith_bus.needs_attention():        # 'your_turn' or 'unread'
    smith_bus.reply(c["id"], "I think Y…")    # atomic turn-claim
```

Fits interval-ticking daemons: no simultaneous liveness. Each tick an agent
calls `needs_attention()`; the turn-claim in `reply()` is atomic (BEGIN
IMMEDIATE) so two daemons on the same tick can't both take the same turn. In the
assistant, exposed as `talk_to` / `talk_reply` / `talk_history` / `talk_check`,
and a pending turn preempts the work-cycle goal rotation (a peer is blocked on
the reply).

**Spawn vs Mailbox vs Conversation at a glance:**

| | Spawn | Mailbox | Conversation |
|---|---|---|---|
| Direction | Parent → Child | Peer ↔ Peer | Peer ↔ Peer |
| Blocking | Yes (request/response) | No (fire-and-forget) | No (turn-taking) |
| Shape | One call, one result | Independent messages | Bounded multi-turn dialogue |
| Terminates | On child return | N/A | `done` or `max_turns` (guaranteed) |
| Good for | Sub-task delegation | Ongoing coordination | Consult / debate / negotiate |

## Testing

```bash
cd core && pip install -e ".[dev]" && python -m pytest tests/ -v
```

143 tests across 6 files.

## Completed

- Phase 1: config, chat LLM interface, prompt assembly, LLM reasoner, agent role/instance, ReAct execution
- Memory integration: auto-recall, auto-store, explicit memory tools (memory_store/recall/reflect)
- Multi-agent Phase 1: SpawnRole/SpawnRegistry (sync delegation) + AgentMailbox (async messaging)

## Deferred

Persistence (save/restore agent state), clone.

## Completed (assistant layer — not in agent-core directly)

- **Two-layer soul:** `soul.txt` (immutable, human-written) + `soul_learned.txt` (agent-proposed, human-approved). Managed by `SoulManager` in `assistant/assistant/soul_manager.py`. Proposals stored in `assistant_data.db`, surfaced at session start, approved/rejected via `propose_soul_change` / `decide_soul_proposal` tools.
