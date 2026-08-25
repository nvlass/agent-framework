# agent-framework

A **local-first toolkit for building autonomous AI agents with memory** — written
to be *read*. Each layer is a small, standalone Python package you can use on its
own or wire together into a complete agent.

This is not trying to out-feature LangChain or LlamaIndex. It's a different thing:
a compact, legible codebase where you can follow every layer end to end — how an
agent remembers, reflects, plans, acts, and coordinates with other agents —
without wading through abstraction. If you're learning how agents actually work
under the hood, or you want a hackable base you fully understand, this is for you.

Local-first by design: it runs against a local `llama.cpp` server, and optionally
against cloud APIs (Fireworks, Anthropic, xAI/Grok) — the same code, swap the
backend. It runs on modest hardware, down to a Raspberry Pi.

## The layers

```
┌─────────────────────────────────────────────────────────────┐
│                    agent-core  (core/)                       │
│        Identity/soul, orchestration, multi-agent, safety     │
├─────────────────────────────────────────────────────────────┤
│                 agent-patterns  (patterns/)                  │
│          ReAct, Plan-and-Execute, Reflexion loops            │
├─────────────────────────────────────────────────────────────┤
│                   agent-mind  (mind/)                        │
│        Goals, planning, introspection, metacognition         │
├────────────────────────────┬────────────────────────────────┤
│    agent-memory  (mem/)     │      agent-tools  (tools/)      │
│  Episodic memory, vector    │   Tool definition, registry,    │
│  recall, reflection,        │   sandboxed execution,          │
│  learning, consolidation    │   permissions                   │
└────────────────────────────┴────────────────────────────────┘
```

Each package is independently installable and separately tested. Depend on just
`agent-memory` if that's all you need, or `agent-core` to pull the whole stack.

| Package | What it does | Tests |
|---------|--------------|-------|
| `mem/` — **agent-memory** | Episodic store (SQLite + vectors), reflection, HDBSCAN consolidation, cross-domain analogy | 213 |
| `tools/` — **agent-tools** | Tool schema, registry, sandboxed executor, per-tool permissions | 115 |
| `mind/` — **agent-mind** | Goal stack, planner, introspection triggers | 118 |
| `patterns/` — **agent-patterns** | ReAct, Plan-and-Execute, Reflexion — composable loops | 71 |
| `core/` — **agent-core** | Soul/identity, prompt assembly, orchestration, spawn + mailbox multi-agent | 143 |

## Two worked examples

- **`assistant/`** — a conversational personal assistant (CLI + Textual TUI) that
  remembers across sessions, researches on its own (curiosity + work-cycle
  daemons), reflects on what it learns, and can act autonomously. Configured with
  a YAML file and a plain-text "soul". This is where the framework is exercised
  most heavily.
- **`examples/news-agent/`** — a smaller agent that scans news sources and
  composes a daily digest. A good first read: one agent, a handful of tools.

## Quick start

Requires Python 3.11+.

```bash
# Use just the memory layer
cd mem && pip install -e ".[dev]" && python -m pytest tests/ -v
python -m agent_memory.memory_cli health

# Or run the assistant example
cd assistant && pip install -e .
cp assistant.yaml my-agent.yaml        # edit: model, soul, tools
python main.py --config my-agent.yaml  # CLI
python tui.py  --config my-agent.yaml  # Textual TUI
```

The assistant talks to an LLM backend. Point it at a local `llama.cpp` server, or
set an API key for a cloud provider (`FIREWORKS_API_KEY`, `ANTHROPIC_API_KEY`, or
`XAI_API_KEY`) and name the model in your YAML — the provider is auto-detected
from the model id. See `assistant/assistant.yaml` for the fully-commented config
template and `assistant/SETUP.md` for setup details.

## Design principles

- **Separation & overridability** — every component swaps out cleanly; sensible
  defaults, full override. Depend on interfaces, not implementations.
- **Local-first** — no cloud dependency required; runs on a Pi.
- **Legibility over cleverness** — read the code, understand the agent.
- **Test-backed** — ~660 tests across the packages; mock LLMs so logic is testable
  without a model.

Per-package design notes and rationale live in each directory's `CLAUDE.md`.

## Status

All layers are at Phase 1+ and working, with the assistant example in active use.
It's a learning project as much as a tool — understanding over feature-count. Rough
edges exist and are documented honestly in the roadmap (`CLAUDE.md`).

## License

MIT — see [LICENSE](./LICENSE).
