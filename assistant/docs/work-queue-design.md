# Design note: unified work queue for the autonomous loop

*Status: proposal — decisions flagged inline for Nikos before implementation.*
*2026-08-26*

## Problem

The work cycle currently has **three** ways work reaches it, wired ad-hoc:

1. **Reactive:** `prompt_queue` (scheduled tasks / injected user prompts), drained
   every ~5s in `_run`.
2. **Proactive rotation:** `_pick_goal` cycles `todos → research → interests →
   dream`, interval-gated.
3. **A special case:** conversation turns are checked every tick and *preempt*
   the rotation (`_goal_from_conversations` ahead of `_pick_goal`).

Every new source of work needs bespoke wiring, and "conversation preempts
everything" is a hardcoded exception rather than a general rule. This unifies all
three into one priority-ordered stream, so adding a source is declaring a
producer + a priority, not editing the loop.

## Shape

```python
@dataclass(order=True)
class WorkItem:
    priority: int                 # lower = more urgent (heapq order)
    seq: int                      # tiebreak: FIFO within a priority
    kind: str = field(compare=False)      # "user" | "conversation" | "mailbox" |
                                          # "scheduled" | "todos" | "research" | ...
    payload: dict = field(compare=False)  # what the handler needs
    key: str = field(compare=False)       # idempotency key (dedup)
```

- **Producers** put items on the queue. A producer either *pushes* (user prompt,
  scheduler firing) or is *polled* each tick (bus: conversation your-turn,
  mailbox unread). Polling producers are the "poll the bus periodically and
  enqueue" idea.
- **Consumer** = the work cycle. Each tick it pops the highest-priority item and
  dispatches on `kind` to the matching handler (`_goal_from_<kind>` today — a
  dict `{kind: handler}` registry, i.e. the multimethod).

Producers and consumers decouple *noticing* work from *doing* it — the part
currently tangled inside `_run`.

## Two-tier priority (the core decision)

Not everything belongs in the same ordering discipline:

- **Urgent tier — strict priority, deterministic.** Something/someone is blocked.
  No randomness here: a lottery that occasionally ignores a waiting peer is a bug.
  ```
  user prompt  <  conversation your-turn  <  mailbox  <  scheduled task
  ```
  (`<` = more urgent). These always preempt the background tier.

- **Background tier — weighted lottery.** Self-directed work with no deadline.
  When the urgent tier is empty, pick one via
  `random.choices(sources, weights=significance)`. No source starves (nonzero
  ticket → eventually picked), significance biases without hard-ordering, and the
  "arbitrary long wait" downside is harmless here (nothing is blocked on a dream
  cycle). This is the same pattern the curiosity engine already uses for
  `strategy_weights` — consistent, not novel.
  ```
  todos, research, interests, dream   (weighted random among themselves)
  ```

**Decisions (resolved 2026-08-27, Nikos):**

1. **`todos` → background, high weight.** A commitment, but no deadline — so it
   competes in the lottery with a heavy ticket (chores dominate, never fully
   starve creativity).
2. **Scheduled task preempts conversation your-turn.** (Nikos' call, over the
   your-turn lean.) So the **urgent tier order** (lower = served first) is:

   | kind | priority |
   |------|----------|
   | `user` (live prompt) | 10 |
   | `scheduled` | 20 |
   | `conversation` your-turn | 30 |
   | `mailbox` | 40 |

   *Implication to keep in mind:* a long scheduled task can make a blocked peer
   wait. Acceptable for now; if it bites, a scheduled task could yield to a
   your-turn when the peer has been waiting > N.

3. **Background weights: sensible defaults now → config next → `SOUL_LEARNABLE`
   later.** Starting defaults (config-overridable):

   | kind | weight |
   |------|--------|
   | `todos` | 3.0 |
   | `research` | 1.5 |
   | `interests` | 1.0 |
   | `dream` | 0.5 |

## Idempotency

Polling producers will re-see the same item until it's handled — so producers
must not enqueue a duplicate. Guard with `key` (e.g.
`conversation:2:turn:pipin`): skip if a live item with that key is already
queued or in-flight. Cleared when the item completes.

## Persistence — queue as a materialized view

The queue is **in-memory and rebuildable**, not a source of truth. On startup it
re-polls the durable sources (conversation bus DB, `TaskScheduler`, `TodoDB`) and
repopulates. Only genuinely ephemeral **user prompts** may need their own
persistence (as `prompt_queue`/`TaskScheduler` already do). So a restart loses
nothing that mattered.

## Learnable weights

The background-tier significance weights are a natural `SOUL_LEARNABLE` knob: the
agent can *propose* adjusting its own chore-vs-daydream balance ("I've done todos
90% of cycles and my research stalled — raise research's ticket"), reviewed and
approved like any soul change. Starvation policy becomes part of the agent's
evolving temperament rather than a fixed constant.

## Migration (incremental, low-blast-radius)

1. Add `WorkItem` + a `WorkQueue` (heap for urgent, weighted-draw for background)
   as a standalone unit **with tests** — no loop changes yet.
2. Rewrite `_run` to: poll producers → drain one item → dispatch. The existing
   `_goal_from_*` methods become the handler registry (mostly unchanged bodies).
3. Delete the conversation special-case and the `_pick_goal` rotation — both are
   now just producers + priorities.
4. Verify against a real daemon run (the failure modes here only show up live).

Step 1 is safe and self-contained; steps 2–3 are the core-loop change and want
the verify skill + a live daemon check before landing.

## What this is, in one line

Turn "don't starve creativity" from a scheduler hack into a *personality
parameter*, and turn "add a work source" from editing the loop into declaring a
producer.
