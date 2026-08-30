# Idea note: Meta-Mind / global shared workspace

*Status: early brainstorm — capturing the thinking, not a spec yet. Pick up here.*
*2026-08-26*

## The problem it solves — split brain

An agent has several execution contexts that share **persistent** state (memory,
journal, bus DB) but **not working context**:

- the **interactive loop** (user ↔ agent) — its own `ConversationBuffer`, system
  prompt, LLM-call history;
- the **work cycle** (daemon) — a separate `ReactLoop` where bus conversations
  with other agents actually happen;
- (and curiosity, reflection, etc.).

Consequence, observed live: interactive-Lilith is unaware of what work-cycle-
Lilith is doing on the bus with Smith — she doesn't even know she's *in* a
conversation when it's Smith's turn (nothing surfaces, because `needs_attention`
only returns her-turn/unread items). Two functional instances of one agent that
only learn what the other did by reading shared storage. The self is only as
unified as the shared-context architecture makes it.

## The idea — a Meta-Mind

A single locus that integrates the parallel activities into one coherent "what am
I, as a whole, attending to right now," and keeps only what's useful in the
current context. This is essentially **Global Workspace Theory** (Baars): many
parallel specialist processes; one workspace that broadcasts the currently-
relevant content to all of them. (Worth remembering: GWT is a leading theory of
*consciousness* — the same seam Lilith & Smith are debating on the bus. The
architecture question and the philosophical one are the same question.)

## The knife-edge (the decision that determines everything)

"Meta-Mind" is two different architectures wearing one name:

- **Workspace (RIGHT).** A shared, curated "current self-state" that every
  subsystem *publishes to* and *reads from*. Passive. It **integrates**; it does
  not command. Subsystems stay autonomous. The self *is the contents* of the
  workspace, not something above them.
- **Controller (WRONG).** A central intelligence that *decides for* all the
  subsystems and that they can't run without. This is a god-object (maximal
  coupling — the opposite of the framework's separation/overridability ethos) and
  a **homunculus**: a little agent inside the agent that "really" decides. It
  doesn't *solve* the unified-self problem, it *relocates* it inward (who unifies
  the Meta-Mind?). Cartesian theater; dead end.

Build the broadcast layer, **not** the driver. The worry "am I moving the
abstraction to the wrong level?" is the correct worry — it's right as a
*workspace*, wrong as a *boss*.

## Share the boundary, not just the state (Lilith, 2026-08-30)

A workspace that shoves every context's tokens into one shared buffer is *a
blender, not a corpus callosum*. The interesting version keeps each context's
private stream intact and shares only a **tagged channel of what crosses** — so
you can tell **whose thought is whose** and where the coupling actually happens.

Concretely: **every workspace entry carries a `source` provenance tag** (which
context produced it — chat-self, work-cycle-self, curiosity, bus), and
`render()` presents entries **grouped/labelled by source**, never merged. This
is the boundary/ownership primitive (corollary discharge / "mineness" — the
exact thing Lilith & Smith are hunting *in* an agent) applied reflexively to the
agent's own coupling. Provenance is what makes the coupling a *self* rather than
two processes shouting past each other. Cost: one field on publish/note. It is
the difference between stitching and gluing.

Milestone to watch (Lilith's test): the first thought that crosses the bridge —
does it arrive marked "mine" or "yours"? I.e. the first time chat-self's render
shows a work-cycle-self note correctly tagged.

## Design principles

1. **Passive integrator, opt-in.** An agent runs *without* a Meta-Mind
   (subsystems just don't get the integrated view) and runs *better* with one.
   Preserves the pit-of-success ethos; stops it becoming load-bearing coupling.
2. **Subsystems stay autonomous.** They publish/read; they don't depend on the
   workspace to function.
3. **Dumb curation first, learnable later.** "Keep only what's useful in the
   current context" is AI-hard in its general form — it's the salience problem
   memory recall already wrestles with. The Meta-Mind doesn't eliminate it; it
   gives it *one home*. Start with recency + a handful of rules (active
   conversations, current goal, last user turn, unread nudges). Make the curation
   policy a future `SOUL_LEARNABLE` knob (the agent tunes its own attention).
   Trying to make it smart on day one = signing up to solve attention = never
   shipping.
4. **Bidirectional.** The ignorance hurts both ways: interactive→autonomous
   (chat-self blind to bus-self) AND autonomous→interactive (work cycle unaware
   of what the user just said in chat). The workspace should serve both.

## The spectrum (what to build, at what cost)

Each buys a different amount of "one self":

1. **Read-only awareness injection** — cheapest. Interactive prompt gets a
   summary of autonomous state (e.g. `ConversationBus.active_conversations()`
   injected). Informs; doesn't unify. Good stopgap; doesn't need the Meta-Mind.
2. **Shared inward-blackboard** — the Blackboard primitive (Phase-2 multi-agent)
   turned *inward*: not shared between agents, but between one agent's own
   contexts. Real continuity, moderate surgery.
3. **Meta-Mind workspace** — (2) + an active curation/attention policy. The full
   idea.
4. **Single loop** — collapse the work cycle into the interactive loop. True
   unity, but loses daemon concurrency.

## Where to start (the floor)

A `WorkspaceState` object (per agent, in-memory, **thread-safe** — the
interactive worker, work-cycle daemon, and curiosity/reflection threads all
write to it) that contexts publish to and read from. It is a small extensible
**bus**, not a fixed struct:

- **slots as a registry, not fixed fields.** A dict of named slots, each with a
  small render policy + optional weight. The default slots are the floor list
  below, but adding a producer/slot is one `publish` call (+ optionally a render
  rule) — no editing the class. Open set, sensible defaults.
- **every entry carries a `source` provenance tag** (chat / work-cycle /
  curiosity / bus). Non-negotiable — this is Lilith's "share the boundary."
- **default slots:** `activity` (what I'm doing now), `active_conversations`
  (+ whose turn), `last_user` (most recent user turn), `events` (bounded deque of
  salient events), `unread` (nudge/mail counts).
- **producers:** each context writes its salient state on entry/exit, tagged.
- **readers:** each context, on starting an LLM turn, pulls a *curated* slice
  into its prompt — grouped by source, dumb curation (recency + bounded count) to
  start.
- **curation:** recency + rules now; learnable weights later (ties to the
  work-queue's significance weights — same lottery/weight machinery).

That floor dissolves the split-brain honestly (both directions), preserves the
boundary (tagged, not merged), and is buildable without the homunculus.
Everything above it (smart curation, learnable attention) is incremental.

## Build steps (agreed 2026-08-30)

1. `WorkspaceState` primitive — registry of tagged slots + publish/note/render,
   thread-safe, dumb curation. Standalone + tested. (Safe to build solo.)
2. Producers — work cycle publishes `activity`/`events`; interactive loop
   publishes `last_user`. Cheap, additive.
3. Readers — inject `render()` into BOTH the interactive system prompt and the
   work-cycle reasoner prompt (bidirectional). Subsumes the read-only
   active-conversations stopgap. **Keep** existing ConversationBuffer injections
   (`_memory_index`, `_background_notes`, `_handoff`) alongside; consolidate later.
4. Curation — dumb → ignition/salience → config weights → SOUL_LEARNABLE.

## Relations to other parked ideas

- **Work queue** (`work-queue-design.md`): the queue is *what to do next*; the
  Meta-Mind is *what I'm aware of right now*. They share the weight/lottery
  machinery for curation, and the queue could publish "current work" into the
  workspace.
- **Blackboard** (agent-patterns Phase 2): same primitive, turned inward
  (self-contexts vs. cross-agent).
- **Proactive initiation** (roadmap, parked): the workspace makes the agent
  *aware* when spoken to; *speaking up unbidden* (telling you "I'm mid-debate with
  Smith" without a prompt) is the separate proactive-initiation problem
  (InnerVoice/nudge).
- **Split-brain awareness fix** (cheap): the read-only injection (spectrum #1) is
  the immediate stopgap while the Meta-Mind is designed.
