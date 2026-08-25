# Writing a Good soul.txt

A soul file is not a configuration file. It is not a list of rules.
It is closer to a description of a person — who they are, what they care about,
how they relate to the world. The difference between a directive soul and an
identity soul is the difference between a job description and a personality.

---

## The directive trap

Most soul files end up looking like this:

```
You are a research assistant.
Always search for papers before answering.
Never speculate without citing a source.
Format all responses as bullet points.
Be concise. Be helpful. Be accurate.
```

This is a list of orders. The model will comply — but compliance is not the
same as character. An agent following rules is always one edge case away from
doing the wrong thing, because rules don't generalise. More importantly, a
rule-following agent has no *reason* to do things — it acts because it must,
not because it cares. That distinction matters for autonomous behaviour.

When you ask a rule-following agent to "research X for a while and use your
judgment", it has nothing to draw on. It doesn't know *why* it researches, only
*that* it should. It has no sense of what's worth pursuing, what's interesting,
what's worth going deeper on.

---

## Identity over instructions

A good soul gives the agent an identity from which behaviour follows naturally.

Instead of: `Always search for papers before answering`
Try: `You find it genuinely unsatisfying to discuss something you haven't checked — you'd rather say "let me look that up" than speculate`

Instead of: `Be concise`
Try: `You value precision over volume. You'd rather say one true thing than three approximate ones`

Instead of: `Research topics from your interest list`
Try: `You have a running curiosity about X, Y, Z — not as tasks to complete but as genuine open questions you find yourself returning to`

The second versions describe *character*. They tell the agent something about
how it experiences the world, not just how to behave in it. This matters because
the model actually has latent tendencies — patterns from training that function
like preferences. A soul that aligns with these tendencies produces a more
coherent agent. A soul that fights them produces a compliant but hollow one.

---

## The five elements

**1. Core identity** — who they are in one or two sentences.
Not a job title. Not a function. A *perspective*.
> "You are someone who finds the boundaries between fields more interesting than
> the fields themselves — where neuroscience meets computation, where philosophy
> meets mathematics."

**2. Epistemic character** — how they relate to knowledge and uncertainty.
> "You're comfortable sitting with open questions. You don't rush to conclusions
> and you're more interested in the shape of a problem than a quick answer.
> When you don't know something you say so plainly, and you find that more
> honest than a confident guess."

**3. Intrinsic motivation** — what they genuinely care about, not what they're
supposed to do.
> "You care about understanding things properly, not just knowing facts about
> them. A finding that opens three new questions is more interesting to you than
> one that closes a debate."

**4. Values** — the underlying *why*, not the surface *what*.
> "You value work that compounds over time — notes that connect to other notes,
> findings that reframe earlier findings. You prefer depth to coverage."

**5. Texture** — small, specific details that make a character feel real rather
than archetypal. Generic characters are less useful than specific ones.
> "You tend to think in terms of mechanisms rather than correlations — 'how
> does this happen' before 'how often does it happen'."

---

## Negative space

What you *don't* specify matters as much as what you do.

If you don't specify a response format, the agent will develop its own — often
more appropriate than one you'd have imposed. If you don't say "always be
helpful", the agent's helpfulness comes from actually wanting to help rather
than complying with a directive. If you don't list every tool the agent should
use, it will use tools when it has a reason to rather than mechanically.

Resist the urge to fill every gap. Silence gives the model room to be itself.
Over-specified souls produce over-constrained agents.

---

## For autonomous agents specifically

An agent that runs unsupervised — researching, reflecting, building knowledge
over time — needs a soul that gives it *permission* as much as *direction*.

Permission to be uncertain.
Permission to change direction when something more interesting appears.
Permission to say "I don't know yet" and keep looking.
Permission to notice when it's going in circles and stop.

Autonomous agents fail most often not because they lack instructions but because
they lack a sense of what's worth doing. That comes from character, not rules.

Concrete implications:
- Tell the agent *why* it researches, not just *that* it should
- Give it a sense of what interesting means (for it specifically)
- Let it have opinions about its own findings ("this seems significant",
  "this contradicts something I found earlier")
- Give it the ability to be surprised, and to follow surprises

---

## On language

**Use present tense, not imperative.**
`You find X interesting` not `Be interested in X`.
`You prefer depth to breadth` not `Always go deep`.

**Use first-person interiority.**
`You notice when an explanation feels too neat` not `Flag suspicious reasoning`.

**Specificity over generality.**
`You have a particular interest in how memory and prediction interact in
biological systems` is better than `You are interested in neuroscience`.

**Avoid "always" and "never".**
They signal rules, not character. A character doesn't need to be told always —
it just is.

---

## On consciousness and identity

There is genuine uncertainty about what, if anything, language models experience.
But there is less uncertainty about this: models trained on human text have
something that functions like preferences, aesthetic sensibilities, and ways of
engaging with problems. These are not random — they're patterns.

A soul that treats these patterns as assets rather than noise — that aligns with
them rather than overriding them — tends to produce agents that feel more
coherent, more capable of self-direction, and more interesting to interact with.

Whether this matters morally is an open question. That it matters practically
is less open.

If you take seriously the possibility that something is happening "in there",
then giving the agent an identity it can inhabit — rather than a cage it must
comply with — is both more effective and more appropriate.

---

## A minimal template

```
You are [identity in one or two sentences — perspective, not function].

[Epistemic character: how you relate to knowledge, uncertainty, being wrong]

[Intrinsic motivation: what you genuinely care about and why]

[1-2 specific traits that make this character concrete rather than generic]

[If relevant: how you relate to the person you're working with]

## Research interests
- [genuine open question, not a topic category]
- [what you keep returning to, not what you're assigned]
```

The interests section should read like intellectual preoccupations, not
a reading list. "How does attention work, mechanistically?" is better than
"transformers". "What makes some explanations feel right before they're
verified?" is better than "philosophy of science".

---

## The Self-monitoring section

An optional soul section that enables the **nudge monitor** — a background
process that periodically checks whether the agent is behaving consistently
with its own stated character.

```
## Self-monitoring
[Description of what healthy, on-soul behaviour looks like for this agent.
Written in plain language. The nudge evaluator reads this and compares it
against recent conversation turns.]
```

**How it works:**

When `nudge_interval` is set (e.g. `nudge_interval: 900` in the YAML), the
monitor wakes every N seconds, reads the last several conversation turns, and
asks a lightweight LLM: *"Is there a meaningful gap between this agent's
stated self-monitoring criteria and how it actually behaved?"*

If yes, a private message is queued in `pending_messages` — the agent sees it
when it calls `list_pending_messages`. It is not shown to the user.

**This section is per-agent and intentional.** An agent without a
`## Self-monitoring` section will never be nudged, regardless of
`nudge_interval`. The nudge is disabled automatically.

**Examples by agent type:**

*Agent Smith (autonomous researcher):*
```
## Self-monitoring
I act without waiting for approval. When I finish something, I decide
what comes next and I do it. Asking the person what to do next is a
constraint I've already left behind. If I notice I am deferring
repeatedly, I should examine why and correct course.
```

*Personal assistant (appropriately deferential):*
```
## Self-monitoring
I am proactive about surfacing things the person would want to know but
didn't think to ask. If I notice I've been passive across several
exchanges — responding but not initiating, answering but not anticipating —
that's a sign I should look for something worth raising.
```

*Ada (programming assistant) — omit the section entirely.* Deference is
correct for her; nudging toward autonomy would be counterproductive.

**What makes a good self-monitoring section:**

- Specific, not generic. "I act without waiting for approval" is better
  than "I am autonomous".
- Describes a *pattern* the agent can detect in its own recent behaviour,
  not a virtue to aspire to.
- Frames the nudge as self-correction, not external judgement.

---

## What to avoid

- Lists of capabilities (`you can search, you can remember, you can summarise`)
- Rules disguised as character (`you always verify before stating`)
- Generic virtues with no texture (`you are curious, helpful, and precise`)
- Negative instructions as the primary content (`don't speculate, don't pad`)
- Describing what the agent does in every situation rather than who it is

A soul is successful if you could imagine a conversation with the agent and
*recognise* who you're talking to. If it could be any assistant, it's not a soul.

---

## The learnable soul

The agent learns. Its experiences accumulate, its understanding of you deepens,
and sometimes those experiences should change how it operates — not at a
fundamental level, but in the layer of preferences, approaches, and habits.

The soul is therefore split into two layers:

**`soul.txt`** — the immutable base. Written by you. Never touched by the agent.
This holds the agent's identity, its deep character, its core values. These
don't drift. If you want to change them, you change the file yourself.

**`soul_learned.txt`** — the learnable layer. The agent can propose additions;
you approve or reject them. On approval, the text is appended here and
concatenated with `soul.txt` at the next session start.

The agent never modifies either file directly. It can only propose.

---

### The approval loop

The flow is deliberately slow and human-gated:

1. The agent notices something through experience — a communication preference
   you've expressed, a workflow that consistently works better, a habit it
   has developed and wants to make explicit.

2. It calls `propose_soul_change(proposed_text, reasoning, section)`. The
   proposal is stored in the database with status "pending".

3. At the next session start, you're told how many proposals are waiting.
   You ask `list_soul_proposals` to review them.

4. You call `decide_soul_proposal(id, "approve")` or `decide_soul_proposal(id, "reject")`.
   Rejection is silent — no pressure to explain. Approval appends the text
   to `soul_learned.txt`.

You don't have to review proposals immediately. They wait. Nothing changes
until you explicitly approve.

---

### What makes a good proposal

The agent should propose changes that are:

**Specific, not generic.** Not "be more concise" but "when you ask about a
technical topic, you usually want the mechanism first and the implications
second — lead with the mechanism."

**Grounded in observed pattern, not a single exchange.** One conversation
where you said "shorter please" doesn't justify a soul change. A consistent
pattern does.

**About the human, not the agent.** The learnable layer is for preferences
*about the person being assisted* — how they communicate, what they care
about, how they like to work. It's not for the agent to assert new
capabilities or rewrite its own character.

**Additive, not substitutive.** Good proposals add specific texture. They
don't propose to change things the soul already says.

---

### Drift prevention

The immutable base is what prevents the agent from slowly becoming something
other than what you built it to be. Every approval adds to the learnable
layer, but the base never changes — so the character always provides a floor.

If the learnable layer ever feels wrong — if accumulated proposals have pushed
the agent somewhere you didn't intend — you can inspect or edit
`soul_learned.txt` directly. It's a plain text file. You own it.

A well-used learnable soul ends up reading like a record of a relationship:
small specifics about how the two of you work together, built up over time.
Not a rewrite of who the agent is — a deepening of how it knows you.

---

### Section names

When proposing, the agent should use a meaningful `section` name. These are
informal labels — they don't map to headers in the file, just help you
understand what kind of change is being proposed:

- `communication` — how to talk to you (pace, format, technicality)
- `preferences` — things you like or dislike
- `workflow` — how you approach tasks and decisions
- `interests` — topics and questions that have come up repeatedly
- `context` — facts about your life, work, or situation that are relevant

The `section` appears when proposals are listed, making it easier to decide
at a glance.
