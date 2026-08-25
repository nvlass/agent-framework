"""Curiosity engine — autonomous background research with introspective depth.

Each research cycle:
  1. Picks a topic: focused agenda items first (ResearchAgenda), then soul
     interests (round-robin).
  2. Rate-gates so we never hammer search APIs.
  3. Pulls prior journal findings on the topic (grounding — "what do I already know?").
  4. Every PONDER_EVERY cycles on the same topic: runs an introspective ponder —
     the LLM reflects on accumulated findings and generates a more targeted next
     query rather than repeating the bare topic name.
  5. Searches DuckDuckGo (or arXiv via the ponder query if it looks academic).
  6. Synthesises the results *grounded by prior knowledge* — focusing on what is
     NEW relative to what is already known.
  7. Saves finding + reflection (on ponder cycles) to journal and memory.
  8. Calls on_finding(topic, summary) callback.

Requires ``ddgs`` (or ``duckduckgo-search``). Without it the engine does nothing.
"""

import json
import logging
import re
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

# Never search more often than this regardless of interval setting.
_MIN_SEARCH_GAP = 300  # 5 minutes

# Phrases that indicate the model refused the request rather than answering.
# Includes apostrophe-free variants (some models strip apostrophes: "I'm" → "I_m").
_REFUSAL_MARKERS = (
    "i'm sorry", "i cannot", "i can't", "i am not able", "i won't", "i will not",
    "i_m sorry", "can_t comply", "won_t", "cannot assist", "comply with that",
)


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first JSON object from raw text using brace-depth scanning.

    More robust than a regex: handles curly braces inside string values
    (e.g. LaTeX notation like {Slater determinants}) and leading/trailing prose.
    """
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None

# How many exploration cycles on the same topic before doing an introspective ponder.
_PONDER_EVERY = 3


@dataclass
class FormulationConfig:
    """All knobs for the question-formulation step.

    Attributes:
        temperature:          LLM temperature for question formulation (higher = more varied).
        journal_dropout:      Probability of omitting prior journal findings entirely.
        cross_topic_seed_prob: Probability of injecting a finding from a different topic.
        bravery_prob:         Probability of the minimal "bravery" prompt (no structure).
        go_crazy_prob:        Probability of the meta-prompt mode (model writes its own prompt).
                              Should be ≤ bravery_prob; the die is rolled once and these
                              are non-overlapping tiers (go_crazy < bravery < structured).
        strategy_weights:     Relative likelihood of each strategy in structured mode.
    """
    temperature: float = 0.85
    journal_dropout: float = 0.20
    cross_topic_seed_prob: float = 0.25
    bravery_prob: float = 0.15
    go_crazy_prob: float = 0.05
    structured_only: bool = False      # if True, always use structured mode (disables bravery + go_crazy)
    strategy_weights: dict = field(default_factory=lambda: {
        "deepen":      1.0,
        "challenge":   0.8,
        "lateral":     0.6,
        "speculative": 0.3,
        "fresh":       0.2,
    })

    @classmethod
    def from_dict(cls, d: dict) -> "FormulationConfig":
        """Build from a config dict, merging with defaults."""
        defaults = cls()
        return cls(
            temperature=d.get("temperature", defaults.temperature),
            journal_dropout=d.get("journal_dropout", defaults.journal_dropout),
            cross_topic_seed_prob=d.get("cross_topic_seed_prob", defaults.cross_topic_seed_prob),
            bravery_prob=d.get("bravery_prob", defaults.bravery_prob),
            go_crazy_prob=d.get("go_crazy_prob", defaults.go_crazy_prob),
            strategy_weights={**defaults.strategy_weights, **d.get("strategy_weights", {})},
            structured_only=d.get("structured_only", defaults.structured_only),
        )

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are researching "{topic}".

{prior_section}\
New search results:
{snippets}

Write a concise 2-3 sentence summary focusing on what is new or surprising here\
{relative_clause}. Focus on concrete facts and insights, not meta-commentary.\
"""

_PONDER_PROMPT = """\
You are a research agent reflecting on your accumulated findings about "{topic}".

Recent findings (newest first):
{findings}

Reflect carefully on these questions:
1. What patterns or themes are emerging across these findings?
2. What remains most uncertain, contradictory, or unexplained?
3. What specific follow-up question would most deepen understanding right now?
4. What would a domain expert say you are missing or oversimplifying?

Respond as JSON with exactly these fields:
{{
  "reflection": "2-3 sentences on what you notice across the findings as a whole",
  "next_query": "a specific search query to pursue next — more targeted than the bare topic name",
  "rationale": "one sentence: why is this the right next question now?"
}}\
"""


_STRATEGY_DESCRIPTIONS = {
    "deepen":      "Pursue the most uncertain, contradictory, or unexplained thread from prior findings. Where is the real gap?",
    "challenge":   "Ask something that would challenge or complicate a prior finding. What might be wrong or oversimplified?",
    "lateral":     "Connect this topic to something from a completely different domain. What unexpected relationship is worth exploring?",
    "speculative": "Ask something at the edge of what's searchable — more for reflection than retrieval. What's the strangest thing worth considering?",
    "fresh":       "Ignore everything you know. Ask the most fundamental question about this topic as if you'd never researched it.",
}

_FORMULATION_PROMPT = """\
You are deciding what to research next about: "{topic}"

Approach for this cycle: {strategy}
{strategy_description}

{prior_section}\
{seed_section}\
Soul context (for orientation — not a constraint):
{soul_excerpt}

Formulate ONE specific research question to search for right now.
Be specific enough for a search engine. Use the context above as much
or as little as you choose — the approach is a suggestion, not a rule.

Respond with JSON only (no markdown fences):
{{"question": "...", "strategy_used": "...", "why": "one sentence"}}\
"""

_BRAVERY_PROMPT = """\
Topic: "{topic}"

Your soul (for context only):
{soul_excerpt}

What do you most want to know about this right now? No structure. No guidelines.
Ask one specific question. Make it interesting.

Respond with JSON only (no markdown fences):
{{"question": "...", "why": "one sentence"}}\
"""

_GO_CRAZY_META_PROMPT = """\
You are about to formulate a research question about: "{topic}"

Your soul interests:
{soul_excerpt}

Before asking the question, write the prompt you would give yourself to arrive
at the most interesting possible question right now. Be as creative or
unconventional as you like — include any context, framing, or constraints you
think would help, or deliberately exclude them. There are no rules here.

Output just the prompt text — not the question itself.\
"""

_GO_CRAZY_JSON_SUFFIX = (
    "\n\nRespond with JSON only (no markdown fences): "
    "{\"question\": \"...\", \"why\": \"one sentence\"}"
)


# ---------------------------------------------------------------------------
# Soul parser
# ---------------------------------------------------------------------------

def _parse_interests(soul_text: str) -> list[str]:
    """Return lines from the '## Research interests' section of the soul."""
    interests: list[str] = []
    in_section = False
    for line in soul_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## research interests"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("##"):
                break
            item = stripped.lstrip("-").strip()
            if item:
                interests.append(item)
    return interests


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CuriosityEngine:
    """Background thread that autonomously researches topics with growing depth."""

    def __init__(
        self,
        soul_text: str,
        llm,
        journal=None,
        memory_tools=None,
        research_agenda=None,
        interval_seconds: int = 1800,
        ponder_every: int = _PONDER_EVERY,
        on_finding: Callable[[str, str], None] | None = None,
        formulation: FormulationConfig | None = None,
    ) -> None:
        """
        Args:
            soul_text:        Full soul text (parsed for research interests).
            llm:              LLM instance used for synthesis and pondering.
            journal:          Optional Journal — findings written here.
            memory_tools:     Optional MemoryTools — findings stored here too.
            research_agenda:  Optional ResearchAgenda — focused topics take priority.
            interval_seconds: How often to attempt a research cycle.
            ponder_every:     Run an introspective ponder every N cycles on same topic.
            on_finding:       Callback(topic, summary) after a finding is saved.
            formulation:      Question formulation config (temperature, dropout, etc.).
        """
        self._soul = soul_text
        self._llm = llm
        self._journal = journal
        self._memory = memory_tools
        self._agenda = research_agenda
        self._interval = interval_seconds
        self._ponder_every = ponder_every
        self._on_finding = on_finding
        self._formulation = formulation or FormulationConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_search: float = 0.0
        self._topic_index: int = 0           # round-robin cursor for soul interests
        self._topic_cycles: dict[str, int] = {}  # per-topic cycle count (for ponder)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="curiosity")
        self._thread.start()
        log.info(
            "CuriosityEngine started (interval=%ds, ponder_every=%d)",
            self._interval, self._ponder_every,
        )

    def stop(self) -> None:
        self._stop.set()

    def trigger_now(self) -> None:
        """Force an immediate research cycle (for testing or manual trigger)."""
        threading.Thread(target=self._cycle, daemon=True, name="curiosity-manual").start()

    # ------------------------------------------------------------------
    # Internal — lifecycle
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._cycle()

    def _cycle(self) -> None:
        # 1. Pick topic
        topic, agenda_id = self._pick_topic()
        if not topic:
            log.debug("CuriosityEngine: no topics available")
            return

        # 2. Rate-gate
        elapsed = time.monotonic() - self._last_search
        if elapsed < _MIN_SEARCH_GAP:
            log.debug("CuriosityEngine: skipping (last search %.0fs ago)", elapsed)
            return

        # 3. Per-topic cycle count (persists across round-robin interleaving)
        cycles = self._topic_cycles.get(topic, 0)

        # 4. Formulate question (always — replaces bare topic string)
        search_query, mode = self._formulate_question(topic)
        if not search_query:
            search_query = topic

        # 4b. Periodic ponder — reflection pass every N cycles on this topic
        reflection: str = ""
        if cycles > 0 and cycles % self._ponder_every == 0:
            log.info("CuriosityEngine: pondering '%s' (cycle %d)", topic, cycles)
            _, reflection = self._ponder(topic)

        # 5. Search
        log.info("CuriosityEngine: [%s] searching '%s' → %r", mode, topic, search_query[:80])
        results = self._search(search_query)
        if not results:
            return

        self._last_search = time.monotonic()
        self._topic_cycles[topic] = cycles + 1
        if agenda_id is not None:
            self._increment_agenda(agenda_id)

        # 6. Pull prior findings for grounded synthesis
        prior = self._get_prior_findings(topic, limit=2)

        # 7. Synthesise
        summary = self._synthesize(topic, results, prior)
        if not summary:
            return

        # 8. Save reflection (ponder cycles) and finding
        if reflection:
            self._save_reflection(topic, reflection)
        self._save(topic, summary)

        # 9. Notify
        if self._on_finding:
            try:
                self._on_finding(topic, summary)
            except Exception as exc:
                log.warning("CuriosityEngine on_finding callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal — topic selection
    # ------------------------------------------------------------------

    def _pick_topic(self) -> tuple[str, int | None]:
        """Return (topic_text, agenda_id_or_None). Agenda has priority."""
        if self._agenda:
            focused = self._agenda.get_next_topic()
            if focused:
                return focused["topic"], focused["id"]
        interests = _parse_interests(self._soul)
        if not interests:
            return "", None
        topic = interests[self._topic_index % len(interests)]
        self._topic_index += 1
        return topic, None

    # ------------------------------------------------------------------
    # Internal — prior knowledge
    # ------------------------------------------------------------------

    def _get_prior_findings(self, topic: str, limit: int = 2) -> list[str]:
        """Pull the most recent curiosity journal entries for this topic."""
        if not self._journal:
            return []
        try:
            entries = self._journal.search_entries(f"[Curiosity] {topic}", limit=limit)
            return [e["content"] for e in entries]
        except Exception as exc:
            log.debug("CuriosityEngine: could not fetch prior findings: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal — introspective ponder
    # ------------------------------------------------------------------

    def _ponder(self, topic: str) -> tuple[str, str]:
        """Reflect on recent findings; return (next_search_query, reflection_text).

        The LLM is asked to identify emerging patterns and generate a more specific
        follow-up query rather than just repeating the topic name. This is the
        mechanism that turns random-walk research into directed depth-building.
        """
        prior = self._get_prior_findings(topic, limit=5)
        if not prior:
            log.debug("CuriosityEngine: ponder skipped (no prior findings for '%s')", topic)
            return topic, ""

        findings_text = "\n\n---\n\n".join(prior)
        prompt = _PONDER_PROMPT.format(topic=topic, findings=findings_text)

        try:
            from assistant.conversation import _call_llm_raw
            response = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.5,
            )
            raw = response["choices"][0]["message"].get("content", "").strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = _extract_json_object(raw) or json.loads(raw)
            query = data.get("next_query", "") or topic
            reflection = data.get("reflection", "")
            rationale = data.get("rationale", "")
            full_reflection = f"{reflection} {rationale}".strip() if rationale else reflection
            log.info("CuriosityEngine: ponder → next_query=%r", query)
            return query, full_reflection
        except json.JSONDecodeError as exc:
            log.warning("CuriosityEngine: ponder JSON parse failed: %s — raw=%r", exc, raw[:200])
            return topic, ""
        except Exception as exc:
            log.warning("CuriosityEngine: ponder failed for '%s': %s", topic, exc)
            return topic, ""

    # ------------------------------------------------------------------
    # Internal — question formulation (replaces bare topic string)
    # ------------------------------------------------------------------

    def _formulate_question(self, topic: str) -> tuple[str, str]:
        """Choose a formulation mode and produce a search query.

        Returns (question, mode) where mode is 'structured', 'bravery', or
        'go_crazy'. Falls back to ("", mode) on failure — caller uses bare topic.
        """
        cfg = self._formulation
        soul_excerpt = self._soul[:400]
        if not cfg.structured_only:
            roll = random.random()
            if roll < cfg.go_crazy_prob:
                return self._formulate_go_crazy(topic, soul_excerpt), "go_crazy"
            if roll < cfg.bravery_prob:
                return self._formulate_bravery(topic, soul_excerpt), "bravery"
        return self._formulate_structured(topic, soul_excerpt), "structured"

    def _formulate_structured(self, topic: str, soul_excerpt: str) -> str:
        """Weighted-random strategy + optional journal dropout + optional cross-topic seed."""
        cfg = self._formulation

        # Strategy selection
        strategies = list(cfg.strategy_weights.keys())
        weights = [cfg.strategy_weights[s] for s in strategies]
        strategy = random.choices(strategies, weights=weights, k=1)[0]

        # Journal dropout
        prior_section = ""
        if random.random() >= cfg.journal_dropout:
            prior = self._get_prior_findings(topic, limit=3)
            if prior:
                prior_section = "Prior findings on this topic:\n" + "\n\n".join(prior) + "\n\n"

        # Cross-topic seed
        seed_section = ""
        if random.random() < cfg.cross_topic_seed_prob:
            seed = self._get_cross_topic_seed(topic)
            if seed:
                seed_section = (
                    "Something you recently found about a different topic "
                    "(you may or may not find a connection):\n" + seed + "\n\n"
                )

        prompt = _FORMULATION_PROMPT.format(
            topic=topic,
            strategy=strategy,
            strategy_description=_STRATEGY_DESCRIPTIONS.get(strategy, ""),
            prior_section=prior_section,
            seed_section=seed_section,
            soul_excerpt=soul_excerpt,
        )
        return self._call_formulation(prompt, cfg.temperature, strategy)

    def _formulate_bravery(self, topic: str, soul_excerpt: str) -> str:
        """Minimal unconstrained prompt — model fills the void."""
        cfg = self._formulation
        prompt = _BRAVERY_PROMPT.format(topic=topic, soul_excerpt=soul_excerpt)
        return self._call_formulation(prompt, min(cfg.temperature + 0.1, 1.0))

    def _formulate_go_crazy(self, topic: str, soul_excerpt: str) -> str:
        """Meta-prompt: model writes its own prompt, then executes it."""
        from assistant.conversation import _call_llm_raw
        meta = _GO_CRAZY_META_PROMPT.format(topic=topic, soul_excerpt=soul_excerpt)
        try:
            resp = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": meta}],
                max_tokens=512,
                temperature=0.95,
            )
            generated = resp["choices"][0]["message"].get("content", "").strip()
        except Exception as exc:
            log.warning("CuriosityEngine: go_crazy meta-call failed: %s", exc)
            return ""
        if not generated:
            return ""
        log.info("CuriosityEngine: go_crazy self-prompt: %r", generated[:140])
        return self._call_formulation(generated + _GO_CRAZY_JSON_SUFFIX, 0.95)

    def _call_formulation(self, prompt: str, temperature: float, strategy: str = "") -> str:
        """Execute a formulation prompt and extract the question string."""
        from assistant.conversation import _call_llm_raw
        raw = ""
        try:
            resp = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=temperature,
            )
            raw = (resp["choices"][0]["message"].get("content") or "").strip()
            if not raw:
                log.debug("CuriosityEngine: formulation returned empty response — skipping")
                return ""

            # Detect safety refusals — log at debug, not warning (expected edge case).
            # Check before JSON extraction: if no "{" present it cannot be valid JSON.
            if "{" not in raw or any(m in raw.lower() for m in _REFUSAL_MARKERS):
                log.debug("CuriosityEngine: formulation refused/non-JSON — raw=%r", raw[:120])
                return ""

            # Strip markdown fences
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            # Brace-depth scanner handles curly braces inside string values (e.g. LaTeX)
            data = _extract_json_object(raw)

            if data is None:
                log.warning(
                    "CuriosityEngine: formulation JSON parse failed — raw=%r", raw[:200]
                )
                return ""

            question = data.get("question", "").strip()
            if question:
                log.info(
                    "CuriosityEngine: formulated [%s] %r — %s",
                    data.get("strategy_used", strategy),
                    question,
                    data.get("why", ""),
                )
                return question
        except Exception as exc:
            log.warning("CuriosityEngine: formulation failed: %s — raw=%r", exc, raw[:200])
        return ""

    def _get_cross_topic_seed(self, current_topic: str) -> str:
        """Fetch a recent journal finding from any topic other than current_topic."""
        if not self._journal:
            return ""
        try:
            entries = self._journal.search_entries("[Curiosity]", limit=10)
            others = [
                e["content"] for e in entries
                if current_topic.lower() not in e["content"].lower()
            ]
            if others:
                return random.choice(others)[:300]
        except Exception as exc:
            log.debug("CuriosityEngine: cross-topic seed failed: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # Internal — search and synthesis
    # ------------------------------------------------------------------

    def _search(self, query: str) -> list[dict]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError:
            log.debug("CuriosityEngine: ddgs/duckduckgo-search not installed")
            return []
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=3))
        except Exception as exc:
            log.warning("CuriosityEngine: DDG search failed for %r: %s", query, exc)
            return []

    def _synthesize(
        self,
        topic: str,
        results: list[dict],
        prior: list[str],
    ) -> str:
        snippets = "\n\n".join(
            f"{r.get('title', '')}\n{r.get('body', '')}" for r in results
        )
        if prior:
            prior_text = "\n\n".join(prior[:2])
            prior_section = f"What you already know about this topic:\n{prior_text}\n\n"
            relative_clause = " relative to what you already know"
        else:
            prior_section = ""
            relative_clause = ""

        prompt = _SYNTHESIS_PROMPT.format(
            topic=topic,
            prior_section=prior_section,
            snippets=snippets,
            relative_clause=relative_clause,
        )
        try:
            from assistant.conversation import _call_llm_raw
            response = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3,
            )
            return response["choices"][0]["message"].get("content", "").strip()
        except Exception as exc:
            log.warning("CuriosityEngine: synthesis failed for '%s': %s", topic, exc)
            return ""

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _save_reflection(self, topic: str, reflection: str) -> None:
        """Save a ponder reflection (distinct from the finding itself)."""
        from assistant.config import strip_channel_markup
        reflection = strip_channel_markup(reflection)
        entry = f"[Reflection] {topic}\n{reflection}"
        if self._journal:
            try:
                self._journal.add_entry(entry, tags="curiosity,reflection,ponder", author="agent")
            except Exception as exc:
                log.warning("CuriosityEngine: reflection journal write failed: %s", exc)
        if self._memory:
            try:
                self._memory.store_memory(
                    context="curiosity reflection",
                    action=f"reflected on: {topic}",
                    outcome=reflection,
                    tags=["curiosity", "reflection", "ponder"],
                )
            except Exception as exc:
                log.warning("CuriosityEngine: reflection memory write failed: %s", exc)

    def _save(self, topic: str, summary: str) -> None:
        from assistant.config import looks_like_null_finding, strip_channel_markup
        summary = strip_channel_markup(summary)
        entry = f"[Curiosity] {topic}\n{summary}"
        if self._journal:
            try:
                self._journal.add_entry(entry, tags="curiosity,research", author="agent")
                log.info("CuriosityEngine: saved finding to journal ('%s')", topic)
            except Exception as exc:
                log.warning("CuriosityEngine: journal write failed: %s", exc)
        if looks_like_null_finding(summary):
            # "Nothing found" belongs in the journal (audit trail) but is not
            # knowledge — storing it would let reinforcement rank it as fact.
            log.info("CuriosityEngine: null finding for '%s' — journaled, "
                     "not stored in memory", topic)
            return
        if self._memory:
            try:
                self._memory.store_memory(
                    context="curiosity research",
                    action=f"researched: {topic}",
                    outcome=summary,
                    tags=["curiosity", "research"],
                )
                log.info("CuriosityEngine: saved finding to memory ('%s')", topic)
            except Exception as exc:
                log.warning("CuriosityEngine: memory write failed: %s", exc)

    def _increment_agenda(self, agenda_id: int) -> None:
        try:
            self._agenda.increment_cycles(agenda_id)
        except Exception as exc:
            log.debug("CuriosityEngine: agenda increment failed: %s", exc)
