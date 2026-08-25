"""L1 auto-extraction — pull atomic facts from recent conversation turns.

Fires every N assistant turns (configurable). Extracted facts are stored in
long-term memory tagged ["atom", "auto-extracted"] so they survive compaction
and are retrievable via recall_similar().

Facts focus on durable signal: user preferences, background, decisions,
constraints, goals. Trivial exchanges produce an empty list and store nothing.
"""

import json
import logging

log = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract durable facts from this conversation excerpt. Focus on:
- User preferences, habits, or working style revealed
- Background context about the user (role, projects, constraints)
- Decisions made or conclusions reached
- Goals or intentions stated
- Anything an assistant would want to remember in a future session

Rules:
- Write each fact as a single sentence ("User prefers...", "User is working on...", "User decided...")
- Only include facts with lasting value — skip greetings, filler, and one-off requests
- If nothing worth remembering occurred, return an empty list

Return ONLY a JSON array of strings. Example:
["User prefers concise responses.", "User is building a local-first agent framework in Python."]

Conversation:
{turns}
"""


class ExtractionPass:
    """Extracts atomic facts from recent conversation turns and stores them in memory.

    Call maybe_extract() after each completed assistant turn. It counts turns
    internally and only runs the LLM extraction every ``every_n_turns`` calls.
    """

    def __init__(self, llm, memory_tools, every_n_turns: int = 5) -> None:
        self._llm = llm
        self._memory = memory_tools
        self._every_n = every_n_turns
        self._turn_count = 0

    @property
    def enabled(self) -> bool:
        return self._memory is not None and self._every_n > 0

    def maybe_extract(self, buffer) -> int:
        """Call after each assistant turn. Returns facts stored (0 most calls)."""
        if not self.enabled:
            return 0
        self._turn_count += 1
        if self._turn_count < self._every_n:
            return 0
        self._turn_count = 0
        return self._run(buffer)

    def _run(self, buffer) -> int:
        from assistant.conversation import _call_llm_raw

        turns_text = buffer.recent_turns_text(n=self._every_n)
        if not turns_text:
            return 0

        prompt = _EXTRACTION_PROMPT.format(turns=turns_text)
        try:
            resp = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )
            raw = (resp["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            log.warning("ExtractionPass: LLM call failed: %s", exc)
            return 0

        facts = _parse_facts(raw)
        if not facts:
            return 0

        stored = 0
        for fact in facts:
            try:
                self._memory.store_memory(
                    context="conversation",
                    action="auto-extracted fact",
                    outcome=fact,
                    tags=["atom", "auto-extracted"],
                    dedup=True,
                )
                stored += 1
            except Exception as exc:
                log.warning("ExtractionPass: store failed: %s", exc)

        if stored:
            log.info("ExtractionPass: stored %d fact(s)", stored)
        return stored


def _parse_facts(raw: str) -> list[str]:
    """Parse JSON array from LLM response, tolerating markdown fences."""
    if not raw:
        return []
    text = raw
    if "```" in text:
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        facts = json.loads(text[start:end + 1])
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        return []
