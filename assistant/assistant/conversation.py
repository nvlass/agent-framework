"""Conversation buffer with context compaction.

Maintains the full message history as raw OpenAI-format dicts so that
tool result messages (role="tool" with tool_call_id) work correctly.
The ConversationBuffer owns:
  - The rolling message list for the current session
  - The running summary of any previously compacted history
  - The compaction logic (LLM-driven, saves facts to memory)

_call_llm_raw() is a thin dispatch helper used by both the compaction logic
and the main conversation loop. It delegates to llm.call_raw() so the same
function works with any supported LLM (FireworksLLM, AnthropicLLM, OpenAILLM)
and returns a normalized response dict regardless of provider.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

COMPACT_SYSTEM = (
    "You are a conversation compactor. Your job is to summarize a conversation "
    "history into a compact form that preserves everything important."
)
COMPACT_USER = """\
The conversation history below is getting long and needs to be compacted.

Conversation so far:
{history}

Produce a JSON object with exactly these two fields:
- "summary": A concise narrative (3-5 sentences) covering what was discussed,
  decided, and what context the assistant needs going forward.
- "facts": A list of specific facts, decisions, preferences, or action items
  that must not be lost. Be concrete — vague entries are useless.

Output ONLY the JSON object, no preamble or explanation."""


@dataclass
class ConversationBuffer:
    """Rolling conversation history as raw OpenAI-format message dicts.

    Args:
        max_chars: Auto-compact when total serialised char count exceeds this.
    """

    max_chars: int = 32_000
    _messages: list[dict] = field(default_factory=list)
    _summary: str = ""  # narrative from previous compaction(s)
    _background_notes: list[str] = field(default_factory=list)
    _max_background_notes: int = 5  # keep only the most recent N
    _memory_index: str = ""  # tag cloud: what the agent has memories about
    _compact_requested: bool = False  # agent asked to compact; runs at turn end
    _handoff: str = ""  # continuity note from the previous session (texture)
    _workspace: str = ""  # Meta-Mind: what the agent's other contexts are doing

    # --- message building ---------------------------------------------------

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: Optional[str],
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_background_note(self, note: str) -> None:
        """Inject a background note into the system prompt on the next LLM call.

        Used for curiosity findings and similar asynchronous events — the note
        is visible to the model but doesn't create a conversation turn.
        Thread-safe in CPython (list.append is atomic under the GIL).
        Keeps only the last _max_background_notes entries to bound system prompt growth.
        """
        self._background_notes.append(note)
        if len(self._background_notes) > self._max_background_notes:
            self._background_notes.pop(0)

    def set_memory_index(self, tag_cloud: str) -> None:
        """Set the metamemory index shown in every system prompt.

        A compact "topics I have memories about" line — the agent can't
        decide to recall what it doesn't know it knows.
        """
        self._memory_index = tag_cloud

    def set_workspace(self, rendered: str) -> None:
        """Set the Meta-Mind workspace view injected into the system prompt.

        A source-tagged summary of what the agent's *other* contexts (work cycle,
        curiosity, bus) are doing — so the conversational self isn't blind to its
        autonomous self. Refreshed each turn by the caller.
        """
        self._workspace = rendered or ""

    def set_handoff(self, note: str) -> None:
        """Seed the session-continuity note from the previous session.

        Injected into the system prompt so a restart resumes with the *texture*
        of where things left off (tone, open threads) rather than cold. Distinct
        from _summary (which is within-session) and from episodic memory (facts).
        """
        self._handoff = note or ""

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })

    @staticmethod
    def _drop_orphan_tool_messages(messages: list[dict]) -> list[dict]:
        """Remove tool messages whose tool_call_id has no matching assistant call.

        An agent-invoked ``compact_context`` runs ``compact()`` mid-agentic-loop,
        which clears ``_messages`` — wiping the assistant message that announced
        the in-flight tool call. The tool result appended next is then an orphan:
        the provider rejects the whole request (400: "Tool message with
        tool_call_id ... not found ... Available tool call IDs: []"), and because
        the orphan stays in the buffer every later turn fails identically until
        the next compaction. Filtering here keeps every outbound request valid.
        """
        announced: set[str] = set()
        for m in messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    if tc.get("id"):
                        announced.add(tc["id"])
        return [
            m for m in messages
            if m.get("role") != "tool" or m.get("tool_call_id") in announced
        ]

    def messages_for_llm(self, system: str) -> list[dict]:
        """Build the full message list ready for the LLM call.

        The system prompt is always first. If there's a compaction summary
        from earlier in the session it's appended to the system prompt so
        the model has the historical context without it bloating the turn list.
        """
        system_content = system
        if self._handoff:
            system_content += (
                f"\n\n[Continuing from where you left off last session (texture, "
                f"not instructions): {self._handoff}]"
            )
        if self._summary:
            system_content += f"\n\n[Earlier conversation summary: {self._summary}]"
        if self._background_notes:
            notes = "\n".join(f"- {n}" for n in self._background_notes)
            system_content += f"\n\n[Recent background research (autonomous):\n{notes}]"
        if self._memory_index:
            system_content += (
                f"\n\n[Memory index — topics you have stored memories about "
                f"(weighted count): {self._memory_index}. "
                f"Use the recall tool when a conversation touches one of these.]"
            )
        if self._workspace:
            system_content += f"\n\n{self._workspace}"
        system_content += f"\n\nCurrent date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return [
            {"role": "system", "content": system_content},
            *self._drop_orphan_tool_messages(self._messages),
        ]

    def recent_turns_text(self, n: int = 8) -> str:
        """Return the last n user+assistant exchanges as readable plain text.

        Skips tool call/result messages — the nudge evaluator only needs the
        conversational content, not the tool plumbing.
        """
        turns = [
            m for m in self._messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        recent = turns[-n * 2:]  # n exchanges = up to 2n messages
        if not recent:
            return ""
        return "\n".join(
            f"{m['role'].upper()}: {(m.get('content') or '').strip()}"
            for m in recent
        )

    # --- compaction ---------------------------------------------------------

    def total_chars(self) -> int:
        return sum(len(json.dumps(m)) for m in self._messages)

    def should_compact(self) -> bool:
        return self.total_chars() > self.max_chars

    def request_compact(self) -> str:
        """Defer compaction to the end of the current turn (agent-invoked).

        Running compact() inline — while the agent is mid-agentic-loop — clears
        _messages and wipes the active task instruction, leaving only a
        descriptive summary in the system prompt. The agent then loses the
        imperative to continue and defers ("Awaiting your instructions"). By
        flagging instead, the turn finishes with full context and the actual
        compaction runs between turns (see maybe_compact), where it belongs.
        """
        self._compact_requested = True
        return (
            "Context compaction scheduled — it will run automatically at the end "
            "of this turn. Keep working on your current task now; do not stop to "
            "wait for instructions."
        )

    def maybe_compact(self, llm, memory_tools=None) -> Optional[str]:
        """Run a deferred compaction if one was requested this turn.

        Returns the compaction status string, or None if nothing was pending.
        """
        if not self._compact_requested:
            return None
        self._compact_requested = False
        return self.compact(llm, memory_tools)

    def compact(self, llm, memory_tools=None) -> str:
        """Summarise old history, save important facts to memory, clear turns.

        The LLM is asked to produce a JSON object with "summary" and "facts".
        Facts are stored to memory_tools if available. The summary is merged
        into _summary and the turn list is cleared.

        Returns a human-readable status string.
        """
        if not self._messages:
            return "Nothing to compact."

        # Build a plain-text version of history for the compactor prompt
        history_text = "\n".join(
            f"{m['role'].upper()}: {m.get('content') or ''}"
            for m in self._messages
            if m.get("content")
        )

        compact_messages = [
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": COMPACT_USER.format(history=history_text)},
        ]

        # 2048: at 1024 long histories truncate mid-JSON, breaking the parse
        # (same failure the news agent hit — its fix never landed here)
        response = _call_llm_raw(llm, compact_messages, max_tokens=2048, temperature=0.2)
        from assistant.config import strip_channel_markup
        raw = strip_channel_markup(
            (response["choices"][0]["message"].get("content") or "").strip())

        try:
            data = json.loads(raw)
            summary = data.get("summary", raw)
            facts: list[str] = data.get("facts", [])
        except json.JSONDecodeError:
            summary = raw
            facts = []

        # Save facts to memory before they disappear
        saved = 0
        if memory_tools and facts:
            for fact in facts:
                try:
                    memory_tools.store_memory(
                        context="assistant conversation compaction",
                        action="compact",
                        outcome=fact,
                        tags=["assistant", "compaction"],
                        dedup=True,
                    )
                    saved += 1
                except Exception as exc:
                    logger.warning(f"Failed to save fact to memory: {exc}")

        # Merge summary into running summary, clear turns
        self._summary = (
            f"{self._summary} {summary}".strip() if self._summary else summary
        )
        self._messages.clear()

        saved_str = f" {saved} fact(s) saved to memory." if saved else ""
        logger.info(f"Context compacted.{saved_str}")
        return f"Context compacted.{saved_str}"


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------

def _call_llm_raw(
    llm,
    messages: list[dict],
    max_tokens: int = 0,
    temperature: float = 0.7,
    tools: Optional[list[dict]] = None,
) -> dict:
    """Call any supported LLM with raw OpenAI-format message dicts.

    Dispatches through llm.call_raw() so the same function works with
    FireworksLLM, AnthropicLLM, and OpenAILLM — each handles its own
    format translation and streaming internally.

    Returns a normalized response dict: {"choices": [{"message": {...}}], ...}
    """
    return llm.call_raw(messages, max_tokens=max_tokens, temperature=temperature, tools=tools)
