"""Autonomous work cycle — self-directed ReAct sessions during idle time.

Unlike CuriosityEngine's fixed search→synthesize→journal pipeline, each cycle
picks a goal (queued scheduled prompts, pending TODOs, research agenda topics,
soul interests) and hands it to a real ReactLoop over the assistant's tool
registry: the agent decides which tools to use and when it is done. Bounded
by max_iterations per cycle.

Tools excluded from autonomous use: compact_context (conversation-level
concern) and decide_soul_proposal (the human's decision — the agent must
never approve its own proposals).
"""

import json
import logging
import queue
import threading
import time

from agent_mind.goals.model import Goal
from agent_patterns.context import SharedContext
from agent_patterns.react import ReactLoop, ReasonerInterface, ReasoningResult
from agent_tools.core.executor import ToolExecutor
from agent_tools.core.registry import ToolRegistry

from assistant.config import TYPE_TAGS
from assistant.conversation import _call_llm_raw
from assistant.curiosity import _parse_interests

log = logging.getLogger(__name__)

_EXCLUDED_TOOLS = {"compact_context", "decide_soul_proposal"}

_SOUL_EXCERPT_CHARS = 2000

_REASONER_SYSTEM = """\
You are working autonomously — no user is present. You have a goal and a set
of tools. Each turn, either call ONE tool that makes concrete progress toward
the goal, or, if the goal is achieved (or genuinely cannot be advanced),
respond with a final summary of what you accomplished and found.

Rules:
- Never ask questions or wait for input; decide and act.
- Prefer depth over breadth: read sources, don't just collect links.
- Record what matters: journal findings, save important facts to memory.
- If a tool fails twice, try a different approach or wrap up with what you have.

{soul}"""

_REASONER_USER = """\
GOAL: {goal}

{observations}
Decide your next step: call one tool, or give your final summary if done."""


class LLMReasoner(ReasonerInterface):
    """ReasonerInterface backed by a chat LLM with native function calling."""

    def __init__(self, llm, soul_text: str = "", temperature: float = 0.4,
                 max_tokens: int = 2048) -> None:
        self._llm = llm
        self._temperature = temperature
        self._max_tokens = max_tokens
        soul = soul_text.strip()[:_SOUL_EXCERPT_CHARS]
        self._system = _REASONER_SYSTEM.format(
            soul=f"Your identity:\n{soul}" if soul else ""
        ).strip()

    def reason(self, goal, observations: list[str],
               available_tools: list[dict]) -> ReasoningResult:
        if observations:
            obs_lines = "\n".join(
                f"{i}. {o}" for i, o in enumerate(observations, 1)
            )
            obs_block = f"Steps taken so far (tool: result):\n{obs_lines}\n"
        else:
            obs_block = "No steps taken yet.\n"
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": _REASONER_USER.format(
                goal=goal.description, observations=obs_block)},
        ]
        response = _call_llm_raw(
            self._llm, messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            tools=available_tools or None,
        )
        msg = response["choices"][0]["message"]
        thought = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            fn = tool_calls[0].get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    log.warning("Unparseable tool args for %s: %.200s",
                                fn.get("name"), args)
                    args = {}
            return ReasoningResult(thought=thought, action=fn.get("name", ""),
                                   action_args=args)
        return ReasoningResult(thought=thought,
                               answer=thought or "(no output)")


class WorkCycle:
    """Background thread running bounded autonomous ReAct sessions."""

    def __init__(
        self,
        llm,
        executor,
        soul_text: str = "",
        journal=None,
        todo_db=None,
        research_agenda=None,
        memory_tools=None,
        conversation_bus=None,
        interval_seconds: int = 3600,
        max_iterations: int = 8,
        on_cycle=None,
        use_queue: bool = False,
    ) -> None:
        """
        Args:
            llm:              LLM used by the reasoner.
            executor:         ToolExecutor over the full assistant registry —
                              a restricted copy is built internally.
            soul_text:        Full soul text (identity excerpt + interests).
            journal:          Optional Journal — cycle outcomes logged here.
            todo_db:          Optional TodoDB — pending TODOs become goals.
            research_agenda:  Optional ResearchAgenda — focused topics become goals.
            memory_tools:     Optional MemoryTools — enables dream/replay cycles
                              (sampling memory pairs for associative synthesis).
            interval_seconds: Seconds between cycles.
            max_iterations:   Max ReAct steps per cycle.
            on_cycle:         Callback(source, outcome, summary) after each cycle.
        """
        sub_registry = ToolRegistry()
        for tool in executor.registry.list_tools():
            if tool.name not in _EXCLUDED_TOOLS:
                sub_registry.register(tool)
        self._executor = ToolExecutor(sub_registry, permission_checker=lambda _: True)
        self._reasoner = LLMReasoner(llm, soul_text=soul_text)
        self._soul = soul_text
        self._journal = journal
        self._todos = todo_db
        self._agenda = research_agenda
        self._memory = memory_tools
        self._conversations = conversation_bus
        self._interval = interval_seconds
        self._max_iterations = max_iterations
        self._on_cycle = on_cycle
        self._goal_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_cursor = 0
        self._interest_cursor = 0
        # Unified-queue path (opt-in; the default rotation loop is unchanged).
        self._use_queue = use_queue
        self._sched_seq = 0  # unique keys for scheduled items (never deduped)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="work-cycle")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def enqueue_goal(self, description: str) -> None:
        """Queue an externally supplied goal (e.g. a scheduled task prompt).

        Runs ahead of the normal source rotation, within ~5s.
        """
        self._goal_queue.put(description)

    def _run(self) -> None:
        if self._use_queue:
            self._run_queued()
        else:
            self._run_rotation()

    def _run_rotation(self) -> None:
        next_cycle = time.time() + self._interval
        while not self._stop.is_set():
            self._stop.wait(5)
            if self._stop.is_set():
                break
            goal_desc = source = None
            try:
                prompt = self._goal_queue.get_nowait()
                goal_desc = (
                    f"Complete this scheduled task: {prompt}\n"
                    "Use your tools as needed, then summarize the outcome."
                )
                source = "scheduled"
            except queue.Empty:
                # A peer waiting on our turn preempts the normal rotation.
                # (Closed-conversation outcomes are surfaced/acked through the
                # normal attention path — not silently consumed here.)
                conv_goal = self._goal_from_conversations()
                if conv_goal:
                    goal_desc, source = conv_goal, "conversation"
                elif time.time() >= next_cycle:
                    goal_desc, source = self._pick_goal()
                    next_cycle = time.time() + self._interval
            if goal_desc:
                try:
                    self.run_cycle(goal_desc, source)
                except Exception:
                    log.exception("Work cycle failed (source=%s)", source)

    # ------------------------------------------------------------------
    # Unified-queue loop (opt-in via use_queue). Same cadence as the rotation
    # (conversations/scheduled every tick, background interval-paced) but routed
    # through one two-tier queue: urgent strict-priority, background weighted
    # lottery. Background items are lightweight TICKETS (kind only) generated
    # lazily at dispatch, so a side-effectful generator only runs when its kind
    # is actually chosen — preserving cursor/agenda semantics. See
    # docs/work-queue-design.md.
    # ------------------------------------------------------------------

    _URGENT_PRIORITY = {"scheduled": 20, "conversation": 30}
    _BACKGROUND_WEIGHTS = {"todos": 3.0, "research": 1.5, "interests": 1.0, "dream": 0.5}

    def _run_queued(self) -> None:
        from assistant.work_queue import WorkQueue
        q = WorkQueue()
        next_bg = time.time() + self._interval
        while not self._stop.is_set():
            self._stop.wait(5)
            if self._stop.is_set():
                break
            self._enqueue_urgent(q)
            if time.time() >= next_bg and q.pending()["background"] == 0:
                self._enqueue_background(q)
                next_bg = time.time() + self._interval
            item = q.get()
            if item is not None:
                self._dispatch_item(q, item)

    def _enqueue_urgent(self, q) -> None:
        """Drain scheduled prompts and pending conversation turns as urgent items."""
        from assistant.work_queue import WorkItem
        while True:
            try:
                prompt = self._goal_queue.get_nowait()
            except queue.Empty:
                break
            goal = (f"Complete this scheduled task: {prompt}\n"
                    "Use your tools as needed, then summarize the outcome.")
            self._sched_seq += 1
            q.put_urgent(WorkItem("scheduled", payload=goal,
                                  key=f"scheduled:{self._sched_seq}"),
                         priority=self._URGENT_PRIORITY["scheduled"])
        if self._conversations:
            for c in self._conversations.needs_attention(limit=5):
                if c["attention"] == "your_turn":
                    q.put_urgent(WorkItem("conversation", payload=c["id"],
                                          key=f"conversation:{c['id']}"),
                                 priority=self._URGENT_PRIORITY["conversation"])

    def _enqueue_background(self, q) -> None:
        """Enqueue one lightweight ticket per available background source."""
        from assistant.work_queue import WorkItem
        available = {
            "todos": self._todos is not None,
            "research": self._agenda is not None,
            "interests": True,
            "dream": self._memory is not None,
        }
        for kind, weight in self._BACKGROUND_WEIGHTS.items():
            if available.get(kind):
                q.put_background(WorkItem(kind, key=kind), weight=weight)

    def _dispatch_item(self, q, item) -> None:
        """Generate the goal for a selected item (lazily for background) and run it."""
        key = item.dedup_key()
        try:
            if item.kind == "scheduled":
                self.run_cycle(item.payload, "scheduled")
            elif item.kind == "conversation":
                goal = self._conversation_goal_by_id(item.payload)
                if goal:
                    self.run_cycle(goal, "conversation")
            else:  # background: generate now, so only the chosen kind's generator fires
                goal = getattr(self, f"_goal_from_{item.kind}")()
                if goal:
                    self.run_cycle(goal, item.kind)
        except Exception:
            log.exception("Queued work cycle failed (kind=%s)", item.kind)
        finally:
            q.done(key)

    # ------------------------------------------------------------------
    # goal selection
    # ------------------------------------------------------------------

    def _pick_goal(self) -> tuple[str | None, str | None]:
        """Rotate over available goal sources: todos → research → interests → dream."""
        sources = ("todos", "research", "interests", "dream")
        for offset in range(len(sources)):
            source = sources[(self._source_cursor + offset) % len(sources)]
            goal_desc = getattr(self, f"_goal_from_{source}")()
            if goal_desc:
                self._source_cursor = (self._source_cursor + offset + 1) % len(sources)
                return goal_desc, source
        return None, None

    def _goal_from_conversations(self) -> str | None:
        """A pending conversation turn becomes a high-priority goal.

        A peer agent is blocked waiting on our reply, so this runs ahead of the
        normal source rotation (checked every tick, not interval-gated).
        """
        if not self._conversations:
            return None
        pending = [c for c in self._conversations.needs_attention(limit=5)
                   if c["attention"] == "your_turn"]
        if not pending:
            return None
        return self._conversation_goal(pending[0])

    def _conversation_goal_by_id(self, conversation_id) -> str | None:
        """Goal for a SPECIFIC conversation (queued path), if it's still our turn."""
        if not self._conversations:
            return None
        for c in self._conversations.needs_attention(limit=10):
            if c["id"] == conversation_id and c["attention"] == "your_turn":
                return self._conversation_goal(c)
        return None  # peer already got a reply, or it closed — nothing to do

    def _conversation_goal(self, c) -> str:
        hist = self._conversations.history(c["id"])
        transcript = "\n".join(
            f"  {t['turn_no']}. {t['from_agent']}: {t['message']}" for t in hist)
        topic = f" about {c['topic']}" if c["topic"] else ""
        return (
            f"You are in a conversation (#{c['id']}) with {c['last_from']}{topic} "
            f"and it is your turn (turn {c['turn_count']}/{c['max_turns']}). "
            f"The conversation so far:\n{transcript}\n\n"
            f"Reply using the talk_reply tool with conversation_id={c['id']}. "
            "Advance the discussion substantively; if it has reached a natural "
            "conclusion or the turn limit is close, set done=true to close it. "
            "A peer agent is waiting on your reply — do not ignore it."
        )

    def _goal_from_todos(self) -> str | None:
        if not self._todos:
            return None
        listing = self._todos.list_todos("pending")
        if listing.startswith("No "):
            return None
        return (
            "Review your pending TODO list and make concrete progress on ONE "
            "item — pick the most urgent or most actionable:\n\n"
            f"{listing}\n\n"
            "Use your tools to actually do the work, not just plan it. "
            "Record what you did with todo_note, and call todo_done ONLY if "
            "the item is fully complete. Finish with a summary of what you "
            "accomplished."
        )

    def _goal_from_research(self) -> str | None:
        if not self._agenda:
            return None
        topic = self._agenda.get_next_topic()
        if not topic:
            return None
        goal_line = f" Goal: {topic['goal']}" if topic.get("goal") else ""
        self._pending_topic_id = topic["id"]
        return (
            f"Advance your research on '{topic['topic']}'.{goal_line}\n"
            "Search for sources, read at least one in depth with "
            "fetch_readable, and record concrete findings with journal_add "
            f"(tags: research,{topic['topic']}). Prefer new angles over "
            "repeating what your journal already covers — check with "
            "journal_search first. Finish with a summary of what is new."
        )

    def _goal_from_interests(self) -> str | None:
        interests = _parse_interests(self._soul)
        if not interests:
            return None
        topic = interests[self._interest_cursor % len(interests)]
        self._interest_cursor += 1
        return (
            f"Explore something new about one of your core interests: {topic}.\n"
            "Search, read at least one source in depth, and journal concrete "
            "findings (tags: research). Check journal_search first so you "
            "build on prior findings instead of repeating them. Finish with "
            "a summary of what is new."
        )

    def _goal_from_dream(self) -> str | None:
        """Dream/replay: sample two memories from disjoint topics and ask
        whether a meaningful connection exists between them."""
        pair = self._sample_disjoint_memories()
        if not pair:
            return None
        (tags_a, text_a), (tags_b, text_b) = pair
        return (
            "Memory replay: here are two unrelated memories from your "
            "long-term store.\n\n"
            f"Memory A (tags: {tags_a or 'none'}): {text_a}\n\n"
            f"Memory B (tags: {tags_b or 'none'}): {text_b}\n\n"
            "Consider whether there is a genuinely meaningful connection "
            "between them — a shared structure, a way one informs the other, "
            "or a question their combination raises. You may use recall or "
            "web_search to test a suspected connection. If you find a real "
            "one, record it with save_note (tags: association) and finish by "
            "stating it. If they are simply unrelated, say so and finish — "
            "'no connection' is a perfectly good outcome; never force one."
        )

    def _sample_disjoint_memories(self):
        """Pick two random episodes whose topic tags don't overlap."""
        store = getattr(self._memory, "store", None)
        if store is None:
            return None
        try:
            rows = store.conn.execute(
                "SELECT context, action, outcome, tags FROM episodes "
                "ORDER BY RANDOM() LIMIT 40"
            ).fetchall()
        except Exception:
            log.debug("Dream sampling failed", exc_info=True)
            return None
        candidates = []
        for context, action, outcome, tags_json in rows:
            text = (outcome or context or "").strip()
            if len(text) < 30:  # too thin to dream about
                continue
            try:
                tags = {str(t).lower() for t in json.loads(tags_json or "[]")}
            except (json.JSONDecodeError, TypeError):
                tags = set()
            topics = tags - TYPE_TAGS
            candidates.append((topics, ", ".join(sorted(topics)), text[:400]))
        for i, (topics_a, label_a, text_a) in enumerate(candidates):
            for topics_b, label_b, text_b in candidates[i + 1:]:
                if topics_a.isdisjoint(topics_b) and text_a != text_b:
                    return (label_a, text_a), (label_b, text_b)
        return None

    # ------------------------------------------------------------------
    # cycle execution
    # ------------------------------------------------------------------

    def run_cycle(self, goal_desc: str, source: str) -> str:
        """Run one bounded ReAct session. Returns the outcome summary."""
        log.info("Work cycle starting (source=%s): %.120s", source, goal_desc)
        context = SharedContext(goal=Goal(description=goal_desc))
        loop = ReactLoop(self._executor, self._reasoner,
                         max_iterations=self._max_iterations)
        result = loop.run(context)

        if result.success:
            outcome = "completed"
        elif result.reflection_triggered:
            outcome = f"stopped ({result.reflection_triggered.reason})"
        else:
            outcome = "hit iteration limit"
        summary = result.summary.strip() if result.summary else ""
        if not summary and context.observations:
            summary = f"last step: {context.observations[-1][:300]}"

        if source == "research" and getattr(self, "_pending_topic_id", None) is not None:
            self._agenda.increment_cycles(self._pending_topic_id)
            self._pending_topic_id = None

        if self._journal:
            self._journal.add_entry(
                f"[work-cycle/{source}] {outcome} after {result.iterations} "
                f"steps. {summary}",
                tags=f"work-cycle,{source}",
                author="agent",
            )
        log.info("Work cycle %s after %d steps (source=%s)",
                 outcome, result.iterations, source)
        if self._on_cycle:
            try:
                self._on_cycle(source, outcome, summary)
            except Exception:
                log.exception("on_cycle callback failed")
        return summary
