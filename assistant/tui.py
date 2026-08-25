"""Personal assistant — Textual TUI.

Two-panel layout:
  Left:   conversation (user + assistant turns)
  Right:  system log (tool calls, reflections, curiosity, scheduled tasks)
  Bottom: input box

Same flags as main.py:
    python tui.py
    python tui.py --config assistant.yaml
    python tui.py --db assistant.db
    python tui.py --db assistant.db --curiosity-interval 1800

Requires: pip install textual
"""

import argparse
import json
import logging
import logging.handlers
import queue
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, TabbedContent, TabPane

from agent_tools.core.executor import ToolExecutor

from assistant.conversation import ConversationBuffer, _call_llm_raw
from assistant.inner_voice import InnerVoice
from assistant.pending_messages import PendingMessages
from assistant.skills import SkillLibrary
from assistant.soul_manager import SoulManager
from assistant.transcript import Transcript
from assistant.curiosity import CuriosityEngine, FormulationConfig
from assistant.work_cycle import WorkCycle
from assistant.nudge import NudgeMonitor
from assistant.journal import Journal
from assistant.research_agenda import ResearchAgenda
from assistant.scheduler import ReflectionScheduler
from assistant.task_scheduler import TaskScheduler
from assistant.todo import TodoDB
from assistant.tools import build_assistant_tools
from assistant.critic import CriticPass
from assistant.file_tools import FileTools
from assistant.extraction import ExtractionPass

from assistant.config import (
    _DEFAULT_MODEL,
    _SCRIPT_DIR,
    _make_llm,
    build_mail_sender as _build_mail_sender,
    build_conversation_bus as _build_conversation_bus,
    build_mailbox as _build_mailbox,
    build_memory as _build_memory,
    build_router as _build_router,
    build_session_handoff as _build_session_handoff,
    build_spawn_registry as _build_spawn_registry,
    detect_vector_backend as _detect_vector_backend,
    fmt_args as _fmt_args,
    format_tag_cloud as _format_tag_cloud,
    load_config as _load_config,
    provider_for as _provider_for,
    resolve as _resolve,
)

log = logging.getLogger(__name__)


class AssistantApp(App):
    """Textual TUI for the personal assistant."""

    CSS = """
    #panels {
        height: 1fr;
    }
    #chat-log {
        width: 2fr;
        border: solid $primary;
    }
    #sys-tabs {
        width: 1fr;
        border: solid $accent;
    }
    #input-bar {
        height: 3;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit_app", "Quit", priority=True),
        Binding("ctrl+k", "compact_ctx", "Compact context"),
        Binding("ctrl+m", "show_tab('tab-messages')", "Messages"),
        Binding("ctrl+t", "show_tab('tab-todos')", "Todos"),
        Binding("ctrl+j", "show_tab('tab-journal')", "Journal"),
    ]

    def __init__(self, cfg: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cfg = cfg
        # Worker → app: scheduled prompts from TaskScheduler
        self._prompt_queue: queue.Queue[str] = queue.Queue()
        # App (UI thread) → worker: user-typed input
        self._input_queue: queue.Queue[str | None] = queue.Queue()
        # Set by _agent_loop so tab refresh methods can read them from any method
        self._pending_messages = None
        self._soul_manager = None
        self._todo_db = None
        self._journal = None
        self._mailbox = None
        self._conversation_bus = None
        self._session_in: int = 0   # prompt tokens this session
        self._session_out: int = 0  # completion tokens this session

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="panels"):
            chat = RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
            chat.border_title = "Chat"
            yield chat
            with TabbedContent(id="sys-tabs"):
                with TabPane("System", id="tab-system"):
                    yield RichLog(id="sys-log", wrap=True, markup=True, highlight=False)
                with TabPane("Messages", id="tab-messages"):
                    with Vertical():
                        yield Input(placeholder="Search messages…", id="msg-search")
                        yield RichLog(id="msg-log", wrap=True, markup=True, highlight=False)
                with TabPane("Todos", id="tab-todos"):
                    yield RichLog(id="todo-log", wrap=True, markup=True, highlight=False)
                with TabPane("Journal", id="tab-journal"):
                    yield RichLog(id="journal-log", wrap=True, markup=True, highlight=False)
                with TabPane("Proposals", id="tab-proposals"):
                    yield RichLog(id="proposal-log", wrap=True, markup=True, highlight=False)
        yield Input(placeholder="Type a message…", id="input-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._agent_loop, thread=True, name="agent-loop")
        self.query_one("#input-bar", Input).focus()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "input-bar":
            return  # ignore Enter in search boxes
        text = event.value.strip()
        self.query_one("#input-bar", Input).value = ""
        if text:
            self._input_queue.put(text)

    def action_quit_app(self) -> None:
        self._input_queue.put(None)  # signal worker to stop
        self.exit()

    def action_compact_ctx(self) -> None:
        self._input_queue.put("compact")

    # ------------------------------------------------------------------
    # Thread-safe write helpers (called from worker thread)
    # ------------------------------------------------------------------

    def _chat(self, markup: str) -> None:
        """Write a line to the chat panel."""
        self.call_from_thread(self.query_one("#chat-log", RichLog).write, markup)

    def _sys(self, markup: str) -> None:
        """Write a line to the system log panel."""
        self.call_from_thread(self.query_one("#sys-log", RichLog).write, markup)

    def _set_title(self, title: str) -> None:
        self.call_from_thread(setattr, self, "title", title)

    def _set_subtitle(self, subtitle: str) -> None:
        self.call_from_thread(setattr, self, "sub_title", subtitle)

    def _update_alerts(self) -> None:
        """Refresh the header subtitle with counts of items needing attention."""
        parts = []
        if self._pending_messages:
            n = self._pending_messages.count_unread()
            if n:
                parts.append(f"{n} msg{'s' if n > 1 else ''}")
        if self._soul_manager:
            n = self._soul_manager.count_pending()
            if n:
                parts.append(f"{n} proposal{'s' if n > 1 else ''}")
        if self._todo_db:
            pending = self._todo_db.list_todos("pending")
            if not pending.startswith("No "):
                n = len([l for l in pending.splitlines() if l.strip()])
                if n:
                    parts.append(f"{n} todo{'s' if n > 1 else ''}")
        if self._session_in or self._session_out:
            parts.append(f"↑{self._session_in:,} ↓{self._session_out:,} tok")
        self._set_subtitle("  ·  ".join(parts))
        self._update_tab_labels()

    def _update_tab_labels(self) -> None:
        """Refresh tab labels with live counts — safe to call from worker thread."""
        def _do() -> None:
            try:
                tabs = self.query_one("#sys-tabs", TabbedContent)
                if self._pending_messages:
                    n = self._pending_messages.count_unread()
                    tabs.get_tab("tab-messages").label = f"Messages ({n})" if n else "Messages"
                if self._soul_manager:
                    n = self._soul_manager.count_pending()
                    tabs.get_tab("tab-proposals").label = f"Proposals ({n})" if n else "Proposals"
            except Exception:
                pass
        self.call_from_thread(_do)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Refresh data tabs when switched to."""
        pane_id = event.pane.id if event.pane else None
        if pane_id == "tab-messages":
            self._refresh_messages_tab()
        elif pane_id == "tab-todos":
            self._refresh_todos_tab()
        elif pane_id == "tab-journal":
            self._refresh_journal_tab()
        elif pane_id == "tab-proposals":
            self._refresh_proposals_tab()

    def action_show_tab(self, tab_id: str) -> None:
        """Switch to a named tab (used by keybindings)."""
        try:
            self.query_one("#sys-tabs", TabbedContent).active = tab_id
        except Exception:
            pass

    def _refresh_messages_tab(self, search: str = "") -> None:
        log = self.query_one("#msg-log", RichLog)
        log.clear()
        if not self._pending_messages:
            log.write("[dim]No messages yet.[/dim]")
            return

        if search:
            msgs = self._pending_messages.search(search)
            total = len(msgs)
            log.write(f"[dim]Search: [bold]{escape(search)}[/bold] — {total} result(s)[/dim]\n")
        else:
            msgs = self._pending_messages.get_all(limit=100)
            total_all = self._pending_messages.count_all()
            showing = min(len(msgs), 100)
            unread = sum(1 for m in msgs if not m["read"])
            summary = f"{unread} unread" if unread else "all read"
            more = f", {total_all - showing} older not shown" if total_all > showing else ""
            log.write(f"[dim]Showing {showing} of {total_all} ({summary}{more})[/dim]\n")

        for m in msgs:
            ts = (m["created_at"] or "")[:16]
            urgency = m.get("urgency", "low")
            urg_colour = "red" if urgency == "high" else ("yellow" if urgency == "medium" else "")
            urg_tag = f"[{urg_colour}][{urgency}][/{urg_colour}] " if urg_colour else ""
            topic = f"[italic]{escape(m['topic'])}[/italic] — " if m.get("topic") else ""
            prefix = "[bold]●[/bold] " if not m["read"] else "[dim]○[/dim] "
            log.write(f"{prefix}[dim]{ts}[/dim] {urg_tag}{topic}{escape(m['message'])}")

        # Auto-mark unread as read (only on full view, not on search)
        if not search:
            for m in msgs:
                if not m["read"]:
                    try:
                        self._pending_messages.mark_read(m["id"])
                    except Exception:
                        pass
            try:
                self.query_one("#sys-tabs", TabbedContent).get_tab("tab-messages").label = "Messages"
            except Exception:
                pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "msg-search":
            self._refresh_messages_tab(search=event.value.strip())

    def _refresh_todos_tab(self) -> None:
        log = self.query_one("#todo-log", RichLog)
        log.clear()
        if not self._todo_db:
            log.write("[dim]No data yet.[/dim]")
            return
        for status, colour in (("pending", "yellow"), ("in_progress", "cyan")):
            result = self._todo_db.list_todos(status)
            if not result.startswith("No "):
                log.write(f"[bold {colour}]{status.replace('_', ' ').title()}[/bold {colour}]")
                for line in result.splitlines():
                    log.write(f"  {escape(line)}")
                log.write("")

    def _refresh_journal_tab(self) -> None:
        log = self.query_one("#journal-log", RichLog)
        log.clear()
        if not self._journal:
            log.write("[dim]No data yet.[/dim]")
            return
        from datetime import date, timedelta
        for i in range(5):
            day = (date.today() - timedelta(days=i)).isoformat()
            label = "Today" if i == 0 else ("Yesterday" if i == 1 else day)
            result = self._journal.read_day(day)
            if not result.startswith("No "):
                log.write(f"[bold cyan]{label}[/bold cyan]")
                for line in result.splitlines():
                    log.write(f"  {escape(line)}")
                log.write("")

    def _refresh_proposals_tab(self) -> None:
        log = self.query_one("#proposal-log", RichLog)
        log.clear()
        if not self._soul_manager:
            log.write("[dim]No data yet.[/dim]")
            return
        result = self._soul_manager.list_proposals("pending")
        if result.startswith("No "):
            log.write("[dim]No pending proposals.[/dim]")
        else:
            log.write("[bold magenta]Pending soul proposals[/bold magenta]\n")
            for line in result.splitlines():
                log.write(escape(line))

    # ------------------------------------------------------------------
    # Agent loop (runs in worker thread via run_worker)
    # ------------------------------------------------------------------

    def _agent_loop(self) -> None:  # noqa: C901 (complexity ok — mirrors main.py)
        cfg = self._cfg
        model = cfg["model"]
        db = cfg["db"]
        data_db_path = cfg["data_db"]
        soul_path = cfg.get("soul_path")
        reflect_interval = cfg["reflect_interval"]
        curiosity_interval = cfg["curiosity_interval"]
        nudge_interval = cfg.get("nudge_interval", 0)
        max_chars = cfg["max_chars"]

        soul_manager = SoulManager(soul_arg=soul_path, data_db=data_db_path)
        soul, soul_source = soul_manager.load()
        short_model = model.split("/")[-1]
        display_name = cfg.get("name") or "Assistant"
        self._set_title(f"{display_name} ({short_model})")

        llm = _make_llm(model, max_tokens=8192, min_request_interval=1.0,
                        sampling=cfg.get("sampling"))
        if not llm.is_available():
            _provider, _env = _provider_for(model)
            if _env is None:
                self._sys(f"[red bold]Error: cannot reach {_provider} server for "
                          f"model '{model}' — is llama-server running?[/red bold]")
            else:
                self._sys(f"[red bold]Error: {_env} not set (required for "
                          f"{_provider} model '{model}').[/red bold]")
            self.call_from_thread(self.exit)
            return

        router = _build_router(cfg, llm)
        critic = CriticPass(router.for_task("conversation"), soul) if cfg.get("critic_pass") else None
        memory = _build_memory(db, llm=llm) if db else None
        extraction_interval = cfg.get("extraction_interval", 0)
        vec_backend = _detect_vector_backend() if memory else "off"
        data_db = Path(data_db_path)
        transcript = Transcript(data_db.with_suffix("").parent / "assistant_transcript.md")
        # Rolling log file
        log_file = data_db.with_suffix("").parent / "assistant.log"
        _fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(_fh)
        pending_messages = PendingMessages(data_db)
        self._pending_messages = pending_messages
        self._soul_manager = soul_manager
        inner_voice = InnerVoice(llm=router.for_task("inner_voice"), soul_text=soul, pending_messages=pending_messages)
        todo_db = TodoDB(data_db)
        self._todo_db = todo_db
        journal = Journal(data_db)
        self._journal = journal
        research_agenda = ResearchAgenda(data_db)
        task_scheduler = TaskScheduler(data_db, self._prompt_queue)
        buffer = ConversationBuffer(max_chars=max_chars)
        if memory:
            tag_cloud = _format_tag_cloud(memory)
            if tag_cloud:
                buffer.set_memory_index(tag_cloud)

        # Session handoff (opt-in): resume with last session's texture, not cold.
        session_handoff = _build_session_handoff(cfg, data_db.parent)
        if session_handoff:
            _last_note = session_handoff.read_latest()
            if _last_note:
                buffer.set_handoff(_last_note)
                self._sys("[dim][Handoff: resumed from last session's note][/dim]")

        skills_dir = (
            soul_manager._soul_path.parent / "skills"
            if soul_manager._soul_path
            else _SCRIPT_DIR / "skills"
        )
        skill_library = SkillLibrary(skills_dir)
        config_dir = cfg.get("config_dir")
        spawn_registry = _build_spawn_registry(cfg, config_dir=config_dir)
        mailbox = _build_mailbox(cfg, config_dir=config_dir)
        self._mailbox = mailbox
        conversation_bus = _build_conversation_bus(cfg, config_dir=config_dir)
        self._conversation_bus = conversation_bus
        mail_sender = _build_mail_sender(cfg)
        work_dir_str = cfg.get("work_dir")
        file_tools = FileTools(Path(work_dir_str)) if work_dir_str else None
        enabled = cfg.get("tools")
        registry = build_assistant_tools(
            buffer, llm, memory, todo_db, journal,
            task_scheduler=task_scheduler,
            research_agenda=research_agenda,
            soul_manager=soul_manager,
            pending_messages=pending_messages,
            skill_library=skill_library,
            spawn_registry=spawn_registry,
            mailbox=mailbox,
            conversation_bus=conversation_bus,
            file_tools=file_tools,
            mail_sender=mail_sender,
            digest_llm=router.for_task("compaction"),
            enabled=enabled,
            sandbox_cfg=cfg.get("sandbox"),
        )
        extractor = (
            ExtractionPass(router.for_task("conversation"), memory,
                           every_n_turns=extraction_interval)
            if extraction_interval > 0 and memory else None
        )
        executor = ToolExecutor(registry, permission_checker=lambda _: True)
        tool_schemas = registry.to_schemas()

        task_scheduler.start()

        # Background reflection
        reflection_scheduler = None
        if memory and reflect_interval > 0:
            def _do_reflect() -> None:
                try:
                    hours = max(1, (reflect_interval * 2) // 3600)
                    result = memory.reflect_on_recent(hours=hours)
                    if result and result.success:
                        data = result.data or {}
                        insight = data.get("insight", "")
                        takeaway = data.get("actionable_takeaway", "")
                        n = data.get("episodes_analyzed", "?")
                        summary = insight[:120] if insight else result.message
                        # Log and persist first — these must not be lost if the TUI
                        # screen is gone (e.g. after exit() while thread still runs).
                        log.info("Reflection complete (%s episodes): %s", n, insight)
                        if takeaway:
                            log.info("Reflection takeaway: %s", takeaway)
                        transcript.system(f"Reflection ({n} eps): {insight}" +
                                          (f" | Takeaway: {takeaway}" if takeaway else ""))
                        if insight:
                            inner_voice.evaluate(insight, topic="reflection", source="reflection")
                        try:
                            self._sys(f"[dim][Reflection ({n} eps): {escape(summary)}][/dim]")
                        except Exception:
                            pass  # TUI screen may be gone; result is already logged
                except Exception as exc:
                    log.warning("Reflection failed: %s", exc)

            reflection_scheduler = ReflectionScheduler(
                interval_seconds=reflect_interval,
                callback=_do_reflect,
            )
            reflection_scheduler.start()

        # Background nudge monitor
        nudge_monitor = None
        if nudge_interval > 0:
            nudge_monitor = NudgeMonitor(
                llm=router.for_task("nudge"),
                soul_text=soul,
                pending_messages=pending_messages,
                buffer=buffer,
                interval_seconds=nudge_interval,
            )
            nudge_monitor.start()
            if nudge_monitor.enabled:
                self._sys(f"[bold]Nudge  :[/bold] every {nudge_interval}s")

        # Background curiosity
        curiosity = None
        if curiosity_interval > 0:
            curiosity = CuriosityEngine(
                soul_text=soul,
                llm=router.for_task("curiosity"),
                journal=journal,
                memory_tools=memory,
                research_agenda=research_agenda,
                interval_seconds=curiosity_interval,
                formulation=FormulationConfig.from_dict(cfg.get("curiosity_formulation") or {}),
                on_finding=lambda topic, summary: (
                    buffer.add_background_note(f"Researched '{topic}': {summary}"),
                    self._sys(
                        f"[yellow bold][Curiosity: {escape(topic)}][/yellow bold]\n"
                        f"[dim]{escape(summary)}[/dim]"
                    ),
                    transcript.system(f"Curiosity: {topic} — {summary[:120]}"),
                    inner_voice.evaluate(summary, topic=topic, source="curiosity"),
                ),
            )
            curiosity.start()

        # Autonomous work cycle — bounded ReAct sessions over the tool registry
        wc_cfg = cfg.get("work_cycle") or {}
        wc_interval = int(wc_cfg.get("interval", 0))
        work_cycle = None
        if wc_interval > 0:
            work_cycle = WorkCycle(
                llm=router.for_task("curiosity"),
                executor=executor,
                soul_text=soul,
                journal=journal,
                todo_db=todo_db,
                research_agenda=research_agenda,
                memory_tools=memory,
                conversation_bus=conversation_bus,
                interval_seconds=wc_interval,
                max_iterations=int(wc_cfg.get("max_iterations", 8)),
                on_cycle=lambda source, outcome, summary: (
                    buffer.add_background_note(
                        f"[Work cycle/{source}] {outcome}: {summary[:400]}"),
                    self._sys(
                        f"[green bold][Work cycle/{escape(source)}: {escape(outcome)}][/green bold]\n"
                        f"[dim]{escape(summary)}[/dim]"
                    ),
                    transcript.system(f"Work cycle ({source}): {outcome} — {summary[:120]}"),
                    inner_voice.evaluate(summary, topic=source, source="work-cycle")
                    if summary else None,
                ),
            )
            work_cycle.start()

        # Startup info in system log
        self._sys(f"[bold]Model  :[/bold] {short_model}")
        self._sys(f"[bold]Soul   :[/bold] {escape(soul_source)}")
        self._sys(f"[bold]Memory :[/bold] {escape(db) + ' [' + vec_backend + ']' if memory else 'off'}")
        self._sys(f"[bold]Data   :[/bold] {escape(data_db_path)}")
        self._sys(f"[bold]Log    :[/bold] {escape(str(log_file))} | {escape(transcript._path.name)}")
        router_info = router.describe()
        if router_info:
            self._sys(f"[bold]Models :[/bold] {escape(router_info.strip())}")
        if memory:
            self._sys(f"[bold]Reflect:[/bold] every {reflect_interval}s")
        if curiosity:
            self._sys(f"[bold]Curious:[/bold] every {curiosity_interval}s")
        if mailbox:
            self._sys(f"[bold]Mailbox:[/bold] {escape(mailbox.name)} @ {escape(str(cfg.get('mailbox_db', '?')))}")
        if spawn_registry:
            self._sys(f"[bold]Spawn  :[/bold] {escape(', '.join(spawn_registry.names()))}")
        if file_tools:
            self._sys(f"[bold]Files  :[/bold] {escape(str(file_tools.work_dir))}")
        if mail_sender:
            self._sys(f"[bold]Mail   :[/bold] {escape(', '.join(mail_sender.aliases))}")
        if work_cycle:
            self._sys(f"[bold]Work   :[/bold] every {wc_interval}s (max {int(wc_cfg.get('max_iterations', 8))} steps)")
        self._sys("[dim]─────────────────────────────[/dim]")

        # Startup recap: pending messages from inner voice
        unread = pending_messages.format_unread()
        if unread:
            self._sys("[bold magenta]Messages from assistant:[/bold magenta]")
            for line in unread.splitlines():
                self._sys(f"  {escape(line)}")

        # Pending soul proposals
        n_proposals = soul_manager.count_pending()
        if n_proposals > 0:
            self._sys(f"[bold magenta]{n_proposals} pending soul proposal(s) — use list_soul_proposals[/bold magenta]")

        # Pending TODOs + today's journal
        pending = todo_db.list_todos("pending")
        if not pending.startswith("No "):
            self._sys("[bold yellow]Pending TODOs:[/bold yellow]")
            for line in pending.splitlines():
                self._sys(f"  {escape(line)}")

        today_journal = journal.read_day("today")
        if not today_journal.startswith("No "):
            self._sys("[bold yellow]Today's Journal:[/bold yellow]")
            for line in today_journal.splitlines():
                self._sys(f"  {escape(line)}")

        # Auto-recall: surface recent memories before the first turn
        if memory:
            try:
                result = memory.recall_similar(soul[:400], limit=5)
                memories = result.data.get("memories", []) if result.success else []
                for m in memories:
                    buffer.add_background_note(f"From memory: {m}")
                if memories:
                    self._sys(f"[dim]Recalled {len(memories)} memories from previous sessions.[/dim]")
            except Exception as exc:
                log.debug("Auto-recall failed: %s", exc)

        self._update_alerts()

        try:
            while True:
                # Drain scheduled tasks before waiting for user input
                try:
                    scheduled = self._prompt_queue.get_nowait()
                    self._sys(f"[bold cyan][Scheduled task fired][/bold cyan]")
                    self._chat(f"[dim italic][Scheduled][/dim italic] {escape(scheduled[:80])}")
                    user_input = scheduled
                except queue.Empty:
                    # Block until user types something (or 5s timeout to recheck prompt_queue)
                    try:
                        user_input = self._input_queue.get(timeout=5.0)
                    except queue.Empty:
                        continue

                if user_input is None:
                    break
                if user_input.lower() in ("quit", "exit"):
                    self.call_from_thread(self.exit)
                    break
                if user_input.lower() == "compact":
                    result = buffer.compact(router.for_task("compaction"), memory)
                    self._sys(f"[dim][Compact: {escape(str(result))}][/dim]")
                    transcript.system(f"Context compacted: {result}")
                    continue

                transcript.user(user_input)
                self._chat(f"[bold green]You:[/bold green] {escape(user_input)}")
                buffer.add_user(user_input)

                # Auto-compact before LLM call
                if buffer.should_compact():
                    self._sys("[dim][Auto-compacting context…][/dim]")
                    result = buffer.compact(router.for_task("compaction"), memory)
                    transcript.system(f"Auto-compacted: {result}")
                    # Re-add the current user message so it isn't lost in the summary
                    buffer.add_user(user_input)

                # Inject unread nudges into system prompt so the agent actually sees them
                if self._pending_messages:
                    unread_nudges = [m for m in self._pending_messages.get_all(limit=5)
                                     if not m["read"] and m.get("source") == "nudge"]
                    for m in unread_nudges[-2:]:
                        buffer.add_background_note(f"[Self-monitoring] {m['message']}")
                        self._pending_messages.mark_read(m["id"])

                # Inject unread mailbox messages from other agents
                if self._mailbox:
                    unread_mail = self._mailbox.inbox(unread_only=True, limit=3)
                    for m in unread_mail:
                        topic_str = f" [{m['topic']}]" if m.get("topic") else ""
                        buffer.add_background_note(
                            f"[Mailbox from {m['from_agent']}{topic_str}] {m['message']}"
                        )
                        self._mailbox.mark_read(m["id"])

                # Inject conversations awaiting this agent (its turn, or an outcome)
                if self._conversation_bus:
                    for c in self._conversation_bus.needs_attention(limit=3):
                        if c["attention"] == "your_turn":
                            buffer.add_background_note(
                                f"[Conversation #{c['id']} — your turn] {c['last_from']} "
                                f"said: {c['last_message']} (reply with talk_reply; "
                                f"turn {c['turn_count']}/{c['max_turns']})"
                            )
                        else:
                            buffer.add_background_note(
                                f"[Conversation #{c['id']} — {c['state']}] {c['last_from']} "
                                f"said: {c['last_message']}"
                            )
                            self._conversation_bus.acknowledge(c["id"])

                # Agentic tool loop
                while True:
                    try:
                        response = _call_llm_raw(
                            router.for_task("conversation"),
                            buffer.messages_for_llm(soul),
                            tools=tool_schemas or None,
                        )
                    except Exception as exc:
                        self._sys(f"[red][LLM error: {escape(str(exc))}][/red]")
                        break

                    usage = response.get("usage", {})
                    self._session_in += usage.get("prompt_tokens", 0)
                    self._session_out += usage.get("completion_tokens", 0)

                    msg = response["choices"][0]["message"]
                    content = (msg.get("content") or "").strip()
                    tool_calls = msg.get("tool_calls") or []

                    buffer.add_assistant(content or None, tool_calls or None)

                    # Surface assistant text whenever present — even alongside
                    # tool calls (e.g. a final answer emitted in the same message
                    # as a trailing compact_context call). Gating on "no
                    # tool_calls" silently dropped it and compaction ate it.
                    if content:
                        self._chat(f"[bold blue]Assistant:[/bold blue] {escape(content)}")
                        transcript.assistant(content)

                    if not tool_calls:
                        if content:
                            if critic and critic.enabled:
                                flagged = critic.check(content, buffer)
                                if flagged:
                                    self._sys("[dim][Self-check: deference flagged — note injected][/dim]")
                            if extractor and extractor.enabled:
                                n = extractor.maybe_extract(buffer)
                                if n:
                                    self._sys(f"[dim][Extracted {n} fact(s) to memory][/dim]")
                        self._update_alerts()
                        break

                    # Execute each tool call and log to system panel
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}
                        tc_id = tc.get("id", "")

                        self._sys(
                            f"[cyan]⚙ {escape(name)}[/cyan]"
                            f"[dim]({escape(_fmt_args(args))})[/dim]"
                        )
                        try:
                            result = executor.execute(name, **args)
                            output = str(result.output if result.success else result.error or "error")
                        except Exception as exc:
                            output = f"Tool error: {exc}"
                        self._sys(f"  [dim]→ {escape(output[:160])}[/dim]")
                        transcript.tool(name, output)
                        buffer.add_tool_result(tc_id, name, output)

                # Agent-requested compaction runs here, after the turn's work is
                # done — never mid-loop, where it would wipe the active task.
                deferred = buffer.maybe_compact(router.for_task("compaction"), memory)
                if deferred:
                    self._sys(f"[dim][Compact: {escape(str(deferred))}][/dim]")
                    transcript.system(f"Deferred compaction: {deferred}")

        finally:
            # Session handoff note — capture texture BEFORE exit-compaction
            # clears the buffer. Never breaks shutdown.
            if session_handoff:
                try:
                    _convo = buffer.recent_turns_text(n=12)
                    if buffer._summary:
                        _convo = f"[Earlier summary: {buffer._summary}]\n\n{_convo}"
                    if session_handoff.make_note(router.for_task("compaction"), _convo):
                        log.info("Handoff note saved for next session")
                except Exception as exc:
                    log.warning("Handoff note failed: %s", exc)

            # Auto-compact on exit: save this session to long-term memory
            if memory and buffer._messages:
                try:
                    result = buffer.compact(router.for_task("compaction"), memory)
                    log.info("Session saved on exit: %s", result)
                    transcript.system(f"Session saved on exit: {result}")
                except Exception as exc:
                    log.warning("Auto-compact on exit failed: %s", exc)
            task_scheduler.stop()
            if reflection_scheduler:
                reflection_scheduler.stop()
            if curiosity:
                curiosity.stop()
            if nudge_monitor:
                nudge_monitor.stop()
            if work_cycle:
                work_cycle.stop()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal assistant TUI")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite DB for memory (default: assistant_memory.db)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable long-term memory entirely")
    parser.add_argument("--reflect-interval", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--data-db", type=str, default=None)
    parser.add_argument("--soul", type=str, default=None)
    parser.add_argument("--curiosity-interval", type=int, default=None)
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    file_cfg = _load_config(args.config)

    cfg = {
        "model":              _resolve(args.model,              file_cfg, "model",              _DEFAULT_MODEL),
        "db":                 None if args.no_memory else _resolve(args.db, file_cfg, "db", "assistant_memory.db"),
        "data_db":            _resolve(args.data_db,            file_cfg, "data_db",            "assistant_data.db"),
        "soul_path":          _resolve(args.soul,               file_cfg, "soul",               None),
        "reflect_interval":   _resolve(args.reflect_interval,   file_cfg, "reflect_interval",   3600),
        "curiosity_interval": _resolve(args.curiosity_interval, file_cfg, "curiosity_interval", 0),
        "nudge_interval":     file_cfg.get("nudge_interval", 0),
        "max_chars":          _resolve(args.max_chars,          file_cfg, "max_chars",          32_000),
        "name":               file_cfg.get("name"),
        "tools":              file_cfg.get("tools"),
        "sandbox":            file_cfg.get("sandbox"),
        "mailbox_db":         file_cfg.get("mailbox_db"),
        "spawn_roles":        file_cfg.get("spawn_roles"),
        "config_dir":         Path(args.config).parent if args.config else None,
        "critic_pass":          file_cfg.get("critic_pass", False),
        "work_dir":             file_cfg.get("work_dir"),
        "mail":                 file_cfg.get("mail"),
        "work_cycle":           file_cfg.get("work_cycle"),
        "extraction_interval":  file_cfg.get("extraction_interval", 0),
    }

    log_level = _resolve(args.log_level, file_cfg, "log_level", "WARNING")
    logging.basicConfig(level=getattr(logging, log_level))

    app = AssistantApp(cfg=cfg)
    app.run()


if __name__ == "__main__":
    main()
