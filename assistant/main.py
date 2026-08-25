"""Personal assistant — conversational CLI.

A persistent conversation loop backed by a Fireworks fast model, with:
  - Context compaction (agent-initiated or automatic at --max-chars)
  - Long-term memory via agent-memory (optional, --db)
  - Periodic background reflection (--reflect-interval, default 1h)

Usage:
    python main.py
    python main.py --config assistant.yaml
    python main.py --db assistant.db
    python main.py --db assistant.db --reflect-interval 1800
    python main.py --max-chars 16000   # compact sooner

Config file (YAML) keys — all optional, CLI flags override:
    soul, model, db, data_db, max_chars,
    reflect_interval, curiosity_interval, log_level
"""

import argparse
import faulthandler
import json
import logging
import logging.handlers
import queue
import select
import signal
import threading
try:
    import readline
    _READLINE = True
except ImportError:
    readline = None  # type: ignore[assignment]
    _READLINE = False
import sys
from pathlib import Path

from agent_tools.core.executor import ToolExecutor

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
from assistant.conversation import ConversationBuffer, _call_llm_raw
from assistant.inner_voice import InnerVoice
from assistant.pending_messages import PendingMessages
from assistant.skills import SkillLibrary
from assistant.soul_manager import SoulManager
from assistant.transcript import Transcript
from assistant.curiosity import CuriosityEngine, FormulationConfig
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
from assistant.work_cycle import WorkCycle

HISTORY_FILE = Path.home() / ".assistant_history"


def _setup_readline() -> None:
    if not _READLINE:
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    try:
        readline.set_history_length(1000)
    except Exception:
        pass


def _save_readline_history() -> None:
    if not _READLINE:
        return
    try:
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


def _input(prompt: str, timeout: float | None = None) -> str | None:
    """Read a line from stdin with an optional timeout (seconds).

    Returns the input string, or None on timeout.
    Falls back to blocking input() when timeout is None or select unavailable.
    """
    if timeout is not None:
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            ready = select.select([sys.stdin], [], [], timeout)[0]
            if not ready:
                # Erase the prompt so it doesn't litter the screen on each tick
                sys.stdout.write("\r" + " " * len(prompt) + "\r")
                sys.stdout.flush()
                return None
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return line.rstrip("\n")
        except (AttributeError, OSError):
            pass  # select not available (Windows) — fall through
    try:
        return input(prompt)
    except Exception:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return sys.stdin.readline().rstrip("\n")



def _reflect(memory_tools, interval_seconds: int = 3600, inner_voice=None, transcript=None) -> None:
    """Background reflection callback — called by ReflectionScheduler."""
    _log = logging.getLogger(__name__)
    try:
        # Look back 2x the interval so we never miss notes saved just after
        # the previous reflection fired.
        hours = max(1, (interval_seconds * 2) // 3600)
        result = memory_tools.reflect_on_recent(hours=hours)
        if result and result.success:
            data = result.data or {}
            insight = data.get("insight", "")
            takeaway = data.get("actionable_takeaway", "")
            n = data.get("episodes_analyzed", "?")
            summary = insight[:120] if insight else result.message
            print(f"\n[Reflection: {summary}]\n", flush=True)
            _log.info("Reflection complete (%s episodes): %s", n, insight)
            if takeaway:
                _log.info("Reflection takeaway: %s", takeaway)
            if transcript:
                transcript.system(f"Reflection ({n} eps): {insight}" +
                                  (f" | Takeaway: {takeaway}" if takeaway else ""))
            if inner_voice and insight:
                inner_voice.evaluate(insight, topic="reflection", source="reflection")
    except Exception as exc:
        _log.warning("Reflection failed: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Personal assistant")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (CLI flags override individual values)")
    parser.add_argument("--model", type=str, default=None,
                        help="Fireworks model ID")
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite DB path for memory (default: assistant_memory.db)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable long-term memory entirely")
    parser.add_argument("--reflect-interval", type=int, default=None,
                        help="Background reflection interval in seconds (default 3600)")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Auto-compact when conversation exceeds this many chars (default 32000)")
    parser.add_argument("--data-db", type=str, default=None,
                        help="SQLite DB for TODOs and journal (default: ./assistant_data.db)")
    parser.add_argument("--soul", type=str, default=None,
                        help="Path to soul file (default: ./soul.txt or bundled)")
    parser.add_argument("--curiosity-interval", type=int, default=None,
                        help="Curiosity research interval in seconds (0 = off, e.g. 1800)")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default WARNING)")
    parser.add_argument("--daemon", action="store_true",
                        help="Headless mode: no interactive input, driven by "
                             "work cycle + scheduled tasks + mailbox")
    args = parser.parse_args()

    # Load config file, then resolve each setting: CLI > YAML > default
    cfg = _load_config(args.config)
    model            = _resolve(args.model,            cfg, "model",            _DEFAULT_MODEL)
    name             = _resolve(None,                  cfg, "name",             None)
    db               = None if args.no_memory else _resolve(args.db, cfg, "db", "assistant_memory.db")
    data_db_path     = _resolve(args.data_db,          cfg, "data_db",          "assistant_data.db")
    soul_path        = _resolve(args.soul,             cfg, "soul",             None)
    reflect_interval = _resolve(args.reflect_interval, cfg, "reflect_interval", 3600)
    curiosity_interval = _resolve(args.curiosity_interval, cfg, "curiosity_interval", 0)
    nudge_interval   = _resolve(None,                  cfg, "nudge_interval",   0)
    critic_pass        = cfg.get("critic_pass", False)
    extraction_interval = cfg.get("extraction_interval", 0)
    wc_cfg             = cfg.get("work_cycle") or {}
    wc_interval        = int(wc_cfg.get("interval", 0))
    max_chars          = _resolve(args.max_chars,        cfg, "max_chars",        32_000)
    log_level        = _resolve(args.log_level,        cfg, "log_level",        "WARNING")

    logging.basicConfig(level=getattr(logging, log_level))
    # Rolling log file — 5 MB per file, keep 3 backups
    log_file = Path(data_db_path).with_suffix("").parent / "assistant.log"
    _fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)
    faulthandler.enable()

    soul_manager = SoulManager(soul_arg=soul_path, data_db=data_db_path)
    soul, soul_source = soul_manager.load()
    llm = _make_llm(model, max_tokens=8192, min_request_interval=1.0,
                    sampling=cfg.get("sampling"))
    if not llm.is_available():
        _provider, _env = _provider_for(model)
        if _env is None:  # local server: availability is a health check, not a key
            print(f"Error: cannot reach {_provider} server for model '{model}' — "
                  f"is llama-server running and listening on the configured host/port?",
                  file=sys.stderr)
        else:
            print(f"Error: {_env} not set (required for {_provider} model '{model}').",
                  file=sys.stderr)
        sys.exit(1)

    router = _build_router(cfg, llm)
    critic = CriticPass(router.for_task("conversation"), soul) if critic_pass else None
    memory = _build_memory(db, llm=llm) if db else None
    extractor = ExtractionPass(router.for_task("conversation"), memory,
                               every_n_turns=extraction_interval) \
        if extraction_interval > 0 and memory else None
    vec_backend = _detect_vector_backend() if memory else "off"
    data_db = Path(data_db_path)
    transcript = Transcript(data_db.with_suffix("").parent / "assistant_transcript.md")
    pending_messages = PendingMessages(data_db)
    inner_voice = InnerVoice(llm=router.for_task("inner_voice"), soul_text=soul, pending_messages=pending_messages)
    todo_db = TodoDB(data_db)
    journal = Journal(data_db)
    research_agenda = ResearchAgenda(data_db)
    prompt_queue: queue.Queue = queue.Queue()
    task_scheduler = TaskScheduler(data_db, prompt_queue)
    buffer = ConversationBuffer(max_chars=max_chars)
    if memory:
        tag_cloud = _format_tag_cloud(memory)
        if tag_cloud:
            buffer.set_memory_index(tag_cloud)

    # Session handoff (opt-in): resume with last session's texture, not cold.
    session_handoff = _build_session_handoff(cfg, Path(data_db_path).parent)
    if session_handoff:
        _last_note = session_handoff.read_latest()
        if _last_note:
            buffer.set_handoff(_last_note)
            print("[Handoff: resumed from last session's note]")
    skills_dir = (
        soul_manager._soul_path.parent / "skills"
        if soul_manager._soul_path
        else _SCRIPT_DIR / "skills"
    )
    skill_library = SkillLibrary(skills_dir)
    config_dir = Path(args.config).parent if args.config else None
    spawn_registry = _build_spawn_registry(cfg, config_dir=config_dir)
    mailbox = _build_mailbox(cfg, config_dir=config_dir)
    conversation_bus = _build_conversation_bus(cfg, config_dir=config_dir)
    mail_sender = _build_mail_sender(cfg)
    work_dir_str = cfg.get("work_dir")
    file_tools = FileTools(Path(work_dir_str)) if work_dir_str else None
    registry = build_assistant_tools(buffer, llm, memory, todo_db, journal,
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
                                     enabled=cfg.get("tools"),
                                     sandbox_cfg=cfg.get("sandbox"))
    executor = ToolExecutor(registry, permission_checker=lambda _: True)
    tool_schemas = registry.to_schemas()  # OpenAI function-calling format

    task_scheduler.start()

    # Periodic reflection
    scheduler = None
    if memory and reflect_interval > 0:
        scheduler = ReflectionScheduler(
            interval_seconds=reflect_interval,
            callback=lambda: _reflect(memory, reflect_interval, inner_voice, transcript),
        )
        scheduler.start()

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

    # Background curiosity research
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
                print(f"\n[Curiosity: {topic}]\n{summary}\n", flush=True),
                transcript.system(f"Curiosity: {topic} — {summary[:120]}"),
                inner_voice.evaluate(summary, topic=topic, source="curiosity"),
            ),
        )
        curiosity.start()

    # Autonomous work cycle — bounded ReAct sessions over the tool registry
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
                transcript.system(
                    f"Work cycle ({source}): {outcome} — {summary[:120]}"),
                inner_voice.evaluate(summary, topic=source, source="work-cycle")
                if summary else None,
            ),
        )
        work_cycle.start()

    # Startup banner
    cfg_info    = f"  Config : {args.config}" if args.config else ""
    mem_info    = f"  Memory : {db} [{vec_backend}]" if memory else "  Memory : off"
    data_info   = f"  Data   : {data_db_path}  (todos + journal)"
    log_info    = f"  Log    : {log_file} | {transcript._path.name}"
    soul_info   = f"  Soul   : {soul_source}"
    router_info = router.describe()
    ref_info    = (f"  Reflect: every {reflect_interval}s" if memory and reflect_interval > 0
                   else "  Reflect: off" if memory else "")
    cur_info    = f"  Curious: every {curiosity_interval}s" if curiosity else ""
    nudge_info  = f"  Nudge  : every {nudge_interval}s" if nudge_monitor and nudge_monitor.enabled else ""
    files_info  = f"  Files  : {file_tools.work_dir}" if file_tools else ""
    mail_info   = f"  Mail   : {', '.join(mail_sender.aliases)}" if mail_sender else ""
    work_info   = f"  Work   : every {wc_interval}s (max {int(wc_cfg.get('max_iterations', 8))} steps)" if work_cycle else ""
    print(f"{name or 'Assistant'} ({model.split('/')[-1]})")
    if cfg_info:
        print(cfg_info)
    print(soul_info)
    print(mem_info)
    print(data_info)
    print(log_info)
    if router_info:
        print(router_info)
    if ref_info:
        print(ref_info)
    if cur_info:
        print(cur_info)
    if nudge_info:
        print(nudge_info)
    if files_info:
        print(files_info)
    if mail_info:
        print(mail_info)
    if work_info:
        print(work_info)
    print("  Type 'quit' to exit, 'compact' to compact context manually.")
    print()

    # Startup recap: pending messages from inner voice
    unread = pending_messages.format_unread()
    if unread:
        print(unread)
        print()

    # Pending soul proposals
    n_proposals = soul_manager.count_pending()
    if n_proposals > 0:
        print(f"--- {n_proposals} pending soul proposal(s) — use list_soul_proposals to review ---")
        print()

    # Pending TODOs + today's journal
    pending = todo_db.list_todos("pending")
    if not pending.startswith("No "):
        print("--- Pending TODOs ---")
        print(pending)
        print()
    today_journal = journal.read_day("today")
    if not today_journal.startswith("No "):
        print("--- Today's Journal ---")
        print(today_journal)
        print()

    # Auto-recall: surface recent memories before the first turn
    if memory:
        try:
            result = memory.recall_similar(soul[:400], limit=5)
            memories = result.data.get("memories", []) if result.success else []
            for m in memories:
                buffer.add_background_note(f"From memory: {m}")
            if memories:
                print(f"  Recall : {len(memories)} memories surfaced from previous sessions")
        except Exception as exc:
            logging.getLogger(__name__).debug("Auto-recall failed: %s", exc)

    if args.daemon:
        if not (work_cycle or curiosity or scheduler):
            print("Warning: daemon mode with no background activity enabled — "
                  "set work_cycle.interval (or curiosity_interval) in the config.",
                  file=sys.stderr)
        print("  Mode   : daemon — Ctrl-C or SIGTERM to stop\n")
        transcript.system("Daemon mode started")
        stop_event = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        try:
            while not stop_event.is_set():
                try:
                    scheduled = prompt_queue.get(timeout=5)
                except queue.Empty:
                    continue
                if work_cycle:
                    work_cycle.enqueue_goal(scheduled)
                    transcript.system(f"Scheduled task → work cycle: {scheduled[:120]}")
                else:
                    journal.add_entry(
                        f"[scheduled task fired but no work cycle is enabled to run it] {scheduled}",
                        tags="scheduler", author="agent")
        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down.")
            task_scheduler.stop()
            for t in (scheduler, curiosity, nudge_monitor, work_cycle):
                if t:
                    t.stop()
        return

    _setup_readline()

    session_in = 0
    session_out = 0

    try:
        while True:
            # Drain any scheduled prompts before asking for user input
            try:
                scheduled = prompt_queue.get_nowait()
                print(f"\n[Scheduled task]: {scheduled[:80]}\n", flush=True)
                user_input = scheduled
            except queue.Empty:
                try:
                    user_input = _input("You: ", timeout=30.0)
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break
                if user_input is None:  # select timeout — loop back to check queue
                    continue
                user_input = user_input.strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye.")
                break
            if user_input.lower() == "compact":
                result = buffer.compact(router.for_task("compaction"), memory)
                print(f"[{result}]")
                transcript.system(f"Context compacted: {result}")
                continue

            transcript.user(user_input)
            buffer.add_user(user_input)

            # Auto-compact before the LLM call so it never sees a bloated context
            if buffer.should_compact():
                print("[Auto-compacting context...]", flush=True)
                result = buffer.compact(router.for_task("compaction"), memory)
                transcript.system(f"Auto-compacted: {result}")
                # Re-add the current user message so it isn't lost in the summary
                buffer.add_user(user_input)

            # Inject unread nudges into system prompt so the agent actually sees them
            if pending_messages:
                unread_nudges = [m for m in pending_messages.get_all(limit=5)
                                 if not m["read"] and m.get("source") == "nudge"]
                for m in unread_nudges[-2:]:  # at most 2, most recent first
                    buffer.add_background_note(f"[Self-monitoring] {m['message']}")
                    pending_messages.mark_read(m["id"])

            # Inject unread mailbox messages from other agents
            if mailbox:
                unread_mail = mailbox.inbox(unread_only=True, limit=3)
                for m in unread_mail:
                    topic_str = f" [{m['topic']}]" if m.get("topic") else ""
                    buffer.add_background_note(
                        f"[Mailbox from {m['from_agent']}{topic_str}] {m['message']}"
                    )
                    mailbox.mark_read(m["id"])

            # Inject conversations awaiting this agent (its turn, or a closed outcome)
            if conversation_bus:
                for c in conversation_bus.needs_attention(limit=3):
                    if c["attention"] == "your_turn":
                        buffer.add_background_note(
                            f"[Conversation #{c['id']} — your turn] {c['last_from']} "
                            f"said: {c['last_message']} "
                            f"(reply with talk_reply; turn {c['turn_count']}/{c['max_turns']})"
                        )
                    else:  # closed/unread outcome — surface once, then ack
                        buffer.add_background_note(
                            f"[Conversation #{c['id']} — {c['state']}] {c['last_from']} "
                            f"said: {c['last_message']}"
                        )
                        conversation_bus.acknowledge(c["id"])

            # Agentic tool loop — keep calling until the model stops using tools
            while True:
                try:
                    response = _call_llm_raw(
                        router.for_task("conversation"),
                        buffer.messages_for_llm(soul),
                        tools=tool_schemas or None,
                    )
                except Exception as exc:
                    print(f"\n[LLM error: {exc}]\n")
                    break

                usage = response.get("usage", {})
                session_in += usage.get("prompt_tokens", 0)
                session_out += usage.get("completion_tokens", 0)

                msg = response["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                tool_calls = msg.get("tool_calls") or []

                # Record the assistant turn (with tool_calls if present)
                buffer.add_assistant(content or None, tool_calls or None)

                # Surface assistant text whenever present — even alongside tool
                # calls. A model may emit its final answer in the SAME message as
                # a trailing tool call (commonly compact_context when wrapping
                # up); gating display on "no tool_calls" silently dropped that
                # answer, and the deferred compaction then summarised it away.
                if content:
                    print(f"\nAssistant: {content}\n")
                    transcript.assistant(content)

                if not tool_calls:
                    if content:
                        if critic and critic.enabled:
                            critic.check(content, buffer)
                        if extractor and extractor.enabled:
                            extractor.maybe_extract(buffer)
                    if session_in or session_out:
                        print(f"  [↑{session_in:,} ↓{session_out:,} tokens this session]")
                    break

                # Execute each tool call and feed results back
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    tc_id = tc.get("id", "")

                    print(f"  [Tool: {name}({_fmt_args(args)})]", flush=True)
                    try:
                        result = executor.execute(name, **args)
                        output = str(result.output if result.success else result.error or "error")
                    except Exception as exc:
                        output = f"Tool error: {exc}"
                    transcript.tool(name, output)
                    buffer.add_tool_result(tc_id, name, output)

            # Agent-requested compaction runs here, after the turn's work is
            # done — never mid-loop, where it would wipe the active task.
            deferred = buffer.maybe_compact(router.for_task("compaction"), memory)
            if deferred:
                print(f"[{deferred}]", flush=True)
                transcript.system(f"Deferred compaction: {deferred}")

    finally:
        # Session handoff note — capture the session's texture BEFORE the
        # exit-compaction below clears the buffer. Never breaks shutdown.
        if session_handoff:
            try:
                _convo = buffer.recent_turns_text(n=12)
                if buffer._summary:
                    _convo = f"[Earlier summary: {buffer._summary}]\n\n{_convo}"
                if session_handoff.make_note(router.for_task("compaction"), _convo):
                    print("[Handoff note saved for next session]")
            except Exception as exc:
                logging.getLogger(__name__).warning("Handoff note failed: %s", exc)

        # Auto-compact on exit: save this session to long-term memory
        if memory and buffer._messages:
            try:
                result = buffer.compact(router.for_task("compaction"), memory)
                print(f"\n[Session saved: {result}]")
                transcript.system(f"Session saved on exit: {result}")
            except Exception as exc:
                logging.getLogger(__name__).warning("Auto-compact on exit failed: %s", exc)
        task_scheduler.stop()
        if scheduler:
            scheduler.stop()
        if curiosity:
            curiosity.stop()
        if nudge_monitor:
            nudge_monitor.stop()
        if work_cycle:
            work_cycle.stop()
        _save_readline_history()




if __name__ == "__main__":
    main()
