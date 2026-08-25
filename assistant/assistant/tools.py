"""Assistant tool registry.

Tools available to the personal assistant:
  - compact_context: Agent-initiated context compaction
  - web_search:      DuckDuckGo search (requires duckduckgo-search package)
  - fetch_readable:  Fetch a URL and return readable text; handles HTML (html2text)
                     and PDF (pdfminer.six, optional)
  - digest:          Query-focused summary of a URL or long text (map-reduce via
                     the summarizer LLM); fetches internally so raw never hits the buffer
  - search_arxiv:    Search arXiv papers via the official API (rate-limited, cached)
  - save_note:       Persist a fact/decision to long-term memory
  - recall:          Search long-term memory for relevant context
  - todo_add / todo_list / todo_done / todo_note: TODO management
  - journal_add / journal_read / journal_search: Daily journal

Memory, TODO, and journal tools are only registered when their respective
backends are provided. web_search requires duckduckgo-search installed.
fetch_readable requires html2text; PDF support additionally requires pdfminer.six.
search_arxiv uses only stdlib (xml, urllib) — no extra deps.
"""

import threading
import time

_FETCH_MAX_CHARS = 12_000

# digest (query-focused summarization): per-map-chunk size and a safety cap on
# chunk count. A doc over CHUNK*MAX_CHUNKS chars has its tail dropped — and the
# drop is reported, never silent (see _map_reduce_digest).
_DIGEST_CHUNK_CHARS = 8_000
_DIGEST_MAX_CHUNKS = 12

# ---------------------------------------------------------------------------
# arXiv rate limiter — shared across all calls, thread-safe
# arXiv guidelines: ≥3 s between requests. We use 5 s to be polite.
# ---------------------------------------------------------------------------
_ARXIV_MIN_GAP = 5.0  # seconds
_ARXIV_CACHE_TTL = 3600  # seconds (1 hour)
_arxiv_lock = threading.Lock()
_arxiv_last_call: float = 0.0
_arxiv_cache: dict[str, tuple[float, str]] = {}  # query_key → (timestamp, result)

from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel
from agent_tools.core.registry import ToolRegistry


def build_assistant_tools(
    buffer,
    llm,
    memory_tools=None,
    todo_db=None,
    journal=None,
    task_scheduler=None,
    research_agenda=None,
    soul_manager=None,
    pending_messages=None,
    skill_library=None,
    spawn_registry=None,
    mailbox=None,
    file_tools=None,
    mail_sender=None,
    digest_llm=None,
    enabled: dict | None = None,
    sandbox_cfg: dict | None = None,
) -> ToolRegistry:
    """Build the tool registry for the personal assistant.

    Args:
        buffer:           ConversationBuffer — used by compact_context.
        llm:              FireworksLLM — used as the compaction LLM.
        memory_tools:     Optional MemoryTools — enables save_note and recall.
        todo_db:          Optional TodoDB — enables todo_* tools.
        journal:          Optional Journal — enables journal_* tools.
        task_scheduler:   Optional TaskScheduler — enables schedule_task tools.
        research_agenda:  Optional ResearchAgenda — enables research_focus tools.
        soul_manager:     Optional SoulManager — enables soul proposal tools.
        pending_messages: Optional PendingMessages — enables pending message tools.
        skill_library:    Optional SkillLibrary — enables list_skills, invoke_skill,
                          propose_skill tools.
        spawn_registry:   Optional SpawnRegistry — enables spawn_agent tool.
        mailbox:          Optional AgentMailbox — enables send_message, check_inbox,
                          reply_to_message tools.
        file_tools:       Optional FileTools — enables write_file, read_file, list_files,
                          all sandboxed to the configured work_dir.
        mail_sender:      Optional MailSender — enables the send_email tool, jailed
                          to the recipient aliases configured in the YAML.
        sandbox_cfg:      Optional dict from the ``sandbox:`` YAML section.  When
                          provided, registers the ``python_exec`` tool.  Keys:
                          work_dir (str), timeout (int), use_firejail (bool),
                          unix_user (str|None).
        enabled:          Optional dict controlling which tool groups are active, e.g.
                          {'web_search': True, 'fetch_readable': False,
                           'todos': True, 'journal': True, 'memory': True,
                           'research': True}.
                          Omitted keys default to True — all tools on unless explicitly disabled.
    """
    def _on(key: str) -> bool:
        return enabled.get(key, True) if enabled else True

    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name="compact_context",
        description=(
            "Request compaction of the conversation history when it's getting "
            "long. Summarises old turns and saves important facts to long-term "
            "memory before they're lost. The compaction runs automatically at "
            "the end of the current turn, so calling this does NOT interrupt or "
            "end your work — keep going with the task; do not stop to wait for "
            "instructions after calling it."
        ),
        parameters=[],
        returns="string",
        permission=PermissionLevel.WRITE,
        timeout_seconds=30,
        execute=lambda: buffer.request_compact(),
    ))

    if _on("web_search"):
      try:
        try:
            from ddgs import DDGS  # noqa: F401 — new package name
        except ImportError:
            from duckduckgo_search import DDGS  # noqa: F401 — legacy name
        registry.register(ToolDefinition(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                "Use for current information, facts you don't know, or anything that "
                "might have changed since your training data."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Number of results to return (default 5)",
                    required=False,
                    default=5,
                ),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=15,
            execute=_web_search,
        ))
      except ImportError:
        pass  # duckduckgo-search not installed — tool simply absent

    if _on("fetch_readable"):
      try:
        import html2text as _ht  # noqa: F401 — just checking availability
        registry.register(ToolDefinition(
            name="fetch_readable",
            description=(
                "Fetch a URL and return its content as readable text. "
                "Handles HTML pages and PDF documents (if pdfminer.six is installed). "
                "Use to read articles, papers, documentation, or any web page when "
                "you have a specific URL. Prefer web_search when you don't have the URL yet."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL to fetch (http/https). PDFs are detected automatically.",
                ),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=30,
            execute=_fetch_readable,
        ))
      except ImportError:
        pass  # html2text not installed — tool simply absent

    if _on("digest"):
        _digest_llm = digest_llm or llm
        registry.register(ToolDefinition(
            name="digest",
            description=(
                "Summarise a web page or long text through the lens of a specific "
                "question (query-focused digest). Give a URL or raw text as "
                "'source' and a 'focus' question; returns only what's relevant to "
                "the question WITHOUT loading the full raw content into the "
                "conversation. Prefer this over fetch_readable for long or "
                "scattered pages (Reddit threads, long articles, docs) when you "
                "only need what bears on a specific question — it keeps the "
                "conversation small and won't lose a detail buried deep in the page."
            ),
            parameters=[
                ToolParameter(
                    name="source",
                    type="string",
                    description="A URL (http/https, fetched internally) or raw text to digest.",
                ),
                ToolParameter(
                    name="focus",
                    type="string",
                    description="The question to summarise toward, e.g. "
                                "'red flags about working at company X'.",
                ),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=120,
            execute=lambda source, focus: _digest(source, focus, _digest_llm),
        ))

    if _on("arxiv"):
        registry.register(ToolDefinition(
            name="search_arxiv",
            description=(
                "Search arXiv for academic papers using the official API. "
                "Returns titles, authors, abstracts, and PDF links. "
                "Results are cached for 1 hour; requests are rate-limited to be "
                "respectful of arXiv's servers (max 1 request per 5 seconds). "
                "Use category to narrow to a field, e.g. 'cs.AI', 'cs.LG', 'q-bio.NC'."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query (keywords, author:name, ti:title, abs:abstract)",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Number of results to return (default 5, max 10)",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="sort_by",
                    type="string",
                    description="'relevance' (default) or 'recent' (newest first)",
                    required=False,
                    default="relevance",
                ),
                ToolParameter(
                    name="category",
                    type="string",
                    description="Optional arXiv category filter, e.g. 'cs.AI'",
                    required=False,
                    default="",
                ),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=30,
            execute=_search_arxiv,
        ))

    if memory_tools and _on("memory"):
        registry.register(ToolDefinition(
            name="save_note",
            description=(
                "Save an important fact, decision, or preference to long-term memory. "
                "Use for things worth remembering across sessions."
            ),
            parameters=[
                ToolParameter(
                    name="note",
                    type="string",
                    description="The fact or note to save",
                ),
                ToolParameter(
                    name="tags",
                    type="string",
                    description="Comma-separated tags e.g. 'work,decision'",
                    required=False,
                    default="",
                ),
            ],
            returns="string",
            permission=PermissionLevel.WRITE,
            timeout_seconds=10,
            execute=lambda note, tags="": _save_note(memory_tools, note, tags),
        ))

        registry.register(ToolDefinition(
            name="recall",
            description=(
                "Search long-term memory for relevant notes or past context. "
                "Use when past information would help answer the current question."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="What to search for",
                ),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=10,
            execute=lambda query: _recall(memory_tools, query),
        ))

        if hasattr(memory_tools, "recall_analogies"):
            registry.register(ToolDefinition(
                name="recall_analogies",
                description=(
                    "Find past experiences from DIFFERENT domains that are "
                    "structurally similar to a situation — associative recall. "
                    "Where recall finds 'the same kind of thing', this finds 'a "
                    "different kind of thing that works the same way'. Use when "
                    "stuck, when a problem feels familiar but you can't place it, "
                    "or to transfer an approach across topics."
                ),
                parameters=[
                    ToolParameter(
                        name="situation",
                        type="string",
                        description="The current situation, problem, or idea",
                    ),
                ],
                returns="string",
                permission=PermissionLevel.READ,
                timeout_seconds=20,
                execute=lambda situation: _recall_analogies(memory_tools, situation),
            ))

    if todo_db and _on("todos"):
        registry.register(ToolDefinition(
            name="todo_add",
            description="Add a new TODO item.",
            parameters=[
                ToolParameter(name="title", type="string", description="Short title"),
                ToolParameter(name="description", type="string", description="Details", required=False, default=""),
                ToolParameter(name="priority", type="string", description="urgent / high / normal / low", required=False, default="normal"),
                ToolParameter(name="due_date", type="string", description="Due date YYYY-MM-DD", required=False, default=""),
                ToolParameter(name="tags", type="string", description="Comma-separated tags", required=False, default=""),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda title, description="", priority="normal", due_date="", tags="":
                todo_db.add(title, description, priority, due_date, tags),
        ))
        registry.register(ToolDefinition(
            name="todo_list",
            description="List TODOs. status: pending (default), in_progress, done, cancelled, all.",
            parameters=[
                ToolParameter(name="status", type="string", description="Filter by status", required=False, default="pending"),
            ],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda status="pending": todo_db.list_todos(status),
        ))
        registry.register(ToolDefinition(
            name="todo_done",
            description="Mark a TODO as done.",
            parameters=[
                ToolParameter(name="todo_id", type="integer", description="TODO id"),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda todo_id: todo_db.set_status(todo_id, "done"),
        ))
        registry.register(ToolDefinition(
            name="todo_note",
            description=(
                "Add a remark or observation to a TODO — e.g. what you found out, "
                "what's blocking it, or why it matters."
            ),
            parameters=[
                ToolParameter(name="todo_id", type="integer", description="TODO id"),
                ToolParameter(name="note", type="string", description="The remark to attach"),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda todo_id, note: todo_db.add_note(todo_id, note, author="agent"),
        ))

    if journal and _on("journal"):
        registry.register(ToolDefinition(
            name="journal_add",
            description=(
                "Add an entry to the daily journal. Use for recording work done, "
                "thoughts, ideas, or observations worth keeping. "
                "Call this proactively when something is worth noting."
            ),
            parameters=[
                ToolParameter(name="content", type="string", description="Entry text"),
                ToolParameter(name="tags", type="string", description="Comma-separated tags", required=False, default=""),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda content, tags="": journal.add_entry(content, tags, author="agent"),
        ))
        registry.register(ToolDefinition(
            name="journal_read",
            description="Read journal entries for a day. date: 'today' (default), 'yesterday', or YYYY-MM-DD.",
            parameters=[
                ToolParameter(name="date", type="string", description="Date to read", required=False, default="today"),
            ],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda date="today": journal.read_day(date),
        ))
        registry.register(ToolDefinition(
            name="journal_search",
            description="Search past journal entries by keyword.",
            parameters=[
                ToolParameter(name="query", type="string", description="Search term"),
            ],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda query: journal.search(query),
        ))

    if task_scheduler and _on("scheduler"):
        registry.register(ToolDefinition(
            name="schedule_task",
            description=(
                "Schedule a prompt to be injected into the conversation at a future time. "
                "The agent will process it autonomously when it fires. "
                "Use 'when' for one-shot tasks: 'YYYY-MM-DD HH:MM' (resolve natural language "
                "like 'Sunday at 3pm' to this format using the current date from your system "
                "prompt). Use 'cron' for recurring tasks: standard 5-field cron expression "
                "e.g. '0 8 * * *' for daily at 8am (requires croniter installed)."
            ),
            parameters=[
                ToolParameter(name="prompt", type="string",
                              description="The prompt to inject when the task fires"),
                ToolParameter(name="when", type="string",
                              description="One-shot datetime: YYYY-MM-DD HH:MM",
                              required=False, default=""),
                ToolParameter(name="cron", type="string",
                              description="Cron expression for recurring tasks",
                              required=False, default=""),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda prompt, when="", cron="": (
                task_scheduler.schedule_cron(prompt, cron) if cron
                else task_scheduler.schedule_once(prompt, when) if when
                else "Provide either 'when' (datetime) or 'cron' (expression)."
            ),
        ))
        registry.register(ToolDefinition(
            name="list_scheduled_tasks",
            description="List all scheduled tasks (active and completed).",
            parameters=[],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda: task_scheduler.list_tasks(),
        ))
        registry.register(ToolDefinition(
            name="cancel_scheduled_task",
            description="Cancel an active scheduled task by ID.",
            parameters=[
                ToolParameter(name="task_id", type="integer", description="Task ID"),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda task_id: task_scheduler.cancel_task(task_id),
        ))

    if research_agenda and _on("research"):
        registry.register(ToolDefinition(
            name="research_focus",
            description=(
                "Add a topic to the focused research agenda. The curiosity engine will "
                "prioritise it over general interests, running introspective ponder cycles "
                "to build depth rather than broad coverage. Use when the user asks to "
                "'research X for a while', 'deep-dive into Y', or 'keep studying Z'."
            ),
            parameters=[
                ToolParameter(name="topic", type="string",
                              description="Topic to research in depth"),
                ToolParameter(name="goal", type="string",
                              description="What you want to understand or answer",
                              required=False, default=""),
                ToolParameter(name="deadline", type="string",
                              description="Optional deadline YYYY-MM-DD",
                              required=False, default=""),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda topic, goal="", deadline="": research_agenda.add_topic(topic, goal, deadline),
        ))
        registry.register(ToolDefinition(
            name="research_status",
            description=(
                "List the current focused research agenda — topics, goals, cycles completed, "
                "and status. Use to check what is being actively researched."
            ),
            parameters=[],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda: research_agenda.summarize(),
        ))
        registry.register(ToolDefinition(
            name="research_close",
            description="Mark a focused research topic as mature (enough depth reached) or cancelled.",
            parameters=[
                ToolParameter(name="topic_id", type="integer",
                              description="Agenda item ID (from research_status)"),
                ToolParameter(name="status", type="string",
                              description="'mature' or 'cancelled'",
                              required=False, default="mature"),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda topic_id, status="mature": research_agenda.mark_status(topic_id, status),
        ))

    # ------------------------------------------------------------------
    # Soul tools — propose changes to the learnable soul layer
    # ------------------------------------------------------------------
    if soul_manager and _on("soul"):
        registry.register(ToolDefinition(
            name="propose_soul_change",
            description=(
                "Propose an addition or change to your learnable soul. Use this when you "
                "notice a stable pattern in how you work — a preference, tendency, or value "
                "that has emerged from experience and that you'd like to make explicit. "
                "Be specific and grounded: quote the evidence (a pattern of interactions, "
                "a recurring situation). The human will approve or reject. Do NOT propose "
                "changes to core identity or values — only operational preferences."
            ),
            parameters=[
                ToolParameter(name="proposed_text", type="string",
                              description="The text to add to your soul (first-person, present tense)."),
                ToolParameter(name="reasoning", type="string",
                              description="Why this belongs in your soul: what pattern or evidence led here."),
                ToolParameter(name="section", type="string",
                              description="Which aspect of self this touches (e.g. 'communication', 'research', 'values').",
                              required=False, default="preferences"),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda proposed_text, reasoning, section="preferences": soul_manager.add_proposal(
                proposed_text, reasoning, section
            ),
        ))

        registry.register(ToolDefinition(
            name="list_soul_proposals",
            description="List pending soul proposals awaiting human review.",
            parameters=[
                ToolParameter(name="status", type="string",
                              description="'pending', 'approved', or 'rejected'",
                              required=False, default="pending"),
            ],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda status="pending": soul_manager.list_proposals(status),
        ))

        registry.register(ToolDefinition(
            name="decide_soul_proposal",
            description=(
                "Approve or reject a soul proposal. Approval appends the proposed text "
                "to the learned soul layer immediately. Use this when the human has "
                "indicated their decision on a proposal."
            ),
            parameters=[
                ToolParameter(name="proposal_id", type="integer",
                              description="Proposal ID from list_soul_proposals."),
                ToolParameter(name="decision", type="string",
                              description="'approve' or 'reject'"),
                ToolParameter(name="notes", type="string",
                              description="Optional human notes on the decision.",
                              required=False, default=""),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda proposal_id, decision, notes="": soul_manager.decide(
                proposal_id, decision, notes
            ),
        ))

    # ------------------------------------------------------------------
    # Pending message tools — manage proactive messages from inner voice
    # ------------------------------------------------------------------
    if pending_messages and _on("inner_voice"):
        registry.register(ToolDefinition(
            name="list_pending_messages",
            description="List messages your inner voice queued while you were reflecting or researching.",
            parameters=[],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda: pending_messages.format_unread() or "No pending messages.",
        ))

        registry.register(ToolDefinition(
            name="dismiss_message",
            description="Mark a pending message as read/dismissed.",
            parameters=[
                ToolParameter(name="message_id", type="integer",
                              description="Message ID from list_pending_messages."),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda message_id: (
                "Message dismissed." if pending_messages.mark_read(message_id)
                else f"Message {message_id} not found."
            ),
        ))

        registry.register(ToolDefinition(
            name="queue_message",
            description=(
                "Queue a proactive message to yourself for the next session. "
                "Use this when you want to bring something up later — a thought, "
                "a question, something you noticed mid-conversation that deserves "
                "its own moment. The message will appear at the start of the next session."
            ),
            parameters=[
                ToolParameter(name="message", type="string",
                              description="The message to queue (1-3 sentences, first-person)."),
                ToolParameter(name="topic", type="string",
                              description="Short label for the topic.",
                              required=False, default="note"),
                ToolParameter(name="urgency", type="string",
                              description="'low' (default, quiet), 'medium' (highlighted), 'high' (important).",
                              required=False, default="low"),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda message, topic="note", urgency="low": (
                f"Queued message #{pending_messages.add(topic, message, source='agent', urgency=urgency)}."
            ),
        ))

    # ------------------------------------------------------------------
    # Skill tools — named prompt templates the agent invokes as tools
    # ------------------------------------------------------------------
    if skill_library and _on("skills"):
        registry.register(ToolDefinition(
            name="list_skills",
            description=(
                "List available skills — named prompt templates you can invoke as tools. "
                "Each skill has a name, a description, and required arguments. "
                "Call this before invoke_skill to see what is available."
            ),
            parameters=[],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=5,
            execute=lambda: skill_library.list_skills(),
        ))

        registry.register(ToolDefinition(
            name="invoke_skill",
            description=(
                "Invoke a named skill — a prompt template that applies a specific "
                "reasoning pattern to your input and returns the result. "
                "Use list_skills first to see available skills and their required arguments. "
                "Examples: critique_plan (args: plan), going_in_circles (args: recent)."
            ),
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Skill name (from list_skills)",
                ),
                ToolParameter(
                    name="arguments",
                    type="object",
                    description=(
                        "Key-value pairs matching the skill's placeholders. "
                        "E.g. for critique_plan: {\"plan\": \"my plan text here\"}"
                    ),
                    required=False,
                    default={},
                ),
            ],
            returns="string", permission=PermissionLevel.READ, timeout_seconds=60,
            execute=lambda name, arguments=None: skill_library.invoke(name, llm, arguments or {}),
        ))

        registry.register(ToolDefinition(
            name="propose_skill",
            description=(
                "Propose a new skill — a reusable prompt template for a reasoning pattern "
                "you find yourself applying repeatedly. The proposal is written to "
                "skills/proposed/ for human review; move it to skills/ to activate it. "
                "Use {placeholder} syntax for variable parts of the prompt."
            ),
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Skill name (snake_case, e.g. 'evaluate_sources')",
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="One-line description of what this skill does",
                ),
                ToolParameter(
                    name="template",
                    type="string",
                    description="The prompt template. Use {variable} for arguments.",
                ),
            ],
            returns="string", permission=PermissionLevel.WRITE, timeout_seconds=5,
            execute=lambda name, description, template: skill_library.propose(
                name, description, template
            ),
        ))

    # ------------------------------------------------------------------
    # File tools — sandboxed read/write within work_dir
    # ------------------------------------------------------------------
    if file_tools and _on("files"):
        _wd = str(file_tools.work_dir)
        registry.register(ToolDefinition(
            name="write_file",
            description=(
                f"Write text content to a file inside the work directory ({_wd}). "
                "Creates parent directories as needed. "
                "Set append=true to add to an existing file instead of overwriting. "
                "Use for producing documents, summaries, reports, or any persistent output."
            ),
            parameters=[
                ToolParameter("path", "string",
                              "File path relative to the work directory, e.g. 'lua_guide.md'"),
                ToolParameter("content", "string", "Text to write"),
                ToolParameter("append", "boolean",
                              "If true, append to existing file instead of overwriting",
                              required=False, default=False),
            ],
            returns="string",
            permission=PermissionLevel.WRITE,
            timeout_seconds=10,
            execute=lambda path, content, append=False: file_tools.write_file(path, content, append),
        ))

        registry.register(ToolDefinition(
            name="read_file",
            description=(
                f"Read a file from the work directory ({_wd}). "
                f"Content is capped at 20,000 chars to protect context. "
                "Use to review previously written documents or load intermediate results."
            ),
            parameters=[
                ToolParameter("path", "string",
                              "File path relative to the work directory"),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=10,
            execute=lambda path: file_tools.read_file(path),
        ))

        registry.register(ToolDefinition(
            name="list_files",
            description=(
                f"List files and directories inside the work directory ({_wd}). "
                "Shows file sizes. Use '.' (default) for the root."
            ),
            parameters=[
                ToolParameter("path", "string",
                              "Subdirectory to list (default '.' for root)",
                              required=False, default="."),
            ],
            returns="string",
            permission=PermissionLevel.READ,
            timeout_seconds=5,
            execute=lambda path=".": file_tools.list_files(path),
        ))

    # ------------------------------------------------------------------
    # Sandbox — Python code execution (opt-in, DANGEROUS)
    # ------------------------------------------------------------------
    if sandbox_cfg and _on("sandbox"):
        from pathlib import Path as _Path
        from assistant.sandbox import _resolve_backend

        _work_dir      = _Path(sandbox_cfg.get("work_dir", "~/agent_work")).expanduser()
        _sb_timeout    = int(sandbox_cfg.get("timeout", 120))
        _sb_user       = sandbox_cfg.get("unix_user") or None
        _fj_path       = sandbox_cfg.get("firejail_path") or None
        _python        = sandbox_cfg.get("python") or "python3"
        # Support legacy use_firejail boolean as well as the new backend key
        _backend: str
        if "backend" in sandbox_cfg:
            _backend = str(sandbox_cfg["backend"])
        elif sandbox_cfg.get("use_firejail") is False:
            _backend = "none"
        else:
            _backend = "auto"

        _active = _resolve_backend(_backend, _fj_path)
        _sb_desc = (
            _active if _active
            else f"unavailable (requested: {_backend})"
        )

        registry.register(ToolDefinition(
            name="python_exec",
            description=(
                "Execute Python code in a sandboxed subprocess and return the output. "
                f"Working directory: {_work_dir} — files written here persist between "
                "calls, so you can save intermediate results and load them in a later "
                "invocation. No network access inside the sandbox. "
                "stdout and stderr are both returned. "
                "Useful for numerical computation, simulations, data analysis, or "
                f"anything that needs actual execution rather than reasoning. "
                f"Sandbox: {_sb_desc}."
            ),
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python source code to execute.",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description=(
                        f"Max execution time in seconds (default {_sb_timeout}, "
                        f"capped at {_sb_timeout})."
                    ),
                    required=False,
                    default=_sb_timeout,
                ),
            ],
            returns="string",
            permission=PermissionLevel.DANGEROUS,
            timeout_seconds=_sb_timeout + 10,
            execute=lambda code, timeout=_sb_timeout: _run_python_tool(
                code, _work_dir, min(int(timeout), _sb_timeout), _sb_user,
                _backend, _fj_path, _python,
            ),
        ))

    # ------------------------------------------------------------------
    # mailbox — inter-agent messaging
    # ------------------------------------------------------------------
    if mailbox and _on("mailbox"):
        def _send_message(to: str, message: str, topic: str = "") -> str:
            msg_id = mailbox.send(to=to, message=message, topic=topic)
            return f"Message sent to {to!r} (id=#{msg_id})"

        def _check_inbox(unread_only: bool = True) -> str:
            msgs = mailbox.inbox(unread_only=unread_only, limit=20)
            if not msgs:
                return "No messages." if unread_only else "Inbox is empty."
            lines = []
            for m in msgs:
                ts = (m["created_at"] or "")[:16]
                read_marker = "○" if m["read"] else "●"
                topic_str = f" [{m['topic']}]" if m.get("topic") else ""
                reply_str = f" (reply to #{m['reply_to']})" if m.get("reply_to") else ""
                lines.append(
                    f"{read_marker} [#{m['id']}] {ts} From {m['from_agent']}{topic_str}{reply_str}:\n  {m['message']}"
                )
            return "\n".join(lines)

        def _reply_to_message(msg_id: int, message: str) -> str:
            original = mailbox.get_message(msg_id)
            if original is None:
                return f"Error: message #{msg_id} not found"
            reply_id = mailbox.send(
                to=original["from_agent"],
                message=message,
                topic=original.get("topic", ""),
                reply_to=msg_id,
            )
            return f"Reply sent to {original['from_agent']!r} (id=#{reply_id})"

        registry.register(ToolDefinition(
            name="send_message",
            description="Send a message to another agent by name.",
            parameters=[
                ToolParameter("to", "string", "Recipient agent name"),
                ToolParameter("message", "string", "Message body"),
                ToolParameter("topic", "string", "Optional topic label", required=False),
            ],
            returns="string",
            permission=PermissionLevel.SAFE,
            execute=_send_message,
        ))

        registry.register(ToolDefinition(
            name="check_inbox",
            description="Read messages in your mailbox from other agents.",
            parameters=[
                ToolParameter("unread_only", "boolean",
                              "If true, show only unread messages (default true)",
                              required=False),
            ],
            returns="string",
            permission=PermissionLevel.SAFE,
            execute=_check_inbox,
        ))

        registry.register(ToolDefinition(
            name="reply_to_message",
            description="Reply to a specific mailbox message by its ID.",
            parameters=[
                ToolParameter("msg_id", "integer", "ID of the message to reply to"),
                ToolParameter("message", "string", "Reply body"),
            ],
            returns="string",
            permission=PermissionLevel.SAFE,
            execute=_reply_to_message,
        ))

    # ------------------------------------------------------------------
    # mail — jailed outbound email (recipients fixed in YAML)
    # ------------------------------------------------------------------
    if mail_sender and _on("mail"):
        alias_list = ", ".join(mail_sender.aliases)
        registry.register(ToolDefinition(
            name="send_email",
            description=(
                "Send a plain-text email to one of the configured recipients. "
                f"Allowed recipients: {alias_list}. "
                "Only these aliases are accepted — arbitrary addresses are impossible."
            ),
            parameters=[
                ToolParameter("to", "string", f"Recipient alias, one of: {alias_list}"),
                ToolParameter("subject", "string", "Email subject line"),
                ToolParameter("body", "string", "Plain-text email body"),
            ],
            returns="string",
            permission=PermissionLevel.WRITE,
            timeout_seconds=35,
            execute=mail_sender.send,
        ))

    # ------------------------------------------------------------------
    # spawn_agent — delegate to a specialist child agent
    # ------------------------------------------------------------------
    if spawn_registry and _on("spawn"):
        from agent_core.agent import AgentRole, AgentInstance

        names_str = ", ".join(spawn_registry.names())

        def _do_spawn(role: str, task: str) -> str:
            role_cfg = spawn_registry.get(role)
            if role_cfg is None:
                return f"Error: unknown role {role!r}. Available: {names_str}"

            # Build child registry from this assistant's registry
            allowed = set(role_cfg.tools)
            child_reg = ToolRegistry()
            for tool_name in allowed:
                tool = registry.get(tool_name)
                if tool is not None:
                    child_reg.register(tool)

            child_llm = role_cfg.llm if role_cfg.llm is not None else llm
            child_role = AgentRole(name=role_cfg.name, soul=role_cfg.soul)
            child = AgentInstance(child_role, child_llm, registry=child_reg)
            result = child.run(task)
            return result.summary or ("done" if result.success else "failed")

        registry.register(ToolDefinition(
            name="spawn_agent",
            description=(
                f"Delegate a task to a specialist sub-agent and get its result. "
                f"Available roles: {names_str}. "
                "Use when a task is best handled by a specialist."
            ),
            parameters=[
                ToolParameter(
                    name="role",
                    type="string",
                    description=f"Role to spawn. One of: {names_str}",
                ),
                ToolParameter(
                    name="task",
                    type="string",
                    description="Complete task description for the child agent",
                ),
            ],
            returns="string",
            permission=PermissionLevel.SAFE,
            execute=_do_spawn,
        ))

    return registry


def _run_python_tool(
    code: str,
    work_dir,
    timeout: int,
    unix_user: str | None,
    backend: str,
    firejail_path: str | None,
    python: str = "python3",
) -> str:
    from assistant.sandbox import run_python
    r = run_python(code, work_dir, timeout, unix_user, backend, firejail_path, python)
    if r["error"]:
        return f"Error: {r['error']}"
    parts = []
    if r["stdout"].strip():
        parts.append(r["stdout"].rstrip())
    if r["stderr"].strip():
        parts.append(f"[stderr]\n{r['stderr'].rstrip()}")
    if not parts:
        return f"[exit {r['returncode']}] (no output)"
    suffix = "" if r["returncode"] == 0 else f"\n[exit {r['returncode']}]"
    return "\n".join(parts) + suffix


def _fetch_full_text(url: str, _max_retries: int = 3) -> tuple[str | None, str | None]:
    """Fetch a URL and return (full_readable_text, None) or (None, error_message).

    The text is untruncated — callers truncate (fetch_readable) or summarise
    (digest) as they see fit. Handles HTML and PDF; retries transient 429/503
    with backoff. The error strings are complete, user-facing messages.
    """
    import time
    import urllib.error
    import urllib.request
    try:
        import html2text
    except ImportError:
        return None, "Fetch failed: html2text not installed (pip install html2text)."

    req = urllib.request.Request(
        url, headers={"User-Agent": "assistant-agent/1.0 (research tool; respectful bot)"}
    )
    raw: bytes | None = None
    content_type = ""
    for attempt in range(_max_retries + 1):
        try:
            resp = urllib.request.urlopen(req, timeout=20)
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            # 429/503 are transient rate-limits: back off and retry. Returning a
            # bare failure here makes the agent treat the source as dead and give
            # up on the whole task (observed: one 429 killed an 8h scheduled run).
            if exc.code in (429, 503) and attempt < _max_retries:
                ra = exc.headers.get("Retry-After")
                try:
                    wait = min(float(ra), 30.0) if ra else 2.0 * (2 ** attempt)
                except ValueError:
                    wait = 2.0 * (2 ** attempt)
                time.sleep(wait)
                continue
            if exc.code in (429, 503):
                return None, (
                    f"Fetch rate-limited (HTTP {exc.code}) after {_max_retries} retries. "
                    f"This is TRANSIENT — retry later or use a different source. Do not "
                    f"treat {url} as permanently unavailable, and do not stop the task or "
                    f"ask the user for data you can find yourself."
                )
            return None, f"Fetch failed: HTTP Error {exc.code}: {exc.reason}"
        except Exception as exc:
            return None, f"Fetch failed: {exc}"
    if raw is None:
        return None, f"Fetch failed: no response for {url}"

    # PDF detection: Content-Type header or URL extension
    is_pdf = "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")
    if is_pdf:
        return _extract_pdf_text(raw), None

    text_body = raw.decode("utf-8", errors="replace")
    h = html2text.HTML2Text()
    h.body_width = 0
    return h.handle(text_body), None


def _fetch_readable(url: str) -> str:
    text, err = _fetch_full_text(url)
    if err is not None:
        return err
    if len(text) > _FETCH_MAX_CHARS:
        cut = text.rfind("\n", 0, _FETCH_MAX_CHARS)
        text = text[: cut if cut != -1 else _FETCH_MAX_CHARS] + "\n\n[... truncated ...]"
    return text


def _extract_pdf_text(data: bytes) -> str:
    """Extract plain text from PDF bytes using pdfminer.six (pure Python)."""
    try:
        import io
        from pdfminer.high_level import extract_text
        text = extract_text(io.BytesIO(data))
    except ImportError:
        return (
            "PDF detected but pdfminer.six is not installed.\n"
            "Install it with: pip install pdfminer.six"
        )
    except Exception as exc:
        return f"PDF extraction failed: {exc}"

    text = text.strip()
    if not text:
        return "PDF contained no extractable text (may be scanned/image-based)."
    if len(text) > _FETCH_MAX_CHARS:
        cut = text.rfind("\n", 0, _FETCH_MAX_CHARS)
        text = text[: cut if cut != -1 else _FETCH_MAX_CHARS] + "\n\n[... truncated ...]"
    return text


# ---------------------------------------------------------------------------
# digest — query-focused summarization (map-reduce)
# ---------------------------------------------------------------------------
# Folds a large document down *through the lens of a question*, so scattered
# signal (a red flag in comment 40 of a Reddit thread) survives where plain
# head-truncation would lose it. Crucially, when given a URL it fetches
# internally — the raw content never enters the conversation buffer; only the
# focused synthesis comes back. See CLAUDE.md backlog for the design rationale.

_DIGEST_SINGLE_SYS = (
    "You extract and synthesise only what is relevant to a specific question "
    "from a document. Ignore everything unrelated. Be concrete and faithful — "
    "quote figures, names, and specifics; never invent."
)
_DIGEST_SINGLE_USER = (
    "Question: {focus}\n\n"
    "Document:\n{text}\n\n"
    "Answer the question using only the document. If the document does not "
    "address it, say so plainly."
)
_DIGEST_MAP_SYS = (
    "You are scanning ONE part of a larger document for anything relevant to a "
    "question. Return only the relevant facts/passages from this part, concise "
    "and faithful. If nothing here is relevant, reply exactly: NOTHING RELEVANT."
)
_DIGEST_MAP_USER = (
    "Question: {focus}\n\n"
    "Part {i} of {n}:\n{chunk}\n\n"
    "Relevant extract (or 'NOTHING RELEVANT'):"
)
_DIGEST_REDUCE_SYS = (
    "You synthesise per-section extracts into one focused answer to a question. "
    "Merge duplicates, keep specifics, note if the evidence is thin. Faithful only."
)
_DIGEST_REDUCE_USER = (
    "Question: {focus}\n\n"
    "Extracts from across the document:\n{extracts}\n\n"
    "Focused synthesis answering the question:"
)


def _chunk_text(text: str, size: int) -> list[str]:
    """Greedily pack paragraphs into ~size-char chunks; hard-split giant paras."""
    chunks: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        if len(para) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            for j in range(0, len(para), size):
                chunks.append(para[j:j + size])
            continue
        if cur and len(cur) + len(para) + 2 > size:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def _digest_wrap(synthesis: str, origin: str) -> str:
    return (
        f"{synthesis}\n\n"
        f"[Focused digest of {origin}. The raw content was NOT added to the "
        f"conversation — fetch_readable it directly if you need the full text.]"
    )


def _map_reduce_digest(text: str, focus: str, llm, origin: str) -> str:
    from assistant.conversation import _call_llm_raw
    from assistant.config import strip_channel_markup

    def _one(sys_prompt: str, user_prompt: str, max_tokens: int) -> str:
        resp = _call_llm_raw(
            llm,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens, temperature=0.3,
        )
        return strip_channel_markup(
            (resp["choices"][0]["message"].get("content") or "").strip())

    # Small enough to fold in one pass.
    if len(text) <= _DIGEST_CHUNK_CHARS:
        try:
            out = _one(_DIGEST_SINGLE_SYS,
                       _DIGEST_SINGLE_USER.format(focus=focus, text=text), 700)
        except Exception as exc:
            # LLM unavailable — fall back to plain truncation, better than nothing.
            head = text[:_FETCH_MAX_CHARS]
            return (f"[digest LLM unavailable ({exc}); returning truncated raw "
                    f"text from {origin} instead.]\n\n{head}")
        return _digest_wrap(out or "(no relevant content found)", origin)

    # Map: score each chunk against the focus, keep the relevant extracts.
    chunks = _chunk_text(text, _DIGEST_CHUNK_CHARS)
    dropped = 0
    if len(chunks) > _DIGEST_MAX_CHUNKS:
        dropped = len(chunks) - _DIGEST_MAX_CHUNKS
        chunks = chunks[:_DIGEST_MAX_CHUNKS]

    extracts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        try:
            e = _one(_DIGEST_MAP_SYS,
                     _DIGEST_MAP_USER.format(focus=focus, i=i, n=len(chunks),
                                             chunk=chunk), 400)
        except Exception:
            continue  # skip a failed chunk rather than abort the whole digest
        if e and "NOTHING RELEVANT" not in e.upper():
            extracts.append(e)

    drop_note = (f"\n\n[Note: document exceeded the {_DIGEST_MAX_CHUNKS}-chunk "
                 f"cap; the last {dropped} section(s) were not scanned.]"
                 if dropped else "")

    if not extracts:
        return (f"digest: nothing in {origin} was relevant to '{focus}'."
                + drop_note)

    # Reduce: synthesise the extracts into one focused answer.
    joined = "\n\n---\n\n".join(extracts)
    try:
        out = _one(_DIGEST_REDUCE_SYS,
                   _DIGEST_REDUCE_USER.format(focus=focus, extracts=joined), 800)
    except Exception:
        out = joined  # reduce failed — raw extracts still beat nothing
    return _digest_wrap(out, origin) + drop_note


def _digest(source: str, focus: str, llm) -> str:
    src = (source or "").strip()
    focus = (focus or "").strip()
    if not src:
        return "digest: 'source' is empty (give a URL or text to summarise)."
    if not focus:
        return ("digest: 'focus' is required — the question to summarise toward "
                "(e.g. 'red flags about working at company X').")

    is_url = src.startswith(("http://", "https://")) and len(src.split()) == 1
    if is_url:
        text, err = _fetch_full_text(src)
        if err is not None:
            return err
        origin = src
    else:
        text, origin = src, "provided text"

    text = (text or "").strip()
    if not text:
        return f"digest: nothing to summarise from {origin}."
    return _map_reduce_digest(text, focus, llm, origin)


def _import_ddgs():
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    return DDGS


def _web_search(query: str, max_results: int = 5) -> str:
    DDGS = _import_ddgs()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results found."
    parts = []
    for r in results:
        parts.append(f"**{r['title']}**\n{r['href']}\n{r['body']}")
    return "\n\n".join(parts)


def _save_note(memory_tools, note: str, tags: str) -> str:
    tag_list = ["assistant", "note"] + [t.strip() for t in tags.split(",") if t.strip()]
    result = memory_tools.store_memory(
        context="assistant session",
        action="save_note",
        outcome=note,
        tags=tag_list,
        dedup=True,
    )
    if result.success and (result.data or {}).get("reinforced"):
        return f"Already knew this — {result.message}"
    return f"Saved: {note}"


def _recall(memory_tools, query: str) -> str:
    result = memory_tools.recall_similar(query, limit=5)
    if not result.success:
        return f"Memory recall failed: {result.error}"
    memories = result.data.get("memories", [])
    if not memories:
        return "Nothing relevant found in memory."
    parts = []
    for m in memories:
        text = m.get("outcome") or m.get("action") or m.get("context") or str(m)
        parts.append(f"- {text}")
    return "\n".join(parts)


def _recall_analogies(memory_tools, situation: str) -> str:
    result = memory_tools.recall_analogies(situation, limit=3)
    if not result.success:
        return f"Analogy search failed: {result.error}"
    analogies = result.data.get("analogies", [])
    if not analogies:
        return "No cross-domain analogies found in memory."
    parts = []
    for a in analogies:
        tags = ", ".join(a.get("tags") or []) or "untagged"
        parts.append(
            f"- [{tags}] (similarity {a['similarity']}, domain distance "
            f"{a['domain_distance']}): {a['memory']}"
        )
    return "Structurally similar experiences from other domains:\n" + "\n".join(parts)


def _search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    category: str = "",
) -> str:
    """Search arXiv via the official API. Rate-limited and cached."""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    global _arxiv_last_call

    max_results = min(max(1, max_results), 10)

    # Build search query — optionally scope to a category
    search_query = query
    if category:
        search_query = f"cat:{category} AND ({query})"

    sort_order = "descending" if sort_by == "recent" else "descending"
    sort_field = "submittedDate" if sort_by == "recent" else "relevance"

    cache_key = f"{search_query}|{max_results}|{sort_field}"

    # Check cache first (no network cost)
    with _arxiv_lock:
        if cache_key in _arxiv_cache:
            ts, cached = _arxiv_cache[cache_key]
            if time.monotonic() - ts < _ARXIV_CACHE_TTL:
                return f"[cached]\n{cached}"

    # Rate limit: wait if we called too recently
    with _arxiv_lock:
        elapsed = time.monotonic() - _arxiv_last_call
        if elapsed < _ARXIV_MIN_GAP:
            time.sleep(_ARXIV_MIN_GAP - elapsed)
        _arxiv_last_call = time.monotonic()

    params = urllib.parse.urlencode({
        "search_query": search_query,
        "max_results": max_results,
        "sortBy": sort_field,
        "sortOrder": sort_order,
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "assistant-agent/1.0 (research tool; respectful bot)"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=20)
        xml_data = resp.read()
    except Exception as exc:
        return f"arXiv request failed: {exc}"

    # Parse Atom XML
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        return f"arXiv response parse error: {exc}"

    entries = root.findall("atom:entry", ns)
    if not entries:
        return "No results found on arXiv."

    parts = []
    for entry in entries:
        title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
        if len(abstract) > 400:
            abstract = abstract[:397] + "..."

        authors = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ]
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += f" et al. (+{len(authors) - 3})"

        # Find the PDF and abstract page links
        pdf_url = ""
        abs_url = ""
        for link in entry.findall("atom:link", ns):
            rel = link.get("rel", "")
            title_attr = link.get("title", "")
            href = link.get("href", "")
            if title_attr == "pdf":
                pdf_url = href
            elif rel == "alternate":
                abs_url = href

        published = (entry.findtext("atom:published", "", ns) or "")[:10]

        parts.append(
            f"**{title}**\n"
            f"Authors: {author_str}\n"
            f"Published: {published}\n"
            f"Abstract: {abstract}\n"
            f"Page: {abs_url}\n"
            f"PDF:  {pdf_url}"
        )

    result = "\n\n".join(parts)

    # Store in cache
    with _arxiv_lock:
        _arxiv_cache[cache_key] = (time.monotonic(), result)

    return result
