"""Shared setup helpers used by both main.py (headless) and tui.py (Textual)."""

import re
import sys
from pathlib import Path

_DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-120b"

# Resolved at import time so callers can use it as a constant
_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # assistant/ root


def load_config(config_path: str | None) -> dict:
    """Load a YAML config file. Returns empty dict if not provided."""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError:
        print(
            "Warning: pyyaml not installed — ignoring --config. Run: pip install pyyaml",
            file=sys.stderr,
        )
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve(cli_val, cfg: dict, key: str, default):
    """Return first set value: CLI arg → YAML config → hardcoded default."""
    if cli_val is not None:
        return cli_val
    return cfg.get(key, default)


def load_soul(soul_arg: str | None = None) -> tuple[str, str]:
    """Load soul text. Returns (soul_text, source_path_str).

    Priority:
      1. --soul <path>  (explicit CLI argument)
      2. ./soul.txt     (current working directory)
      3. soul.txt next to the assistant package root
      4. Hardcoded fallback
    """
    candidates: list[Path] = []
    if soul_arg:
        candidates.append(Path(soul_arg))
    candidates.append(Path.cwd() / "soul.txt")
    candidates.append(_SCRIPT_DIR / "soul.txt")

    for path in candidates:
        if path.exists():
            return path.read_text().strip(), str(path)
    return "You are a helpful personal assistant.", "<default>"


def detect_vector_backend() -> str:
    """Return the best available vector backend: 'chromadb', 'sqlite-vec', or 'lite'."""
    try:
        import chromadb  # noqa: F401
        return "chromadb"
    except ImportError:
        pass
    try:
        import sqlite_vec  # noqa: F401
        return "sqlite-vec"
    except ImportError:
        pass
    return "lite"


def build_memory(db_path: str, llm=None):
    """
    Build a memory backend, auto-detecting the best available vector store.

    Priority: chromadb → sqlite-vec → LiteMemory (keyword-only, no vectors).
    Falls back to LiteMemory if agent-memory itself is not installed.
    """
    backend = detect_vector_backend()

    if backend in ("chromadb", "sqlite-vec"):
        try:
            from agent_memory import MemoryStore, MemoryTools
            from assistant.llm_adapter import FireworksEmbeddingGenerator, FireworksMemoryLLM

            embedding_gen = FireworksEmbeddingGenerator()
            store = MemoryStore(
                db_path=db_path,
                embedding_generator=embedding_gen,
                backend=backend,
            )
            memory_llm = FireworksMemoryLLM(llm) if llm else None
            return MemoryTools(store=store, llm=memory_llm)
        except ImportError:
            pass  # agent-memory not installed, fall through to LiteMemory

    from assistant.lite_memory import LiteMemory
    print(
        "Note: using lightweight SQLite memory (install chromadb or sqlite-vec for semantic recall).",
        file=sys.stderr,
    )
    return LiteMemory(db_path, llm=llm)


class ModelRouter:
    """Routes assistant tasks to (potentially different) LLM instances.

    If a single model is configured, all tasks use it (default behaviour).
    If a ``models:`` section exists in the YAML config, specific tasks can
    be overridden to use a different model — e.g. a capable model for
    reflection while keeping a fast model for conversation.

    Supported task names: conversation, curiosity, inner_voice, compaction.
    (reflection is reserved for when memory LLM routing is added.)

    YAML example::

        model: accounts/fireworks/models/gpt-oss-120b  # default
        models:
            compaction: claude-sonnet-4-6
            curiosity:  accounts/fireworks/models/deepseek-v3p2
    """

    TASKS = ("conversation", "curiosity", "inner_voice", "compaction", "nudge")

    def __init__(self, default_llm, overrides: dict | None = None) -> None:
        self._default = default_llm
        self._overrides: dict = overrides or {}

    def for_task(self, task: str):
        """Return the LLM assigned to *task*, falling back to the default."""
        return self._overrides.get(task, self._default)

    @property
    def default(self):
        return self._default

    def describe(self) -> str:
        """Human-readable summary of overrides for the startup banner."""
        if not self._overrides:
            return ""
        parts = []
        for task, llm in self._overrides.items():
            parts.append(f"{task}={llm._model.split('/')[-1]}")
        return "  Models : " + ", ".join(parts)


def provider_for(model_id: str) -> tuple[str, str]:
    """Return (provider_label, api_key_env_var) for a model id.

    Mirrors the dispatch in :func:`_make_llm` so callers can print an accurate
    "missing key" message without duplicating the prefix rules.
    """
    if model_id.startswith("claude"):
        return "Anthropic", "ANTHROPIC_API_KEY"
    if model_id.startswith("grok"):
        return "xAI", "XAI_API_KEY"
    return "Fireworks", "FIREWORKS_API_KEY"


def _make_llm(model_id: str, max_tokens: int = 8192, min_request_interval: float = 1.0):
    """Instantiate the right LLM class from a model ID string.

    Dispatch rules (checked in order):
    - ``claude*``  → AnthropicLLM         (reads ANTHROPIC_API_KEY)
    - ``grok*``    → OpenAILLM at api.x.ai (reads XAI_API_KEY)
    - everything else → FireworksLLM       (reads FIREWORKS_API_KEY)
    """
    from agent_core.llm_cloud import AnthropicLLM, FireworksLLM, OpenAILLM
    if model_id.startswith("claude"):
        return AnthropicLLM(model=model_id, max_tokens=max_tokens)
    if model_id.startswith("grok"):
        import os
        # base_url must NOT include /v1 — OpenAILLM appends /v1/chat/completions.
        return OpenAILLM(
            model=model_id,
            base_url="https://api.x.ai",
            api_key=os.environ.get("XAI_API_KEY", ""),
            max_tokens=max_tokens,
        )
    return FireworksLLM(model=model_id, max_tokens=max_tokens,
                        min_request_interval=min_request_interval)


def build_mailbox(cfg: dict, config_dir: "Path | None" = None):
    """Build an AgentMailbox from config.  Returns None if not configured.

    Requires both ``name:`` and ``mailbox_db:`` to be set in the YAML.
    ``mailbox_db`` may be relative (resolved against the config file directory)
    or absolute.  Use an absolute path when multiple agents on the same machine
    need to share the same file.
    """
    db_path = cfg.get("mailbox_db")
    agent_name = cfg.get("name")
    if not db_path or not agent_name:
        return None
    from agent_core.mailbox import AgentMailbox
    path = Path(db_path)
    if not path.is_absolute() and config_dir:
        path = config_dir / path
    return AgentMailbox(db_path=path, agent_name=str(agent_name))


def build_conversation_bus(cfg: dict, config_dir: "Path | None" = None):
    """Build a ConversationBus from config.  Returns None if not configured.

    Shares the mailbox's requirements and file — ``name:`` + ``mailbox_db:`` —
    since conversations live in the same shared channel as mailbox messages.
    No separate config needed: any agent with a mailbox also gets structured
    turn-taking conversations.
    """
    db_path = cfg.get("mailbox_db")
    agent_name = cfg.get("name")
    if not db_path or not agent_name:
        return None
    from agent_core.conversation import ConversationBus
    path = Path(db_path)
    if not path.is_absolute() and config_dir:
        path = config_dir / path
    return ConversationBus(db_path=path, agent_name=str(agent_name))


def build_session_handoff(cfg: dict, default_dir):
    """Build a SessionHandoff from config. Returns None unless opted in.

    Opt-in via ``session_handoff: true`` — agents with a soul-level handoff
    protocol simply leave it unset and are unaffected. Overridable:
      - ``handoff_file:``   path (relative → resolved against *default_dir*)
      - ``handoff_prompt:`` note-synthesis instructions (own voice / texture)
      - ``handoff_max_entries:`` log soft-cap (default 20)
    Default file is ``<default_dir>/<name>_handoff.md``.
    """
    if not cfg.get("session_handoff"):
        return None
    from assistant.session_handoff import SessionHandoff, DEFAULT_HANDOFF_PROMPT
    hf = cfg.get("handoff_file")
    if hf:
        path = Path(hf).expanduser()
        if not path.is_absolute():
            path = Path(default_dir) / path
    else:
        name = str(cfg.get("name") or "assistant")
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).lower()
        path = Path(default_dir) / f"{safe}_handoff.md"
    return SessionHandoff(
        path,
        prompt=cfg.get("handoff_prompt") or DEFAULT_HANDOFF_PROMPT,
        max_entries=int(cfg.get("handoff_max_entries", 20)),
    )


_END_TOKENS_RE = re.compile(r"<\|(?:end|return|start|call)\|>(?:assistant)?")

# Phrases that mark a research summary as "nothing was found". Checked against
# the head of the text only — a real finding may mention absences later on.
_NULL_FINDING_MARKERS = (
    "no new ", "no additional", "do not contain", "does not contain",
    "no relevant", "nothing new", "contain no new", "no fresh ",
    "did not return", "no concrete new", "no significant new",
)


def strip_channel_markup(text: str) -> str:
    """Remove gpt-oss harmony channel markup that leaks into LLM output.

    Keeps only the content after the last <|message|> marker (the final
    channel — analysis channels precede it), then drops residual tokens.
    Text without markup passes through untouched.
    """
    if "<|message|>" not in text and "<|channel|>" not in text:
        return text
    text = text.split("<|message|>")[-1]
    return _END_TOKENS_RE.sub("", text).strip()


def looks_like_null_finding(text: str) -> bool:
    """True if a research summary says 'nothing was found'.

    Such summaries belong in the journal (audit trail) but must not be
    stored as memories — absence of findings is not knowledge, and
    reinforcement would rank it as if it were.
    """
    head = text.strip().lower()[:220]
    return any(marker in head for marker in _NULL_FINDING_MARKERS)


# Tags that mark an entry's type/origin, not its topic — excluded from the
# tag cloud and from dream-replay topic comparison.
TYPE_TAGS = {
    "assistant", "note", "atom", "auto-extracted", "agent", "journal-summary",
    "consolidated", "research", "curiosity", "association", "work-cycle",
    "reflection", "journal", "entry", "finding",
}


def format_tag_cloud(memory, limit: int = 25) -> str:
    """Compact 'tag(count)' metamemory line from a MemoryTools instance.

    Returns "" when the backend doesn't support tag aggregation (LiteMemory)
    or the store is empty.
    """
    store = getattr(memory, "store", None)
    if store is None or not hasattr(store, "get_tag_counts"):
        return ""
    try:
        counts = store.get_tag_counts(skip_tags=TYPE_TAGS)
    except Exception:
        return ""
    return ", ".join(f"{tag}({n})" for tag, n in counts[:limit])


def build_mail_sender(cfg: dict):
    """Build a MailSender from the ``mail:`` YAML section.  None if absent.

    Requires ``mail.recipients`` (alias -> address map).  The agent can only
    send to configured aliases — addresses are never taken from the LLM.
    """
    mail_cfg = cfg.get("mail") or {}
    recipients = mail_cfg.get("recipients") or {}
    if not recipients:
        return None
    from assistant.mail import MailSender
    return MailSender(
        recipients={str(k): str(v) for k, v in recipients.items()},
        from_addr=str(mail_cfg.get("from", "assistant@localhost")),
        sendmail_path=str(mail_cfg.get("sendmail_path", "/usr/sbin/sendmail")),
        subject_prefix=str(mail_cfg.get("subject_prefix", "")),
        max_per_day=int(mail_cfg.get("max_per_day", 20)),
    )


def build_spawn_registry(cfg: dict, config_dir: "Path | None" = None):
    """Build a SpawnRegistry from the ``spawn_roles:`` YAML section.

    Returns ``None`` if no spawn_roles are configured.

    Args:
        cfg:        Full YAML config dict.
        config_dir: Directory of the config file, used to resolve relative
                    soul paths.  Pass ``Path(config_path).parent`` from main.
    """
    spawn_cfg = cfg.get("spawn_roles") or {}
    if not spawn_cfg:
        return None
    from agent_core.spawn import SpawnRegistry
    return SpawnRegistry.from_config(
        spawn_cfg,
        soul_base_dir=config_dir,
        make_llm_fn=_make_llm,
    )


def build_router(cfg: dict, default_llm) -> ModelRouter:
    """Build a ModelRouter from config.

    If ``cfg`` has no ``models`` key, returns a router that sends all tasks
    to *default_llm* (zero cost, same as before).
    """
    models_cfg = cfg.get("models") or {}
    if not models_cfg:
        return ModelRouter(default_llm)

    overrides = {}
    for task, model_id in models_cfg.items():
        if task not in ModelRouter.TASKS:
            print(f"Warning: unknown model routing task '{task}' — ignored.", file=sys.stderr)
            continue
        overrides[task] = _make_llm(str(model_id))

    return ModelRouter(default_llm, overrides)


def fmt_args(args: dict) -> str:
    """Format tool args for display — truncate long values."""
    parts = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 40:
            sv = sv[:37] + "..."
        parts.append(f"{k}={sv!r}")
    return ", ".join(parts)
