#!/usr/bin/env python3
"""One-time journal consolidation: distil accumulated research into long-term memory.

Reads journal entries from data_db, groups them by topic tag, summarizes
each group with the configured LLM, and stores the results in the memory DB
tagged ["journal-summary", topic, "consolidated"].

After this runs, the summaries are retrievable via recall_similar() and
survive context compaction. Raw journal entries are left untouched.

Usage:
    python journal_to_memory.py --config adadb/assistant.yaml --list-topics
    python journal_to_memory.py --config adadb/assistant.yaml --dry-run
    python journal_to_memory.py --config adadb/assistant.yaml
    python journal_to_memory.py --config adadb/assistant.yaml --topic consciousness
    python journal_to_memory.py --config adadb/assistant.yaml --model accounts/fireworks/models/gpt-oss-120b
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Tags that identify entry type, not topic — skip when choosing the topic group.
_SKIP_TAGS = {"research", "curiosity", "note", "auto-extracted", "atom", "agent",
              "journal", "entry", "finding", "reflection"}

_CHUNK_SIZE = 50  # entries per LLM summarization call

_CHUNK_PROMPT = """\
Summarize the key findings from these {n} research journal entries on the topic "{topic}".

Extract the most important facts, patterns, and concrete insights in 4-6 bullet points.
Be specific — preserve actual findings, not just "research was done on X".
Note any recurring themes or contradictions across entries.

Entries (chronological):
{entries}
"""

_SYNTHESIS_PROMPT = """\
You have {n} partial research summaries on the topic "{topic}":

{summaries}

Write a cohesive 2-3 paragraph synthesis covering:
- What is now understood about this topic
- Patterns and themes that emerged
- Open questions or areas worth investigating further

Write in first person as the researcher ("I have found...", "The evidence suggests...").
Be concrete and specific — this will be stored as a memory and retrieved in future sessions.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate journal entries into long-term memory"
    )
    parser.add_argument("--config", required=True,
                        help="Path to assistant YAML config (e.g. adadb/assistant.yaml)")
    parser.add_argument("--list-topics", action="store_true",
                        help="Show discovered topics and entry counts, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run LLM summarization but do not write to memory")
    parser.add_argument("--topic",
                        help="Only process this topic (default: all)")
    parser.add_argument("--chunk-size", type=int, default=_CHUNK_SIZE,
                        help=f"Entries per LLM call (default {_CHUNK_SIZE})")
    parser.add_argument("--model",
                        help="Override the model from config")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip topics that already have a journal-summary in memory")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        _die("pyyaml not installed — run: pip install pyyaml")

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        _die(f"Config not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    config_dir = config_path.parent

    data_db = _resolve_path(cfg.get("data_db", "assistant_data.db"), config_dir)
    memory_db_str = cfg.get("db")
    if not memory_db_str:
        _die("No 'db' key in config — add 'db: ada.db' to enable memory")
    memory_db = _resolve_path(memory_db_str, config_dir)

    if not data_db.exists():
        _die(f"Journal DB not found: {data_db}")

    print(f"Journal DB : {data_db}")
    print(f"Memory DB  : {memory_db}")

    entries_by_topic = _load_journal(data_db)

    if not entries_by_topic:
        print("No journal entries found.")
        return

    if args.list_topics:
        print(f"\nDiscovered topics ({len(entries_by_topic)}):")
        for topic, entries in sorted(entries_by_topic.items(), key=lambda x: -len(x[1])):
            date_range = f"{entries[0]['date']} – {entries[-1]['date']}"
            print(f"  {topic:<30} {len(entries):>4} entries  [{date_range}]")
        return

    if args.topic:
        if args.topic not in entries_by_topic:
            available = ", ".join(sorted(entries_by_topic))
            _die(f"Topic '{args.topic}' not found. Available: {available}")
        entries_by_topic = {args.topic: entries_by_topic[args.topic]}

    model = args.model or cfg.get("model", "accounts/fireworks/models/gpt-oss-120b")
    print(f"Model      : {model}")
    print(f"Chunk size : {args.chunk_size}")
    if args.dry_run:
        print("Mode       : dry-run (no writes)")
    print()

    llm = _build_llm(model)
    memory_tools = None if args.dry_run else _build_memory(str(memory_db), llm)

    existing_topics: set[str] = set()
    if args.skip_existing and memory_tools:
        existing_topics = _existing_summary_topics(str(memory_db))
        if existing_topics:
            print(f"Skipping already-consolidated topics: {', '.join(sorted(existing_topics))}\n")

    total_stored = 0
    for topic, entries in sorted(entries_by_topic.items(), key=lambda x: -len(x[1])):
        if topic in existing_topics:
            continue
        stored = _process_topic(topic, entries, llm, memory_tools, args.chunk_size, args.dry_run)
        total_stored += stored

    print(f"\n{'═' * 52}")
    if args.dry_run:
        print("Dry run complete — nothing written to memory.")
    else:
        print(f"Done. {total_stored} topic summary/summaries stored in memory.")
        print("Use recall_similar() or ask the agent to recall a topic to verify.")


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------

def _load_journal(data_db: Path) -> dict[str, list[dict]]:
    conn = sqlite3.connect(str(data_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT date, time, content, tags, author FROM journal_entries "
        "ORDER BY date, time"
    ).fetchall()
    conn.close()

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        topic = _primary_topic(tags)
        by_topic[topic].append({
            "date": row["date"],
            "time": row["time"],
            "content": row["content"],
        })
    return dict(by_topic)


def _primary_topic(tags: list) -> str:
    for tag in tags:
        t = str(tag).lower().strip()
        if t and t not in _SKIP_TAGS:
            return t
    return "general"


# ---------------------------------------------------------------------------
# Topic processing
# ---------------------------------------------------------------------------

def _process_topic(
    topic: str,
    entries: list[dict],
    llm,
    memory_tools,
    chunk_size: int,
    dry_run: bool,
) -> int:
    print(f"{'─' * 52}")
    print(f"Topic: {topic}  ({len(entries)} entries)")

    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
    print(f"  {len(chunks)} chunk(s) of up to {chunk_size} entries")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarising chunk {i + 1}/{len(chunks)}...", end=" ", flush=True)
        summary = _summarize_chunk(topic, chunk, llm)
        if summary:
            chunk_summaries.append(summary)
            print("✓")
        else:
            print("empty — skipped")

    if not chunk_summaries:
        print(f"  No usable summaries for '{topic}' — skipping.")
        return 0

    if len(chunk_summaries) == 1:
        final = chunk_summaries[0]
    else:
        print(f"  Synthesising {len(chunk_summaries)} summaries...", end=" ", flush=True)
        final = _synthesize(topic, chunk_summaries, llm)
        print("✓")

    # Preview
    preview = final[:400].replace("\n", " ")
    print(f"  Preview: {preview}{'...' if len(final) > 400 else ''}")

    if dry_run or memory_tools is None:
        return 0

    try:
        memory_tools.store_memory(
            context=f"journal consolidation: {topic}",
            action="synthesized research findings",
            outcome=final,
            tags=["journal-summary", topic, "consolidated"],
        )
        print(f"  Stored to memory ✓")
        return 1
    except Exception as exc:
        print(f"  ERROR storing: {exc}", file=sys.stderr)
        return 0


def _summarize_chunk(topic: str, entries: list[dict], llm) -> str:
    lines = [f"[{e['date']} {e['time']}] {e['content']}" for e in entries]
    prompt = _CHUNK_PROMPT.format(
        topic=topic, n=len(entries), entries="\n".join(lines)
    )
    return _call_llm(llm, prompt, max_tokens=1024)


def _synthesize(topic: str, summaries: list[str], llm) -> str:
    joined = "\n\n---\n\n".join(f"Summary {i+1}:\n{s}" for i, s in enumerate(summaries))
    prompt = _SYNTHESIS_PROMPT.format(topic=topic, n=len(summaries), summaries=joined)
    return _call_llm(llm, prompt, max_tokens=2048)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_llm(llm, prompt: str, max_tokens: int) -> str:
    from assistant.conversation import _call_llm_raw
    try:
        resp = _call_llm_raw(
            llm,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return (resp["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:
        print(f"\n  LLM call failed: {exc}", file=sys.stderr)
        return ""


def _build_llm(model: str):
    from assistant.config import _make_llm
    return _make_llm(model)


def _build_memory(db_path: str, llm):
    from assistant.config import build_memory as _build_memory_cfg
    return _build_memory_cfg(db_path, llm=llm)


def _resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def _existing_summary_topics(db_path: str) -> set[str]:
    """Return topics that already have a journal-summary in the memory DB."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT tags FROM episodes WHERE tags LIKE '%journal-summary%'"
        ).fetchall()
        conn.close()
        topics = set()
        for (tags_json,) in rows:
            try:
                tags = json.loads(tags_json or "[]")
                for t in tags:
                    if t not in ("journal-summary", "consolidated"):
                        topics.add(t)
            except json.JSONDecodeError:
                pass
        return topics
    except Exception:
        return set()


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
