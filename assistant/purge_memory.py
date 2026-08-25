#!/usr/bin/env python3
"""Purge junk memories: channel-markup leaks and 'nothing found' findings.

Historical stores accumulated two kinds of junk that dedup correctly kept
(they are genuine, heavily-reinforced duplicates — of garbage):

- LLM chat-template leaks (harmony ``<|channel|>`` markup stored verbatim)
- "no new results" research summaries stored as if they were knowledge

Both are now blocked at write time (strip_channel_markup /
looks_like_null_finding); this script removes what's already stored.
Deletes matching episodes (SQLite row + vector entry) and reflections.

Run once per agent after dedup_memory.py. With chromadb, run from the
agent's working directory so the vector dir (data/memory_vectors, relative
to cwd — same as at runtime) resolves to the same place; with sqlite-vec
the vectors live inside the DB file and cwd doesn't matter.

    python purge_memory.py --config souls/ada.yaml --dry-run
    python purge_memory.py --config souls/ada.yaml
    python purge_memory.py --config souls/ada.yaml --like "%custom junk%"
"""

import argparse
import sys
from pathlib import Path

_DEFAULT_LIKES = [
    "%<|channel|>%",
    "%<|message|>%",
    "%no new technical data%",
    "%do not contain any additional%",
    "%does not contain any new%",
    "%contain no new%",
    "%no relevant results%",
    "%no new information%",
]

_SHOW = 20  # matches to display


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _where(column: str, n_patterns: int) -> str:
    return " OR ".join([f"{column} LIKE ?"] * n_patterns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove junk memories (template leaks, null findings)")
    parser.add_argument("--config", required=True,
                        help="Agent YAML config (provides db path)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report matches without deleting")
    parser.add_argument("--like", action="append", default=[],
                        help="Additional SQL LIKE pattern (repeatable)")
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

    memory_db_str = cfg.get("db")
    if not memory_db_str:
        _die("No 'db' key in config — nothing to purge")
    memory_db = Path(memory_db_str).expanduser()
    if not memory_db.is_absolute():
        memory_db = config_path.parent / memory_db
    if not memory_db.exists():
        _die(f"Memory DB not found: {memory_db}")

    from assistant.config import detect_vector_backend
    from agent_memory.memory_store import MemoryStore

    backend = detect_vector_backend()
    if backend not in ("chromadb", "sqlite-vec"):
        _die("No vector backend available (install chromadb or sqlite-vec)")

    patterns = _DEFAULT_LIKES + args.like
    print(f"Memory DB : {memory_db} [{backend}]")
    print(f"Patterns  : {len(patterns)}")
    if args.dry_run:
        print("Mode      : dry-run (no writes)")

    store = MemoryStore(db_path=str(memory_db), embedding_generator=None,
                        backend=backend)
    cursor = store.conn.cursor()

    ep_rows = cursor.execute(
        f"SELECT id, COALESCE(occurrence_count, 1), substr(outcome, 1, 80) "
        f"FROM episodes WHERE {_where('outcome', len(patterns))}",
        patterns,
    ).fetchall()
    print(f"\nEpisodes matching: {len(ep_rows)}")
    for ep_id, occ, head in ep_rows[:_SHOW]:
        print(f"  #{ep_id} (x{occ}): {head}")
    if len(ep_rows) > _SHOW:
        print(f"  ... and {len(ep_rows) - _SHOW} more")

    refl_rows = cursor.execute(
        f"SELECT id, COALESCE(occurrence_count, 1), substr(insight, 1, 80) "
        f"FROM reflections WHERE {_where('insight', len(patterns))}",
        patterns,
    ).fetchall()
    print(f"\nReflections matching: {len(refl_rows)}")
    for r_id, occ, head in refl_rows[:_SHOW]:
        print(f"  #{r_id} (x{occ}): {head}")
    if len(refl_rows) > _SHOW:
        print(f"  ... and {len(refl_rows) - _SHOW} more")

    if args.dry_run:
        print(f"\nDry-run: would remove {len(ep_rows)} episodes, "
              f"{len(refl_rows)} reflections.")
        store.close()
        return

    for ep_id, _, _ in ep_rows:
        cursor.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
        store._vec.delete(ep_id)
    for r_id, _, _ in refl_rows:
        cursor.execute("DELETE FROM reflections WHERE id = ?", (r_id,))
    store.conn.commit()
    store.close()
    print(f"\nRemoved {len(ep_rows)} episodes, {len(refl_rows)} reflections.")


if __name__ == "__main__":
    main()
