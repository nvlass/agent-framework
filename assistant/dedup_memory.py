#!/usr/bin/env python3
"""One-time memory cleanup: re-embed episodes and merge near-duplicates.

Two phases:

1. Re-embed. Episode embeddings were historically computed from
   context+action only, so notes and L1 atoms — whose content lives in
   outcome — were indistinguishable in vector space. Every episode embedding
   is regenerated under the current formula (context+action+outcome).

2. Dedup. Episodes, then reflections, are greedy-clustered by cosine
   similarity. Each group keeps its earliest member, occurrence counts are
   summed into it (repetition becomes an importance signal instead of
   noise), and the rest are deleted from SQLite and the vector store.

Usage:
    python dedup_memory.py --config adadb/assistant.yaml --dry-run
    python dedup_memory.py --config adadb/assistant.yaml
    python dedup_memory.py --config adadb/assistant.yaml --skip-reembed
    python dedup_memory.py --config adadb/assistant.yaml --episode-threshold 0.92
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_BATCH = 32


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _resolve_path(p: str, base: Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else base / path


def _embed_texts(generator, texts: list[str]) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), _BATCH):
        chunk = texts[i:i + _BATCH]
        embs.extend(generator.generate_embeddings_batch(chunk))
        print(f"  embedded {min(i + _BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    matrix = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _greedy_groups(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster row indices: each group is [keeper, dup, dup...], keeper first.

    Rows must be ordered oldest-first so the earliest memory wins.
    """
    sims = matrix @ matrix.T
    n = matrix.shape[0]
    assigned: set[int] = set()
    groups = []
    for i in range(n):
        if i in assigned:
            continue
        members = [i] + [
            j for j in range(i + 1, n)
            if j not in assigned and sims[i, j] >= threshold
        ]
        assigned.update(members)
        if len(members) > 1:
            groups.append(members)
    return groups


def _reembed_episodes(store, rows, matrix, dry_run: bool) -> None:
    if dry_run:
        print(f"  (dry-run) would re-embed {len(rows)} episodes")
        return
    for (ep_id, *_), emb in zip(rows, matrix):
        store._vec.delete(ep_id)
        store._vec.add(ep_id, emb)
    print(f"  re-embedded {len(rows)} episodes")


def _dedup_episodes(store, rows, matrix, threshold: float, dry_run: bool) -> int:
    groups = _greedy_groups(matrix, threshold)
    if not groups:
        print("  no duplicate episodes found")
        return 0
    removed = 0
    cursor = store.conn.cursor()
    for members in groups:
        keeper_row = rows[members[0]]
        keeper_id = keeper_row[0]
        dup_ids = [rows[j][0] for j in members[1:]]
        counts = cursor.execute(
            f"SELECT COALESCE(SUM(COALESCE(occurrence_count, 1)), 0) FROM episodes "
            f"WHERE id IN ({','.join('?' * len(members))})",
            [rows[j][0] for j in members],
        ).fetchone()[0]
        sample = (keeper_row[3] or keeper_row[1] or "")[:90].replace("\n", " ")
        print(f"  keep #{keeper_id} (x{counts}) ← merge {dup_ids}: {sample}")
        if dry_run:
            removed += len(dup_ids)
            continue
        cursor.execute(
            "UPDATE episodes SET occurrence_count = ?, "
            "last_confirmed = CURRENT_TIMESTAMP WHERE id = ?",
            (counts, keeper_id),
        )
        cursor.execute(
            f"DELETE FROM episodes WHERE id IN ({','.join('?' * len(dup_ids))})",
            dup_ids,
        )
        for dup_id in dup_ids:
            store._vec.delete(dup_id)
        removed += len(dup_ids)
    store.conn.commit()
    return removed


def _dedup_reflections(store, generator, threshold: float, dry_run: bool) -> int:
    cursor = store.conn.cursor()
    rows = cursor.execute(
        "SELECT id, reflection_type, insight, COALESCE(occurrence_count, 1) "
        "FROM reflections ORDER BY id"
    ).fetchall()
    rows = [r for r in rows if r[2]]
    if not rows:
        print("  no reflections in store")
        return 0
    removed = 0
    types = sorted({r[1] for r in rows})
    for rtype in types:
        subset = [r for r in rows if r[1] == rtype]
        if len(subset) < 2:
            continue
        matrix = _embed_texts(generator, [r[2] for r in subset])
        groups = _greedy_groups(matrix, threshold)
        for members in groups:
            keeper = subset[members[0]]
            dup_ids = [subset[j][0] for j in members[1:]]
            total = sum(subset[j][3] for j in members)
            sample = keeper[2][:90].replace("\n", " ")
            print(f"  [{rtype}] keep #{keeper[0]} (x{total}) ← merge {dup_ids}: {sample}")
            if dry_run:
                removed += len(dup_ids)
                continue
            cursor.execute(
                "UPDATE reflections SET occurrence_count = ?, "
                "last_confirmed = CURRENT_TIMESTAMP WHERE id = ?",
                (total, keeper[0]),
            )
            cursor.execute(
                f"DELETE FROM reflections WHERE id IN ({','.join('?' * len(dup_ids))})",
                dup_ids,
            )
            removed += len(dup_ids)
    store.conn.commit()
    if removed == 0:
        print("  no duplicate reflections found")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-embed and deduplicate an agent's memory store")
    parser.add_argument("--config", required=True,
                        help="Agent YAML config (provides db path)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--skip-reembed", action="store_true",
                        help="Skip phase 1 (embeddings already current)")
    parser.add_argument("--episode-threshold", type=float, default=0.90,
                        help="Cosine similarity for episode duplicates (default 0.90)")
    parser.add_argument("--reflection-threshold", type=float, default=0.85,
                        help="Cosine similarity for reflection duplicates (default 0.85)")
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
        _die("No 'db' key in config — nothing to deduplicate")
    memory_db = _resolve_path(memory_db_str, config_path.parent)
    if not memory_db.exists():
        _die(f"Memory DB not found: {memory_db}")

    from assistant.config import detect_vector_backend
    from assistant.llm_adapter import FireworksEmbeddingGenerator
    from agent_memory.memory_store import MemoryStore, _episode_search_text

    backend = detect_vector_backend()
    if backend not in ("chromadb", "sqlite-vec"):
        _die("No vector backend available (install chromadb or sqlite-vec)")

    print(f"Memory DB : {memory_db} [{backend}]")
    if args.dry_run:
        print("Mode      : dry-run (no writes)")

    generator = FireworksEmbeddingGenerator()
    store = MemoryStore(
        db_path=str(memory_db),
        embedding_generator=generator,
        backend=backend,
    )

    rows = store.conn.execute(
        "SELECT id, context, action, outcome FROM episodes ORDER BY id"
    ).fetchall()
    print(f"\nPhase 1 — re-embed ({len(rows)} episodes)")
    matrix = None
    if rows:
        texts = [_episode_search_text(c, a, o or "") for _, c, a, o in rows]
        matrix = _embed_texts(generator, texts)
        if args.skip_reembed:
            print("  skipped (--skip-reembed)")
        else:
            _reembed_episodes(store, rows, matrix, args.dry_run)

    print(f"\nPhase 2 — dedup episodes (threshold {args.episode_threshold})")
    ep_removed = 0
    if matrix is not None:
        ep_removed = _dedup_episodes(
            store, rows, matrix, args.episode_threshold, args.dry_run)

    print(f"\nPhase 3 — dedup reflections (threshold {args.reflection_threshold})")
    refl_removed = _dedup_reflections(
        store, generator, args.reflection_threshold, args.dry_run)

    verb = "would remove" if args.dry_run else "removed"
    print(f"\nDone: {verb} {ep_removed} duplicate episodes, "
          f"{refl_removed} duplicate reflections.")
    store.close()


if __name__ == "__main__":
    main()
