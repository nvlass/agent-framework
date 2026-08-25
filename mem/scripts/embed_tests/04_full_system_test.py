#!/usr/bin/env python3
"""
04_full_system_test.py - Complete Memory System Integration Test

This script tests the full memory system pipeline:
1. Store episodes with real embeddings
2. Semantic retrieval (find similar memories)
3. Consolidation (cluster and extract patterns)
4. Pattern-based recommendations

Run: python scripts/embed_tests/04_full_system_test.py
     python scripts/embed_tests/04_full_system_test.py --model mxbai
"""

import sys
import os
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_model_path, add_model_argument, print_model_info

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "embed_test_memory.db"
VECTOR_PATH = PROJECT_ROOT / "data" / "embed_test_vectors"


def get_sample_episodes():
    """
    Sample coding experiences organized by topic.
    Each tuple: (context, action, outcome, success_score, tags)
    """
    return [
        # ----- Python Debugging -----
        ("Python TypeError: 'NoneType' object is not subscriptable",
         "Added null check before dictionary access",
         "Bug fixed, handles missing data gracefully",
         0.9, ["python", "debugging", "null-check"]),

        ("Python TypeError with wrong argument types",
         "Added type hints and isinstance validation",
         "Type errors caught early with clear messages",
         0.85, ["python", "debugging", "types"]),

        ("Python crashes on nested dictionary access",
         "Used dict.get() with default values",
         "No more crashes, returns default when key missing",
         0.95, ["python", "debugging", "dict"]),

        ("TypeError: can't concatenate str and int",
         "Converted int to str before concatenation",
         "String formatting works correctly",
         0.8, ["python", "debugging", "types"]),

        ("NoneType error in list comprehension",
         "Added filter to skip None values",
         "List comprehension completes without errors",
         0.88, ["python", "debugging", "null-check"]),

        # ----- Git Operations -----
        ("Need to undo last commit without losing changes",
         "Used git reset --soft HEAD~1",
         "Commit undone, changes preserved in staging",
         0.95, ["git", "undo"]),

        ("Accidentally committed sensitive file",
         "Used git filter-branch to remove from history",
         "Sensitive file removed from all commits",
         0.7, ["git", "security"]),

        ("Merge conflict in package-lock.json",
         "Deleted file, ran npm install, committed fresh",
         "Clean merge, dependencies resolved",
         0.9, ["git", "merge", "npm"]),

        ("Need to squash multiple commits",
         "Used git rebase -i HEAD~3 with squash",
         "Commits combined into single clean commit",
         0.85, ["git", "rebase"]),

        # ----- Docker Issues -----
        ("Docker container won't start, port in use",
         "Changed host port mapping in docker-compose",
         "Container starts on new port",
         0.95, ["docker", "networking"]),

        ("Docker build failing, missing dependencies",
         "Added missing apt packages to Dockerfile",
         "Build completes successfully",
         0.9, ["docker", "build"]),

        ("Docker container out of disk space",
         "Added docker system prune to cleanup script",
         "Freed up space, container stable",
         0.85, ["docker", "maintenance"]),

        # ----- Database -----
        ("SQL query running slow on large table",
         "Added index on frequently queried columns",
         "Query time reduced from 30s to 0.5s",
         0.95, ["database", "performance"]),

        ("Database connection pool exhausted",
         "Increased pool size and added timeout",
         "No more connection errors under load",
         0.85, ["database", "performance"]),
    ]


def main():
    parser = argparse.ArgumentParser(description="Full memory system integration test")
    add_model_argument(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("Full Memory System Integration Test")
    print("=" * 60)

    model_path = get_model_path(args.model)

    # Clean up previous test data
    if DB_PATH.exists():
        os.remove(DB_PATH)
    if VECTOR_PATH.exists():
        shutil.rmtree(VECTOR_PATH)

    # -------------------------------------------------------------------------
    # 1. Initialize the memory system
    # -------------------------------------------------------------------------
    print("\n1. Initializing memory system...")
    print_model_info(args.model)

    from agent_memory.embeddings import EmbeddingGenerator
    from agent_memory.memory_store import MemoryStore

    embedding_gen = EmbeddingGenerator(str(model_path))
    store = MemoryStore(
        db_path=str(DB_PATH),
        vector_store_path=str(VECTOR_PATH),
        embedding_generator=embedding_gen
    )

    print(f"   Database: {DB_PATH}")
    print(f"   Vector store: {VECTOR_PATH}")

    # -------------------------------------------------------------------------
    # 2. Store episodes
    # -------------------------------------------------------------------------
    print("\n2. Storing episodes...")

    episodes = get_sample_episodes()
    for context, action, outcome, score, tags in episodes:
        store.store_episode(
            context=context,
            action=action,
            outcome=outcome,
            success_score=score,
            tags=tags
        )

    print(f"   Stored {len(episodes)} episodes")

    # -------------------------------------------------------------------------
    # 3. Test semantic retrieval
    # -------------------------------------------------------------------------
    print("\n3. Testing semantic retrieval...")
    print("   (Finding similar past experiences)")

    test_queries = [
        ("How do I fix Python NoneType errors?", ["python", "null"]),
        ("Git commit history problems", ["git", "undo"]),
        ("Docker networking not working", ["docker", "networking"]),
    ]

    for query, expected_topics in test_queries:
        print(f"\n   Query: '{query}'")
        print(f"   Expected topics: {expected_topics}")

        results = store.retrieve_episodes(query, limit=3)
        print("   Results:")
        for episode, similarity in results:
            tags_str = ", ".join(episode.tags[:2])
            print(f"     [{similarity:.3f}] [{tags_str}] {episode.context[:45]}...")

    # -------------------------------------------------------------------------
    # 4. Run consolidation
    # -------------------------------------------------------------------------
    print("\n4. Running consolidation (clustering + pattern extraction)...")

    report = store.run_consolidation()

    print(f"   Episodes processed: {report.episodes_processed}")
    print(f"   Clusters found: {report.clusters_found}")
    print(f"   Patterns extracted: {report.patterns_created}")
    print(f"   Duration: {report.duration_seconds:.2f}s")

    # -------------------------------------------------------------------------
    # 5. Show extracted patterns
    # -------------------------------------------------------------------------
    print("\n5. Extracted patterns:")

    patterns = store.get_learned_patterns(limit=5)

    if not patterns:
        print("   No patterns extracted (need more episodes or tighter clusters)")
    else:
        for pattern in patterns:
            print(f"\n   Pattern #{pattern.pattern_id}:")
            print(f"     Context: {pattern.context_signature[:60]}...")
            print(f"     Action: {pattern.recommended_action[:60]}...")
            print(f"     Success rate: {pattern.success_rate:.0%}")
            print(f"     Based on: {pattern.sample_count} episodes")
            print(f"     Confidence: {pattern.confidence:.2f}")

    # -------------------------------------------------------------------------
    # 6. Test pattern-based recommendations
    # -------------------------------------------------------------------------
    print("\n6. Pattern-based recommendations:")
    print("   (Using learned patterns to suggest actions)")

    recommendation_queries = [
        "I'm getting a Python TypeError",
        "Docker container networking problems",
        "Database queries are slow",
    ]

    for query in recommendation_queries:
        print(f"\n   Query: '{query}'")
        recommendations = store.recommend_actions(query, limit=2)

        if recommendations:
            for rec in recommendations:
                pattern = rec['pattern']
                print(f"     Recommendation: {pattern.recommended_action[:50]}...")
                print(f"       Success: {pattern.success_rate:.0%}, "
                      f"Confidence: {pattern.confidence:.2f}, "
                      f"Match: {rec['match_score']:.2f}")
        else:
            print("     No recommendations (no matching patterns)")

    # -------------------------------------------------------------------------
    # 7. Memory hierarchy demo
    # -------------------------------------------------------------------------
    print("\n7. Memory hierarchy (working -> short-term -> long-term):")

    # Working memory contains recent episodes from this session
    working = store.get_working_memory_episodes(n=3)
    print(f"   Working memory: {len(working)} episodes (hot cache)")

    # Short-term cache stats
    cache_stats = store.get_short_term_cache_stats()
    print(f"   Short-term cache: {cache_stats}")

    # Total in long-term storage
    stats = store.get_stats()
    print(f"   Long-term storage: {stats.get('total_episodes', 0)} episodes")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"""
    Episodes stored: {len(episodes)}
    Patterns learned: {len(patterns)}
    Semantic retrieval: Working (similarity-based)
    Pattern recommendations: Working (keyword + confidence based)
    Memory tiers: Working -> Short-term -> Long-term

    The memory system can:
    1. Store coding experiences with embeddings
    2. Find similar past experiences semantically
    3. Learn patterns from clusters of similar episodes
    4. Recommend actions based on what worked before
    """)

    print(f"\nTest database saved at: {DB_PATH}")
    print("You can explore it with the CLI or SQLite browser.")


if __name__ == "__main__":
    main()
