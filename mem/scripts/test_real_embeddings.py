#!/usr/bin/env python3
"""
Test the memory system with real embeddings.
Creates sample episodes, runs consolidation, and tests pattern-based retrieval.
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import src as a package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "nomic-embed-text-v1.5.Q8_0.gguf")
DB_PATH = str(Path(__file__).parent.parent / "data" / "test_real_memory.db")
VECTOR_PATH = str(Path(__file__).parent.parent / "data" / "test_real_vectors")


def create_test_episodes():
    """Sample episodes from various coding scenarios"""
    return [
        # Python debugging cluster
        ("Debugging Python TypeError: 'NoneType' object is not subscriptable",
         "Added null check before accessing dictionary key",
         "Fixed the bug, code now handles missing data gracefully",
         0.9, ["python", "debugging", "null-check"]),

        ("Python TypeError when calling function with wrong argument types",
         "Added type hints and isinstance check",
         "Type error caught early, better error messages",
         0.85, ["python", "debugging", "types"]),

        ("Python script crashes with TypeError in nested dictionary access",
         "Used dict.get() with default value instead of direct access",
         "No more crashes, returns default when key missing",
         0.95, ["python", "debugging", "dict"]),

        ("TypeError in Python: can't concatenate str and int",
         "Converted int to str using str() before concatenation",
         "String formatting works correctly now",
         0.8, ["python", "debugging", "types"]),

        ("Debugging NoneType error in Python list comprehension",
         "Added filter to skip None values in comprehension",
         "List comprehension completes without errors",
         0.88, ["python", "debugging", "null-check"]),

        # Git workflow cluster
        ("Need to undo last git commit without losing changes",
         "Used git reset --soft HEAD~1 to keep changes staged",
         "Commit undone, changes preserved in staging",
         0.95, ["git", "version-control"]),

        ("Accidentally committed sensitive file to git",
         "Used git filter-branch to remove file from history",
         "Sensitive file removed from all commits",
         0.7, ["git", "security"]),

        ("Git merge conflict in package-lock.json",
         "Deleted file, ran npm install, committed result",
         "Clean merge, dependencies resolved correctly",
         0.9, ["git", "npm", "merge"]),

        ("Need to squash last 3 git commits into one",
         "Used git rebase -i HEAD~3 with squash option",
         "Commits combined into single clean commit",
         0.85, ["git", "version-control"]),

        ("Git branch diverged from main, too many conflicts",
         "Rebased onto main incrementally, resolving conflicts step by step",
         "Branch successfully updated with main changes",
         0.75, ["git", "rebase"]),

        # Docker cluster
        ("Docker container won't start, port already in use",
         "Changed host port mapping in docker-compose.yml",
         "Container starts on new port",
         0.95, ["docker", "networking"]),

        ("Docker build failing due to missing dependencies",
         "Added missing apt packages to Dockerfile",
         "Build completes successfully",
         0.9, ["docker", "build"]),

        ("Docker container running out of disk space",
         "Added docker system prune to cleanup script",
         "Freed up disk space, container stable",
         0.85, ["docker", "maintenance"]),

        ("Docker network issues between containers",
         "Created custom bridge network for containers",
         "Containers can communicate via network",
         0.88, ["docker", "networking"]),

        # API/HTTP cluster
        ("REST API returning 500 error on POST request",
         "Added request body validation and error handling",
         "API returns proper 400 for invalid input, 200 for valid",
         0.9, ["api", "debugging", "validation"]),

        ("API timeout when processing large files",
         "Implemented chunked upload and background processing",
         "Large files handled without timeout",
         0.8, ["api", "performance"]),

        ("CORS error when calling API from frontend",
         "Added CORS headers to API response",
         "Frontend can now call API successfully",
         0.95, ["api", "cors", "frontend"]),

        # Database cluster
        ("SQL query running slow on large table",
         "Added index on frequently queried columns",
         "Query time reduced from 30s to 0.5s",
         0.95, ["database", "performance", "sql"]),

        ("Database connection pool exhausted",
         "Increased pool size and added connection timeout",
         "No more connection errors under load",
         0.85, ["database", "performance"]),

        ("SQLite database locked during concurrent writes",
         "Implemented write queue with single writer",
         "No more locking errors, writes serialized",
         0.9, ["database", "sqlite", "concurrency"]),
    ]


def main():
    import shutil

    # Clean up previous test data
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(VECTOR_PATH):
        shutil.rmtree(VECTOR_PATH)

    print("=" * 60)
    print("Testing Memory System with Real Embeddings")
    print("=" * 60)

    # Initialize embedding generator
    print(f"\n1. Initializing memory store...")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Database: {DB_PATH}")

    from agent_memory.embeddings import EmbeddingGenerator
    from agent_memory.memory_store import MemoryStore

    print("   Loading embedding model...")
    embedding_gen = EmbeddingGenerator(MODEL_PATH)

    store = MemoryStore(
        db_path=DB_PATH,
        vector_store_path=VECTOR_PATH,
        embedding_generator=embedding_gen
    )

    # Store episodes
    print(f"\n2. Storing {len(create_test_episodes())} episodes...")
    episodes = create_test_episodes()
    episode_ids = []

    for i, (context, action, outcome, score, tags) in enumerate(episodes):
        ep_id = store.store_episode(
            context=context,
            action=action,
            outcome=outcome,
            success_score=score,
            tags=tags
        )
        episode_ids.append(ep_id)
        print(f"   Stored episode {ep_id}: {context[:50]}...")

    print(f"\n   Total episodes stored: {len(episode_ids)}")

    # Test semantic retrieval
    print("\n3. Testing semantic retrieval...")

    test_queries = [
        "How to fix Python type errors?",
        "Git commit history problems",
        "Docker container networking issues",
        "Database performance optimization"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = store.retrieve_episodes(query, limit=3)
        for ep, similarity in results:
            print(f"     [{similarity:.3f}] {ep.context[:60]}...")

    # Run consolidation
    print("\n4. Running consolidation...")
    report = store.run_consolidation()
    print(f"   Episodes processed: {report.episodes_processed}")
    print(f"   Clusters found: {report.clusters_found}")
    print(f"   Patterns created: {report.patterns_created}")
    print(f"   Duration: {report.duration_seconds:.2f}s")

    # Show extracted patterns
    print("\n5. Extracted patterns:")
    patterns = store.get_learned_patterns(limit=10)
    for pattern in patterns:
        print(f"\n   Pattern #{pattern.pattern_id}:")
        print(f"     Context: {pattern.context_signature}")
        print(f"     Recommended action: {pattern.recommended_action[:60]}...")
        print(f"     Success rate: {pattern.success_rate:.0%}")
        print(f"     Sample count: {pattern.sample_count}")
        print(f"     Confidence: {pattern.confidence:.2f}")

    # Test pattern-based recommendations
    print("\n6. Testing pattern-based recommendations...")

    recommendation_queries = [
        "I have a Python TypeError",
        "Docker container won't connect to database",
        "Git merge has conflicts"
    ]

    for query in recommendation_queries:
        print(f"\n   Query: '{query}'")
        recommendations = store.recommend_actions(query, limit=2)
        if recommendations:
            for rec in recommendations:
                pattern = rec['pattern']
                print(f"     → {pattern.recommended_action[:60]}...")
                print(f"       (success: {pattern.success_rate:.0%}, confidence: {pattern.confidence:.2f}, match: {rec['match_score']:.2f})")
        else:
            print("     No recommendations found")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    stats = store.get_stats()
    print(f"Total episodes: {stats.get('total_episodes', 'N/A')}")
    print(f"Total patterns: {stats.get('total_patterns', 'N/A')}")
    avg_score = stats.get('avg_success_score')
    print(f"Average success score: {avg_score:.2f}" if avg_score else "Average success score: N/A")

    print("\nTest complete!")
    print(f"\nTo explore further, use the CLI:")
    print(f"  EMBEDDING_MODEL_PATH={MODEL_PATH} python -m src.memory_cli --db {DB_PATH} --vector {VECTOR_PATH} stats")
    print(f"  EMBEDDING_MODEL_PATH={MODEL_PATH} python -m src.memory_cli --db {DB_PATH} --vector {VECTOR_PATH} patterns")


if __name__ == "__main__":
    main()
