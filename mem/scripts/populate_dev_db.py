#!/usr/bin/env python3
"""
Populate development database with sample data.

Usage:
    python scripts/populate_dev_db.py              # Use default paths
    python scripts/populate_dev_db.py --clean      # Clean and recreate
    python scripts/populate_dev_db.py --no-model   # Skip embedding model (faster, no semantic search)
"""

import argparse
import os
import shutil
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "nomic-embed-text-v1.5.Q8_0.gguf"
DB_PATH = PROJECT_ROOT / "data" / "agent_memory.db"
VECTOR_PATH = PROJECT_ROOT / "data" / "memory_vectors"


def get_sample_episodes():
    """
    Sample episodes covering various coding scenarios.
    Returns list of (context, action, outcome, success_score, tags)
    """
    return [
        # Python debugging - successful
        ("Python TypeError: 'NoneType' object is not subscriptable",
         "Added null check before accessing dictionary key",
         "Bug fixed, code handles missing data gracefully",
         0.95, ["python", "debugging", "null-check"]),

        ("Python TypeError when calling function with wrong argument types",
         "Added type hints and isinstance validation",
         "Type errors caught early with clear messages",
         0.85, ["python", "debugging", "types"]),

        ("Python script crashes with TypeError in nested dictionary access",
         "Used dict.get() with default value instead of direct access",
         "No more crashes, returns sensible default",
         0.90, ["python", "debugging", "dict"]),

        ("TypeError: can't concatenate str and int in Python",
         "Converted int to str using f-string formatting",
         "String formatting works correctly",
         0.80, ["python", "debugging", "types"]),

        ("NoneType error in Python list comprehension",
         "Added filter to skip None values in comprehension",
         "List comprehension completes without errors",
         0.88, ["python", "debugging", "null-check"]),

        # Python debugging - failures (for learning)
        ("Python script hangs with no error message",
         "Added print statements to debug",
         "Still couldn't find the issue, took too long",
         0.20, ["python", "debugging", "performance"]),

        ("Python memory error on large dataset",
         "Tried loading entire file into memory",
         "Script crashed, should have used streaming",
         0.15, ["python", "debugging", "memory"]),

        # Git operations
        ("Need to undo last git commit without losing changes",
         "Used git reset --soft HEAD~1",
         "Commit undone, changes preserved in staging",
         0.95, ["git", "undo"]),

        ("Accidentally committed sensitive file to git",
         "Used git filter-branch to remove file from history",
         "Sensitive file removed from all commits",
         0.70, ["git", "security"]),

        ("Git merge conflict in package-lock.json",
         "Deleted file, ran npm install, committed fresh lockfile",
         "Clean merge, dependencies resolved correctly",
         0.90, ["git", "merge", "npm"]),

        ("Need to squash last 3 git commits into one",
         "Used git rebase -i HEAD~3 with squash option",
         "Commits combined into single clean commit",
         0.85, ["git", "rebase"]),

        ("Git branch diverged significantly from main",
         "Rebased onto main incrementally, resolving conflicts step by step",
         "Branch updated with main changes, history clean",
         0.75, ["git", "rebase"]),

        # Docker
        ("Docker container won't start, port already in use",
         "Changed host port mapping in docker-compose.yml",
         "Container starts successfully on new port",
         0.95, ["docker", "networking"]),

        ("Docker build failing due to missing dependencies",
         "Added missing apt packages to Dockerfile",
         "Build completes successfully",
         0.90, ["docker", "build"]),

        ("Docker container running out of disk space",
         "Added docker system prune to cleanup script",
         "Freed up space, container stable",
         0.85, ["docker", "maintenance"]),

        ("Docker network issues between containers",
         "Created custom bridge network for container communication",
         "Containers can now communicate via network",
         0.88, ["docker", "networking"]),

        # API/HTTP
        ("REST API returning 500 error on POST request",
         "Added request body validation and proper error handling",
         "API returns 400 for invalid input, 200 for valid",
         0.90, ["api", "debugging", "validation"]),

        ("API timeout when processing large file uploads",
         "Implemented chunked upload with background processing",
         "Large files handled without timeout",
         0.80, ["api", "performance"]),

        ("CORS error when calling API from frontend",
         "Added CORS headers to API middleware",
         "Frontend can now call API successfully",
         0.95, ["api", "cors", "frontend"]),

        # Database
        ("SQL query running extremely slow on large table",
         "Added composite index on frequently queried columns",
         "Query time reduced from 30s to 0.5s",
         0.95, ["database", "performance", "sql"]),

        ("Database connection pool exhausted under load",
         "Increased pool size and added connection timeout",
         "No more connection errors under heavy load",
         0.85, ["database", "performance"]),

        ("SQLite database locked during concurrent writes",
         "Implemented write queue with single writer pattern",
         "No more locking errors, writes serialized properly",
         0.90, ["database", "sqlite", "concurrency"]),

        # Testing
        ("Unit tests failing intermittently",
         "Added proper test isolation and mocking",
         "Tests now pass consistently",
         0.85, ["testing", "debugging"]),

        ("Integration tests too slow",
         "Used test database with smaller dataset",
         "Test suite runs 5x faster",
         0.80, ["testing", "performance"]),
    ]


def main():
    parser = argparse.ArgumentParser(description="Populate development database")
    parser.add_argument("--clean", action="store_true", help="Remove existing database and start fresh")
    parser.add_argument("--no-model", action="store_true", help="Skip embedding model (no semantic search)")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    parser.add_argument("--vectors", type=str, default=str(VECTOR_PATH), help="Vector store path")
    args = parser.parse_args()

    db_path = Path(args.db)
    vector_path = Path(args.vectors)

    print("=" * 60)
    print("Populating Development Database")
    print("=" * 60)

    # Clean if requested
    if args.clean:
        print("\nCleaning existing data...")
        if db_path.exists():
            os.remove(db_path)
            print(f"  Removed {db_path}")
        if vector_path.exists():
            shutil.rmtree(vector_path)
            print(f"  Removed {vector_path}")

    # Initialize
    print("\nInitializing memory store...")

    from agent_memory import MemoryStore, EmbeddingGenerator, DomainLearner

    embedding_gen = None
    if not args.no_model:
        if MODEL_PATH.exists():
            print(f"  Loading embedding model: {MODEL_PATH.name}")
            embedding_gen = EmbeddingGenerator(str(MODEL_PATH))
        else:
            print(f"  Warning: Model not found at {MODEL_PATH}")
            print(f"  Run: python scripts/setup_embedding_model.py")
            print(f"  Continuing without embeddings...")

    store = MemoryStore(
        db_path=str(db_path),
        vector_store_path=str(vector_path),
        embedding_generator=embedding_gen,
    )

    print(f"  Database: {db_path}")
    print(f"  Vectors: {vector_path}")

    # Store episodes (requires embedding model)
    episodes = get_sample_episodes()
    if embedding_gen:
        print(f"\nStoring {len(episodes)} episodes...")

        for i, (context, action, outcome, score, tags) in enumerate(episodes):
            store.store_episode(
                context=context,
                action=action,
                outcome=outcome,
                success_score=score,
                tags=tags,
            )
            if (i + 1) % 5 == 0:
                print(f"  Stored {i + 1}/{len(episodes)} episodes")

        print(f"  Done: {len(episodes)} episodes stored")

        # Run consolidation
        print("\nRunning consolidation...")
        report = store.run_consolidation()
        print(f"  Clusters found: {report.clusters_found}")
        print(f"  Patterns created: {report.patterns_created}")
    else:
        print("\nSkipping episodes and consolidation (no embedding model)")
        print("  Episodes require embeddings for semantic search")
        print("  Run without --no-model to populate episodes")

    # Seed domain keywords (learnable markers)
    print("\nSeeding domain keywords...")
    learner = DomainLearner(store)
    num_keywords = learner.seed_default_domains()
    print(f"  Seeded {num_keywords} keywords across {len(store.get_all_domains())} domains")

    # Store problem types (Phase 5)
    print("\nStoring problem types...")
    problem_types = [
        ("python_debugging", "Debugging Python runtime errors",
         ["TypeError", "ValueError", "exception handling", "stack trace"]),
        ("git_workflow", "Git version control operations",
         ["commit", "merge", "rebase", "branch management"]),
        ("docker_operations", "Docker container management",
         ["container", "image", "port mapping", "networking"]),
        ("api_development", "REST API development and debugging",
         ["HTTP", "endpoints", "request/response", "CORS"]),
        ("database_optimization", "Database performance and queries",
         ["SQL", "indexing", "query optimization", "connection pooling"]),
        ("testing_strategies", "Software testing approaches",
         ["unit tests", "integration tests", "mocking", "fixtures"]),
    ]

    problem_type_ids = {}
    for name, description, characteristics in problem_types:
        type_id = store.store_problem_type(
            name=name,
            description=description,
            characteristics=characteristics,
        )
        problem_type_ids[name] = type_id
        print(f"  Created problem type: {name} (id={type_id})")

    # Link similar problem types
    print("\nLinking similar problem types...")
    links = [
        ("python_debugging", "testing_strategies"),  # Both involve finding bugs
        ("docker_operations", "api_development"),     # Both involve networking
        ("python_debugging", "api_development"),      # Both involve error handling
    ]
    for type1, type2 in links:
        store.link_similar_problem_types(
            problem_type_ids[type1],
            problem_type_ids[type2]
        )
        print(f"  Linked: {type1} <-> {type2}")

    # Store sample adaptations
    print("\nStoring sample adaptations...")
    adaptations = [
        {
            "source_context": "Python TypeError: 'NoneType' object is not subscriptable",
            "target_context": "Docker container fails with null environment variable",
            "original_strategy": "Added null check before accessing dictionary key",
            "adapted_strategy": "Added default values for environment variables in docker-compose.yml",
            "reasoning": "Both cases involve handling missing/null values defensively",
            "success_score": 0.85,
            "source_type": "python_debugging",
            "target_type": "docker_operations",
        },
        {
            "source_context": "Git merge conflict in package-lock.json",
            "target_context": "Docker compose conflict in multi-service setup",
            "original_strategy": "Deleted file, ran npm install, committed fresh lockfile",
            "adapted_strategy": "Deleted conflicting compose file, rebuilt from individual service configs",
            "reasoning": "Both involve regenerating auto-managed configuration files",
            "success_score": 0.80,
            "source_type": "git_workflow",
            "target_type": "docker_operations",
        },
        {
            "source_context": "SQL query running extremely slow on large table",
            "target_context": "API endpoint timing out on data retrieval",
            "original_strategy": "Added composite index on frequently queried columns",
            "adapted_strategy": "Added pagination and caching to the API response",
            "reasoning": "Both address performance issues with large data access",
            "success_score": 0.90,
            "source_type": "database_optimization",
            "target_type": "api_development",
        },
    ]

    for adapt in adaptations:
        store.store_adaptation(
            source_context=adapt["source_context"],
            target_context=adapt["target_context"],
            original_strategy=adapt["original_strategy"],
            adapted_strategy=adapt["adapted_strategy"],
            adaptation_reasoning=adapt["reasoning"],
            success_score=adapt["success_score"],
            source_problem_type_id=problem_type_ids.get(adapt["source_type"]),
            target_problem_type_id=problem_type_ids.get(adapt["target_type"]),
        )
    print(f"  Stored {len(adaptations)} sample adaptations")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    stats = store.get_stats()
    print(f"  Total episodes: {stats.get('total_episodes', 'N/A')}")
    print(f"  Learned patterns: {len(store.get_learned_patterns())}")
    print(f"  Reflections: {store.count_reflections()}")
    print(f"  Problem types: {len(store.get_all_problem_types())}")
    print(f"  Adaptations: {len(store.get_adaptations())}")

    avg_score = stats.get('avg_success_score')
    if avg_score:
        print(f"  Avg success score: {avg_score:.2f}")

    store.close()

    print("\nDone! Use the CLI to explore:")
    print(f"  python -m agent_memory.memory_cli stats")
    print(f"  python -m agent_memory.memory_cli search 'python error'")
    print(f"  python -m agent_memory.memory_cli patterns")


if __name__ == "__main__":
    main()
