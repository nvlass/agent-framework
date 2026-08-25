"""
Example usage of the Agent Memory System - Phase 1

This demonstrates basic episodic memory storage and retrieval.

Before running this, you'll need a GGUF embedding model.
Download one from HuggingFace, for example:
- all-MiniLM-L6-v2 (small, fast)
- bge-small-en (good quality/speed tradeoff)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import EmbeddingGenerator
from memory_store import MemoryStore
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    console.print("\n[bold cyan]Agent Memory System - Phase 1 Demo[/bold cyan]\n")

    # NOTE: You need to provide a path to a GGUF embedding model
    # For testing without a model, set use_mock=True
    use_mock = True

    if use_mock:
        console.print("[yellow]Using mock embeddings (no model loaded)[/yellow]\n")
        # Create mock generator for testing
        import numpy as np

        class MockGenerator:
            def generate_embedding(self, text, use_cache=True):
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(384).astype(np.float32)
                return emb / np.linalg.norm(emb)

        embedding_gen = MockGenerator()
    else:
        # For real usage, specify your model path
        model_path = "path/to/your/model.gguf"
        console.print(f"[green]Loading model: {model_path}[/green]")
        embedding_gen = EmbeddingGenerator(model_path=model_path)

    # Initialize memory store
    memory_store = MemoryStore(embedding_generator=embedding_gen)

    console.print("[bold green]Step 1: Storing sample episodes[/bold green]\n")

    # Store some example episodes
    episodes_data = [
        {
            "context": "User asked how to implement a binary search algorithm",
            "action": "Provided Python code with detailed comments",
            "outcome": "User understood and implemented successfully",
            "success_score": 0.9,
            "tags": ["coding", "algorithms", "python"],
        },
        {
            "context": "User reported authentication bug",
            "action": "Identified missing null check in login handler",
            "outcome": "Bug fixed, tests passed",
            "success_score": 1.0,
            "tags": ["debugging", "authentication"],
        },
        {
            "context": "User needed React component optimization",
            "action": "Recommended React.memo and useMemo",
            "outcome": "Performance improved by 40%",
            "success_score": 0.95,
            "tags": ["react", "performance"],
        },
        {
            "context": "User confused about Docker networking",
            "action": "Explained bridge networks and port mapping",
            "outcome": "User still had issues with DNS resolution",
            "success_score": 0.6,
            "tags": ["docker", "networking"],
        },
    ]

    for i, episode_data in enumerate(episodes_data, 1):
        episode_id = memory_store.store_episode(**episode_data)
        console.print(f"  [dim]Stored episode {episode_id}: {episode_data['context'][:60]}...[/dim]")

    console.print(f"\n[green]✓ Stored {len(episodes_data)} episodes[/green]\n")

    # Show statistics
    console.print("[bold green]Step 2: Memory statistics[/bold green]\n")
    stats = memory_store.get_stats()

    stats_table = Table(show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    for key, value in stats.items():
        stats_table.add_row(key.replace("_", " ").title(), str(value))

    console.print(stats_table)
    console.print()

    # Semantic search
    console.print("[bold green]Step 3: Semantic search[/bold green]\n")

    queries = [
        "How do I optimize React performance?",
        "Help with debugging authentication issues",
        "Docker networking problems",
    ]

    for query in queries:
        console.print(f"[cyan]Query:[/cyan] {query}")
        results = memory_store.retrieve_episodes(query, limit=2)

        if results:
            for i, (episode, similarity) in enumerate(results, 1):
                console.print(f"  [dim]{i}. (similarity: {similarity:.2f})[/dim] {episode.context[:70]}...")
        else:
            console.print("  [dim]No results found[/dim]")

        console.print()

    # Recent episodes
    console.print("[bold green]Step 4: Recent episodes[/bold green]\n")

    recent = memory_store.get_recent_episodes(hours=24, limit=5)
    console.print(f"Found {len(recent)} episodes in last 24 hours:\n")

    recent_table = Table()
    recent_table.add_column("ID", style="cyan")
    recent_table.add_column("Context", style="white")
    recent_table.add_column("Success", style="green")
    recent_table.add_column("Tags", style="yellow")

    for episode in recent:
        recent_table.add_row(
            str(episode.id),
            episode.context[:50] + "..." if len(episode.context) > 50 else episode.context,
            f"{episode.success_score:.2f}" if episode.success_score else "N/A",
            ", ".join(episode.tags[:3]),
        )

    console.print(recent_table)

    # Cleanup
    memory_store.close()

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print(f"[dim]Database stored at: {memory_store.db_path}[/dim]\n")


if __name__ == "__main__":
    main()
