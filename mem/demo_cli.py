"""
Demo script for Memory CLI - Uses mock embeddings for easy testing

This script demonstrates the CLI without requiring an actual embedding model.
It creates a mock memory store with sample data, then you can run CLI commands.

Usage:
    python demo_cli.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memory_store import MemoryStore
from rich.console import Console

console = Console()


class MockEmbeddingGenerator:
    """Mock embedding generator for demos (no model needed)"""

    def __init__(self):
        self.dimension = 384

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate deterministic 'embedding' based on text hash"""
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(self.dimension).astype(np.float32)
        return emb / np.linalg.norm(emb)


def create_demo_store() -> MemoryStore:
    """Create a memory store with sample data for demos"""
    console.print("[cyan]Creating demo memory store with sample data...[/cyan]\n")

    store = MemoryStore(embedding_generator=MockEmbeddingGenerator())

    # Add sample episodes
    sample_episodes = [
        {
            "context": "User asked how to implement a binary search algorithm",
            "action": "Provided Python code with detailed comments explaining the algorithm",
            "outcome": "User understood and successfully implemented it",
            "success_score": 0.9,
            "tags": ["coding", "algorithms", "python"],
        },
        {
            "context": "User reported a bug in the authentication system",
            "action": "Analyzed code and identified missing null check",
            "outcome": "Bug was fixed and tests passed",
            "success_score": 1.0,
            "tags": ["debugging", "authentication", "bugfix"],
        },
        {
            "context": "User needed help with database schema design",
            "action": "Suggested normalized schema with foreign keys",
            "outcome": "User implemented but had performance issues",
            "success_score": 0.5,
            "tags": ["database", "schema", "design"],
        },
        {
            "context": "User asked about React component optimization",
            "action": "Recommended using React.memo and useMemo hooks",
            "outcome": "Performance improved significantly",
            "success_score": 0.95,
            "tags": ["react", "performance", "optimization"],
        },
        {
            "context": "User wanted to learn about Docker basics",
            "action": "Explained containers, images, and provided a simple Dockerfile",
            "outcome": "User successfully containerized their application",
            "success_score": 0.85,
            "tags": ["docker", "containers", "devops"],
        },
        {
            "context": "Advanced Python metaclasses question",
            "action": "Explained metaclass usage with examples",
            "outcome": "User still confused about practical applications",
            "success_score": 0.6,
            "tags": ["python", "advanced", "teaching"],
        },
        {
            "context": "Teaching SQL joins to beginner",
            "action": "Used visual diagrams to explain different join types",
            "outcome": "User understood and could write join queries",
            "success_score": 0.85,
            "tags": ["sql", "beginner", "teaching"],
        },
    ]

    for episode in sample_episodes:
        store.store_episode(**episode)

    console.print(f"[green]✓ Created demo store with {len(sample_episodes)} episodes[/green]\n")

    return store


def interactive_demo():
    """Run interactive CLI demo"""
    store = create_demo_store()

    console.print("[bold cyan]Demo Memory CLI[/bold cyan]")
    console.print("The demo store is ready. Try these commands:\n")

    commands = [
        "stats                          - Show memory statistics",
        "search 'Python' --limit 3      - Search for Python-related episodes",
        "recent --hours 24              - Show recent episodes",
        "show 1                         - Show episode #1 in detail",
        "tags                           - Show tag usage statistics",
        "export demo_episodes.json      - Export all episodes to JSON",
        "",
        "Type 'exit' or 'quit' to stop",
    ]

    for cmd in commands:
        console.print(f"  [dim]{cmd}[/dim]")

    console.print()

    # Import CLI functions
    from memory_cli import (
        command_stats,
        command_search,
        command_recent,
        command_show,
        command_export,
        command_tags,
    )
    import argparse

    command_map = {
        'stats': command_stats,
        'search': command_search,
        'recent': command_recent,
        'show': command_show,
        'export': command_export,
        'tags': command_tags,
    }

    while True:
        try:
            user_input = input("\n[demo-cli] > ").strip()

            if not user_input or user_input in ['exit', 'quit']:
                console.print("\n[cyan]Goodbye![/cyan]")
                break

            # Parse command
            parts = user_input.split()
            cmd = parts[0]

            if cmd not in command_map:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                continue

            # Build args namespace
            args = argparse.Namespace()

            if cmd == 'stats':
                pass  # No args needed

            elif cmd == 'search':
                if len(parts) < 2:
                    console.print("[red]Usage: search <query> [--limit N][/red]")
                    continue
                # Handle quoted queries
                if user_input.count("'") >= 2 or user_input.count('"') >= 2:
                    # Extract quoted query
                    quote_char = "'" if "'" in user_input else '"'
                    start = user_input.index(quote_char) + 1
                    end = user_input.index(quote_char, start)
                    args.query = user_input[start:end]
                else:
                    args.query = parts[1]

                args.limit = 5
                if '--limit' in parts:
                    idx = parts.index('--limit')
                    if idx + 1 < len(parts):
                        args.limit = int(parts[idx + 1])

            elif cmd == 'recent':
                args.hours = 24
                args.limit = 10
                if '--hours' in parts:
                    idx = parts.index('--hours')
                    if idx + 1 < len(parts):
                        args.hours = int(parts[idx + 1])
                if '--limit' in parts:
                    idx = parts.index('--limit')
                    if idx + 1 < len(parts):
                        args.limit = int(parts[idx + 1])

            elif cmd == 'show':
                if len(parts) < 2:
                    console.print("[red]Usage: show <episode_id>[/red]")
                    continue
                args.episode_id = int(parts[1])

            elif cmd == 'export':
                if len(parts) < 2:
                    console.print("[red]Usage: export <filename>[/red]")
                    continue
                args.output = parts[1]

            elif cmd == 'tags':
                pass  # No args needed

            # Execute command
            command_map[cmd](args, store)

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    store.close()


if __name__ == "__main__":
    interactive_demo()
