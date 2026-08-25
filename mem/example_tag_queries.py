"""
Demonstration of SQLite JSON tag querying capabilities

This shows how we can efficiently query episodes by tags using
SQLite's native JSON functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from memory_store import MemoryStore
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main():
    console.print("\n[bold cyan]SQLite JSON Tag Querying Demo[/bold cyan]\n")

    # Create mock generator
    import numpy as np

    class MockGenerator:
        def generate_embedding(self, text, use_cache=True):
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(384).astype(np.float32)
            return emb / np.linalg.norm(emb)

    # Initialize store
    store = MemoryStore(embedding_generator=MockGenerator())

    # Add diverse episodes with various tags
    console.print("[dim]Adding sample episodes...[/dim]\n")

    episodes = [
        {
            "context": "User learning Python basics",
            "action": "Taught variables and data types",
            "outcome": "User understood concepts",
            "success_score": 0.9,
            "tags": ["python", "beginner", "teaching"],
        },
        {
            "context": "Advanced Python metaclasses question",
            "action": "Explained metaclass usage with examples",
            "outcome": "User still confused",
            "success_score": 0.6,
            "tags": ["python", "advanced", "teaching"],
        },
        {
            "context": "React component rendering issue",
            "action": "Fixed infinite render loop",
            "outcome": "Bug resolved",
            "success_score": 1.0,
            "tags": ["react", "debugging", "bugfix"],
        },
        {
            "context": "Docker container won't start",
            "action": "Fixed port binding conflict",
            "outcome": "Container running",
            "success_score": 0.95,
            "tags": ["docker", "debugging", "devops"],
        },
        {
            "context": "Teaching SQL joins to beginner",
            "action": "Used visual diagrams to explain",
            "outcome": "User understood",
            "success_score": 0.85,
            "tags": ["sql", "beginner", "teaching"],
        },
    ]

    for ep in episodes:
        store.store_episode(**ep)

    console.print(f"[green]✓ Added {len(episodes)} episodes[/green]\n")

    # Demo 1: Find episodes by single tag
    console.print("[bold]1. Find all Python-related episodes[/bold]")
    python_eps = store.get_episodes_by_tag("python")
    console.print(f"   Found {len(python_eps)} episodes with tag 'python'\n")

    # Demo 2: Find episodes for beginners
    console.print("[bold]2. Find all beginner-friendly content[/bold]")
    beginner_eps = store.get_episodes_by_tag("beginner")
    for ep in beginner_eps:
        console.print(f"   - {ep.context[:60]}... (tags: {', '.join(ep.tags)})")
    console.print()

    # Demo 3: Find debugging episodes
    console.print("[bold]3. Find all debugging episodes[/bold]")
    debug_eps = store.get_episodes_by_tag("debugging")
    console.print(f"   Found {len(debug_eps)} debugging episodes\n")

    # Demo 4: Multiple tags (ANY match)
    console.print("[bold]4. Find episodes about Python OR React[/bold]")
    lang_eps = store.get_episodes_by_tags(["python", "react"], match_all=False)
    console.print(f"   Found {len(lang_eps)} episodes\n")

    # Demo 5: Multiple tags (ALL must match)
    console.print("[bold]5. Find episodes that are BOTH Python AND beginner[/bold]")
    py_beginner = store.get_episodes_by_tags(["python", "beginner"], match_all=True)
    console.print(f"   Found {len(py_beginner)} episodes")
    for ep in py_beginner:
        console.print(f"   - {ep.context}")
    console.print()

    # Demo 6: Get all tags with counts
    console.print("[bold]6. Show all tags with usage counts[/bold]\n")

    tags = store.get_all_tags()

    table = Table(title="Tag Usage Statistics")
    table.add_column("Tag", style="cyan")
    table.add_column("Count", style="green", justify="right")

    for tag, count in tags:
        table.add_row(tag, str(count))

    console.print(table)
    console.print()

    # Demo 7: SQL behind the scenes
    console.print("[bold]7. The SQL behind tag querying[/bold]\n")

    sql_examples = """
[cyan]Single tag query:[/cyan]
SELECT * FROM episodes
WHERE EXISTS (
    SELECT 1 FROM json_each(episodes.tags)
    WHERE json_each.value = 'python'
)

[cyan]Multiple tags (ANY):[/cyan]
SELECT * FROM episodes
WHERE EXISTS (
    SELECT 1 FROM json_each(episodes.tags)
    WHERE json_each.value IN ('python', 'react')
)

[cyan]All tags with counts:[/cyan]
SELECT json_each.value as tag, COUNT(*) as count
FROM episodes, json_each(episodes.tags)
GROUP BY tag
ORDER BY count DESC
    """

    panel = Panel(sql_examples, title="SQLite JSON Queries", border_style="dim")
    console.print(panel)

    # Cleanup
    store.close()

    console.print("\n[bold green]✓ Demo complete![/bold green]")
    console.print("[dim]All queries use SQLite's native JSON functions for efficiency[/dim]\n")


if __name__ == "__main__":
    main()
