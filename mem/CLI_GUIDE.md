# Memory CLI Implementation Guide

This guide will help you implement the Memory CLI Explorer step by step.

## Architecture Note

The CLI uses **pure dependency injection** - all command functions receive a `MemoryStore` instance as a parameter. This makes the code:
- ✅ Fully testable (no mocking needed)
- ✅ No global state
- ✅ No boolean flags like `use_mock`
- ✅ Clean separation of concerns

Production code reads configuration and creates the store. Tests create test stores and pass them directly.

## Getting Started

You'll be implementing 6 functions in `src/memory_cli.py`:

1. `setup_parser()` - Set up command-line argument parsing
2. `command_stats()` - Display memory statistics
3. `command_search()` - Search for episodes
4. `command_recent()` - Show recent episodes
5. `command_show()` - Display detailed episode view
6. `command_export()` - Export episodes to JSON

## Implementation Order (Recommended)

### Step 1: `setup_parser()` - The Foundation

This creates the argument parser with all subcommands. Start here because you need this to run any commands.

**What you need to do:**
```python
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent Memory Explorer")
    subparsers = parser.add_subparsers(dest='command')

    # Add stats subcommand
    stats_parser = subparsers.add_parser('stats', help='Show memory statistics')

    # Add search subcommand
    search_parser = subparsers.add_parser('search', help='Search for episodes')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--limit', type=int, default=5, help='Max results')

    # Add remaining subcommands (recent, show, export)...

    return parser
```

**Test it:**
```bash
python src/memory_cli.py --help
python src/memory_cli.py stats --help
python src/memory_cli.py search --help
```

---

### Step 2: `command_stats()` - Easiest Command

Show memory statistics using `store.get_stats()`.

**What you need to do:**
```python
def command_stats(args, store):
    # Get stats dictionary
    stats = store.get_stats()

    # Create a nice table
    table = Table(title="Memory Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add rows
    for key, value in stats.items():
        # Format the key nicely (replace _ with space, title case)
        metric_name = key.replace('_', ' ').title()
        table.add_row(metric_name, str(value))

    console.print(table)
```

**Test it:**
```bash
# First add some data using example_usage.py
python example_usage.py

# Then run stats
python src/memory_cli.py stats
```

**Expected output:**
```
Memory Statistics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric              ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Episodes      │ 4     │
│ ...                 │ ...   │
└─────────────────────┴───────┘
```

---

### Step 3: `command_show()` - Single Episode Detail

Display detailed view of one episode.

**What you need to do:**
```python
def command_show(args, store):
    episode = store.get_episode_by_id(args.episode_id)

    if episode is None:
        console.print(f"[red]Episode {args.episode_id} not found[/red]")
        return

    # Format the episode details
    details = f"""
[cyan]Episode ID:[/cyan] {episode.id}
[cyan]Timestamp:[/cyan] {episode.timestamp}
[cyan]Context:[/cyan] {episode.context}
[cyan]Action:[/cyan] {episode.action}
[cyan]Outcome:[/cyan] {episode.outcome}
[cyan]Success Score:[/cyan] {episode.success_score}
[cyan]Tags:[/cyan] {', '.join(episode.tags)}
    """

    panel = Panel(details, title=f"Episode {episode.id}", border_style="blue")
    console.print(panel)
```

**Test it:**
```bash
python src/memory_cli.py show 1
python src/memory_cli.py show 9999  # Should show error
```

---

### Step 4: `command_recent()` - Time-based Query

Show recent episodes within a time window.

**What you need to do:**
```python
def command_recent(args, store):
    episodes = store.get_recent_episodes(hours=args.hours, limit=args.limit)

    if not episodes:
        console.print("[yellow]No episodes found in time window[/yellow]")
        return

    console.print(f"\nFound {len(episodes)} episodes in last {args.hours} hours:\n")

    # Create table
    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Time", style="dim")
    table.add_column("Context", style="white")
    table.add_column("Score", style="green")

    for episode in episodes:
        # Format timestamp nicely (e.g., "2 hours ago")
        time_str = episode.timestamp.strftime("%Y-%m-%d %H:%M")

        # Truncate context if too long
        context = episode.context[:60] + "..." if len(episode.context) > 60 else episode.context

        score = f"{episode.success_score:.2f}" if episode.success_score else "N/A"

        table.add_row(str(episode.id), time_str, context, score)

    console.print(table)
```

**Test it:**
```bash
python src/memory_cli.py recent --hours 24 --limit 10
python src/memory_cli.py recent  # Uses defaults
```

---

### Step 5: `command_search()` - Semantic Search

Search for episodes using semantic similarity.

**What you need to do:**
```python
def command_search(args, store):
    console.print(f"[cyan]Searching for:[/cyan] {args.query}\n")

    results = store.retrieve_episodes(args.query, limit=args.limit)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"Found {len(results)} results:\n")

    # Display results
    for i, (episode, similarity) in enumerate(results, 1):
        console.print(f"[bold]{i}. [dim](similarity: {similarity:.3f})[/dim][/bold]")
        console.print(f"   [cyan]Context:[/cyan] {episode.context[:80]}...")
        console.print(f"   [cyan]Action:[/cyan] {episode.action[:80]}...")
        console.print(f"   [cyan]Tags:[/cyan] {', '.join(episode.tags)}")
        console.print()
```

**Test it:**
```bash
python src/memory_cli.py search "Python programming" --limit 5
python src/memory_cli.py search "React optimization"
```

---

### Step 6: `command_export()` - JSON Export

Export all episodes to a JSON file.

**What you need to do:**
```python
def command_export(args, store):
    episodes = store.get_all_episodes()

    # Convert to JSON-serializable format
    data = [episode.to_dict() for episode in episodes]

    # Write to file
    try:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)  # default=str handles datetime

        console.print(f"[green]✓ Exported {len(data)} episodes to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing file: {e}[/red]")
```

**Test it:**
```bash
python src/memory_cli.py export episodes.json
cat episodes.json | head -20
```

---

## Testing Your Implementation

### Manual Testing
```bash
# 1. Generate sample data
python example_usage.py

# 2. Test each command
python src/memory_cli.py stats
python src/memory_cli.py show 1
python src/memory_cli.py recent --hours 24
python src/memory_cli.py search "Python" --limit 3
python src/memory_cli.py export output.json
```

### Automated Testing
```bash
python -m pytest tests/test_cli.py -v
```

---

## Hints & Tips

### Working with argparse
```python
# Positional argument
parser.add_argument('name', type=str, help='Description')

# Optional argument with default
parser.add_argument('--limit', type=int, default=5, help='Max results')

# Flag (boolean)
parser.add_argument('--verbose', action='store_true', help='Verbose output')
```

### Working with Rich Tables
```python
from rich.table import Table

table = Table(title="My Table")
table.add_column("Col1", style="cyan")
table.add_column("Col2", style="green")
table.add_row("value1", "value2")
console.print(table)
```

### Working with Rich Panels
```python
from rich.panel import Panel

panel = Panel("Content here", title="Title", border_style="blue")
console.print(panel)
```

### Formatting Datetimes
```python
# Basic format
time_str = episode.timestamp.strftime("%Y-%m-%d %H:%M:%S")

# Relative time (you'd need to calculate)
from datetime import datetime
delta = datetime.now() - episode.timestamp
hours_ago = delta.total_seconds() / 3600
time_str = f"{hours_ago:.1f} hours ago"
```

---

## Common Issues

**Issue:** `NameError: name 'Table' is not defined`
**Fix:** Add `from rich.table import Table` at the top

**Issue:** JSON export fails with datetime
**Fix:** Use `json.dump(data, f, default=str)` to handle datetime serialization

**Issue:** Parser shows no commands
**Fix:** Make sure you added `dest='command'` to `add_subparsers()`

---

## When You're Done

You should be able to:
- ✅ Run `python src/memory_cli.py --help` and see all commands
- ✅ View statistics with `stats` command
- ✅ Search episodes with `search` command
- ✅ View recent episodes with `recent` command
- ✅ Show detailed episode with `show` command
- ✅ Export episodes with `export` command
- ✅ Pass all tests: `python -m pytest tests/test_cli.py -v`

---

## Example Session

```bash
# Generate data
$ python example_usage.py

# Explore the data
$ python src/memory_cli.py stats
Memory Statistics
┏━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric         ┃ Value ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Episodes │ 4     │
└────────────────┴───────┘

$ python src/memory_cli.py search "React" --limit 2
Searching for: React

Found 2 results:
...

$ python src/memory_cli.py show 1
╭─────── Episode 1 ───────╮
│ Context: User asked...  │
│ ...                     │
╰─────────────────────────╯
```

---

Good luck! Start with `setup_parser()`, then work through the commands in order. Shout when you want me to review!
