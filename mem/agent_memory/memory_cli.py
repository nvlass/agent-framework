"""
Memory CLI Explorer - Command-line interface for agent memory system

This tool provides various commands to explore and query the memory store.

Usage:
    python -m agent_memory.memory_cli stats
    python -m agent_memory.memory_cli search "docker networking" --limit 5
    python -m agent_memory.memory_cli recent --hours 48
    python -m agent_memory.memory_cli patterns
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .memory_store import MemoryStore, Episode, Reflection
from .embeddings import LlamaCppEmbeddingGenerator
from .consolidation import LearnedPattern
from .metrics import MemoryMetrics
from .config import MemoryConfig, load_config, get_config_template, create_default_config_file

console = Console()

# Default paths (relative to package location or working directory)
DEFAULT_DB_PATH = Path("data/agent_memory.db")
DEFAULT_VECTOR_PATH = Path("data/memory_vectors")
DEFAULT_MODEL_NAME = "nomic-embed-text-v1.5.Q8_0.gguf"


def find_model_path() -> Optional[Path]:
    """
    Find embedding model in standard locations.

    Search order:
    1. EMBEDDING_MODEL_PATH environment variable
    2. models/ directory in current working directory
    3. models/ directory relative to package
    """
    # Environment variable
    env_path = os.getenv('EMBEDDING_MODEL_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Current working directory models/
    cwd_model = Path("models") / DEFAULT_MODEL_NAME
    if cwd_model.exists():
        return cwd_model

    # Package-relative models/
    pkg_model = Path(__file__).parent.parent / "models" / DEFAULT_MODEL_NAME
    if pkg_model.exists():
        return pkg_model

    return None


def create_memory_store(
    db_path: Optional[str] = None,
    vector_path: Optional[str] = None,
    require_embeddings: bool = False,
) -> MemoryStore:
    """
    Create and initialize memory store with sensible defaults.

    Args:
        db_path: Database path (uses default if None)
        vector_path: Vector store path (uses default if None)
        require_embeddings: If True, raise error if no model found

    Returns:
        Initialized MemoryStore instance
    """
    # Use defaults if not specified
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    vectors = Path(vector_path) if vector_path else DEFAULT_VECTOR_PATH

    # Find embedding model
    model_path = find_model_path()
    embedding_gen = None

    if model_path:
        console.print(f"[dim]Loading model: {model_path.name}[/dim]")
        embedding_gen = LlamaCppEmbeddingGenerator(model_path=str(model_path))
    elif require_embeddings:
        raise RuntimeError(
            "No embedding model found. Please either:\n"
            "  1. Set EMBEDDING_MODEL_PATH environment variable\n"
            "  2. Run: python scripts/setup_embedding_model.py\n"
            "  3. Place model in models/ directory"
        )
    else:
        console.print("[yellow]No embedding model found - semantic search disabled[/yellow]")

    return MemoryStore(
        db_path=str(db),
        vector_store_path=str(vectors),
        embedding_generator=embedding_gen,
    )


# ============================================================================
# TODO: IMPLEMENT THESE COMMAND FUNCTIONS
# ============================================================================

def command_stats(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Display memory store statistics

    TODO: Implement this function to show:
    - Total number of episodes
    - Episodes with outcomes
    - Scored episodes
    - Average success score
    - Vector store size

    Hints:
    - Use store.get_stats() to get statistics dictionary
    - Use rich.table.Table to format output nicely
    - Look at example_usage.py for table examples

    Args:
        args: Command-line arguments (none needed for stats)
        store: MemoryStore instance
    """
    stats = store.get_stats()
    table = Table(title='Memory Statistics')
    table.add_column('Metric', style='cyan')
    table.add_column('Value', style='green')

    for metric, val in stats.items():
        metric_name = metric.replace('_', ' ').title()
        table.add_row(metric_name, str(val))
    console.print(table)


def command_search(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Search for episodes matching a query

    TODO: Implement this function to:
    - Get search query from args.query
    - Retrieve episodes using store.retrieve_episodes()
    - Display results in a formatted table or list
    - Show similarity scores

    Hints:
    - args.query contains the search string
    - args.limit contains max results (default 5)
    - Results are tuples of (Episode, similarity_score)
    - Consider showing: ID, context preview, similarity, tags

    Args:
        args: Command-line arguments with query and limit
        store: MemoryStore instance
    """
    results = store.retrieve_episodes(args.query, limit=args.limit)
    if not results:
        console.print('[yellow]No results found[/yellow]')
        return

    console.print(f'Found {len(results)} results:\n')

    for i, (episode, similarity) in enumerate(results, 1):
        console.print(f'[bold]{i}. [dim](similarity: {similarity:.3f})[/dim][/bold]')
        console.print(f'  [cyan]Context:[/cyan] {episode.context[:80]}...')
        console.print(f'  [cyan]Action:[/cyan] {episode.action[:80]}...')
        console.print(f'  [cyan]Tags:[/cyan] {", ".join(episode.tags)}')
        console.print()

def command_recent(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show recent episodes within time window

    TODO: Implement this function to:
    - Get time window from args.hours
    - Retrieve recent episodes using store.get_recent_episodes()
    - Display in a formatted table

    Hints:
    - args.hours contains the time window (default 24)
    - args.limit contains max results (default 10)
    - Show timestamp, context, action, success score

    Args:
        args: Command-line arguments with hours and limit
        store: MemoryStore instance
    """
    episodes = store.get_recent_episodes(hours=args.hours, limit=args.limit)

    if not episodes:
        console.print("[yellow]No episodes found in time window[/yellow]")
        return

    table = Table()
    table.add_column('ID', style='cyan')
    table.add_column('Time', style='dim')
    table.add_column('Context', style='white')
    table.add_column('Score', style='green')

    for episode in episodes:
        time_str = episode.timestamp.strftime("%Y-%m-%d %H:%M")
        context = episode.context[:60] + "..." if len(episode.context) > 60 else episode.context
        score = f'{episode.success_score:.2f}' if episode.success_score else 'N/A'
        table.add_row(str(episode.id), time_str, context, score)

    console.print(table)

def command_show(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show detailed view of a specific episode

    TODO: Implement this function to:
    - Get episode ID from args.episode_id
    - Retrieve episode using store.get_episode_by_id()
    - Display all episode details in a nice format
    - Handle case where episode doesn't exist

    Hints:
    - Use rich.panel.Panel for a nice bordered display
    - Show all fields: id, timestamp, context, action, outcome,
      success_score, tags, embedding_id
    - Format timestamp nicely

    Args:
        args: Command-line arguments with episode_id
        store: MemoryStore instance
    """
    episode = store.get_episode_by_id(args.episode_id)

    if episode is None:
        console.print(f'[red]Episode {args.episode_id} not found[/red]')
        return
    details = f"""
    [cyan]Episode ID:[/cyan] {episode.id}
    [cyan]Timestamp:[/cyan] {episode.timestamp}
    [cyan]Context:[/cyan] {episode.context}
    [cyan]Action:[/cyan] {episode.action}
    [cyan]Outcome:[/cyan] {episode.outcome}
    [cyan]Success Score:[/cyan] {episode.success_score}
    [cyan]Tags:[/cyan] {', '.join(episode.tags)}
    """
    panel = Panel(details, title=f'Episode {episode.id}', border_style='blue')
    console.print(panel)

def command_export(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Export episodes to JSON file

    TODO: Implement this function to:
    - Get all episodes from store
    - Convert to JSON-serializable format
    - Write to file specified in args.output
    - Show success message with episode count

    Hints:
    - Use store.get_all_episodes()
    - Each Episode has a to_dict() method
    - Use json.dump() with indent=2 for readability
    - Handle file writing errors gracefully

    Args:
        args: Command-line arguments with output filename
        store: MemoryStore instance
    """
    episodes = store.get_all_episodes()
    data = [episode.to_dict() for episode in episodes]
    try:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        console.print(f'[green]Exported {len(data)} episodes to {output_path}')
    except Exception as e:
        console.print(f'[red]Error writing file: {e}[/red]')


def command_tags(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show all tags with usage statistics

    Displays all unique tags across episodes with their usage counts,
    sorted by count descending.

    Args:
        args: Command-line arguments (none needed for tags)
        store: MemoryStore instance
    """
    tags = store.get_all_tags()

    if not tags:
        console.print("[yellow]No tags found[/yellow]")
        return

    table = Table(title="Tag Statistics")
    table.add_column("Tag", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="dim", justify="right")

    total_episodes = store.get_episode_count()

    for tag, count in tags:
        percentage = (count / total_episodes * 100) if total_episodes > 0 else 0
        table.add_row(tag, str(count), f"{percentage:.1f}%")

    console.print(table)
    console.print(f"\n[dim]Total unique tags: {len(tags)}[/dim]")


def command_consolidate(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Run memory consolidation pipeline

    Clusters similar episodes and extracts learned patterns.

    Args:
        args: Command-line arguments with options
        store: MemoryStore instance
    """
    console.print("[bold]Running memory consolidation...[/bold]\n")

    # Run consolidation
    report = store.run_consolidation(
        hours_back=args.hours,
        auto_trigger=args.auto,
        episode_threshold=args.threshold,
        time_threshold_hours=args.time_threshold
    )

    # Show results
    if report.patterns_created == 0 and report.episodes_processed == 0:
        if args.auto:
            console.print("[yellow]Consolidation not triggered (thresholds not met)[/yellow]")
        else:
            console.print("[yellow]No episodes to consolidate[/yellow]")
        return

    table = Table(title="Consolidation Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Episodes Processed", str(report.episodes_processed))
    table.add_row("Clusters Found", str(report.clusters_found))
    table.add_row("Patterns Created", str(report.patterns_created))
    table.add_row("Noise Episodes", str(report.noise_episodes))
    table.add_row("Duration", f"{report.duration_seconds:.2f}s")

    console.print(table)

    # Show patterns summary
    if report.patterns:
        console.print("\n[bold]Patterns Created:[/bold]\n")
        for i, pattern in enumerate(report.patterns, 1):
            console.print(f"[cyan]{i}.[/cyan] {pattern.context_signature}")
            console.print(f"   Recommended: {pattern.recommended_action[:60]}...")
            console.print(f"   Success Rate: {pattern.success_rate:.0%} | "
                         f"Confidence: {pattern.confidence:.2f} | "
                         f"Samples: {pattern.sample_count}")
            console.print()


def command_consolidation_status(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show consolidation status

    Displays last consolidation time and episodes since.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    last_consolidation, episodes_since = store._get_consolidation_metadata()

    table = Table(title="Consolidation Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if last_consolidation:
        time_since = datetime.now() - last_consolidation
        hours_since = time_since.total_seconds() / 3600.0
        table.add_row("Last Consolidation", last_consolidation.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Hours Since", f"{hours_since:.1f}")
    else:
        table.add_row("Last Consolidation", "Never")
        table.add_row("Hours Since", "N/A")

    table.add_row("Episodes Since Last", str(episodes_since))

    # Get pattern count
    patterns = store.get_learned_patterns()
    table.add_row("Total Patterns", str(len(patterns)))

    console.print(table)


def command_patterns(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    List learned patterns

    Args:
        args: Command-line arguments with filtering options
        store: MemoryStore instance
    """
    patterns = store.get_learned_patterns(
        min_confidence=args.min_confidence,
        min_success_rate=args.min_success,
        limit=args.limit
    )

    if not patterns:
        console.print("[yellow]No patterns found[/yellow]")
        return

    table = Table(title=f"Learned Patterns ({len(patterns)} found)")
    table.add_column("ID", style="dim")
    table.add_column("Context", style="cyan")
    table.add_column("Action", style="white")
    table.add_column("Success", style="green", justify="right")
    table.add_column("Conf", style="yellow", justify="right")
    table.add_column("N", style="dim", justify="right")

    for pattern in patterns:
        context = pattern.context_signature[:40] + "..." if len(pattern.context_signature) > 40 else pattern.context_signature
        action = pattern.recommended_action[:30] + "..." if len(pattern.recommended_action) > 30 else pattern.recommended_action

        table.add_row(
            str(pattern.pattern_id),
            context,
            action,
            f"{pattern.success_rate:.0%}",
            f"{pattern.confidence:.2f}",
            str(pattern.sample_count)
        )

    console.print(table)


def command_pattern(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show detailed view of a specific pattern

    Args:
        args: Command-line arguments with pattern_id
        store: MemoryStore instance
    """
    # Get all patterns and find the one we want
    patterns = store.get_learned_patterns()
    pattern = None
    for p in patterns:
        if p.pattern_id == args.pattern_id:
            pattern = p
            break

    if pattern is None:
        console.print(f"[red]Pattern {args.pattern_id} not found[/red]")
        return

    # Display pattern details
    panel_content = f"""
[cyan]Pattern ID:[/cyan] {pattern.pattern_id}
[cyan]Context Signature:[/cyan] {pattern.context_signature}
[cyan]Recommended Action:[/cyan] {pattern.recommended_action}
[cyan]Success Rate:[/cyan] {pattern.success_rate:.0%}
[cyan]Confidence:[/cyan] {pattern.confidence:.2f}
[cyan]Sample Count:[/cyan] {pattern.sample_count}
[cyan]Source Episodes:[/cyan] {', '.join(map(str, pattern.source_episode_ids[:10]))}{'...' if len(pattern.source_episode_ids) > 10 else ''}
[cyan]Created:[/cyan] {pattern.created_at.strftime("%Y-%m-%d %H:%M") if pattern.created_at else 'N/A'}

[bold]Full Description:[/bold]
{pattern.pattern_description}
"""
    panel = Panel(panel_content, title=f"Pattern {pattern.pattern_id}", border_style="blue")
    console.print(panel)


def command_recommend(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Get action recommendations based on learned patterns

    Args:
        args: Command-line arguments with context
        store: MemoryStore instance
    """
    advice = store.get_action_advice(
        context=args.context,
        goal=args.goal
    )

    if not advice['recommendations']:
        console.print("[yellow]No matching patterns found.[/yellow]")
        console.print("[dim]Try running consolidation to learn from past episodes.[/dim]")
        return

    console.print(f"\n[bold]Action Recommendations[/bold] ({advice['patterns_matched']} patterns matched)\n")

    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Recommended Action", style="cyan")
    table.add_column("Success", style="green", justify="right")
    table.add_column("Confidence", style="yellow", justify="right")
    table.add_column("Reason", style="dim")

    for i, rec in enumerate(advice['recommendations'], 1):
        action = rec['action'][:50] + "..." if len(rec['action']) > 50 else rec['action']
        table.add_row(
            str(i),
            action,
            f"{rec['success_rate']:.0%}",
            f"{rec['confidence']:.2f}",
            rec['reason'][:30] + "..." if len(rec['reason']) > 30 else rec['reason']
        )

    console.print(table)
    console.print(f"\n[dim]Overall confidence: {advice['confidence']:.2f}[/dim]")


def command_forget(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Apply forgetting policy or forget specific episode

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    if args.episode_id:
        # Forget specific episode
        success = store.forget_episode(
            args.episode_id,
            reason=args.reason or 'manual',
            summary=args.summary
        )
        if success:
            console.print(f"[green]Episode {args.episode_id} archived to forgotten_memories[/green]")
        else:
            console.print(f"[red]Episode {args.episode_id} not found[/red]")
    else:
        # Apply policy
        console.print("[bold]Applying forgetting policy...[/bold]\n")

        report = store.apply_forgetting_policy(
            age_threshold_days=args.age,
            min_success_for_keep=args.keep_success,
            max_failure_for_keep=args.keep_failure,
            max_forget=args.max,
            dry_run=args.dry_run
        )

        if args.dry_run:
            console.print("[yellow]DRY RUN - No episodes were forgotten[/yellow]\n")

        table = Table(title="Forgetting Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Candidates Found", str(report['candidates_found']))
        if not args.dry_run:
            table.add_row("Episodes Forgotten", str(report['forgotten']))

        console.print(table)

        if report['episodes']:
            console.print("\n[bold]Episodes to forget:[/bold]")
            ep_table = Table()
            ep_table.add_column("ID", style="dim")
            ep_table.add_column("Context", style="white")
            ep_table.add_column("Score", style="green")
            ep_table.add_column("Age (days)", style="yellow")

            for ep in report['episodes'][:10]:
                ep_table.add_row(
                    str(ep['id']),
                    ep['context'],
                    f"{ep['success_score']:.2f}" if ep['success_score'] else "N/A",
                    str(ep['age_days'])
                )

            console.print(ep_table)

            if len(report['episodes']) > 10:
                console.print(f"\n[dim]... and {len(report['episodes']) - 10} more[/dim]")


def command_forgotten(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show forgotten memories

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    memories = store.get_forgotten_memories(
        reason=args.reason,
        limit=args.limit
    )

    if not memories:
        console.print("[yellow]No forgotten memories found[/yellow]")
        return

    table = Table(title=f"Forgotten Memories ({len(memories)} found)")
    table.add_column("ID", style="dim")
    table.add_column("Original", style="cyan")
    table.add_column("Reason", style="yellow")
    table.add_column("Summary", style="white")
    table.add_column("Forgotten At", style="dim")

    for mem in memories:
        table.add_row(
            str(mem['id']),
            str(mem['original_id']),
            mem['reason'],
            mem['summary'][:40] + "..." if mem['summary'] and len(mem['summary']) > 40 else mem['summary'] or "N/A",
            mem['forgotten_at'][:16] if mem['forgotten_at'] else "N/A"
        )

    console.print(table)


def command_memory_tiers(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show memory tier statistics

    Displays episode counts across memory tiers.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    table = Table(title="Memory Tier Statistics")
    table.add_column("Tier", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Description", style="dim")

    # Working memory
    working_count = len(store.working_memory.get_all())
    table.add_row(
        "Working Memory",
        str(working_count),
        "Hot cache (most recent)"
    )

    # Short-term cache stats
    cache_stats = store.get_short_term_cache_stats()
    table.add_row(
        "Short-Term Cache",
        str(cache_stats.get('cached_queries', 0)) + " queries",
        "TTL-based cache"
    )

    # Long-term (total episodes)
    total_episodes = store.get_episode_count()
    table.add_row(
        "Long-Term Storage",
        str(total_episodes),
        "All episodes (SQLite + ChromaDB)"
    )

    # Patterns
    patterns = store.get_learned_patterns()
    table.add_row(
        "Learned Patterns",
        str(len(patterns)),
        "Consolidated knowledge"
    )

    console.print(table)


def command_reflections(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    List reflections with optional filtering.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    reflection_type = args.type if hasattr(args, 'type') else None
    reflections = store.get_reflections(
        reflection_type=reflection_type,
        limit=args.limit
    )

    if not reflections:
        console.print("[yellow]No reflections found[/yellow]")
        return

    table = Table(title=f"Reflections ({len(reflections)} found)")
    table.add_column("ID", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Episode", style="yellow")
    table.add_column("Insight", style="white")
    table.add_column("Created", style="dim")

    for ref in reflections:
        type_emoji = {
            'success_analysis': '[green]success[/green]',
            'failure_analysis': '[red]failure[/red]',
            'pattern_discovery': '[blue]pattern[/blue]'
        }.get(ref.reflection_type, ref.reflection_type)

        table.add_row(
            str(ref.id),
            type_emoji,
            str(ref.trigger_episode_id) if ref.trigger_episode_id else "N/A",
            ref.insight[:50] + "..." if len(ref.insight) > 50 else ref.insight,
            ref.created_at.strftime('%Y-%m-%d %H:%M') if ref.created_at else "N/A"
        )

    console.print(table)


def command_reflection(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show detailed reflection by ID.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    reflection = store.get_reflection_by_id(args.id)

    if reflection is None:
        console.print(f"[red]Reflection #{args.id} not found[/red]")
        return

    # Type indicator
    type_emoji = {
        'success_analysis': '[green]Success Analysis[/green]',
        'failure_analysis': '[red]Failure Analysis[/red]',
        'pattern_discovery': '[blue]Pattern Discovery[/blue]'
    }.get(reflection.reflection_type, reflection.reflection_type)

    console.print(Panel(
        f"[bold]{type_emoji}[/bold]\n\n"
        f"[cyan]Trigger Episode:[/cyan] {reflection.trigger_episode_id or 'N/A'}\n"
        f"[cyan]Created:[/cyan] {reflection.created_at.strftime('%Y-%m-%d %H:%M:%S') if reflection.created_at else 'N/A'}",
        title=f"Reflection #{reflection.id}",
        box=box.ROUNDED
    ))

    # Insight
    console.print(Panel(
        reflection.insight,
        title="Insight",
        box=box.SIMPLE
    ))

    # Causal factors
    if reflection.causal_chain:
        table = Table(title="Causal Factors", box=box.SIMPLE)
        table.add_column("Factor", style="white")
        table.add_column("Contribution", style="cyan")
        table.add_column("Confidence", style="yellow")

        for cf in reflection.causal_chain:
            contrib_style = {
                'positive': '[green]positive[/green]',
                'negative': '[red]negative[/red]',
                'neutral': '[dim]neutral[/dim]'
            }.get(cf.contribution, cf.contribution)

            table.add_row(
                cf.factor,
                contrib_style,
                f"{cf.confidence:.0%}"
            )

        console.print(table)

    # Actionable takeaway
    if reflection.actionable_takeaway:
        console.print(Panel(
            reflection.actionable_takeaway,
            title="Actionable Takeaway",
            box=box.SIMPLE,
            style="green"
        ))


# ============================================================================
# PHASE 6: Health, Config, and Domain Commands
# ============================================================================

def command_health(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show system health report.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    metrics = MemoryMetrics(store)

    if args.json:
        report = metrics.get_health_report()
        console.print(json.dumps(report, indent=2, default=str))
    else:
        console.print(metrics.get_summary())


def command_config_show(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show current configuration.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    config = load_config(args.file if hasattr(args, 'file') else None)

    if args.template:
        console.print(get_config_template())
    else:
        # Show current config as YAML-like format
        config_dict = config.to_dict()

        for section_name, section in config_dict.items():
            console.print(f"\n[bold cyan]{section_name}:[/bold cyan]")
            for key, value in section.items():
                console.print(f"  {key}: [green]{value}[/green]")


def command_config_create(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Create default configuration file.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    output_path = args.output or "agent_memory.yaml"
    create_default_config_file(output_path)
    console.print(f"[green]Created configuration file: {output_path}[/green]")


def command_domains(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show domain keywords.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    if args.domain:
        keywords_dict = store.get_domain_keywords_with_weights(args.domain)
    else:
        keywords_dict = store.get_domain_keywords_with_weights()

    if not keywords_dict:
        console.print("[yellow]No domain keywords found[/yellow]")
        console.print("[dim]Run: python scripts/populate_dev_db.py to seed defaults[/dim]")
        return

    for domain, keywords in keywords_dict.items():
        console.print(f"\n[bold cyan]{domain}[/bold cyan] ({len(keywords)} keywords)")

        if args.verbose:
            table = Table(box=box.SIMPLE)
            table.add_column("Keyword", style="white")
            table.add_column("Weight", style="green", justify="right")

            for keyword, weight in sorted(keywords.items(), key=lambda x: -x[1]):
                table.add_row(keyword, f"{weight:.2f}")

            console.print(table)
        else:
            # Compact view - just list keywords
            kw_list = sorted(keywords.keys())
            console.print(f"  {', '.join(kw_list[:15])}{'...' if len(kw_list) > 15 else ''}")

    console.print(f"\n[dim]Total: {len(keywords_dict)} domains[/dim]")


def command_adaptations(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show strategy adaptations.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    adaptations = store.get_adaptations(
        min_success=args.min_success if hasattr(args, 'min_success') else None,
        limit=args.limit
    )

    if not adaptations:
        console.print("[yellow]No adaptations found[/yellow]")
        return

    table = Table(title=f"Strategy Adaptations ({len(adaptations)} found)")
    table.add_column("ID", style="dim")
    table.add_column("Source Context", style="cyan")
    table.add_column("Target Context", style="white")
    table.add_column("Success", style="green", justify="right")

    for adapt in adaptations:
        source = adapt['source_context'][:30] + "..." if len(adapt['source_context']) > 30 else adapt['source_context']
        target = adapt['target_context'][:30] + "..." if len(adapt['target_context']) > 30 else adapt['target_context']
        success = f"{adapt['success_score']:.0%}" if adapt['success_score'] else "N/A"

        table.add_row(str(adapt['id']), source, target, success)

    console.print(table)


def command_adaptation(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show detailed adaptation by ID.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    adaptations = store.get_adaptations(limit=1000)
    adaptation = None
    for a in adaptations:
        if a['id'] == args.id:
            adaptation = a
            break

    if adaptation is None:
        console.print(f"[red]Adaptation #{args.id} not found[/red]")
        return

    panel_content = f"""
[cyan]Source Context:[/cyan]
{adaptation['source_context']}

[cyan]Target Context:[/cyan]
{adaptation['target_context']}

[cyan]Original Strategy:[/cyan]
{adaptation['original_strategy']}

[cyan]Adapted Strategy:[/cyan]
{adaptation['adapted_strategy']}

[cyan]Reasoning:[/cyan]
{adaptation['adaptation_reasoning'] or 'N/A'}

[cyan]Outcome:[/cyan] {adaptation['outcome'] or 'Not recorded'}
[cyan]Success Score:[/cyan] {f"{adaptation['success_score']:.0%}" if adaptation['success_score'] else 'N/A'}
[cyan]Created:[/cyan] {adaptation['created_at'] or 'N/A'}
"""

    console.print(Panel(panel_content, title=f"Adaptation #{args.id}", border_style="blue"))


def command_problem_types(args: argparse.Namespace, store: MemoryStore) -> None:
    """
    Show problem types.

    Args:
        args: Command-line arguments
        store: MemoryStore instance
    """
    problem_types = store.get_all_problem_types(limit=args.limit if hasattr(args, 'limit') else None)

    if not problem_types:
        console.print("[yellow]No problem types found[/yellow]")
        return

    table = Table(title=f"Problem Types ({len(problem_types)} found)")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Characteristics", style="dim")

    for pt in problem_types:
        chars = ", ".join(pt['characteristics'][:3])
        if len(pt['characteristics']) > 3:
            chars += "..."

        table.add_row(
            str(pt['id']),
            pt['name'],
            pt['description'][:40] + "..." if len(pt['description'] or '') > 40 else pt['description'] or "",
            chars
        )

    console.print(table)


# ============================================================================
# ARGUMENT PARSER SETUP
# ============================================================================

def setup_parser() -> argparse.ArgumentParser:
    """
    Set up argument parser with all subcommands

    TODO: Implement this function to create ArgumentParser with subcommands:

    Main parser:
    - Description: "Agent Memory Explorer"
    - Add --config option for configuration file path
    - Create subparsers with dest='command'

    Subcommands to add:

    1. stats
       - Help: "Show memory statistics"
       - No arguments needed

    2. search
       - Help: "Search for episodes"
       - Argument: query (positional, type=str, help="Search query")
       - Argument: --limit (optional, type=int, default=5, help="Max results")

    3. recent
       - Help: "Show recent episodes"
       - Argument: --hours (optional, type=int, default=24, help="Time window in hours")
       - Argument: --limit (optional, type=int, default=10, help="Max results")

    4. show
       - Help: "Show detailed episode"
       - Argument: episode_id (positional, type=int, help="Episode ID")

    5. export
       - Help: "Export episodes to JSON"
       - Argument: output (positional, type=str, help="Output filename")

    Hints:
    - parser.add_subparsers() creates subcommand handler
    - For each subcommand: subparser = subparsers.add_parser(name, help=...)
    - Add arguments to each subparser
    - Don't forget to add --config to the main parser before creating subparsers
    - Look at argparse documentation if needed

    Returns:
        Configured ArgumentParser instance
    """
    # TODO: Your code here
    parser = argparse.ArgumentParser(description="Agent Memory Explorer")

    # Add global options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )

    # Create subparsers...
    subparsers = parser.add_subparsers(dest='command')
    # Add each command...
    stats_parser = subparsers.add_parser('stats', help='Show memory statistics')

    search_parser = subparsers.add_parser('search', help='Search for episodes')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--limit', type=int, default=5, help='Max results')

    recent_parser = subparsers.add_parser('recent', help='Show recent episodes')
    recent_parser.add_argument('--hours', type=int, default=24, help='Time window in hours')
    recent_parser.add_argument('--limit', type=int, default=10, help='Max results')

    show_parser = subparsers.add_parser('show', help='Show detailed episode')
    show_parser.add_argument('episode_id', type=int, help='Episode ID')

    export_parser = subparsers.add_parser('export', help='Export episodes to JSON')
    export_parser.add_argument('output', type=str, help='Output filename')

    tags_parser = subparsers.add_parser('tags', help='Show tag statistics')

    # Consolidation commands
    consolidate_parser = subparsers.add_parser('consolidate', help='Run memory consolidation')
    consolidate_parser.add_argument('--hours', type=int, default=168, help='Hours of episodes to consolidate (default: 168 = 1 week)')
    consolidate_parser.add_argument('--auto', action='store_true', help='Only run if thresholds are met')
    consolidate_parser.add_argument('--threshold', type=int, default=100, help='Episode count trigger (for --auto)')
    consolidate_parser.add_argument('--time-threshold', type=float, default=24.0, help='Hours trigger (for --auto)')

    consolidation_status_parser = subparsers.add_parser('consolidation-status', help='Show consolidation status')

    # Pattern commands
    patterns_parser = subparsers.add_parser('patterns', help='List learned patterns')
    patterns_parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence (0.0-1.0)')
    patterns_parser.add_argument('--min-success', type=float, default=0.0, help='Minimum success rate (0.0-1.0)')
    patterns_parser.add_argument('--limit', type=int, default=20, help='Max patterns to show')

    pattern_parser = subparsers.add_parser('pattern', help='Show detailed pattern')
    pattern_parser.add_argument('pattern_id', type=int, help='Pattern ID')

    # Memory tier command
    tiers_parser = subparsers.add_parser('tiers', help='Show memory tier statistics')

    # Action recommendation command
    recommend_parser = subparsers.add_parser('recommend', help='Get action recommendations for a context')
    recommend_parser.add_argument('context', type=str, help='Current situation/context')
    recommend_parser.add_argument('--goal', type=str, help='Optional goal to achieve')

    # Forgetting commands
    forget_parser = subparsers.add_parser('forget', help='Apply forgetting policy or forget specific episode')
    forget_parser.add_argument('--episode-id', type=int, help='Forget specific episode by ID')
    forget_parser.add_argument('--reason', type=str, default='low_utility', help='Reason for forgetting')
    forget_parser.add_argument('--summary', type=str, help='Summary for specific episode')
    forget_parser.add_argument('--age', type=int, default=30, help='Age threshold in days (policy mode)')
    forget_parser.add_argument('--keep-success', type=float, default=0.8, help='Keep episodes with success >= this')
    forget_parser.add_argument('--keep-failure', type=float, default=0.3, help='Keep episodes with success <= this')
    forget_parser.add_argument('--max', type=int, default=50, help='Max episodes to forget')
    forget_parser.add_argument('--dry-run', action='store_true', help='Show what would be forgotten without forgetting')

    forgotten_parser = subparsers.add_parser('forgotten', help='Show forgotten memories')
    forgotten_parser.add_argument('--reason', type=str, help='Filter by reason')
    forgotten_parser.add_argument('--limit', type=int, default=20, help='Max results')

    # Reflection commands
    reflections_parser = subparsers.add_parser('reflections', help='List reflections')
    reflections_parser.add_argument('--type', type=str, choices=['success_analysis', 'failure_analysis', 'pattern_discovery'],
                                    help='Filter by reflection type')
    reflections_parser.add_argument('--limit', type=int, default=20, help='Max results')

    reflection_parser = subparsers.add_parser('reflection', help='Show detailed reflection')
    reflection_parser.add_argument('id', type=int, help='Reflection ID to show')

    # Phase 6: Health command
    health_parser = subparsers.add_parser('health', help='Show system health report')
    health_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Phase 6: Config commands
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    config_parser.add_argument('--template', action='store_true', help='Show configuration template')
    config_parser.add_argument('--file', type=str, help='Config file to load')

    config_create_parser = subparsers.add_parser('config-create', help='Create default configuration file')
    config_create_parser.add_argument('--output', type=str, default='agent_memory.yaml', help='Output file path')

    # Phase 6: Domain commands
    domains_parser = subparsers.add_parser('domains', help='Show domain keywords')
    domains_parser.add_argument('--domain', type=str, help='Filter by domain name')
    domains_parser.add_argument('--verbose', '-v', action='store_true', help='Show keyword weights')

    # Phase 6: Adaptation commands
    adaptations_parser = subparsers.add_parser('adaptations', help='Show strategy adaptations')
    adaptations_parser.add_argument('--min-success', type=float, help='Filter by minimum success score')
    adaptations_parser.add_argument('--limit', type=int, default=20, help='Max results')

    adaptation_parser = subparsers.add_parser('adaptation', help='Show detailed adaptation')
    adaptation_parser.add_argument('id', type=int, help='Adaptation ID')

    # Phase 6: Problem types command
    problem_types_parser = subparsers.add_parser('problem-types', help='Show problem types')
    problem_types_parser.add_argument('--limit', type=int, default=50, help='Max results')

    return parser


# ============================================================================
# MAIN FUNCTION (This is complete - no need to modify)
# ============================================================================

def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()

    # Show help if no command specified
    if not hasattr(args, 'command') or args.command is None:
        parser.print_help()
        return

    # Map commands to functions
    commands = {
        'stats': command_stats,
        'search': command_search,
        'recent': command_recent,
        'show': command_show,
        'export': command_export,
        'tags': command_tags,
        'consolidate': command_consolidate,
        'consolidation-status': command_consolidation_status,
        'patterns': command_patterns,
        'pattern': command_pattern,
        'tiers': command_memory_tiers,
        'recommend': command_recommend,
        'forget': command_forget,
        'forgotten': command_forgotten,
        'reflections': command_reflections,
        'reflection': command_reflection,
        # Phase 6 commands
        'health': command_health,
        'config': command_config_show,
        'config-create': command_config_create,
        'domains': command_domains,
        'adaptations': command_adaptations,
        'adaptation': command_adaptation,
        'problem-types': command_problem_types,
    }

    # Commands that require embeddings for semantic search
    embedding_required_commands = {'search', 'recommend', 'consolidate'}

    # Initialize memory store
    try:
        console.print("[dim]Initializing memory store...[/dim]")
        require_embeddings = args.command in embedding_required_commands
        with create_memory_store(require_embeddings=require_embeddings) as store:
            # Execute command
            command_func = commands.get(args.command)
            if command_func:
                command_func(args, store)
            else:
                console.print(f"[red]Unknown command: {args.command}[/red]")
                parser.print_help()
    except RuntimeError as e:
        # Configuration errors
        console.print(f"[red]Configuration Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
