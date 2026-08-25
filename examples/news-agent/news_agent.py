"""News digest agent — fetches sites, summarizes, emails a daily digest.

Uses native tool calling (OpenAI-compatible tools API) when available,
falling back to text parsing for CLI backends.

Memory integration:
    When --db is provided, the agent remembers previous runs via agent-memory.
    This gives it two capabilities:
    1. AUTO-RECALL: Before each run, AgentInstance queries the memory store
       for the top 3 episodes similar to the current task. These get injected
       as initial observations in the SharedContext, so the LLM sees them in
       its first prompt ("Relevant past experiences: ..."). This lets the agent
       notice recurring stories or avoid repeating yesterday's mistakes.
    2. AUTO-STORE: After each run, AgentInstance saves an episode recording
       what task was run, how many iterations it took, and whether it succeeded.
       Tagged ["auto"] so they can be filtered later.
    The agent can also call memory_store/memory_recall/memory_reflect as
    explicit tools mid-run if it decides to (they're registered in the
    tool registry automatically).

Requires:
    - llama-server running with a capable model (must support function calling)
    - /usr/sbin/sendmail available (provided by postfix)
    - html2text (pip install html2text)

Usage:
    python news_agent.py
    python news_agent.py --port 7788
    python news_agent.py --db news_memory.db   # enable memory persistence
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import html2text

from agent_core import AgentRole, AgentInstance, AgentConfig, ReactConfig
from agent_core.llm import ChatLLMInterface, ChatMessage
from agent_core.llm_cloud import AnthropicLLM, OpenAILLM, FireworksLLM
from agent_core.llm_llamacpp import LlamaCppServerLLM
from agent_tools.core.registry import ToolRegistry
from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel

SCRIPT_DIR = Path(__file__).resolve().parent

# Max chars fed into the summarizer LLM. Raw content never enters the main
# agent context — only the compact summary does. Can be set high since the
# summarizer model (e.g. Gemma-4-E2B, 128K ctx) handles large inputs well.
FETCH_MAX_CHARS = 100_000

SUMMARIZE_SYSTEM = "You are a news summarizer. Extract the key stories concisely."
SUMMARIZE_USER = """\
Summarize the top 5-8 stories from this news page as bullet points.
For each story include the title, a one-sentence summary, and the URL if visible.
Output only the bullet points, no preamble.

Page content:
{content}"""


@dataclass
class SummaryCache:
    """File-backed JSON cache for page summaries with TTL."""

    cache_file: Path
    ttl_hours: int = 12
    _data: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_file.exists():
            self._data = json.loads(self.cache_file.read_text())

    def get(self, url: str) -> str | None:
        entry = self._data.get(url)
        if not entry:
            return None
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if datetime.now() - fetched_at > timedelta(hours=self.ttl_hours):
            return None
        return entry["summary"]

    def set(self, url: str, summary: str) -> None:
        self._data[url] = {"summary": summary, "fetched_at": datetime.now().isoformat()}
        self.cache_file.write_text(json.dumps(self._data, indent=2))


def _summarize_with_llm(llm: ChatLLMInterface, content: str) -> str:
    """Call the LLM to summarize raw page content into bullet points."""
    messages = [
        ChatMessage(role="system", content=SUMMARIZE_SYSTEM),
        ChatMessage(role="user", content=SUMMARIZE_USER.format(content=content)),
    ]
    response = llm.chat(messages, max_tokens=20480, temperature=0.3)
    return response.content

ARTICLE_MAX_CHARS = 8_000  # per-article truncation for deep-dive

LIST_ARTICLES_SYSTEM = "You are a news indexer. Extract article links from news homepages."
LIST_ARTICLES_USER = """\
Extract the top 8-12 article links from this news page.
For each article output a JSON object with "title" and "url".
Construct the full absolute URL (include domain and protocol) if the link is relative.
The base URL of this page is: {base_url}

Output ONLY a JSON array, no other text:
[{{"title": "...", "url": "..."}}, ...]

Page content:
{content}"""

SYNTHESIZE_SYSTEM = "You are an analytical journalist synthesizing coverage from multiple news sources."
SYNTHESIZE_USER = """\
Multiple news sources covered this story: {topic}

{articles}

Write a synthesis that:
1. States the core facts all sources agree on (1-2 sentences)
2. Notes different angles, emphasis, or framing between sources
3. Highlights any contradictions or unique perspectives

Be concise — 2-3 short paragraphs. Include the source names."""

DEEP_DIVE_TASK = """\
You are conducting a DEEP-DIVE news analysis. Complete ALL steps in order:

1. Call list_articles for EACH site below to get article links.
2. Review the lists. Identify 5-7 MAJOR STORIES covered by 2+ sources.
3. For each major story:
   a. Call fetch_article for the relevant URL from each source that covers it.
   b. Call synthesize_topic with the topic name and a JSON array of those URLs.
4. Compose the full digest:
   - Date header: {date}
   - One section per major story (headline + synthesis + source links)
5. Call send_email with subject "Deep-Dive News Analysis {date}" and the complete digest as body.
   Write the full digest inline as the value of body — do not leave it empty.
6. Confirm completion with a final Answer.

Sites:
{{sites}}
"""
# Note: {sites} is formatted separately in main()
DEEP_DIVE_TASK = DEEP_DIVE_TASK.replace("{{sites}}", "{sites}")


def _extract_article_links(llm: ChatLLMInterface, content: str, base_url: str) -> str:
    """Call LLM to extract article {title, url} pairs from page content. Returns JSON string."""
    messages = [
        ChatMessage(role="system", content=LIST_ARTICLES_SYSTEM),
        ChatMessage(role="user", content=LIST_ARTICLES_USER.format(content=content, base_url=base_url)),
    ]
    response = llm.chat(messages, max_tokens=2048, temperature=0.1)
    return response.content


NEWS_SITES = [
    "https://www.iefimerida.gr/",
    "https://www.pronews.gr/",
    "https://news.ycombinator.com",
    "https://www.theverge.com/",
    "https://www.bbc.com/news",
    "https://www.wired.com",
    "https://techcrunch.com/",
]

TASK = """\
You MUST complete these steps in order:

1. Call fetch_and_summarize once for EACH site listed below. Do not skip any.
2. Call get_summary for each site to retrieve the full summaries.
3. Compose a concise daily digest (under 500 words) grouping stories by topic. \
Each topic must be a header, with bullet points beneath it. \
Include the article URL for each story. \
Note stories that appear on multiple sources.
4. Write the full digest text (all topics, bullets, URLs) and then call send_email
   with subject "Daily News Digest" and the complete digest text as the value of body.
   The body parameter must contain the entire formatted email — write it all inline.
5. ONLY THEN give your final Answer confirming the email was sent.

Sites to scan:
{sites}
"""


def _fetch_readable(url: str) -> str:
    """Fetch a URL and return its content as readable markdown text."""
    req = urllib.request.Request(url, headers={"User-Agent": "news-agent/1.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode("utf-8", errors="replace")
    h = html2text.HTML2Text()
    h.body_width = 0
    text = h.handle(html)
    if len(text) > FETCH_MAX_CHARS:
        # Truncate at last newline before the limit
        cut = text.rfind("\n", 0, FETCH_MAX_CHARS)
        if cut == -1:
            cut = FETCH_MAX_CHARS
        text = text[:cut] + "\n\n[... truncated ...]"
    return text


def _send_email(subject: str, body: str) -> str:
    """Send email by piping the full message to sendmail via stdin.

    Recipients and sender come from env vars so no addresses are hardcoded:
        NEWS_EMAIL_TO    comma-separated recipients (default: you@example.com)
        NEWS_EMAIL_FROM  sender (default: news-agent@localhost)
    """
    to_addr = os.environ.get("NEWS_EMAIL_TO", "you@example.com")
    from_addr = os.environ.get("NEWS_EMAIL_FROM", "news-agent@localhost")
    message = (
        f"To: {to_addr}\n"
        f"Subject: {subject}\n"
        f"From: {from_addr}\n"
        f"\n"
        f"{body}"
    )
    result = subprocess.run(
        ["/usr/sbin/sendmail", "-t"],
        input=message,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return f"sendmail failed: {result.stderr}"
    return "Email sent successfully"


def build_tools(llm: ChatLLMInterface, summarizer_llm: ChatLLMInterface | None = None) -> ToolRegistry:
    """Create the tool registry with fetch_and_summarize, get_summary, and send_email.

    Args:
        llm: Main agent LLM (used as fallback summarizer if summarizer_llm not provided).
        summarizer_llm: Optional dedicated LLM for summarization. When provided, a
            separate llama-server instance handles summarization — freeing the main
            model's context and allowing a smaller/faster model for that task.
    """
    registry = ToolRegistry()
    cache = SummaryCache(cache_file=SCRIPT_DIR / ".news_cache.json")
    _summarizer = summarizer_llm or llm

    def _fetch_and_summarize(url: str) -> str:
        cached = cache.get(url)
        if cached:
            return f"[cached] Summary for {url}:\n{cached}"
        content = _fetch_readable(url)
        summary = _summarize_with_llm(_summarizer, content)
        cache.set(url, summary)
        preview = summary[:300] + "..." if len(summary) > 300 else summary
        return f"Summary for {url}:\n{preview}"

    registry.register(ToolDefinition(
        name="fetch_and_summarize",
        description=(
            "Fetch a URL, summarize its top stories via LLM, and cache the result. "
            "Returns a compact bullet-point summary. Call this once per site."
        ),
        parameters=[
            ToolParameter(name="url", type="string", description="URL to fetch and summarize"),
        ],
        returns="string",
        permission=PermissionLevel.READ,
        timeout_seconds=60,
        execute=_fetch_and_summarize,
    ))

    registry.register(ToolDefinition(
        name="get_summary",
        description=(
            "Retrieve the full cached summary for a previously fetched URL. "
            "Use this to re-read a summary when composing the digest."
        ),
        parameters=[
            ToolParameter(name="url", type="string", description="URL to retrieve summary for"),
        ],
        returns="string",
        permission=PermissionLevel.READ,
        timeout_seconds=5,
        execute=lambda url: cache.get(url) or f"No summary cached for {url} — call fetch_and_summarize first.",
    ))

    registry.register(ToolDefinition(
        name="send_email",
        description=(
            "Send the daily news digest email. "
            "You MUST provide both subject and body. "
            "Write the complete formatted digest text directly as the value of 'body' — "
            "do not leave it empty or defer composing it."
        ),
        parameters=[
            ToolParameter(name="subject", type="string", description="Email subject line"),
            ToolParameter(
                name="body",
                type="string",
                description=(
                    "REQUIRED. The complete formatted email body as a plain text string. "
                    "Write the full digest here, inline, as the value of this parameter. "
                    "Include all stories grouped by topic with URLs. Must not be empty."
                ),
            ),
        ],
        returns="string",
        permission=PermissionLevel.DANGEROUS,
        timeout_seconds=10,
        execute=lambda subject, body: _send_email(subject, body),
    ))

    return registry


def build_deep_dive_tools(
    llm: ChatLLMInterface,
    summarizer_llm: ChatLLMInterface | None = None,
) -> ToolRegistry:
    """Build tool registry for deep-dive mode: list_articles, fetch_article, synthesize_topic, send_email."""
    registry = ToolRegistry()
    links_cache = SummaryCache(cache_file=SCRIPT_DIR / ".news_links_cache.json")
    article_texts: dict[str, str] = {}  # in-memory: url → full article text
    _summarizer = summarizer_llm or llm

    def _list_articles(url: str) -> str:
        cached = links_cache.get(url)
        if cached:
            return f"[cached] Article links for {url}:\n{cached}"
        content = _fetch_readable(url)
        links_json = _extract_article_links(_summarizer, content, base_url=url)
        links_cache.set(url, links_json)
        return f"Article links for {url}:\n{links_json}"

    def _fetch_article(url: str) -> str:
        if url in article_texts:
            preview = article_texts[url][:400]
            return f"[cached] Article at {url}:\n{preview}..."
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "news-agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            h = html2text.HTML2Text()
            h.body_width = 0
            text = h.handle(html)
            if len(text) > ARTICLE_MAX_CHARS:
                cut = text.rfind("\n", 0, ARTICLE_MAX_CHARS)
                text = text[: cut if cut != -1 else ARTICLE_MAX_CHARS] + "\n[truncated]"
            article_texts[url] = text
            preview = text[:400]
            return f"Article at {url}:\n{preview}..."
        except Exception as exc:
            return f"Failed to fetch {url}: {exc}"

    def _synthesize_topic(topic: str, article_urls_json: str) -> str:
        try:
            urls = json.loads(article_urls_json)
        except json.JSONDecodeError as exc:
            return f"Invalid JSON for article_urls: {exc}"
        parts = []
        for url in urls:
            text = article_texts.get(url)
            if not text:
                parts.append(f"Source: {url}\n[Not fetched — call fetch_article first]")
            else:
                parts.append(f"Source: {url}\n\n{text[:3000]}")
        combined = "\n\n---\n\n".join(parts)
        messages = [
            ChatMessage(role="system", content=SYNTHESIZE_SYSTEM),
            ChatMessage(role="user", content=SYNTHESIZE_USER.format(topic=topic, articles=combined)),
        ]
        response = _summarizer.chat(messages, max_tokens=1024, temperature=0.4)
        return response.content

    registry.register(ToolDefinition(
        name="list_articles",
        description=(
            "Fetch a news site's homepage and extract a list of article links (title + URL). "
            "Returns a JSON array. Call once per site."
        ),
        parameters=[
            ToolParameter(name="url", type="string", description="Homepage URL to scan for article links"),
        ],
        returns="string",
        permission=PermissionLevel.READ,
        timeout_seconds=60,
        execute=_list_articles,
    ))

    registry.register(ToolDefinition(
        name="fetch_article",
        description=(
            "Fetch the full text of a single article URL for deep analysis. "
            "You MUST call this for each article URL before calling synthesize_topic."
        ),
        parameters=[
            ToolParameter(name="url", type="string", description="Article URL to fetch"),
        ],
        returns="string",
        permission=PermissionLevel.READ,
        timeout_seconds=30,
        execute=_fetch_article,
    ))

    registry.register(ToolDefinition(
        name="synthesize_topic",
        description=(
            "Synthesize multi-source coverage of a topic. "
            "Provide the story topic and a JSON array of article URLs previously fetched with fetch_article. "
            "Returns a synthesized analysis showing different source perspectives."
        ),
        parameters=[
            ToolParameter(name="topic", type="string", description="Story topic or headline"),
            ToolParameter(
                name="article_urls_json",
                type="string",
                description='JSON array of article URLs e.g. ["https://...", "https://..."]',
            ),
        ],
        returns="string",
        permission=PermissionLevel.READ,
        timeout_seconds=60,
        execute=_synthesize_topic,
    ))

    registry.register(ToolDefinition(
        name="send_email",
        description=(
            "Send the deep-dive news analysis email. "
            "You MUST provide both subject and body. "
            "Write the complete formatted digest text directly as the value of 'body' — "
            "do not leave it empty or defer composing it."
        ),
        parameters=[
            ToolParameter(name="subject", type="string", description="Email subject line"),
            ToolParameter(
                name="body",
                type="string",
                description=(
                    "REQUIRED. The complete formatted email body as a plain text string. "
                    "Include all story sections with synthesis and source URLs. Must not be empty."
                ),
            ),
        ],
        returns="string",
        permission=PermissionLevel.DANGEROUS,
        timeout_seconds=10,
        execute=lambda subject, body: _send_email(subject, body),
    ))

    return registry


def _build_memory(db_path: str):
    """Create a MemoryTools instance backed by a SQLite file.

    MemoryTools is the agent-facing API from agent-memory. It wraps a
    MemoryStore (the raw DB layer) and provides high-level operations:
      - store_memory(context, action, outcome, tags)
      - recall_similar(query, limit)  — vector similarity search
      - reflect_on_recent(hours, focus) — LLM-driven reflection

    When passed to AgentInstance(memory=...), two things happen:
      1. The memory tools get registered in the tool registry so the agent
         can call them explicitly during its ReAct loop.
      2. AgentInstance.run() wraps pattern execution with auto-recall
         (before) and auto-store (after). See module docstring for details.

    Returns None if agent-memory is not installed (graceful degradation).
    """
    try:
        from agent_memory import MemoryStore, MemoryTools
    except ImportError:
        print("Warning: agent-memory not installed, running without memory", file=sys.stderr)
        return None

    store = MemoryStore(db_path=db_path)
    return MemoryTools(store=store)


def main():
    parser = argparse.ArgumentParser(description="News digest agent")
    parser.add_argument("--port", type=int, default=7788, help="llama-server port (main agent LLM)")
    parser.add_argument("--summarizer-port", type=int, default=None,
                        help="llama-server port for summarization LLM (optional, defaults to --port)")
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite DB path for memory persistence (e.g. news_memory.db)")
    parser.add_argument("--deep-dive", action="store_true",
                        help="Deep-dive mode: fetch individual articles and synthesize multi-source perspectives")
    parser.add_argument("--check-fetch", metavar="URL",
                        help="Fetch a URL, run html2text, and print the result (no LLM, no agent)")
    args = parser.parse_args()

    if args.check_fetch:
        text = _fetch_readable(args.check_fetch)
        print(f"--- fetch_readable({args.check_fetch}) ---")
        print(f"Length: {len(text)} chars")
        print()
        print(text)
        return

    llm = FireworksLLM(
        model='accounts/fireworks/models/deepseek-v3p2',
        max_tokens=131028,
        extra_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",}
    )
    if not llm.is_available():
        print(f"Error: llama-server not reachable on port {args.port}", file=sys.stderr)
        print("Start it with: llama-server -m model.gguf --port", args.port, file=sys.stderr)
        sys.exit(1)

    # summarizer_llm = None
    # if args.summarizer_port:
    #     summarizer_llm = LlamaCppServerLLM(port=args.summarizer_port, timeout_seconds=120, max_tokens=2048)
    #     if not summarizer_llm.is_available():
    #         print(f"Warning: summarizer llama-server not reachable on port {args.summarizer_port}, "
    #               f"falling back to main LLM", file=sys.stderr)
    #         summarizer_llm = None
    summarizer_llm = FireworksLLM(
        model='accounts/fireworks/models/gpt-oss-120b',
        max_tokens=131000,
        extra_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",}
    )

    config = AgentConfig(react=ReactConfig(max_iterations=30), log_level="DEBUG")
    role = AgentRole.from_soul_file("news-analyst", SCRIPT_DIR / "soul.txt", config=config)

    sites_list = "\n".join(f"- {url}" for url in NEWS_SITES)
    if args.deep_dive:
        registry = build_deep_dive_tools(llm, summarizer_llm)
        task = DEEP_DIVE_TASK.format(
            sites=sites_list,
            date=datetime.now().strftime("%Y-%m-%d"),
        )
    else:
        registry = build_tools(llm, summarizer_llm)
        task = TASK.format(sites=sites_list)

    # Build memory if --db provided. This gives the agent persistence across
    # runs: it will recall relevant past digests and store each run's outcome.
    memory = _build_memory(args.db) if args.db else None

    # Allow all registered tools — we curated this registry ourselves.
    # memory tools (memory_store, memory_recall, memory_reflect) get added
    # automatically by AgentInstance when memory is not None.
    agent = AgentInstance(
        role, llm, registry=registry,
        permission_checker=lambda t: True,
        memory=memory,
    )

    mode = "deep-dive" if args.deep_dive else "digest"
    mem_status = f", memory={args.db}" if args.db else ""
    sum_status = f", summarizer=port {args.summarizer_port}" if summarizer_llm else ""
    print(f"Running news agent [{mode}] (agent=port {args.port}{sum_status}{mem_status})...")
    result = agent.run(task)

    print(f"\nDone — success={result.success}")
    print(f"Summary: {result.summary}")
    if hasattr(result, "iterations"):
        print(f"Iterations: {result.iterations}")


if __name__ == "__main__":
    main()
