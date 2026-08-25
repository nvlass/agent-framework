# News Digest Agent

A daily news digest agent that fetches news sites, identifies top stories, and emails a concise summary. First real example exercising agent-core, agent-tools, and agent-patterns end-to-end.

## Prerequisites

1. **llama-server** running with a model that supports function calling (Qwen 2.5 7B+ recommended):
   ```bash
   llama-server -m ~/models/your-model.gguf --port 7788
   ```

2. **sendmail** available (for sending email):
   ```bash
   # Linux: sudo apt install postfix
   # macOS: postfix is built-in, just needs configuration
   ```

3. **Framework packages** installed:
   ```bash
   cd /path/to/agents
   pip install -e core/ -e tools/ -e mind/ -e patterns/ -e mem/
   pip install html2text
   ```

## How It Works

The agent uses **native tool calling** via llama-server's OpenAI-compatible `tools`/`tool_choice` API. Tool schemas are passed in the API request rather than as text in the prompt, so the model returns structured `tool_calls` responses instead of fragile `Action:`/`Action Args:` text.

If the LLM doesn't return `tool_calls` (e.g. when using the CLI backend), the reasoner falls back to text parsing automatically.

## Configuration

- **News sites**: Edit `NEWS_SITES` list in `news_agent.py`
- **Recipient**: Edit `RECIPIENT` in `send_email.sh`
- **LLM port**: Pass `--port` flag (default: 7788)

## Usage

```bash
python news_agent.py
python news_agent.py --port 8080

# Enable memory — agent remembers previous runs
python news_agent.py --db news_memory.db

# Debug logging to see native tool calls and memory operations
AGENT_LOG_LEVEL=DEBUG python news_agent.py --db news_memory.db
```

## Memory

When `--db` is provided, the agent uses agent-memory for persistence across runs:

- **Auto-recall**: Before each run, the 3 most relevant past episodes are injected into the prompt. The agent sees them as "Relevant past experiences" and can use them to notice recurring stories or avoid past mistakes.
- **Auto-store**: After each run, an episode is saved with the task, iteration count, and outcome.
- **Explicit tools**: The agent can also call `memory_store`, `memory_recall`, and `memory_reflect` mid-run if it decides to.

The DB file is a SQLite database created automatically on first run.

## Scheduling (cron)

```bash
# Run daily at 8am (with memory for cross-day story tracking)
0 8 * * * cd /path/to/agents/examples/news-agent && python news_agent.py --db news_memory.db >> /tmp/news-agent.log 2>&1
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ConnectionError` on startup | Start llama-server first |
| `sendmail: command not found` | Install/configure postfix |
| Agent loops without sending email | Model may be too small — try a larger model |
| Digest is too long / unfocused | Tweak `soul.txt` or reduce `max_iterations` |
| No "Native tool call" in debug logs | Model may not support function calling — falls back to text parsing |

## Architecture

```
news_agent.py
├── fetch_readable tool (READ) — fetches news sites, converts to markdown
├── send_email tool (DANGEROUS) — sends digest via sendmail (stdin)
├── soul.txt — agent personality and guidelines
├── MemoryTools (optional) — persistence via agent-memory SQLite DB
│   ├── auto-recall before run → injected as observations
│   └── auto-store after run → episode saved
└── AgentInstance (ReactLoop) — reason → act → observe cycle
    └── LLMReasoner — native tool calling with text parsing fallback
```

## How to Create New Tools

There are two ways to define tools for an agent:

### 1. `create_command_tool` — simple shell commands

Best for tools that run a fixed command with safe, predictable parameters (URLs, filenames, flags):

```python
from agent_tools.tools.shell import create_command_tool
from agent_tools.core.definition import ToolParameter

curl_tool = create_command_tool(
    name="curl_fetch",
    command=["curl", "-s", "-L", "-m", "15", "{url}"],
    parameters=[
        ToolParameter(name="url", type="string", description="URL to fetch"),
    ],
    description="Fetch a URL and return its contents",
    timeout_seconds=20,
)
```

`create_command_tool` validates all parameters against dangerous shell characters (`;|&$<>` etc.) to prevent injection. This is great for safety, but means **parameters cannot contain free-form text** like email bodies, code snippets, or user content with special characters.

### 2. `ToolDefinition` with a Python function — full control

When your tool needs to handle arbitrary text, use stdin/pipes, or do anything beyond a simple command:

```python
import subprocess
from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel

def _send_email(subject: str, body: str) -> str:
    """Send email by piping the full message to sendmail via stdin."""
    message = (
        f"To: recipient@example.com\n"
        f"Subject: {subject}\n"
        f"From: agent@example.com\n"
        f"\n"
        f"{body}"
    )
    result = subprocess.run(
        ["/usr/sbin/sendmail", "-t"],
        input=message,          # body goes via stdin, not args
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return f"sendmail failed: {result.stderr}"
    return "Email sent successfully"

email_tool = ToolDefinition(
    name="send_email",
    description="Send an email digest",
    parameters=[
        ToolParameter(name="subject", type="string", description="Email subject"),
        ToolParameter(name="body", type="string", description="Email body text"),
    ],
    returns="string",
    permission=PermissionLevel.DANGEROUS,
    timeout_seconds=10,
    execute=lambda subject, body: _send_email(subject, body),
)
```

Key advantages over `create_command_tool`:
- **No character restrictions** — body can contain bullets, angle brackets, anything
- **No shell** — `subprocess.run` with a list uses `exec` directly, no injection possible
- **stdin support** — pass large/complex data via `input=` instead of command-line args
- **Full Python** — error handling, retries, multiple steps, whatever you need

### Which to choose?

| Scenario | Use |
|----------|-----|
| Simple command, safe params (URLs, paths) | `create_command_tool` |
| Free-form text in parameters | `ToolDefinition` + Python function |
| Need stdin, pipes, or complex logic | `ToolDefinition` + Python function |
| Read-only, no side effects | Either, set `permission=READ` |
| Side effects (send, write, delete) | Either, set `permission=DANGEROUS` |

Both approaches register identically in the `ToolRegistry` and look the same to the agent.

## Deferred to v2

- Multiple output formats (file, stdout)
