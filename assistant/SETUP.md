# Setup Notes

## System packages (Debian/Ubuntu) — required before building Python via asdf

Without these, the asdf Python build silently omits modules, leading to
`ModuleNotFoundError` or segfaults at runtime.

```bash
sudo apt install \
  build-essential \
  libssl-dev \
  libffi-dev \
  libreadline-dev \
  libsqlite3-dev \
  libbz2-dev \
  libncursesw5-dev \
  liblzma-dev \
  zlib1g-dev
```

After installing, rebuild Python:

```bash
asdf install python 3.13.x   # use whatever version you need
asdf reshim python
```

### What each package unlocks

| Package           | Python module / effect                        |
|-------------------|-----------------------------------------------|
| libssl-dev        | `ssl`, HTTPS via `requests`                   |
| libffi-dev        | `ctypes`, `cffi` — needed by many C extensions |
| libreadline-dev   | `readline` — input history / line editing     |
| libsqlite3-dev    | `sqlite3` — TodoDB, Journal, agent-memory     |
| libbz2-dev        | `bz2`                                         |
| libncursesw5-dev  | `curses`, also needed by readline             |
| liblzma-dev       | `lzma` / `xz`                                 |
| zlib1g-dev        | `zlib`, `gzip`                                |
| build-essential   | gcc/make — required to compile any C extension |

## pip packages (beyond pyproject.toml)

```bash
# Optional: web search and page fetching
pip install duckduckgo-search html2text

# Framework packages (install in this order)
pip install -e "../mem"           # agent-memory with ChromaDB+HDBSCAN (server / Pi 3+)
                                     # Pi Zero: skip this — LiteMemory is used automatically
pip install -e "../tools"         # agent-tools
pip install -e "../mind"        # agent-mind
pip install -e "../patterns"    # agent-patterns
pip install -e "../core"        # agent-core
pip install -e "."                 # assistant itself
```

## Key YAML config options

| Key | Default | Description |
|-----|---------|-------------|
| `model` | gpt-oss-120b | Default model for all tasks |
| `models:` | — | Per-task model overrides (conversation, curiosity, inner_voice, compaction, nudge) |
| `db` | assistant_memory.db | SQLite memory DB (omit or `--no-memory` to disable) |
| `data_db` | assistant_data.db | SQLite DB for todos, journal, scheduler, soul proposals |
| `max_chars` | 32000 | Auto-compact context at this size |
| `reflect_interval` | 3600 | Background reflection every N seconds (requires db) |
| `curiosity_interval` | 0 | Background research every N seconds (0 = off) |
| `nudge_interval` | 0 | Soul-consistency check every N seconds (0 = off, requires `## Self-monitoring` in soul) |
| `name` | — | Agent identity name — required for mailbox routing |
| `mailbox_db` | — | Shared SQLite path for inter-agent messaging — see Multi-agent section below |
| `spawn_roles:` | — | Named child-agent role definitions — see Multi-agent section below |
| `sandbox:` | — | Enable `python_exec` tool — see Sandbox section below |
| `mail:` | — | Enable `send_email` tool, jailed to fixed recipients — see Mail section below |
| `work_cycle:` | — | Autonomous ReAct work sessions — see Work cycle section below |
| `log_level` | WARNING | Console log level (DEBUG/INFO/WARNING/ERROR) |

## Multi-agent

The assistant supports two complementary primitives for coordinating with other agents.

### Spawn — synchronous sub-task delegation

An agent can delegate a task to a specialist child agent and wait for the result.
The child runs, returns an answer, then terminates.

**When to use:** "Go research this and bring me a summary." The parent blocks until done.

**Configure in YAML:**

```yaml
name: agent_smith        # parent agent's identity

spawn_roles:
  researcher:
    soul: souls/researcher.txt   # relative to this config file
    tools: [web_search, fetch_readable, save_note]
  critic:
    soul: souls/critic.txt
    tools: []                    # pure reasoning, no tools
  analyst:
    soul: souls/analyst.txt
    tools: [web_search, fetch_readable]
    model: accounts/fireworks/models/deepseek-v3p2   # optional LLM override
```

Once configured, the `spawn_agent` tool is automatically registered. The agent calls:

```
spawn_agent(role="researcher", task="Summarise recent papers on attention mechanisms")
```

Child tools are restricted to those declared in the role definition. The parent can
narrow further at spawn time but cannot grant tools not listed in the role.
Dangerous tools (`python_exec`, shell) are never auto-inherited.

### Mailbox — async peer-to-peer messaging

Two independently-running agents communicate via a shared SQLite database.
Messages persist until read; neither agent blocks.

**When to use:** Ongoing coordination between agents that run simultaneously —
e.g. Smith-Actor sends drafts to Smith-Critic, which queues feedback asynchronously.

**Configure in YAML (each agent's config file):**

```yaml
# agent_smith.yaml
name: agent_smith
mailbox_db: /shared/agents_mailbox.db   # absolute path, same file for all agents

# agent_critic.yaml
name: agent_critic
mailbox_db: /shared/agents_mailbox.db   # same DB, different name
```

Both `name` and `mailbox_db` must be set. Use an **absolute path** so agents
launched from different directories all find the same file.

Once configured, three tools are automatically registered:

| Tool | Description |
|------|-------------|
| `send_message(to, message, topic="")` | Send a message to another agent by name |
| `check_inbox(unread_only=True)` | List received messages (● = unread, ○ = read) |
| `reply_to_message(msg_id, message)` | Reply to a specific message (threads preserved) |

Unread messages are also **injected as background notes** at the start of each turn,
so the agent sees them without needing to call `check_inbox` explicitly.

### Spawn vs Mailbox at a glance

| | Spawn | Mailbox |
|---|---|---|
| Direction | Parent → Child | Peer ↔ Peer |
| Blocking | Yes — parent waits for result | No — fire-and-forget |
| Child lifetime | Task duration only | N/A — both agents persist |
| Good for | Sub-task delegation | Ongoing coordination |

## Work cycle (autonomous ReAct sessions)

The assistant's autonomous mode. Every `interval` seconds it picks a goal and runs
a bounded reason→act→observe loop (agent-patterns `ReactLoop`) over its full tool
registry — the agent decides which tools to use and when it's done, instead of
following a fixed pipeline. **Disabled by default.**

Goal sources, rotated so no source starves the others:
1. **Scheduled prompts** (daemon mode) — `schedule_task` prompts run as one-shot goals
2. **Pending TODOs** — pick one, make concrete progress, `todo_note` / `todo_done`
3. **Research agenda** — active topics via `research_focus`, findings journaled
4. **Soul interests** — round-robin over `## Research interests`
5. **Dream/replay** (requires memory) — two memories with disjoint topics are
   sampled; the agent judges whether a genuine connection exists and saves it
   (tagged `association`) — "unrelated" is an accepted outcome

```yaml
work_cycle:
  interval: 3600        # 0/absent = off
  max_iterations: 8     # ReAct steps per cycle
```

Outcomes are journaled (tag `work-cycle`), surfaced as background notes in the next
conversation turn, and fed through InnerVoice. `compact_context` and
`decide_soul_proposal` are excluded from autonomous use (the latter so the agent
can never approve its own soul proposals).

### Daemon mode (headless)

```bash
python main.py --config souls/ada.yaml --daemon
```

No interactive input: the agent is driven entirely by the work cycle, curiosity,
reflection, and scheduled tasks. Scheduled prompts are executed through the work
cycle rather than injected into a conversation. Combine with the `mail:` section
so the agent can reach you (e.g. a cron-scheduled "email a digest of today's
journal to nikos"). Stop with SIGTERM or Ctrl-C — suitable for a systemd service.

## Mail (send_email tool)

Jailed outbound email. **Disabled by default.** The agent addresses recipients by
alias only; the alias → address map is fixed in the YAML, so it can never mail an
arbitrary address. Subject newlines are stripped (no header injection), the From:
header is fixed, and a daily send cap applies. Delivery is via the local sendmail
binary with a fixed argument list (`sendmail -t -i`, `shell=False`) — postfix or
any sendmail-compatible MTA works.

```yaml
mail:
  recipients:
    nikos: nikos@example.org        # agent says "nikos", never the address
  from: ada@myserver.example.org
  subject_prefix: "[ada]"           # optional
  max_per_day: 20                   # default 20
  sendmail_path: /usr/sbin/sendmail # default
```

Requires a working local MTA (same setup as the news-agent digest — see
`examples/news-agent/README.md` for postfix notes).

## Sandbox (python_exec tool)

Lets the agent write and run Python code. **Disabled by default.** Opt-in per agent
via the `sandbox:` YAML section.

### 1. Install firejail

```bash
sudo apt install firejail
```

Verify:
```bash
firejail --version
```

### 2. Create a dedicated unix user (recommended)

The agent runs as this user. Its home directory becomes the sandbox boundary.

```bash
sudo useradd -m -s /bin/bash agent_smith
sudo mkdir -p /home/agent_smith/work
sudo chown agent_smith:agent_smith /home/agent_smith/work
sudo chmod 755 /home/agent_smith/work   # assistant process (nvlass) needs traverse access
```

Install Python packages the agent needs as that user:

```bash
sudo -u agent_smith pip install numpy scipy   # or whatever is needed
```

### 3. Passwordless sudo for the agent's main user (optional)

If you run the *assistant process* as a different user (e.g. `nvlass`) but want
it to execute code as `agent_smith`, add a targeted sudoers rule:

```bash
sudo visudo -f /etc/sudoers.d/agent_smith
```

Add (replace `nvlass` with your username):

```
nvlass ALL=(agent_smith) NOPASSWD: /usr/bin/firejail
```

This allows only `firejail` to be run as `agent_smith`, nothing else.

### 4. Configure the YAML

```yaml
sandbox:
  work_dir: /home/agent_smith/work
  timeout: 120          # seconds per execution
  use_firejail: true
  unix_user: agent_smith   # omit if already running as agent_smith
```

### 5. What the sandbox prevents

| Threat | Mitigated? |
|--------|-----------|
| Network access | Yes (`--net=none`) |
| Reading other users' files | Yes (`--private=work_dir`) |
| Privilege escalation | Yes (`--noroot`) |
| Polluting /tmp | Yes (`--private-tmp`) |
| Writing outside work_dir | Yes (firejail home isolation) |
| Unlimited CPU/time | Yes (timeout parameter) |

### 6. What the sandbox does NOT prevent

- Computation-heavy loops consuming all CPU until timeout
- Writing large files inside work_dir (add a quota if needed: `sudo setquota -u agent_smith ...`)
- Reading files the unix user already has access to inside work_dir

## Known issues

- **readline segfault**: caused by `pip install readline` (third-party package).
  Uninstall it — Python uses its built-in readline when `libreadline-dev` was
  present at build time. Without it, readline is simply unavailable (no history),
  but the assistant runs fine.

- **`_ctypes` missing**: `libffi-dev` was not installed before the Python build.
  Fix: `sudo apt install libffi-dev && asdf install python <version>`.

- **ChromaDB/HDBSCAN on ARM (Pi Zero)**: these packages compile native extensions
  and are painful to build on ARMv6 / 512 MB RAM. Run the assistant without
  `--db` to avoid them entirely — all other features work fine.
