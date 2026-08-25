# agent-tools

Unified abstraction for agent tool use.

## Status

**Phase:** Initial development
**Location:** `./tools/`

## Overview

agent-tools provides a consistent interface for agents to interact with the world. It handles:
- Tool definition and registration
- Execution with permissions and sandboxing
- Result validation and error handling
- Async support for long-running operations

## Design Decisions

From the master roadmap, these decisions are resolved:

### 1. Permissions: Per-tool granularity
- Each tool declares its own permission requirements
- Tools can be individually enabled/disabled/sandboxed
- Permission check happens before execution

### 2. Execution: Sync by default, async-capable design
- Synchronous execution as the default (simpler mental model)
- Tool definition includes `async_capable: bool` flag
- Both `execute()` and `execute_async()` methods available
- Interface designed to support async variants

### 3. Long-running tools: Async with process monitoring
- Long-running tools use async execution
- Process health monitoring (is it still alive?)
- Support for streaming output as it arrives
- Timeout handling without blocking the agent

### 4. Tool composition: Start simple, agent-decides strategy
- Initial implementation: simple sequential execution
- Architecture leaves space for both pipelines and chaining
- The agent decides which composition strategy based on context
- Avoid premature abstraction; let patterns emerge

---

## Core Interfaces

### Tool Definition

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

class PermissionLevel(Enum):
    """Permission levels for tool execution."""
    SAFE = "safe"           # No side effects, always allowed
    READ = "read"           # Reads system state, low risk
    WRITE = "write"         # Modifies state, requires approval
    DANGEROUS = "dangerous" # System-level operations, explicit consent

@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str                          # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list] = None        # Allowed values

@dataclass
class ToolDefinition:
    """Complete definition of a tool."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: str = "string"            # Return type description
    permission: PermissionLevel = PermissionLevel.SAFE
    async_capable: bool = False
    timeout_seconds: int = 30

    # Implementation
    execute: Optional[Callable] = None
    execute_async: Optional[Callable] = None

    def to_schema(self) -> dict:
        """Export as JSON schema (OpenAI function calling format)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                        **({"enum": p.enum} if p.enum else {}),
                        **({"default": p.default} if p.default is not None else {}),
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }
```

### Tool Registry

```python
class ToolRegistry:
    """Registry for discovering and managing tools."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, permission_level: Optional[PermissionLevel] = None) -> list[ToolDefinition]:
        """List all tools, optionally filtered by permission level."""
        tools = list(self._tools.values())
        if permission_level:
            tools = [t for t in tools if t.permission == permission_level]
        return tools

    def to_schemas(self) -> list[dict]:
        """Export all tools as JSON schemas for LLM."""
        return [t.to_schema() for t in self._tools.values()]
```

### Tool Executor

```python
@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0

class ToolExecutor:
    """Executes tools with permission checking and error handling."""

    def __init__(self, registry: ToolRegistry, permission_checker: Callable[[ToolDefinition], bool]):
        self.registry = registry
        self.permission_checker = permission_checker

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool synchronously."""
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(success=False, output=None, error=f"Unknown tool: {tool_name}")

        if not self.permission_checker(tool):
            return ToolResult(success=False, output=None, error=f"Permission denied: {tool_name}")

        try:
            start = time.time()
            result = tool.execute(**kwargs)
            duration = int((time.time() - start) * 1000)
            return ToolResult(success=True, output=result, duration_ms=duration)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def execute_async(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool asynchronously."""
        # Similar pattern with async/await
        pass
```

---

## Built-in Tool Categories

### Security by Default

**DANGEROUS tools are never included by default.** They must be explicitly requested
when creating an agent.

**Template-based commands, not string filtering.**
Text-based command filtering (allowlists, prefix matching) is fundamentally broken
as a security model — the attack surface is infinite (variable expansion, command
substitution, pipe injection, encoding tricks, heredocs, etc.). Instead, we use
fixed command templates with `subprocess(shell=False)`, which eliminates the entire
class of injection attacks by design.

```python
# Example: Creating an agent with template-based command tools
registry = ToolRegistry()

# Safe defaults are auto-registered
registry.register_defaults()  # read_file, list_directory, get_env, etc.

# Dangerous tools must be explicitly added as fixed command templates
registry.register(create_command_tool(
    name="git_fetch",
    command=["git", "fetch"],
))

registry.register(create_command_tool(
    name="git_checkout",
    command=["git", "checkout", "{branch}"],
    parameters=[
        ToolParameter(name="branch", type="string", description="Branch name"),
    ],
))

registry.register(create_command_tool(
    name="run_pytest",
    command=["pytest", "{path}"],
    parameters=[
        ToolParameter(name="path", type="string", description="Test path"),
    ],
))

# The agent calls: tool.execute(branch="main")
# Runs: subprocess.run(["git", "checkout", "main"], shell=False)
# No shell interpretation, no injection possible.
```

### Default Tools (auto-registered)

These are safe by default and included in the default registry:

**System Tools (READ/SAFE):**
- `read_file` — Read file contents (READ)
- `list_directory` — List directory contents (READ)
- `get_env` — Read environment variable (READ)

**Code Tools (SAFE):**
- `syntax_check` — Validate code syntax (SAFE)

**Memory Tools (bridge to agent-memory):**
- `memory_store` — Store an episode (WRITE)
- `memory_recall` — Retrieve relevant memories (READ)
- `memory_reflect` — Trigger reflection (WRITE)

**Communication Tools (SAFE):**
- `notify_user` — Send notification to user (SAFE)
- `ask_user` — Request input from user (SAFE)

### Explicit-Only Tools (DANGEROUS)

These are **never auto-registered**. They must be explicitly created as templates:

**Command Execution:**
- `create_command_tool(name, command, parameters)` — Factory for template-based command tools
- Command is a fixed list of strings with optional `{param}` placeholders
- `subprocess.run(shell=False)` — no shell interpretation, no injection possible
- Example: `create_command_tool("git_fetch", ["git", "fetch"])` runs exactly `git fetch`
- Example: `create_command_tool("git_checkout", ["git", "checkout", "{branch}"], ...)` — agent fills in branch

**Code Execution:**
- `create_python_tool(name, allowed_modules, sandbox)` — Factory for scoped Python execution
- Can restrict imports, enable sandbox mode
- Example: `create_python_tool("data_analysis", allowed_modules=["pandas", "numpy"])`

**File Writing:**
- `write_file` — Write to file (WRITE) — must be explicitly registered
- Can be scoped to specific directories: `create_write_tool(allowed_paths=["/tmp", "./output"])`

### Example: Test Agent Configuration

```python
def create_test_agent_tools() -> ToolRegistry:
    """Create tools for the test agent with minimal permissions."""
    registry = ToolRegistry()
    registry.register_defaults()  # Safe tools only

    # Each command is a fixed template — no shell injection possible

    # Git operations (each is a separate tool)
    registry.register(create_command_tool(
        name="git_fetch", command=["git", "fetch"],
    ))
    registry.register(create_command_tool(
        name="git_pull", command=["git", "pull"],
    ))
    registry.register(create_command_tool(
        name="git_status", command=["git", "status"],
    ))
    registry.register(create_command_tool(
        name="git_diff", command=["git", "diff"],
    ))
    registry.register(create_command_tool(
        name="git_checkout",
        command=["git", "checkout", "{branch}"],
        parameters=[
            ToolParameter(name="branch", type="string", description="Branch name"),
        ],
    ))

    # Test runners
    registry.register(create_command_tool(
        name="run_pytest",
        command=["pytest", "{path}"],
        parameters=[
            ToolParameter(name="path", type="string", description="Test path"),
        ],
    ))
    registry.register(create_command_tool(
        name="run_clj_tests", command=["clj", "-A:test-ci"],
    ))

    # Read-only file access is already in defaults
    # No write_file, no python_exec, no generic shell access

    return registry
```

---

## Example Tool: Task Queue

This is a working example of a tool implementation. Use it as a reference for creating new tools and as the foundation for the test agent integration.

### Purpose

A SQLite-based task queue for dispatching work to agents and retrieving results. This enables Claude Code to dispatch testing tasks to a test agent.

### Schema

```sql
-- tools/agent_tools/task_queue_schema.sql

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    result TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
```

### Implementation Sketch

```python
# tools/agent_tools/tools/task_queue.py

import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Task:
    id: int
    task: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class TaskQueue:
    """SQLite-based task queue for agent work dispatch."""

    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)  # Load from schema file

    def add(self, task: str) -> int:
        """Add a task to the queue. Returns task ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO tasks (task) VALUES (?)",
                (task,)
            )
            return cursor.lastrowid

    def get(self, task_id: int) -> Optional[Task]:
        """Get a task by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,)
            ).fetchone()
            return Task(**dict(row)) if row else None

    def list_pending(self) -> List[Task]:
        """List all pending tasks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
            return [Task(**dict(row)) for row in rows]

    def claim_next(self) -> Optional[Task]:
        """Claim the next pending task for execution."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Atomic claim with UPDATE ... RETURNING
            row = conn.execute("""
                UPDATE tasks
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = (SELECT id FROM tasks WHERE status = 'pending' ORDER BY created_at LIMIT 1)
                RETURNING *
            """).fetchone()
            return Task(**dict(row)) if row else None

    def complete(self, task_id: int, result: str) -> None:
        """Mark a task as completed with result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks
                SET status = 'completed', result = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (result, task_id))

    def fail(self, task_id: int, error: str) -> None:
        """Mark a task as failed with error."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks
                SET status = 'failed', error = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (error, task_id))
```

### Tool Definitions

```python
# Register task queue operations as tools

task_add_tool = ToolDefinition(
    name="task_add",
    description="Add a task to the queue for agent execution",
    parameters=[
        ToolParameter(name="task", type="string", description="Task description"),
    ],
    returns="integer (task ID)",
    permission=PermissionLevel.WRITE,
    execute=lambda task: queue.add(task),
)

task_status_tool = ToolDefinition(
    name="task_status",
    description="Get the status of a queued task",
    parameters=[
        ToolParameter(name="task_id", type="integer", description="Task ID"),
    ],
    returns="object with status, result, error",
    permission=PermissionLevel.READ,
    execute=lambda task_id: queue.get(task_id),
)

task_list_tool = ToolDefinition(
    name="task_list",
    description="List tasks by status",
    parameters=[
        ToolParameter(name="status", type="string", description="Filter by status",
                     required=False, enum=["pending", "running", "completed", "failed"]),
    ],
    returns="array of tasks",
    permission=PermissionLevel.READ,
)
```

### CLI Interface

```bash
# Add a task
python -m agent_tools.cli task add "run pytest on mem/tests/ and report failures"
# Output: Task 1 created

# List pending tasks
python -m agent_tools.cli task list --status pending

# Check status
python -m agent_tools.cli task status 1

# Get result
python -m agent_tools.cli task result 1
```

---

## Future Integration Patterns

### MCP Server (future)

When ready for tighter Claude Code integration, expose tools via MCP:

```python
# agent_tools/mcp_server.py

from mcp import Server, Tool

def create_mcp_server(registry: ToolRegistry) -> Server:
    """Create MCP server from tool registry."""
    server = Server("agent-tools")

    for tool in registry.list_tools():
        server.add_tool(Tool(
            name=tool.name,
            description=tool.description,
            schema=tool.to_schema()["parameters"],
            handler=lambda **kwargs: executor.execute(tool.name, **kwargs),
        ))

    return server
```

This allows Claude Code to use tools natively without CLI shelling.

### Real-time Notifications (future)

Instead of polling for task completion:

```python
# Agent runner notifies on completion
def on_task_complete(task: Task):
    # Write to a notification file Claude Code watches
    # Or use filesystem events
    # Or websocket push
    pass
```

### Priority Queue (future)

```sql
ALTER TABLE tasks ADD COLUMN priority INTEGER DEFAULT 0;
CREATE INDEX idx_tasks_priority ON tasks(priority DESC, created_at);
```

---

## Project Structure

```
tools/
├── CLAUDE.md                    # This file
├── pyproject.toml
├── agent_tools/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── definition.py        # ToolDefinition, ToolParameter
│   │   ├── registry.py          # ToolRegistry
│   │   ├── executor.py          # ToolExecutor, ToolResult
│   │   └── permissions.py       # PermissionLevel, permission checking
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── system.py            # System tools (run_command, read_file, etc.)
│   │   ├── code.py              # Code tools (python_exec, syntax_check)
│   │   ├── memory.py            # Memory bridge tools
│   │   └── task_queue.py        # Task queue tool (example)
│   ├── cli.py                   # CLI interface
│   └── schema.sql               # Task queue schema
└── tests/
    ├── __init__.py
    ├── test_definition.py
    ├── test_registry.py
    ├── test_executor.py
    └── test_task_queue.py
```

---

## Development Phases

### Phase 1: Core Infrastructure ✅
- [x] ToolDefinition, ToolParameter dataclasses
- [x] ToolRegistry with register/get/list/register_defaults
- [x] ToolExecutor with permission checking
- [x] 99 tests passing

### Phase 2: Built-in Tools ✅ (partial)
- [x] System tools (read_file, list_directory, get_env)
- [x] Code tools (syntax_check)
- [x] Shell tool factory (create_command_tool — template-based, DANGEROUS)
- [x] Memory bridge (create_memory_tools factory)

### Phase 3: Task Queue Example (deferred)
- [ ] SQLite schema and TaskQueue class
- [ ] Task queue tool definitions
- [ ] CLI for task operations
- [ ] Tests for task queue

### Phase 4: Async Support (future)
- [ ] Async executor
- [ ] Long-running tool support
- [ ] Process monitoring

### Phase 5: Integration (future)
- [ ] MCP server (optional)
- [ ] Agent runner for task queue
- [ ] Documentation and examples

---

## Notes

- Follow the separation & overridability principle: each part should be replaceable
- Use dependency injection, no global state
- Type hints everywhere
- Tests for each component

---

**Last Updated:** 2026-02-08
**Status:** Core complete, built-in tools done, memory bridge next
**Developers:** Nikos & Claude
