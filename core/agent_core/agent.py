"""AgentRole and AgentInstance — the core agent abstraction."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agent_mind.goals.model import Goal, GoalState
from agent_patterns.base import PatternResult
from agent_patterns.context import SharedContext
from agent_patterns.react import ReactLoop
from agent_tools.core.definition import PermissionLevel
from agent_tools.core.executor import ToolExecutor
from agent_tools.core.registry import ToolRegistry

from agent_core.config import AgentConfig
from agent_core.llm import ChatLLMInterface
from agent_core.prompt import PromptAssembler
from agent_core.reasoner import LLMReasoner
from agent_core.soul import Soul, SoulManager
from agent_core.spawn import SpawnRegistry

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def _configure_logging(level_str: str) -> None:
    """Configure logging for all agent framework packages.

    Resolves AGENT_LOG_LEVEL env var (highest priority), then uses level_str.
    Adds a StreamHandler only if none are already configured.
    """
    env_level = os.environ.get("AGENT_LOG_LEVEL", "").strip().upper()
    level_name = env_level if env_level else level_str.upper()
    level = getattr(logging, level_name, logging.WARNING)

    for pkg in ("agent_core", "agent_patterns", "agent_tools"):
        pkg_logger = logging.getLogger(pkg)
        pkg_logger.setLevel(level)
        if not pkg_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            pkg_logger.addHandler(handler)


def _default_permission_checker(tool_def) -> bool:
    """Allow SAFE, READ, and WRITE tools by default."""
    return tool_def.permission in (
        PermissionLevel.SAFE, PermissionLevel.READ, PermissionLevel.WRITE,
    )


@dataclass
class AgentRole:
    """Template for an agent — name, soul, default config."""

    name: str
    soul: str = ""
    config: AgentConfig = field(default_factory=AgentConfig)
    soul_manager: Optional[SoulManager] = field(default=None, repr=False)

    @classmethod
    def from_soul_file(
        cls,
        name: str,
        path: str | Path,
        config: Optional[AgentConfig] = None,
    ) -> "AgentRole":
        """Create a role by loading soul text from a single file (immutable only)."""
        soul = Path(path).read_text()
        return cls(name=name, soul=soul, config=config or AgentConfig())

    @classmethod
    def from_soul_dir(
        cls,
        name: str,
        soul_dir: str | Path,
        config: Optional[AgentConfig] = None,
    ) -> "AgentRole":
        """Create a role from a soul directory (SOUL_IMMUTABLE.md + SOUL_LEARNABLE.md).

        The directory should contain:
            SOUL_IMMUTABLE.md  — core identity, never modified by agent
            SOUL_LEARNABLE.md  — operational guidelines, agent can propose changes

        A SoulManager is attached so the agent can call propose_soul_change.
        """
        manager = SoulManager(Path(soul_dir))
        soul_obj = manager.load()
        return cls(
            name=name,
            soul=soul_obj.merged,
            config=config or AgentConfig(),
            soul_manager=manager,
        )


class AgentInstance:
    """A running agent instance wired from role + LLM + tools."""

    def __init__(
        self,
        role: AgentRole,
        llm: ChatLLMInterface,
        registry: Optional[ToolRegistry] = None,
        assembler: Optional[PromptAssembler] = None,
        permission_checker: Optional[callable] = None,
        memory=None,
        planner=None,
        spawn_registry: Optional[SpawnRegistry] = None,
    ) -> None:
        _configure_logging(role.config.log_level)
        self._role = role
        self._llm = llm

        if registry is not None:
            self._registry = registry
        else:
            self._registry = ToolRegistry()
            self._registry.register_defaults()

        self._planner = planner
        self._memory = memory
        if memory is not None:
            from agent_tools.tools.memory import create_memory_tools
            for tool in create_memory_tools(memory).values():
                self._registry.register(tool)

        self._assembler = assembler or PromptAssembler()
        self._permission_checker = permission_checker or _default_permission_checker
        self._spawn_registry = spawn_registry

        if role.soul_manager is not None:
            self._register_soul_tools(role.soul_manager)

        if spawn_registry:
            self._register_spawn_tool(spawn_registry)

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def run(self, task: str) -> PatternResult:
        """Run a task using the configured pattern.

        Creates a Goal, builds a SharedContext, constructs the pattern,
        and executes it.
        """
        config = self._role.config
        errors = config.validate()
        if errors:
            logger.warning("Invalid config: %s", errors)
            return PatternResult(
                success=False,
                summary=f"Invalid config: {'; '.join(errors)}",
            )

        logger.info("Agent %r running task: %s", self._role.name, task)

        goal = Goal(description=task, state=GoalState.ACTIVE)
        context = SharedContext(goal=goal)

        # Auto-recall relevant memories before execution
        if self._memory is not None:
            try:
                recall_result = self._memory.recall_similar(query=task, limit=3)
                if recall_result.success and recall_result.data:
                    context.observations.append(
                        f"Relevant past experiences:\n{recall_result.data}"
                    )
                    logger.debug("Injected %s recalled memories", len(recall_result.data) if isinstance(recall_result.data, list) else 1)
            except Exception:
                logger.warning("Auto-recall failed", exc_info=True)

        pattern = self._build_pattern(config)
        result = pattern.run(context)
        logger.info("Agent %r finished: success=%s, iterations=%s",
                     self._role.name, result.success, result.iterations)

        # Auto-store episode after execution
        if self._memory is not None:
            try:
                self._memory.store_memory(
                    context=task,
                    action=f"Ran {result.iterations} iterations",
                    outcome=result.summary or ("Success" if result.success else "Failed"),
                    tags=["auto"],
                )
                logger.debug("Auto-stored episode for task")
            except Exception:
                logger.warning("Auto-store failed", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def spawn(
        self,
        role_name: str,
        task: str,
        allow_tools: list[str] | None = None,
    ) -> str:
        """Delegate a task to a child agent and return its result summary.

        The child agent is built from the named ``SpawnRole`` in the spawn
        registry.  It shares the parent's LLM (unless the role overrides it)
        and gets only the tools declared in the role (optionally narrowed by
        ``allow_tools``).

        Args:
            role_name:   Key in the spawn registry (e.g. ``"researcher"``).
            task:        Task description passed to the child's ``run()``.
            allow_tools: Optional override — restrict to this subset of the
                         role's declared tools.  ``None`` = use role defaults.

        Returns:
            The child's ``PatternResult.summary`` string, or an error message.
        """
        if self._spawn_registry is None:
            return "Error: no spawn registry configured on this agent"

        role = self._spawn_registry.get(role_name)
        if role is None:
            available = ", ".join(self._spawn_registry.names()) or "none"
            return f"Error: unknown role {role_name!r}. Available: {available}"

        child_registry = self._build_child_registry(role, allow_tools)
        child_llm = role.llm if role.llm is not None else self._llm
        child_role = AgentRole(name=role.name, soul=role.soul)
        child = AgentInstance(child_role, child_llm, registry=child_registry)

        logger.info("Spawning child %r for task: %s", role_name, task[:80])
        result = child.run(task)
        logger.info("Child %r done: success=%s summary=%r", role_name, result.success, result.summary)
        return result.summary or ("done" if result.success else "failed")

    def _build_child_registry(self, spawn_role, allow_tools: list[str] | None) -> ToolRegistry:
        """Build a ToolRegistry for the child containing only allowed tools."""
        allowed = set(allow_tools if allow_tools is not None else spawn_role.tools)
        child_reg = ToolRegistry()
        for tool_name in allowed:
            tool = self._registry.get(tool_name)
            if tool is not None:
                child_reg.register(tool)
            else:
                logger.warning(
                    "Spawn role %r: tool %r not in parent registry — skipped",
                    spawn_role.name, tool_name,
                )
        return child_reg

    def _register_spawn_tool(self, spawn_registry: SpawnRegistry) -> None:
        """Register a spawn_agent tool so the agent can delegate via tool calls."""
        from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel

        names_str = ", ".join(spawn_registry.names())

        def _spawn(role: str, task: str) -> str:
            return self.spawn(role, task)

        self._registry.register(ToolDefinition(
            name="spawn_agent",
            description=(
                f"Delegate a task to a specialist sub-agent and get the result. "
                f"Available roles: {names_str}. "
                "Use this when a task is best handled by a specialist."
            ),
            parameters=[
                ToolParameter(
                    name="role",
                    type="string",
                    description=f"Role to spawn. One of: {names_str}",
                ),
                ToolParameter(
                    name="task",
                    type="string",
                    description="Complete task description for the child agent",
                ),
            ],
            returns="string",
            permission=PermissionLevel.SAFE,
            execute=_spawn,
        ))

    # ------------------------------------------------------------------
    # Soul
    # ------------------------------------------------------------------

    def reload_soul(self) -> None:
        """Reload soul from disk (picks up approved proposals)."""
        if self._role.soul_manager is None:
            return
        soul_obj = self._role.soul_manager.load()
        self._role.soul = soul_obj.merged
        if hasattr(self, '_reasoner'):
            self._reasoner._soul = soul_obj.merged

    def _register_soul_tools(self, manager: SoulManager) -> None:
        """Register propose_soul_change tool so agent can propose learnable changes."""
        from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel

        def _propose(section: str, proposed: str, reasoning: str) -> str:
            current = manager.load().learnable
            p = manager.propose_change(
                section=section,
                current=current,
                proposed=proposed,
                reasoning=reasoning,
            )
            return (
                f"Proposal {p.id!r} recorded (status: pending). "
                f"A human must approve it before it takes effect. "
                f"Run: python -m agent_core.soul_cli approve {p.id}"
            )

        self._registry.register(ToolDefinition(
            name="propose_soul_change",
            description=(
                "Propose a change to your operational guidelines (SOUL_LEARNABLE). "
                "Changes require human approval and do not take effect immediately. "
                "Use when you notice a pattern that would improve your performance."
            ),
            parameters=[
                ToolParameter(
                    name="section",
                    type="string",
                    description="Label for what you're changing (e.g. 'verbosity', 'tool_use_strategy')",
                ),
                ToolParameter(
                    name="proposed",
                    type="string",
                    description="The full proposed new content for SOUL_LEARNABLE.md",
                ),
                ToolParameter(
                    name="reasoning",
                    type="string",
                    description="Why this change would improve your performance",
                ),
            ],
            returns="string",
            permission=PermissionLevel.WRITE,
            execute=_propose,
        ))

    def _build_pattern(self, config: AgentConfig):
        """Construct the execution pattern from config."""
        soul = self._role.soul or config.soul

        reasoner = LLMReasoner(
            llm=self._llm,
            assembler=self._assembler,
            soul=soul,
        )

        executor = ToolExecutor(
            registry=self._registry,
            permission_checker=self._permission_checker,
        )

        logger.info("Building pattern: %s", config.pattern)
        if config.pattern == "react":
            return ReactLoop(
                tool_executor=executor,
                reasoner=reasoner,
                max_iterations=config.react.max_iterations,
            )

        elif config.pattern == "plan_and_execute":
            from agent_patterns.plan_and_execute import PlanAndExecute
            if self._planner:
                planner = self._planner
            else:
                from agent_mind.planning.planner import LLMPlanner
                from agent_core.llm import ChatMessage
                planner = LLMPlanner(
                    llm_fn=lambda prompt: self._llm.chat(
                        [ChatMessage(role="user", content=prompt)]
                    ).content,
                    available_tools=self._registry.to_schemas(),
                    soul=soul,
                )
            return PlanAndExecute(
                planner=planner,
                tool_executor=executor,
                max_replans=config.max_replans,
                reasoner=reasoner,
            )

        elif config.pattern == "reflexion":
            from agent_patterns.reflexion import ReflexionLoop
            from agent_core.llm import ChatMessage
            inner = ReactLoop(
                tool_executor=executor,
                reasoner=reasoner,
                max_iterations=config.reflexion.max_iterations_per_attempt,
            )
            return ReflexionLoop(
                pattern=inner,
                reflect_fn=lambda prompt: self._llm.chat(
                    [ChatMessage(role="user", content=prompt)]
                ).content,
                max_attempts=config.reflexion.max_attempts,
            )

        raise ValueError(f"Unsupported pattern: {config.pattern!r}")
