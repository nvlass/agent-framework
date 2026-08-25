"""Spawn — synchronous parent-child agent delegation.

An agent can delegate a sub-task to a specialist child agent and get the
result back inline.  Children are pre-configured in the parent's YAML under
``spawn_roles:``, so the parent knows exactly which roles it can spawn and
what tools each one is allowed to use.

YAML schema::

    spawn_roles:
      researcher:
        soul: souls/researcher.txt     # relative to config file dir, or absolute
        tools: [web_search, fetch_readable, save_note]
      critic:
        soul: souls/critic.txt
        tools: []                      # pure reasoning — no tools needed
      analyst:
        soul: souls/analyst.txt
        tools: [web_search, fetch_readable]
        model: accounts/fireworks/models/deepseek-v3p2  # optional LLM override

Child tool access rules:
- The child only gets the tools listed in its role definition.
- The parent can further restrict at spawn time via ``allow_tools``.
- Dangerous tools are never inherited automatically; they must be declared.
- The parent's ToolRegistry is the source — unknown tool names are skipped
  with a warning rather than crashing.

Future: ``query_available_roles()`` tool for dynamic discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.llm import ChatLLMInterface

log = logging.getLogger(__name__)


@dataclass
class SpawnRole:
    """Configuration for a spawnable child agent role.

    Attrs:
        name:  Role identifier (matches the key in ``spawn_roles:``).
        soul:  Soul text, pre-loaded from the configured file.
        tools: Tool names the child is allowed to use.
               Empty list means the child gets no tools (pure reasoning).
        llm:   Optional LLM override.  ``None`` inherits the parent's LLM.
    """

    name: str
    soul: str = ""
    tools: list[str] = field(default_factory=list)
    llm: "ChatLLMInterface | None" = field(default=None, repr=False)


class SpawnRegistry:
    """Maps role names → SpawnRole configs for a single parent agent.

    Built from the ``spawn_roles:`` YAML section at startup.  Only roles
    listed here can be spawned — no dynamic role creation at runtime.

    Args:
        roles: ``{name: SpawnRole}`` mapping.
    """

    def __init__(self, roles: dict[str, SpawnRole]) -> None:
        self._roles = roles

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> SpawnRole | None:
        return self._roles.get(name)

    def names(self) -> list[str]:
        return list(self._roles.keys())

    def __bool__(self) -> bool:
        return bool(self._roles)

    def __len__(self) -> int:
        return len(self._roles)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        spawn_cfg: dict,
        soul_base_dir: Path | None = None,
        make_llm_fn=None,
    ) -> "SpawnRegistry":
        """Build from the ``spawn_roles:`` YAML dict.

        Args:
            spawn_cfg:     The ``spawn_roles`` dict from YAML.
            soul_base_dir: Base directory for resolving relative soul paths.
            make_llm_fn:   Optional callable ``(model_id: str) → LLM`` used
                           when a role specifies its own ``model:``.
        """
        roles: dict[str, SpawnRole] = {}

        for name, role_cfg in spawn_cfg.items():
            soul_text = cls._load_soul(name, role_cfg.get("soul", ""), soul_base_dir)
            tools: list[str] = role_cfg.get("tools") or []
            llm = None
            model_id = role_cfg.get("model")
            if model_id and make_llm_fn:
                try:
                    llm = make_llm_fn(str(model_id))
                except Exception as exc:
                    log.warning("SpawnRegistry: could not build LLM for role %r: %s", name, exc)

            roles[name] = SpawnRole(name=name, soul=soul_text, tools=tools, llm=llm)
            log.debug("SpawnRegistry: loaded role %r (%d tools)", name, len(tools))

        return cls(roles)

    @staticmethod
    def _load_soul(role_name: str, soul_path: str, base_dir: Path | None) -> str:
        if not soul_path:
            return ""
        path = Path(soul_path)
        if not path.is_absolute() and base_dir:
            path = base_dir / path
        if not path.exists():
            log.warning("SpawnRegistry: role %r soul file not found: %s", role_name, path)
            return ""
        return path.read_text().strip()
