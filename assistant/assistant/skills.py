"""Skill library — named prompt templates the agent can invoke as tools.

Skills are plain .md files in a directory:
  - Filename (without .md) = skill name
  - First line starting with # = description (shown in list_skills)
  - Remaining text = prompt template with {variable} placeholders

The agent calls list_skills() to see what's available, then invoke() to run one.
Proposed skills land in a skills/proposed/ subdirectory for human review before
being moved to the main skills/ directory to activate.

File format example (critique_plan.md):
    # Review a plan for potential issues and gaps

    You are reviewing the following plan...

    Plan: {plan}

    Identify: ...
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class SkillLibrary:
    """Loads and executes prompt-based skills from a directory.

    Skills are reloaded from disk on every invoke() call so edits take
    effect immediately without restarting the assistant.
    """

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._proposed_dir = skills_dir / "proposed"

    # ------------------------------------------------------------------
    # Public API (called by tool wrappers)
    # ------------------------------------------------------------------

    def list_skills(self) -> str:
        """Return a human-readable list of available skills with their args."""
        if not self._dir.exists():
            return f"No skills directory found at {self._dir}."
        skills = []
        for path in sorted(self._dir.glob("*.md")):
            name = path.stem
            desc = self._get_description(path)
            placeholders = self._get_placeholders(self._get_template(path))
            arg_hint = f"  (args: {', '.join(placeholders)})" if placeholders else ""
            skills.append(f"- {name}{arg_hint}: {desc}")
        if not skills:
            return f"No skills in {self._dir}. Add .md files to create skills."
        return "Available skills:\n" + "\n".join(skills)

    def invoke(self, name: str, llm, arguments: Optional[dict] = None) -> str:
        """Load skill template, fill arguments, call LLM, return result.

        Reloads the file on every call so edits are picked up immediately.
        """
        path = self._dir / f"{name}.md"
        if not path.exists():
            available = [p.stem for p in self._dir.glob("*.md")] if self._dir.exists() else []
            hint = f" Available: {', '.join(available)}" if available else ""
            return f"Skill '{name}' not found.{hint}"

        template = self._get_template(path)
        args = arguments or {}
        try:
            prompt = template.format(**args)
        except KeyError as exc:
            placeholders = self._get_placeholders(template)
            return (
                f"Skill '{name}' requires argument {exc}. "
                f"Required args: {', '.join(placeholders)}"
            )

        from assistant.conversation import _call_llm_raw
        try:
            resp = _call_llm_raw(
                llm,
                [{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            result = resp["choices"][0]["message"].get("content", "").strip()
            log.info("Skill '%s' invoked successfully", name)
            return result
        except Exception as exc:
            log.warning("Skill '%s' invocation failed: %s", name, exc)
            return f"Skill '{name}' failed: {exc}"

    def propose(self, name: str, description: str, template: str) -> str:
        """Write a proposed skill to skills/proposed/ for human review.

        The human moves the file to the main skills/ directory to activate it.
        """
        # Sanitise name — alphanumeric + underscores only
        safe_name = re.sub(r"[^\w]", "_", name.strip().lower())
        if not safe_name:
            return "Invalid skill name."

        self._proposed_dir.mkdir(parents=True, exist_ok=True)
        path = self._proposed_dir / f"{safe_name}.md"
        content = f"# {description.strip()}\n\n{template.strip()}\n"
        path.write_text(content, encoding="utf-8")
        return (
            f"Proposed skill '{safe_name}' written to {path}.\n"
            f"Review it, then move to {self._dir}/{safe_name}.md to activate."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_description(self, path: Path) -> str:
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            return first.lstrip("#").strip() if first.startswith("#") else first.strip()
        except Exception:
            return ""

    def _get_template(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            if lines and lines[0].startswith("#"):
                return "\n".join(lines[1:]).strip()
            return text.strip()
        except Exception:
            return ""

    def _get_placeholders(self, template: str) -> list[str]:
        return sorted(set(re.findall(r"\{(\w+)\}", template)))
