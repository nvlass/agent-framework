"""Sandboxed file read/write tools for the assistant's work directory.

All paths are resolved relative to ``work_dir`` and validated to prevent
path traversal. Absolute paths and ``../``-escapes are rejected.
"""

from pathlib import Path

_MAX_READ_CHARS = 20_000


class FileTools:
    def __init__(self, work_dir: Path) -> None:
        self._root = Path(work_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def work_dir(self) -> Path:
        return self._root

    def _safe_resolve(self, path: str) -> Path:
        if Path(path).is_absolute():
            raise ValueError(f"Absolute paths not allowed: {path!r}")
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root):
            raise ValueError(f"Path escapes work directory: {path!r}")
        return target

    def write_file(self, path: str, content: str, append: bool = False) -> str:
        target = self._safe_resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Written"
        return f"{action} {path} ({len(content):,} chars)"

    def read_file(self, path: str) -> str:
        target = self._safe_resolve(path)
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_READ_CHARS:
            text = text[:_MAX_READ_CHARS] + f"\n\n[... truncated at {_MAX_READ_CHARS:,} chars ...]"
        return text

    def list_files(self, path: str = ".") -> str:
        target = self._safe_resolve(path)
        if not target.exists():
            return f"Directory not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(target.iterdir())
        if not entries:
            return f"Empty directory: {path}"
        lines = []
        for e in entries:
            rel = e.relative_to(self._root)
            if e.is_dir():
                lines.append(f"  {rel}/")
            else:
                size = e.stat().st_size
                lines.append(f"  {rel}  ({size:,} bytes)")
        header = "." if path == "." else path
        return f"Work directory ({self._root}):\n" + "\n".join(lines)
