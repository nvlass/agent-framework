"""Sandboxed Python execution for agent tool use.

Supports two sandbox backends, selected via the ``backend`` config key:

  bwrap     (recommended) — bubblewrap; lightweight, no X11 deps.
            Provides: network isolation, filesystem isolation (work_dir only),
            no privilege escalation, clean /tmp.

  firejail  — traditional firejail; more features but heavier.
            Custom install path supported via ``firejail_path`` config key.

  auto      — try bwrap first, fall back to firejail.

  none      — subprocess only; process isolation but no filesystem/network
            restriction.  Only use for trusted agents / local dev.

Without any sandbox the code still runs as unix_user (if configured),
providing unix permission boundaries at minimum.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _find_executable(name: str, hint: str | None = None) -> str | None:
    """Return full path to *name* or None.  Checks *hint* before PATH."""
    if hint:
        p = Path(hint)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which(name)


def bwrap_available() -> bool:
    return _find_executable("bwrap") is not None


def firejail_available(firejail_path: str | None = None) -> bool:
    return _find_executable("firejail", firejail_path) is not None


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _bwrap_cmd(work_dir: Path, script_path: str, python: str = "python3") -> list[str]:
    """Build a bubblewrap command that:
      - Mounts /usr, /lib*, /bin, /sbin, /etc read-only (Python needs these)
      - Mounts work_dir read-write at its real path
      - Hides /home and /root with a tmpfs
      - Gives a clean /tmp and /proc and /dev
      - Unshares all namespaces (network, mount, pid, ipc, uts)
    """
    w = str(work_dir)
    return [
        "bwrap",
        # Core system — read-only
        "--ro-bind",     "/usr",  "/usr",
        "--ro-bind-try", "/lib",  "/lib",    # symlink on Debian 12 → usr/lib, handled fine
        "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind-try", "/bin",  "/bin",
        "--ro-bind-try", "/sbin", "/sbin",
        "--ro-bind",     "/etc",  "/etc",    # Python needs localtime, ssl certs, etc.
        # Agent working directory — read-write
        "--bind", w, w,
        # Isolated scratch space
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--tmpfs", "/root",
        # Kernel pseudo-filesystems Python expects
        "--proc", "/proc",
        "--dev",  "/dev",
        # Unshare everything (network, mount, pid, ipc, uts, cgroup)
        "--unshare-all",
        "--new-session",
        "--",
        python, script_path,
    ]


def _firejail_cmd(
    work_dir: Path,
    script_name: str,
    firejail_path: str | None,
    python: str = "python3",
) -> list[str]:
    """Build a firejail command.  Uses --private=work_dir so work_dir becomes
    the home directory inside the jail; script_name is relative to it."""
    exe = _find_executable("firejail", firejail_path) or "firejail"
    return [
        exe,
        "--quiet",
        "--net=none",
        "--noroot",
        "--private-tmp",
        f"--private={work_dir}",
        python, script_name,
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_python(
    code: str,
    work_dir: str | Path,
    timeout: int = 120,
    unix_user: str | None = None,
    backend: str = "auto",
    firejail_path: str | None = None,
    python: str = "python3",
) -> dict:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code:          Python source code to execute.
        work_dir:      Directory the script runs in (and the only writable
                       location inside bwrap/firejail sandboxes).
        timeout:       Hard wall-clock limit in seconds.
        unix_user:     Run as this user via ``sudo -u <user> -n``.
        backend:       ``"bwrap"`` | ``"firejail"`` | ``"auto"`` | ``"none"``.
        firejail_path: Full path to firejail binary (e.g. /opt/firejail/bin/firejail)
                       if not on PATH.

    Returns:
        dict: stdout, stderr, returncode, error (str|None), sandbox (str).
    """
    work_dir = Path(work_dir).expanduser().resolve()
    if not work_dir.exists():
        raise RuntimeError(
            f"Sandbox work_dir does not exist: {work_dir}\n"
            f"Create it as the agent user: sudo -u {unix_user or 'agent'} mkdir -p {work_dir}"
        )

    # Resolve backend
    resolved = _resolve_backend(backend, firejail_path)
    if resolved is None:
        requested = backend
        return {
            "stdout": "", "stderr": "", "returncode": -1,
            "error": (
                f"No sandbox backend available (requested: {requested}). "
                "Install bubblewrap (apt install bubblewrap) or firejail."
            ),
            "sandbox": "none",
        }

    # For bwrap/firejail: stage the script in work_dir (sandbox can see it there).
    # For subprocess/none: pipe via stdin so the calling user never needs write
    # access to the agent's work_dir.
    if resolved in ("bwrap", "firejail"):
        return _run_with_file(
            code, work_dir, timeout, unix_user, resolved, firejail_path, python
        )
    else:
        return _run_with_stdin(code, work_dir, timeout, unix_user, python)


def _run_with_stdin(
    code: str,
    work_dir: Path,
    timeout: int,
    unix_user: str | None,
    python: str = "python3",
) -> dict:
    """Run code piped via stdin — no file staging needed.

    The calling process never writes to work_dir, so the calling user
    doesn't need write permission there.  cwd is set to work_dir so the
    script can use relative paths and open()/write files in place.
    Resource limits are applied via preexec_fn (LXC-compatible).
    """
    cmd: list[str] = []
    if unix_user:
        cmd += ["sudo", "-u", unix_user, "-n"]
    cmd += [python, "-"]
    sandbox_desc = f"subprocess + rlimits (unix_user={unix_user or 'self'}, python={python})"
    try:
        result = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            cwd=str(work_dir),
            preexec_fn=_make_rlimit_fn(timeout),
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": None,
            "sandbox": sandbox_desc,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "", "stderr": "", "returncode": -1,
            "error": f"Execution timed out after {timeout}s.",
            "sandbox": sandbox_desc,
        }
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return {
            "stdout": "", "stderr": "", "returncode": -1,
            "error": f"Could not start execution: {exc}",
            "sandbox": "none",
        }


def _run_with_file(
    code: str,
    work_dir: Path,
    timeout: int,
    unix_user: str | None,
    resolved: str,
    firejail_path: str | None,
    python: str = "python3",
) -> dict:
    """Stage code as a temp file in work_dir, then run under bwrap/firejail."""
    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="agent_run_", dir=work_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        script_name = Path(script_path).name

        cmd: list[str] = []
        if unix_user:
            cmd += ["sudo", "-u", unix_user, "-n"]

        if resolved == "bwrap":
            cmd += _bwrap_cmd(work_dir, script_path, python)
            sandbox_desc = f"bwrap (unshare-all, bind={work_dir})"
        else:
            cmd += _firejail_cmd(work_dir, script_name, firejail_path, python)
            sandbox_desc = f"firejail (net=none, private={work_dir})"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                cwd=str(work_dir),
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "error": None,
                "sandbox": sandbox_desc,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "", "stderr": "", "returncode": -1,
                "error": f"Execution timed out after {timeout}s.",
                "sandbox": sandbox_desc,
            }
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return {
                "stdout": "", "stderr": "", "returncode": -1,
                "error": f"Could not start execution: {exc}",
                "sandbox": "none",
            }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _make_rlimit_fn(cpu_seconds: int):
    """Return a preexec_fn that sets CPU and file-size resource limits.

    These are plain setrlimit(2) calls — available in LXC containers where
    mount namespaces are forbidden.  Sets:
      RLIMIT_CPU   — CPU-time limit (seconds); kills the process on overrun
      RLIMIT_FSIZE — max file size 256 MB; prevents runaway writes
    """
    _MAX_FSIZE = 256 * 1024 * 1024  # 256 MB

    def _fn():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, resource.error):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FSIZE, _MAX_FSIZE))
        except (ValueError, resource.error):
            pass

    return _fn


def _resolve_backend(requested: str, firejail_path: str | None) -> str | None:
    """Return the backend to actually use, or None if unavailable."""
    if requested == "auto":
        if bwrap_available():
            return "bwrap"
        if firejail_available(firejail_path):
            return "firejail"
        return None
    if requested == "bwrap":
        return "bwrap" if bwrap_available() else None
    if requested == "firejail":
        return "firejail" if firejail_available(firejail_path) else None
    if requested == "none":
        return "none"
    return None
