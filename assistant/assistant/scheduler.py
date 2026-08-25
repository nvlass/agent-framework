"""Periodic background scheduler for agent reflection.

A daemon thread fires a callback on a fixed interval. Used to trigger
memory reflection without blocking the main conversation loop.
"""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ReflectionScheduler:
    """Calls callback() every interval_seconds in a background daemon thread.

    The callback runs in a separate thread — it must not modify shared state
    without locks. Exceptions in the callback are logged and swallowed so the
    scheduler keeps running regardless.

    Args:
        interval_seconds: How often to fire (e.g. 3600 for hourly).
        callback: Zero-argument callable to invoke on each tick.
    """

    def __init__(self, interval_seconds: int, callback: Callable[[], None]) -> None:
        self._interval = interval_seconds
        self._callback = callback
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def start(self) -> None:
        """Start the scheduler. First tick fires after interval_seconds."""
        self._running = True
        self._schedule()
        logger.info(f"Reflection scheduler started (every {self._interval}s)")

    def stop(self) -> None:
        """Stop the scheduler. Safe to call even if not started."""
        self._running = False
        if self._timer:
            self._timer.cancel()

    def trigger_now(self) -> None:
        """Fire the callback immediately in a background thread."""
        threading.Thread(target=self._fire, daemon=True).start()

    def _schedule(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        try:
            self._callback()
        except Exception as exc:
            logger.warning(f"Reflection callback failed: {exc}")
        finally:
            self._schedule()
