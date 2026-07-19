"""Single-instance guard for the recording daemons (menu bar app, watch).

Uses an advisory ``flock`` on a lock file. The kernel drops the lock the
moment the holding process exits — including crashes and SIGKILL — so there
is no stale-lockfile state to clean up. The holder's PID is written into the
file purely for diagnostics.
"""

import fcntl
import os
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.single_instance")

_LOCK_DIR = Path.home() / ".config" / "trnscrb"


class SingleInstance:
    """Exclusive inter-process lock; hold the object for the process lifetime."""

    def __init__(self, name: str = "instance"):
        self._path = _LOCK_DIR / f"{name}.lock"
        self._fh = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> bool:
        """Try to take the lock. False means another instance holds it."""
        if self._fh is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        _log.debug("Instance lock acquired: %s", self._path)
        return True

    def release(self) -> None:
        """Explicit release; also happens automatically on process exit."""
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()

    def holder_pid(self) -> int | None:
        """PID of the current lock holder, best-effort (diagnostics only)."""
        try:
            text = self._path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return int(text) if text.isdigit() else None
