"""Survive being upgraded underneath a running process.

Homebrew installs each version into its own Cellar directory and deletes the
old one, so `brew upgrade` removes the tree a running trnscrb is executing
from. The process keeps going on already-imported modules and only fails when
it next imports something new — and because heavy dependencies are imported
lazily, that failure lands mid-meeting, on the feature rather than the app:

    Diarization skipped: No module named 'torch'

That has cost silent, whole-day outages: transcription kept working while
speaker labels and voiceprints quietly stopped. It is not detectable by
looking at the installed files, which are perfectly healthy — only by asking
whether *this* process's own tree still exists.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.rollout")

_LAUNCH_AGENT = "io.trnscrb.app"


def install_root() -> Path:
    """The environment this process is running out of."""
    return Path(sys.prefix)


def is_stale() -> bool:
    """True when the tree this process runs from has been deleted.

    An upgrade replaces the directory rather than the files inside it, so the
    running process holds a path that no longer resolves. Anything not yet
    imported is unreachable from here on.
    """
    try:
        return not install_root().is_dir()
    except OSError:  # pragma: no cover - defensive
        return False


def installed_version() -> str | None:
    """Version of the trnscrb currently on disk, which may differ from ours."""
    binary = shutil.which("trnscrb")
    if not binary:
        return None
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split()
    return parts[-1].strip() if parts else None


def restart() -> bool:
    """Relaunch trnscrb from the version now on disk. Returns True if launched.

    Prefers launchd, which owns the app's lifecycle when it is installed as a
    LaunchAgent and will bring it back with the same identity — the TCC grants
    are tied to that identity, so a plain re-exec could lose them.
    """
    agent = Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCH_AGENT}.plist"
    if agent.is_file():
        try:
            subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{_LAUNCH_AGENT}"],
                check=True,
                capture_output=True,
                timeout=20,
            )
            _log.info("Asked launchd to restart %s after an upgrade", _LAUNCH_AGENT)
            return True
        except (OSError, subprocess.SubprocessError) as e:
            _log.warning("launchctl kickstart failed (%s); falling back", e)

    binary = shutil.which("trnscrb")
    if not binary:
        _log.error("Upgraded install found, but no trnscrb on PATH to restart")
        return False
    try:
        subprocess.Popen(
            [binary, "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(Path.home()),
        )
        _log.info("Relaunched %s after an upgrade", binary)
        return True
    except OSError as e:
        _log.error("Could not relaunch trnscrb after an upgrade: %s", e)
        return False
