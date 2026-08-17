"""Centralised logging for trnscrb.

Writes to ~/Library/Logs/trnscrb.log with automatic rotation (5 MB × 3 files).
Also emits to stderr so launchd / terminal sessions capture output.

TRNSCRB_LOG_DIR redirects that file. It exists because the test suite wrote
into the real log, and INFO lines from tests read exactly like real events —
"Aligning system audio: it runs 500 ms behind the mic" is a test fixture's
value, but nothing in the log says so, and it cost real time during a live
debugging session.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(os.environ.get("TRNSCRB_LOG_DIR") or Path.home() / "Library" / "Logs")
_LOG_FILE = _LOG_DIR / "trnscrb.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3

_configured = False


def get_logger(name: str = "trnscrb") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)

    if not _configured:
        _configured = True
        logger = logging.getLogger("trnscrb")
        logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler — rotated, always DEBUG
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                _LOG_FILE,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass  # can't write log file — degrade gracefully

        # Stderr handler — only warnings and above
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logging.getLogger(name)
