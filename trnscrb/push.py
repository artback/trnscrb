"""Client-side push — upload finished transcripts to a remote server.

When ``remote_url`` is configured, every saved transcript is pushed in the
background; ``trnscrb sync`` migrates every local file in one shot.  A dead
or unreachable server never raises — the local save always wins.

A ledger (``push_state.json``) remembers which transcripts have made it to
the server.  ``resync`` pushes the difference, and the menu-bar app /
headless watcher run ``resync_loop`` on a timer, so a transcript saved while
the server was down reaches it on its own — no manual ``trnscrb sync``
needed.
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.push")

# Ledger of successful pushes: {filename: {size, mtime, pushed_at}}.
_STATE_FILE = Path.home() / ".config" / "trnscrb" / "push_state.json"

# How often the loop re-checks settings while resync is disabled — a
# `trnscrb config set remote_url …` takes effect without a restart.
_RESYNC_DISABLED_POLL_SECS = 300.0


def load_state() -> dict:
    """Read the push ledger; a missing or corrupt file is an empty ledger."""
    if not _STATE_FILE.exists():
        return {}
    try:
        raw = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    """Write the ledger atomically — a torn file must not lose history."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(_STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def record_pushed(path: Path) -> None:
    """Note a successful push so resync does not repeat it.

    Stores size and mtime: when the file changes (a live transcript is
    rewritten while the meeting runs, an edit on disk) the entry no longer
    matches and the file counts as pending again.
    """
    try:
        st = path.stat()
    except OSError:
        return
    state = load_state()
    state[path.name] = {
        "size": st.st_size,
        "mtime": st.st_mtime,
        "pushed_at": datetime.now().isoformat(),
    }
    _save_state(state)


def _state_matches(path: Path, record: dict) -> bool:
    """True when the ledger entry still describes the current file."""
    try:
        st = path.stat()
    except OSError:
        return False
    return record.get("size") == st.st_size and record.get("mtime") == st.st_mtime


def pending_files() -> list[Path]:
    """Local transcripts not yet pushed (or changed since the last push)."""
    if not is_configured():
        return []
    from trnscrb import storage

    state = load_state()
    pending = []
    for entry in storage.list_transcripts():
        path = Path(entry["path"])
        record = state.get(entry["name"])
        if record is not None and _state_matches(path, record):
            continue
        pending.append(path)
    return pending


def pending_count() -> int:
    """How many local transcripts still need to reach the remote."""
    return len(pending_files())


def last_push() -> tuple[str, str] | None:
    """(filename, pushed_at) of the most recent successful push, or None."""
    state = load_state()
    best: tuple[str, str] | None = None
    for name, record in state.items():
        pushed = str(record.get("pushed_at") or "")
        if pushed and (best is None or pushed > best[1]):
            best = (name, pushed)
    return best


def is_configured() -> bool:
    """True when a remote URL has been configured."""
    return bool(str(settings.get("remote_url") or "").strip())


def configured_url() -> str | None:
    """The configured remote URL, with trailing slash stripped."""
    url = str(settings.get("remote_url") or "")
    return url.rstrip("/") if url else None


def configured_token() -> str | None:
    """The configured remote bearer token, or None."""
    token = str(settings.get("remote_token") or "")
    return token if token else None


def push_file(
    path: Path,
    *,
    url: str | None = None,
    token: str | None = None,
    retries: int = 3,
    backoff_secs: tuple[float, ...] = (1.0, 3.0, 9.0),
    timeout_secs: float = 30.0,
) -> bool:
    """POST a transcript file to the server.

    Returns True on HTTP 200; False on any failure (connection error,
    timeout, 4xx, 5xx). Never raises.  Retries on transient errors using
    the backoff schedule.

    Args:
        path: Local transcript path.
        url: Override the configured remote URL.
        token: Override the configured remote token.
        retries: Number of retry attempts.
        backoff_secs: Seconds to sleep between attempts (last value repeats).
        timeout_secs: Per-request timeout.
    """
    url = url or configured_url()
    token = token or configured_token()
    if not url or not token:
        _log.debug("push_file: not configured (url=%s token=%s)", url, bool(token))
        return False

    text = path.read_text(encoding="utf-8")
    endpoint = f"{url}/api/transcript"

    for attempt in range(retries):
        try:
            r = httpx.post(
                endpoint,
                json={"filename": path.name, "text": text},
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout_secs,
            )
            if r.status_code == 200:
                _log.info("pushed %s (attempt %d)", path.name, attempt + 1)
                record_pushed(path)
                return True
            if 400 <= r.status_code < 500:
                _log.warning("push %s: client error %d", path.name, r.status_code)
                return False
            # 5xx — retry
            _log.warning(
                "push %s: server error %d (attempt %d)", path.name, r.status_code, attempt + 1
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            _log.warning("push %s: %s (attempt %d)", path.name, e, attempt + 1)
        except Exception as e:
            _log.warning("push %s: unexpected error %s (attempt %d)", path.name, e, attempt + 1)

        if attempt < retries - 1:
            sleep_sec = backoff_secs[min(attempt, len(backoff_secs) - 1)]
            time.sleep(sleep_sec)

    _log.error("push %s: all %d attempts failed", path.name, retries)
    return False


def maybe_push(path: Path) -> bool:
    """Start a background daemon thread pushing this file.

    Returns True when a thread was started, False when unconfigured.
    """
    if not is_configured():
        return False

    def _run() -> None:
        try:
            push_file(path)
        except Exception:
            _log.debug("background push failed for %s", path.name, exc_info=True)

    t = threading.Thread(target=_run, daemon=True, name=f"trnscrb-push-{path.name}")
    t.start()
    return True


def sync_all(
    progress=None,
    *,
    backoff_secs: tuple[float, ...] = (1.0, 3.0, 9.0),
) -> tuple[int, int]:
    """Push every transcript in NOTES_DIR to the remote server.

    Args:
        progress: Optional callable(name, ok) invoked per file.
        backoff_secs: Retry backoff schedule.

    Returns:
        (ok_count, failed_count).

    Raises:
        RuntimeError: When remote_url is not configured.
    """
    from trnscrb import storage

    url = configured_url()
    if not url:
        raise RuntimeError(
            "remote_url is not set — configure it with `trnscrb config set remote_url …`"
        )

    ok = 0
    failed = 0
    for entry in storage.list_transcripts():
        name = entry["name"]
        path = Path(entry["path"])
        success = push_file(path, backoff_secs=backoff_secs)
        if progress:
            progress(name, success)
        if success:
            ok += 1
        else:
            failed += 1
    return ok, failed


def resync(
    progress=None,
    *,
    backoff_secs: tuple[float, ...] = (1.0, 3.0, 9.0),
) -> tuple[int, int]:
    """Push the pending transcripts — the ones that have not made it yet.

    Unlike ``sync_all`` (which re-pushes everything) this only sends what the
    ledger says is missing or changed, so a routine run costs nothing when
    the store is in sync.

    Args:
        progress: Optional callable(name, ok) invoked per file.
        backoff_secs: Retry backoff schedule.

    Returns:
        (ok_count, failed_count). (0, 0) when nothing is pending or the
        remote is not configured.
    """
    if not is_configured():
        return 0, 0

    pending = pending_files()
    if not pending:
        return 0, 0

    _log.info("resync: pushing %d pending transcript(s)", len(pending))
    ok = 0
    failed = 0
    for path in pending:
        success = push_file(path, backoff_secs=backoff_secs)
        if progress:
            progress(path.name, success)
        if success:
            ok += 1
        else:
            failed += 1
    return ok, failed


def resync_interval_secs() -> float:
    """Configured resync period in seconds (0 = disabled)."""
    minutes = settings.get("remote_resync_interval_minutes")
    try:
        return float(int(minutes) * 60)
    except (TypeError, ValueError):
        return 300.0


def resync_loop(stop_event: threading.Event, interval_secs: float | None = None) -> None:
    """Push pending transcripts on a timer until ``stop_event`` is set.

    Runs as a daemon thread from the menu-bar app and the headless watcher.
    It re-reads the settings every cycle, so configuring ``remote_url`` (or
    switching the interval to 0) takes effect without a restart.  A slow or
    dead server is harmless: each cycle is best-effort and the next one
    retries whatever is still pending.
    """
    while not stop_event.is_set():
        interval = resync_interval_secs() if interval_secs is None else interval_secs
        if not (is_configured() and interval > 0):
            # Disabled (or not configured): cheap re-check, no network.
            stop_event.wait(_RESYNC_DISABLED_POLL_SECS)
            continue
        try:
            ok, failed = resync()
            if failed:
                _log.warning(
                    "resync: %d pushed, %d failed — retrying next cycle "
                    "(run `trnscrb sync` to force)",
                    ok,
                    failed,
                )
            elif ok:
                _log.debug("resync: pushed %d pending transcript(s)", ok)
        except Exception:
            _log.warning("resync: cycle failed", exc_info=True)
        stop_event.wait(interval)


def start_resync_loop() -> threading.Thread:
    """Start the background resync loop as a daemon thread.

    The loop outlives nothing — it dies with the process, and at most one
    long-running process (the app or the watcher) holds the single-instance
    lock that owns it, so pushes stay idempotent, not concurrent.
    """
    stop_event = threading.Event()
    t = threading.Thread(target=resync_loop, args=(stop_event,), daemon=True, name="trnscrb-resync")
    t.start()
    return t


def remote_transcripts() -> list[dict] | None:
    """GET /api/transcripts — the server-side store, or None if unreachable."""
    url = configured_url()
    token = configured_token()
    if not url or not token:
        return None
    try:
        r = httpx.get(
            f"{url}/api/transcripts",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("transcripts") or []
    except Exception:
        pass
    return None


def fetch_transcript(transcript_id: str) -> str | None:
    """GET /api/transcript/{id} — body of one remote transcript, or None.

    ``transcript_id`` is the filename stem (the ``id`` from
    ``remote_transcripts``); a trailing ``.txt`` is tolerated.
    """
    url = configured_url()
    token = configured_token()
    if not url or not token:
        return None
    id_str = transcript_id[:-4] if transcript_id.endswith(".txt") else transcript_id
    try:
        r = httpx.get(
            f"{url}/api/transcript/{id_str}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def status() -> dict | None:
    """GET the server health endpoint.

    Returns the JSON dict on success, None on any failure.
    """
    url = configured_url()
    token = configured_token()
    if not url or not token:
        return None
    try:
        r = httpx.get(
            f"{url}/api/health",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None
