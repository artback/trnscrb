"""Client-side push — upload finished transcripts to a remote server.

When ``remote_url`` is configured, every saved transcript is pushed in the
background; ``trnscrb sync`` migrates every local file in one shot.  A dead
or unreachable server never raises — the local save always wins.
"""

import threading
import time
from pathlib import Path

import httpx

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.push")


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
