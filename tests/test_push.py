"""Tests for the client-side push (spec: docs/spec-20261008-client-server.md).

Guarantees under test:
  * local mode is untouched: with ``remote_url`` unset, ``maybe_push`` is a
    no-op and a local save never touches the network
  * when configured, finished transcripts are uploaded to the server
  * failures (wrong token, server down) never raise out of the save path —
    the push gives up, logs, and returns False
  * ``sync_all`` migrates every local transcript in one shot
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from trnscrb import push, settings, storage

TOKEN = "push-token-456"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """A real uvicorn process running the server app (auth on)."""
    from trnscrb import server_http

    port = _free_port()
    app = server_http.build_app(token=TOKEN)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="live-trnscrb-push-server")
    thread.start()
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    # Wait for the socket to accept (uvicorn startup) rather than poking the
    # API: the first /api/health request also warms the optional embedding
    # backend (sentence-transformers), which can exceed a short per-poke
    # timeout on a cold CI runner and look like a dead server.
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                break
        except OSError:
            time.sleep(0.2)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("live server did not come up (socket never accepted)")
    # One authenticated health check with a generous timeout: the first
    # request may block while the optional embedding backend imports.
    r = httpx.get(url + "/api/health", headers=headers, timeout=120)
    assert r.status_code == 200, r.text
    yield {"url": url, "token": TOKEN, "server": server}
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def configured(live_server):
    """Point the client at the live server."""
    settings.put("remote_url", live_server["url"])
    settings.put("remote_token", live_server["token"])
    yield live_server
    settings.put("remote_url", "")
    settings.put("remote_token", "")


def _write_local_transcript(name: str, text: str):
    storage.ensure_notes_dir()
    path = storage.NOTES_DIR / name
    storage.save_transcript(path, text)
    return path


# ── local mode: unconfigured = untouched ─────────────────────────────────


def test_not_configured_by_default():
    assert push.is_configured() is False


def test_maybe_push_noop_when_unconfigured(monkeypatch):
    calls = []
    monkeypatch.setattr(push, "push_file", lambda *a, **k: calls.append(a) or True)
    path = _write_local_transcript("2026-10-08_09-00_local.txt", "local only\n")
    assert push.maybe_push(path) is False
    assert calls == []


def test_save_transcript_untouched_when_unconfigured(monkeypatch):
    """The both-modes guarantee: a local save never reaches the network."""
    calls = []
    monkeypatch.setattr(push, "push_file", lambda *a, **k: calls.append(a) or True)
    path = storage.NOTES_DIR / "2026-10-08_09-01_local.txt"
    storage.save_transcript(path, "still local\n")
    assert path.read_text().strip() == "still local"
    assert calls == []


def test_sync_all_unconfigured_raises():
    with pytest.raises(RuntimeError, match="remote_url"):
        push.sync_all()


# ── configured: upload works ──────────────────────────────────────────────


def test_push_file_success(configured):
    path = _write_local_transcript("2026-10-08_09-02_pushed.txt", "Meeting: pushed\n")
    assert push.push_file(path) is True
    r = httpx.get(
        configured["url"] + "/api/transcript/2026-10-08_09-02_pushed",
        headers={"Authorization": f"Bearer {configured['token']}"},
        timeout=5,
    )
    assert r.status_code == 200
    assert "Meeting: pushed" in r.text


def test_fetch_transcript_requires_token(configured):
    _write_local_transcript("2026-10-08_09-02b_secret.txt", "secret\n")
    r = httpx.get(configured["url"] + "/api/transcript/2026-10-08_09-02b_secret", timeout=5)
    assert r.status_code == 401


def test_push_file_wrong_token(configured, monkeypatch):
    settings.put("remote_token", "wrong")
    path = _write_local_transcript("2026-10-08_09-03_bad.txt", "x\n")
    assert push.push_file(path, backoff_secs=(0.01, 0.01)) is False


def test_push_file_server_down_does_not_raise(configured, monkeypatch):
    settings.put("remote_url", "http://127.0.0.1:1")
    path = _write_local_transcript("2026-10-08_09-04_down.txt", "x\n")
    start = time.time()
    assert push.push_file(path, backoff_secs=(0.01, 0.01)) is False
    assert time.time() - start < 10  # retries must not hang


def test_maybe_push_starts_background_worker(configured, monkeypatch):
    started = threading.Event()
    seen = {}

    def fake_push_file(path, **kwargs):
        seen["path"] = path
        started.set()
        return True

    monkeypatch.setattr(push, "push_file", fake_push_file)
    path = _write_local_transcript("2026-10-08_09-05_bg.txt", "background\n")
    assert push.maybe_push(path) is True
    assert started.wait(timeout=5)
    assert seen["path"] == path


def test_save_transcript_hooks_push_when_configured(configured, monkeypatch):
    """The central hook: any local save triggers a (background) push."""
    started = threading.Event()
    seen = []

    def fake_push_file(path, **kwargs):
        seen.append(path)
        started.set()
        return True

    monkeypatch.setattr(push, "push_file", fake_push_file)
    path = storage.NOTES_DIR / "2026-10-08_09-06_hook.txt"
    storage.save_transcript(path, "hooked save\n")
    assert path.read_text().strip() == "hooked save"
    assert started.wait(timeout=5)
    assert seen == [path]


# ── migration: sync_all ───────────────────────────────────────────────────


def test_sync_all_pushes_everything(configured):
    for name in (
        "2026-10-08_10-00_one.txt",
        "2026-10-08_10-01_two.txt",
        "2026-10-08_10-02_three.txt",
    ):
        _write_local_transcript(name, f"Meeting: {name}\n")
    ok, failed = push.sync_all()
    assert failed == 0
    assert ok >= 3
    for name in (
        "2026-10-08_10-00_one.txt",
        "2026-10-08_10-01_two.txt",
        "2026-10-08_10-02_three.txt",
    ):
        r = httpx.get(
            configured["url"] + f"/api/transcript/{name[:-4]}",
            headers={"Authorization": f"Bearer {configured['token']}"},
            timeout=5,
        )
        assert r.status_code == 200, name


def test_sync_all_reports_failures(configured, monkeypatch):
    settings.put("remote_url", "http://127.0.0.1:1")
    _write_local_transcript("2026-10-08_11-00_fail.txt", "x\n")
    ok, failed = push.sync_all(progress=None, backoff_secs=(0.01, 0.01))
    assert ok == 0
    assert failed >= 1


def test_status_reports_server_health(configured):
    info = push.status()
    assert info is not None
    assert info["version"]


def test_status_returns_none_when_unreachable(configured, monkeypatch):
    settings.put("remote_url", "http://127.0.0.1:1")
    assert push.status() is None
