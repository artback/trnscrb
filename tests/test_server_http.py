"""Tests for the network server (``trnscrb serve``) — spec: docs/spec-20261008-client-server.md.

Covers:
  * the REST ingest API (health, upload, list, fetch) through the Starlette app
  * bearer-token auth on every route, including the MCP endpoint
  * the bind policy (no token + non-loopback bind must refuse to start)
  * token bootstrap helpers (generate / resolve precedence)
  * the MCP side end-to-end: a real uvicorn process + the official MCP client,
    proving the Streamable HTTP transport serves the store tools only

The REST tests use starlette's TestClient (no port needed); the MCP tests run
a live uvicorn server in a daemon thread so the wire protocol is exercised for
real.
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from trnscrb import server_http, settings, storage

TOKEN = "test-token-123"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def app():
    """The server app with token auth enabled."""
    return server_http.build_app(token=TOKEN)


@pytest.fixture()
def client(app):
    from starlette.testclient import TestClient

    return TestClient(app)


@pytest.fixture()
def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def live_server():
    """A real uvicorn process running the server app (auth on)."""
    port = _free_port()
    app = server_http.build_app(token=TOKEN)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="live-trnscrb-server")
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


def _seed_transcript(url: str, token: str, filename: str, text: str) -> None:
    r = httpx.post(
        url + "/api/transcript",
        json={"filename": filename, "text": text},
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert r.status_code == 200, r.text


# ── health ────────────────────────────────────────────────────────────────


def test_health(client, auth):
    r = client.get("/api/health", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["version"]
    assert body["transcript_count"] == 0


def test_health_requires_token(client):
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401


# ── upload (POST /api/transcript) ─────────────────────────────────────────


def test_upload_stores_transcript(client, auth):
    r = client.post(
        "/api/transcript",
        headers=auth,
        json={"filename": "2026-10-08_10-00_Standup.txt", "text": "Meeting: Standup\n"},
    )
    assert r.status_code == 200, r.text
    assert storage.read_transcript("2026-10-08_10-00_Standup").startswith("Meeting: Standup")


def test_upload_is_idempotent_overwrite(client, auth):
    name = "2026-10-08_10-00_Standup.txt"
    r1 = client.post("/api/transcript", headers=auth, json={"filename": name, "text": "first\n"})
    r2 = client.post("/api/transcript", headers=auth, json={"filename": name, "text": "second\n"})
    assert r1.status_code == r2.status_code == 200
    assert storage.read_transcript("2026-10-08_10-00_Standup").strip() == "second"


def test_upload_rejects_path_traversal(client, auth):
    r = client.post(
        "/api/transcript", headers=auth, json={"filename": "../evil.txt", "text": "x\n"}
    )
    assert r.status_code == 400
    assert not (storage.NOTES_DIR.parent / "evil.txt").exists()


@pytest.mark.parametrize("bad", ["a/b.txt", "noextension", "", "x" * 201 + ".txt"])
def test_upload_rejects_bad_filenames(client, auth, bad):
    r = client.post("/api/transcript", headers=auth, json={"filename": bad, "text": "x\n"})
    assert r.status_code == 400


def test_upload_rejects_empty_text(client, auth):
    r = client.post("/api/transcript", headers=auth, json={"filename": "empty.txt", "text": ""})
    assert r.status_code == 400
    assert storage.read_transcript("empty") is None


def test_upload_requires_token(client):
    r = client.post("/api/transcript", json={"filename": "a.txt", "text": "x\n"})
    assert r.status_code == 401


def test_upload_accepts_colon_filename(client, auth):
    """The original bug: "…Olgun-1:1-Team-Integration-Meeting.txt" 400'd."""
    name = "2026-03-27_10-58_Olgun-1:1-Team-Integration-Meeting.txt"
    r = client.post("/api/transcript", headers=auth, json={"filename": name, "text": "x\n"})
    assert r.status_code == 200, r.text


# ── filename validator (denylist, not allowlist) ────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "2026-03-27_10-58_A-1:1-B.txt",  # colons are real meeting names
        "2026-03-27 10-58 Brown Bag Session - Team.txt",  # spaces + dashes
        "2026-03-27_10-58_Synchronisation-ümlaut.txt",  # unicode
        "a.txt",
    ],
)
def test_valid_transcript_filename_accepts(name):
    ok, reason = server_http._valid_transcript_filename(name)
    assert ok, reason


@pytest.mark.parametrize(
    "name",
    [
        "../x.txt",  # traversal
        "/abs.txt",  # absolute
        "a\\b.txt",  # backslash separator
        "a\x00b.txt",  # NUL byte
        "noextension",  # missing .txt
        "..txt",  # bare path element
        "b" * 201 + ".txt",  # over 200 chars
        "",  # empty
    ],
)
def test_valid_transcript_filename_rejects(name):
    ok, reason = server_http._valid_transcript_filename(name)
    assert not ok
    assert reason


# ── list / fetch ──────────────────────────────────────────────────────────


def test_list_and_fetch_transcript(client, auth):
    _ = client.post(
        "/api/transcript",
        headers=auth,
        json={"filename": "2026-10-08_11-00_A.txt", "text": "hello A\n"},
    )
    r = client.get("/api/transcripts", headers=auth)
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["transcripts"]]
    assert "2026-10-08_11-00_A" in ids

    r = client.get("/api/transcript/2026-10-08_11-00_A", headers=auth)
    assert r.status_code == 200
    assert "hello A" in r.text


def test_fetch_missing_transcript_404(client, auth):
    assert client.get("/api/transcript/does-not-exist", headers=auth).status_code == 404


def test_mcp_endpoint_requires_token(client):
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert r.status_code == 401


# ── action items (GET/POST /api/action-items) ──────────────────────────────


@pytest.fixture()
def ai_store(tmp_path, monkeypatch):
    """Point the action-item store at a per-test file (never the real one)."""
    from trnscrb import action_items

    store = tmp_path / "action_items.json"
    monkeypatch.setattr(action_items, "_STORE", store)
    return store


def _ai(item_id: str, **kw) -> dict:
    """A well-formed action item, the way record_meeting writes it."""
    item = {
        "id": item_id,
        "text": f"text-{item_id}",
        "owner": "Me",
        "status": "open",
        "created": "2026-01-01",
        "last_seen": "2026-01-01 10:00",
        "meeting_id": "m1",
        "meeting_title": "Standup",
        "note": "",
        "jira": [],
        "github": [],
        "done_date": "",
        "done_reason": "",
    }
    item.update(kw)
    return item


def test_get_action_items_returns_store(client, auth, ai_store):
    from trnscrb import action_items

    action_items.save([_ai("g1", text="ship it"), _ai("g2")])
    r = client.get("/api/action-items", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert [i["id"] for i in body["items"]] == ["g1", "g2"]
    assert body["items"][0]["text"] == "ship it"


def test_get_action_items_empty_when_no_store(client, auth, ai_store):
    r = client.get("/api/action-items", headers=auth)
    assert r.status_code == 200
    assert r.json() == {"items": []}


def test_get_action_items_requires_token(client, ai_store):
    assert client.get("/api/action-items").status_code == 401
    assert (
        client.get("/api/action-items", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )


def test_post_action_items_merges_new_union_and_skip(client, auth, ai_store):
    from trnscrb import action_items

    action_items.save(
        [
            _ai("s1", jira=["PROJ-1"], text="server text", last_seen="2026-01-02 10:00"),
            _ai("s-only", text="server only"),  # must survive untouched
        ]
    )
    r = client.post(
        "/api/action-items",
        headers=auth,
        json={
            "items": [
                _ai(
                    "s1",
                    jira=["PROJ-2"],
                    github=["https://github.com/o/r/issues/9"],
                    text="client text",
                    last_seen="2026-01-03 09:00",
                ),
                _ai("n1"),  # new id → inserted
                {"text": "no id"},  # no id → skipped
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["merged"] == 2
    assert body["skipped"] == 1
    assert body["total"] == 3

    by_id = {i["id"]: i for i in action_items.load()}
    s1 = by_id["s1"]
    assert s1["jira"] == ["PROJ-1", "PROJ-2"]  # union, deduped
    assert s1["github"] == ["https://github.com/o/r/issues/9"]
    assert s1["text"] == "client text"  # non-empty client value wins
    assert s1["last_seen"] == "2026-01-03 09:00"  # later of the two
    assert s1["status"] == "open"
    assert by_id["s-only"]["text"] == "server only"  # server-only item survives
    assert by_id["n1"]["jira"] == []  # new item normalized


def test_post_action_items_new_item_defaults(client, auth, ai_store):
    from trnscrb import action_items

    r = client.post(
        "/api/action-items",
        headers=auth,
        json={"items": [{"id": "n1", "text": "bare", "owner": "Me"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"merged": 1, "skipped": 0, "total": 1}
    item = action_items.load()[0]
    assert item["jira"] == []
    assert item["github"] == []
    assert item["done_date"] == ""
    assert item["done_reason"] == ""


def test_post_action_items_done_wins_both_directions(client, auth, ai_store):
    from trnscrb import action_items

    # Server done, client open → stays done with the server's done fields.
    action_items.save([_ai("d1", status="done", done_date="2026-01-05", done_reason="shipped")])
    r = client.post(
        "/api/action-items",
        headers=auth,
        json={"items": [_ai("d1", status="open", done_date="", done_reason="")]},
    )
    assert r.status_code == 200, r.text
    merged = {i["id"]: i for i in action_items.load()}["d1"]
    assert merged["status"] == "done"
    assert merged["done_date"] == "2026-01-05"
    assert merged["done_reason"] == "shipped"

    # Server open, client done → done with the client's done fields.
    action_items.save([_ai("d2")])
    r = client.post(
        "/api/action-items",
        headers=auth,
        json={"items": [_ai("d2", status="done", done_date="2026-02-01", done_reason="client")]},
    )
    assert r.status_code == 200, r.text
    merged = {i["id"]: i for i in action_items.load()}["d2"]
    assert merged["status"] == "done"
    assert merged["done_date"] == "2026-02-01"
    assert merged["done_reason"] == "client"


def test_post_action_items_done_without_fields_uses_fallbacks(client, auth, ai_store):
    from trnscrb import action_items

    # Client done but carrying no done fields → the server side's fields win.
    action_items.save([_ai("d3", status="done", done_date="2026-01-05", done_reason="shipped")])
    r = client.post(
        "/api/action-items",
        headers=auth,
        json={"items": [_ai("d3", status="done")]},
    )
    assert r.status_code == 200, r.text
    merged = {i["id"]: i for i in action_items.load()}["d3"]
    assert merged["status"] == "done"
    assert merged["done_date"] == "2026-01-05"
    assert merged["done_reason"] == "shipped"

    # Neither side carries done fields → synthetic reason.
    action_items.save([_ai("d4")])
    r = client.post(
        "/api/action-items",
        headers=auth,
        json={"items": [_ai("d4", status="done")]},
    )
    assert r.status_code == 200, r.text
    merged = {i["id"]: i for i in action_items.load()}["d4"]
    assert merged["status"] == "done"
    assert merged["done_date"] == ""
    assert merged["done_reason"] == "resolved via client sync"


# ── unauthenticated (loopback dev mode) app ───────────────────────────────


def test_app_without_token_allows_requests():
    from starlette.testclient import TestClient

    app = server_http.build_app(token=None)
    c = TestClient(app)
    assert c.get("/api/health").status_code == 200
    r = c.post("/api/transcript", json={"filename": "dev-note.txt", "text": "hi\n"})
    assert r.status_code == 200


# ── bind policy / token bootstrap ─────────────────────────────────────────


def test_bind_policy_refuses_unauthenticated_remote():
    with pytest.raises(server_http.BindPolicyError):
        server_http.check_bind_policy("0.0.0.0", None)


def test_bind_policy_allows_unauthenticated_loopback():
    server_http.check_bind_policy("127.0.0.1", None)
    server_http.check_bind_policy("localhost", None)
    server_http.check_bind_policy("::1", None)


def test_bind_policy_insecure_override():
    server_http.check_bind_policy("0.0.0.0", None, insecure=True)


def test_bind_policy_token_allows_remote():
    server_http.check_bind_policy("0.0.0.0", TOKEN)


def test_generate_token():
    t = server_http.generate_token()
    assert len(t) >= 32
    int(t, 16)  # must be hex — raises ValueError otherwise
    assert t != server_http.generate_token()


def test_resolve_token_precedence(monkeypatch):
    monkeypatch.setenv("TRNSCRB_SERVER_TOKEN", "env-token")
    settings.put("server_token", "settings-token")
    assert server_http.resolve_token("cli-token") == "cli-token"
    assert server_http.resolve_token(None) == "env-token"
    monkeypatch.delenv("TRNSCRB_SERVER_TOKEN")
    assert server_http.resolve_token(None) == "settings-token"
    settings.put("server_token", "")
    assert server_http.resolve_token(None) is None


def test_client_setup_snippet():
    text = server_http.client_setup_snippet("http://10.0.0.5:8765", TOKEN)
    assert "http://10.0.0.5:8765" in text
    assert TOKEN in text
    assert "remote_url" in text
    assert "remote_token" in text
    assert "trnscrb sync" in text
    # the paste-ready opencode entry
    assert '"type": "remote"' in text
    assert "/mcp" in text


# ── MCP over Streamable HTTP, end to end ─────────────────────────────────


def _mcp_call(url: str, token: str, tool: str, arguments: dict, extra_setup=None):
    """Initialize a real MCP client session and call one tool."""
    import asyncio

    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"}

    async def run():
        # The installed mcp SDK (2.2.0) takes a pre-configured http client
        # (no `headers=` kwarg) and yields a (read, write) stream pair.
        async with httpx2.AsyncClient(headers=headers) as client:
            async with streamable_http_client(url, http_client=client) as streams:
                read, write = _unpack_streams(streams)
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if extra_setup is not None:
                        extra_setup(session)
                    return await session.call_tool(tool, arguments)

    return asyncio.run(run())


def _unpack_streams(streams):
    """Normalise the SDK's stream yield to (read, write)."""
    if isinstance(streams, tuple):
        read, write = streams[:2]
    else:
        read, write = streams.read_stream, streams.write_stream
    return read, write


def test_mcp_serves_store_tools_only(live_server):
    import asyncio

    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {live_server['token']}"}

    async def run():
        async with httpx2.AsyncClient(headers=headers) as client:
            async with streamable_http_client(
                live_server["url"] + "/mcp", http_client=client
            ) as streams:
                read, write = _unpack_streams(streams)
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    return {t.name for t in tools.tools}

    names = asyncio.run(run())
    assert set(server_http.STORE_TOOLS) <= names
    # machine-local tools must not be served remotely
    assert "start_recording" not in names
    assert "stop_dictation" not in names
    assert "dictation_list" not in names


def test_mcp_get_transcript_after_upload(live_server):
    _seed_transcript(
        live_server["url"],
        live_server["token"],
        "mcp-seed-meeting.txt",
        "Meeting: seed\n\nWe decided to ship the client-server mode.\n",
    )
    result = _mcp_call(
        live_server["url"] + "/mcp",
        live_server["token"],
        "get_transcript",
        {"transcript_id": "mcp-seed-meeting"},
    )
    assert "client-server mode" in result.content[0].text


def test_mcp_search_transcripts_after_upload(live_server):
    _seed_transcript(
        live_server["url"],
        live_server["token"],
        "mcp-search-meeting.txt",
        "Meeting: search\n\nThe deploy pipeline needs a retry queue.\n",
    )
    result = _mcp_call(
        live_server["url"] + "/mcp",
        live_server["token"],
        "search_transcripts",
        {"query": "retry queue"},
    )
    assert "retry queue" in result.content[0].text


def test_mcp_action_items_roundtrip(live_server):
    added = _mcp_call(
        live_server["url"] + "/mcp",
        live_server["token"],
        "add_action_item",
        {"text": "Ship client-server mode"},
    )
    listed = _mcp_call(
        live_server["url"] + "/mcp",
        live_server["token"],
        "list_action_items",
        {},
    )
    assert "Ship client-server mode" in listed.content[0].text
    assert added.content[0].text
