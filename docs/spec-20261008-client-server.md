# Spec: client/server mode — serve transcripts and MCP from a server

Date: 2026-10-08

## Ask

Run trnscrb in client/server mode: capture and transcription keep happening
locally on the MacBook (MLX/Whisper, TCC-permissioned audio capture), while the
**transcript store and the MCP interface are served from a server** (any
Linux box). The user's agents on the Mac query the transcripts that live on
the server; the Mac pushes finished transcripts to it.

## Design constraints

1. **Runnable in both modes, local by default.** Every existing install keeps
   working untouched: with `remote_url` unset, all new code paths are no-ops
   and the local stdio MCP keeps its full toolset. Client/server mode is
   purely additive and opt-in.
2. **Novice-friendly operation.** Nobody should hand-edit pyproject, run apt
   by heart, or generate tokens manually:
   - Server: one command — `install-server.sh` (curlable from the repo) —
     installs everything `serve` needs, generates and stores the token, and
     prints the paste-ready client setup.
   - `trnscrb serve` without a configured token auto-generates one, saves it
     to the `server_token` setting, and prints the client snippet.
   - Client (Mac): `trnscrb config set remote_url …` / `remote_token …`,
     `trnscrb sync`, paste one JSON block into opencode's MCP config. Done.

## Design decisions

### One process on the server: `trnscrb serve`

A single HTTP process serves, on one port (default `8765`):

| Path | Purpose |
| --- | --- |
| `POST/GET/DELETE /mcp` | MCP over Streamable HTTP (stateless) — the store tools |
| `GET /api/health` | version, transcript count, semantic-search availability |
| `POST /api/transcript` | client upload: `{"filename": "...", "text": "..."}` |
| `GET /api/transcripts` | stored transcript list (id, size, modified) |
| `GET /api/transcript/{id}` | one stored transcript's text |

MCP transport: the `mcp` SDK (2.2.0) already ships `starlette`, `uvicorn` and
`sse-starlette` as core dependencies, so **no new pyproject dependencies**.
The app is `MCPServer.streamable_http_app(stateless_http=True)` — stateless so
any number of clients (several opencode instances, Claude Desktop, …) can
connect without a server-side session store — with custom Starlette routes
attached via `MCPServer.custom_route()` (they merge into the same app).

### Server-side MCP toolset = the store subset

`trnscrb/server_http.py` re-registers, on a fresh `MCPServer("trnscrb")`
instance, the store-side tool functions already defined in
`trnscrb/mcp_server.py`:

`list_transcripts`, `get_transcript`, `get_weekly_transcripts`,
`get_weekly_summaries`, `search_transcripts`, `semantic_search`,
`enrich_transcript`, `list_action_items`, `resolve_action_item`,
`link_action_item`, `add_action_item`.

`MCPServer.tool()` returns the function unchanged, so the existing decorated
functions can be registered a second time with zero code duplication. The
local stdio server (`trnscrb server`) is untouched and keeps the full 27-tool
set. Recording/dictation/calendar tools stay machine-local by design (they
drive the Mac's audio hardware and TCC grants); glossary tools stay local
because the glossary is applied by the client's transcriber.

### Auth

Bearer token, enforced by a Starlette middleware on **every** route (including
`/mcp`):

- Token precedence: `--token` flag > `TRNSCRB_SERVER_TOKEN` env >
  `server_token` setting. Comparison is timing-safe
  (`hmac.compare_digest`).
- Fail closed: no token and a non-loopback bind → refuse to start.
  `--insecure` overrides (LAN convenience; logs a loud warning).
- DNS-rebinding protection (the SDK's `TransportSecuritySettings`) is disabled
  explicitly: the token is the security boundary and the Host header varies
  with whatever name the user reaches the server through (hostname, IP,
  reverse proxy). TLS for WAN exposure is out of scope — the supported
  postures are LAN direct or an SSH tunnel / reverse proxy in front.

### Uploads

`POST /api/transcript` writes into the same `NOTES_DIR` the rest of the app
uses (`~/meeting-notes/`), so **storage.py is reused unmodified** — every
store tool works on uploads with no changes. Filename is validated by a strict
pattern (`[A-Za-z0-9][A-Za-z0-9 ._-]*\.txt`, ≤200 chars) plus the same
`is_relative_to` traversal guard `read_transcript` uses; violations → 400,
nothing written. Uploads are idempotent overwrites, so re-pushing after an
enrichment rewrite keeps the server copy current. Body limit 4 MiB (the SDK
default) covers transcripts comfortably.

### Client side

Settings (client): `remote_url` (e.g. `http://10.0.0.5:8765`),
`remote_token`. Both default to `""` — with `remote_url` empty every new code
path is a no-op, so existing installs are unaffected.

- `trnscrb/push.py`:
  - `request_push(path)` — spawned from `storage.save_transcript` (the single
    choke point every save path goes through: meeting pipeline, watch mode,
    dictation, `transcribe`, `retry`, enrichment rewrites). Runs a daemon
    thread; retries 3× with 1/3/9 s backoff; **logs only, never raises** — a
    dead server must not break a local save.
  - `sync_all()` — one-shot push of every transcript in `NOTES_DIR` for the
    `trnscrb sync` migration command.
  - HTTP via `httpx` (already an `mcp` dependency).
- `trnscrb sync` (CLI) — pushes all local transcripts to the configured server
  and reports ok/failed counts; errors out with a clear message when
  `remote_url` is unset.

### Consuming the server's MCP (Mac side)

OpenCode V2 remote MCP, header credential (OAuth disabled because this server
only speaks an API-key header):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "trnscrb": {
        "type": "remote",
        "url": "http://<server>:8765/mcp",
        "oauth": false,
        "headers": { "Authorization": "Bearer {env:TRNSCRB_TOKEN}" }
      }
    }
  }
}
```

Tools appear as `trnscrb_list_transcripts`, `trnscrb_semantic_search`, …
The local `trnscrb` stdio server keeps serving the recording/dictation tools.

### Server-host installation (Linux) — one command

`install-server.sh` (repo root, also curlable) is the supported novice path:

```sh
curl -fsSL https://raw.githubusercontent.com/artback/trnscrb/main/install-server.sh | bash
```

It: installs `libportaudio2` when missing (needed at `sounddevice` import
time), creates `~/trnscrb-server/venv` (via `uv` when available, plain venv
otherwise), installs the minimal serve dependency set (`mcp==2.2.0`, `numpy`,
`sounddevice`) plus `trnscrb --no-deps` (from git main), generates a token and
stores it in the server's `~/.config/trnscrb/settings.json` under
`server_token`, and prints the two paste-ready lines:

```sh
~/trnscrb-server/venv/bin/trnscrb serve --host 0.0.0.0 --port 8765
trnscrb config set remote_url http://<server>:8765   # on the Mac
```

The manual equivalent (for the curious): `apt-get install libportaudio2`,
`pip install "mcp==2.2.0" numpy sounddevice`, `pip install --no-deps trnscrb`.
The macOS-only deps (MLX, pyobjc, rumps) are never imported by the serve path.
`semantic_search` additionally benefits from `sentence-transformers` (+ torch
CPU) on the server; without it the tool degrades to its existing
"not installed" message and keyword search keeps working.

## Files changed

| File | Change |
| --- | --- |
| `trnscrb/server_http.py` | new — network server: app builder, auth middleware, ingest REST routes, token auto-generation, `run()` |
| `trnscrb/push.py` | new — client push (`request_push`, `push_file`, `sync_all`, `status`) |
| `trnscrb/storage.py` | `save_transcript` hooks `push.request_push` (no-op unless configured) |
| `trnscrb/settings.py` | new defaults: `remote_url`, `remote_token`, `server_token` |
| `trnscrb/cli.py` | registers new `serve` and `sync` commands (defined in the modules above) |
| `install-server.sh` | new — one-command server installer for the Linux host |
| `tests/test_server_http.py` | new — app-level tests (auth, ingest, MCP handshake, traversal, token bootstrap) |
| `tests/test_push.py` | new — client push + sync tests against a live in-process server |
| `README.md` | new "Client/server mode" section |

## Edge cases

- **Path traversal in upload filename** → 400, file not written (pattern +
  `is_relative_to` guard, mirroring `storage.read_transcript`).
- **Re-upload of the same transcript** → idempotent overwrite (enrichment
  rewrites, retry syncs).
- **Server unreachable while a meeting ends** → local save is never affected;
  push retries then gives up and logs; `trnscrb sync` picks it up later
  (uploads are full-file, so a later sync repairs any missed push).
- **Enrichment on the server vs. client** → last local save wins on next
  push; acceptable for v1, documented.
- **No token + remote bind** → server refuses to start; `--insecure` is the
  only override and logs a warning.
- **`remote_url` unset** → `request_push` returns before doing anything
  (zero overhead, zero behavior change for existing installs); `trnscrb sync`
  fails with a clear message.
- **Concurrent uploads of the same file** → both write identical content;
  last writer wins, no corruption (plain `write_text`, same bytes).

## Trade-offs

- Re-registering `mcp_server` functions on the network server means importing
  `trnscrb.mcp_server` on the server host (pulls `numpy` + `sounddevice`
  import-time; the ML backends stay lazy). Alternative — extracting the store
  tools into their own module — would touch the stable 900-line local server
  file for no behavioral gain.
- Bearer-token auth instead of the SDK's OAuth flow: one token, zero moving
  parts, works with every MCP client that sends headers; OAuth is overkill
  for a single-user store.
- Stateless MCP sessions: no resumable streams after a connection drop; a
  reconnect re-initializes (cheap for a read-mostly store).