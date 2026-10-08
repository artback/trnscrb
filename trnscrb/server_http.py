"""Network server — serve transcripts and MCP from a server.

Runs via ``trnscrb serve`` and exposes, on one port:

* MCP over Streamable HTTP (the store toolset)
* REST endpoints: health, transcript upload, list, fetch
* Bearer-token auth on every route, including the MCP endpoint
"""

import hmac
import json
import os
import re
import secrets

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

import trnscrb
from trnscrb import settings, storage
from trnscrb.log import get_logger

_log = get_logger("trnscrb.server_http")

# Store-side tools — the subset served remotely.
STORE_TOOLS = (
    "list_transcripts",
    "get_transcript",
    "get_weekly_transcripts",
    "get_weekly_summaries",
    "search_transcripts",
    "semantic_search",
    "enrich_transcript",
    "list_action_items",
    "resolve_action_item",
    "link_action_item",
    "add_action_item",
)

# Filename validation: [A-Za-z0-9] followed by up to 199 chars from the
# allowed set, ending in .txt — total ≤ 200 chars.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,198}\.txt$")


def _import_mcp_tools():
    """Import the store-side tool functions from the mcp server module."""
    import trnscrb.mcp_server as mcp_mod

    tools = {}
    for name in STORE_TOOLS:
        fn = getattr(mcp_mod, name, None)
        if fn is not None:
            tools[name] = fn
    return tools


def build_app(token: str | None = None, *, insecure: bool = False):
    """Build the ASGI app for ``trnscrb serve``.

    Wraps the MCPServer's Streamable HTTP transport in a bearer-token auth
    middleware.  Returns a callable ASGI app (not a bare Starlette instance)
    so that auth is applied to every route.

    Args:
        token: Bearer token for auth.  ``None`` = no auth (loopback dev mode).
        insecure: When True, allow bind to non-loopback without a token.

    Returns:
        An ASGI app with auth middleware + MCP + REST routes.
    """
    server = MCPServer("trnscrb", version=trnscrb.__version__)

    # Register store-side tools on this server instance.
    mcp_tools = _import_mcp_tools()
    for fn in mcp_tools.values():
        server.tool()(fn)

    # ── Custom REST routes ───────────────────────────────────────────────────

    # Helper: resolve a transcript_id/path safely (traversal guard).
    def _resolve_id(id_str: str):
        path = (storage.NOTES_DIR / f"{id_str}.txt").resolve()
        if not path.is_relative_to(storage.NOTES_DIR.resolve()):
            _log.warning("Path traversal blocked for %r", id_str)
            return None
        return path

    @server.custom_route("/api/health", methods=["GET"])
    async def health_endpoint(request: Request):
        transcript_count = len(list(storage.NOTES_DIR.glob("*.txt")))

        semantic_search_available = False
        try:
            from trnscrb import semantic_search as sem

            semantic_search_available = sem.available()
        except Exception:
            pass

        return JSONResponse(
            {
                "version": trnscrb.__version__,
                "transcript_count": transcript_count,
                "semantic_search_available": semantic_search_available,
            }
        )

    @server.custom_route("/api/transcript", methods=["POST"])
    async def ingest_transcript(request: Request):
        body = await request.json()
        filename = body.get("filename", "")
        text = body.get("text", "")

        if not filename or not text:
            return JSONResponse({"detail": "filename and text are required"}, status_code=400)

        if not _FILENAME_RE.match(filename):
            return JSONResponse({"detail": f"Invalid filename: {filename!r}"}, status_code=400)

        path = _resolve_id(filename[:-4])  # strip .txt
        if path is None:
            return JSONResponse({"detail": "Path traversal blocked"}, status_code=400)

        # Write directly, not via storage.save_transcript: that path carries
        # the client push hook, and an upload would re-push itself in a loop
        # whenever this process also has remote_url configured. The atomic
        # temp-file + replace keeps concurrent readers (and concurrent uploads
        # of the same file) from ever seeing a half-written or empty file.
        content = text if text.endswith("\n") else text + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / (path.name + ".part")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
        return JSONResponse({"stored": filename})

    @server.custom_route("/api/transcripts", methods=["GET"])
    async def list_transcripts_endpoint(request: Request):
        return JSONResponse({"transcripts": storage.list_transcripts()})

    @server.custom_route("/api/transcript/{id}", methods=["GET"])
    async def fetch_transcript_endpoint(request: Request):
        id_str = request.path_params["id"]
        path = _resolve_id(id_str)
        if path is None or not path.exists():
            return PlainTextResponse("Not found", status_code=404)
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    # ── MCP transport ────────────────────────────────────────────────────────
    inner = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )

    # ── Auth middleware (wraps everything) ───────────────────────────────────

    class _AuthMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, token_val: str | None = None):
            super().__init__(app)
            self._token = token_val

        async def dispatch(self, request: Request, call_next):
            if self._token is None:
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse(
                    {"detail": "Missing or invalid Authorization header"},
                    status_code=401,
                )
            provided = auth[7:]
            if not hmac.compare_digest(provided, self._token):
                return JSONResponse({"detail": "Invalid token"}, status_code=401)
            return await call_next(request)

    app = _AuthMiddleware(inner, token_val=token)
    return app


def generate_token() -> str:
    """Generate a random hex token (24 bytes → 48 hex chars)."""
    return secrets.token_hex(24)


def resolve_token(cli_token: str | None = None) -> str | None:
    """Resolve the server token from flag > env > settings.

    Args:
        cli_token: The --token flag value (may be None).

    Returns:
        The resolved token, or None when unconfigured.
    """
    if cli_token:
        return cli_token
    env = os.environ.get("TRNSCRB_SERVER_TOKEN")
    if env:
        return env
    s = settings.get("server_token")
    if s and str(s).strip():
        return str(s)
    return None


class BindPolicyError(RuntimeError):
    """Raised when the bind policy is violated (no token + non-loopback host)."""


def check_bind_policy(host: str, token: str | None, *, insecure: bool = False) -> None:
    """Raise ``BindPolicyError`` when token is None/empty and host is not loopback.

    Args:
        host: The bind host (e.g. "127.0.0.1", "0.0.0.0").
        token: The resolved server token.
        insecure: When True, skip the loopback requirement.
    """
    loopback = host in ("127.0.0.1", "::1", "localhost")
    if token or insecure or loopback:
        return
    raise BindPolicyError(
        f"Cannot bind to {host!r} without a token. "
        "Use --token, set the TRNSCRB_SERVER_TOKEN env var, "
        "or configure server_token."
    )


def start(
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    insecure: bool = False,
    **uvicorn_kwargs,
) -> None:
    """Start the server, blocking.

    Checks bind policy, builds the app with the resolved token, and runs
    via uvicorn.

    Args:
        host: Bind address (default 127.0.0.1).
        port: Bind port (default 8765).
        token: Bearer token for auth (None = no auth).
        insecure: Allow non-loopback without a token (logs warning).
        uvicorn_kwargs: Passed through to ``uvicorn.run``.
    """
    import uvicorn

    check_bind_policy(host, token, insecure=insecure)

    app = build_app(token=token, insecure=insecure)

    log_level = uvicorn_kwargs.pop("log_level", "info")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        **uvicorn_kwargs,
    )


def client_setup_snippet(url: str, token: str) -> str:
    """Return paste-ready client setup text.

    Args:
        url: The server base URL (e.g. http://10.0.0.5:8765).
        token: The bearer token.

    Returns:
        A string with config set commands + opencode MCP JSON block.
    """
    mcp_url = f"{url}/mcp"
    lines = [
        f"trnscrb config set remote_url {url}",
        f"trnscrb config set remote_token {token}",
        "trnscrb sync",
        "",
        "## OpenCode MCP config",
        "",
        "Paste this into your opencode.json mcp.servers section:",
        "",
        "```json",
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "mcp": {
                    "servers": {
                        "trnscrb": {
                            "type": "remote",
                            "url": mcp_url,
                            "oauth": False,
                            "headers": {"Authorization": "Bearer {env:TRNSCRB_TOKEN}"},
                        }
                    }
                },
            },
            indent=2,
        ),
        "```",
    ]
    return "\n".join(lines)
