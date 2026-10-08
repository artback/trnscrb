#!/usr/bin/env bash
# install-server.sh — One-command Linux server installer for trnscrb.
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/artback/trnscrb/main/install-server.sh | bash

set -euo pipefail

SERVER_DIR="$HOME/trnscrb-server"
VENV_DIR="$SERVER_DIR/venv"
HOST="${TRNSCRB_SERVER_HOST:-0.0.0.0}"
PORT="${TRNSCRB_SERVER_PORT:-8765}"
TOKEN=""

# ── 1. libportaudio2 ─────────────────────────────────────────────────────────
if dpkg -l libportaudio2 2>/dev/null | grep -q '^ii'; then
    echo "  ✓ libportaudio2 already installed"
elif command -v apt-get &>/dev/null; then
    if [ "$(id -u)" -eq 0 ]; then
        echo "  Installing libportaudio2 …"
        apt-get install -y libportaudio2
    else
        echo "  libportaudio2 not found."
        echo "  Run as root (or sudo): apt-get install -y libportaudio2"
    fi
else
    echo "  ⚠ No apt-get found — install libportaudio2 manually."
fi

# ── 2. Python venv ───────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "  Creating venv with uv …"
    uv venv "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
        "mcp==2.2.0" numpy sounddevice
    PIP_CMD="uv pip install --python $VENV_DIR/bin/python"
elif command -v python3 &>/dev/null; then
    echo "  Creating venv …"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install \
        "mcp==2.2.0" numpy sounddevice
    PIP_CMD="$VENV_DIR/bin/pip install"
else
    echo "  ✗ Need python3 or uv to create a venv."
    exit 1
fi

# ── 3. trnscrb (no extra deps) ──────────────────────────────────────────────
if [ -f pyproject.toml ] && [ -d trnscrb ]; then
    echo "  Installing trnscrb from this checkout (no-deps) …"
    $PIP_CMD --no-deps .
else
    echo "  Installing trnscrb from git (no-deps) …"
    $PIP_CMD --no-deps "git+https://github.com/artback/trnscrb.git"
fi

# ── 4. Generate and store token ───────────────────────────────────────────────
echo "  Generating token …"
TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(24))")
"$VENV_DIR/bin/python" -c "
from trnscrb import settings
settings.put('server_token', '$TOKEN')
"

# ── 5. Print setup instructions ──────────────────────────────────────────────
SERVER_URL="http://${HOST}:${PORT}"

echo ""
echo "  ════════════════════════════════════════════════════════════"
echo "  Server install complete."
echo ""
echo "  1. Start the server (run as a background service):"
echo "     $VENV_DIR/bin/trnscrb serve --host $HOST --port $PORT"
echo ""
echo "  2. On your Mac, configure the remote:"
echo "     trnscrb config set remote_url $SERVER_URL"
echo "     trnscrb config set remote_token $TOKEN"
echo "     trnscrb sync"
echo ""
echo "  3. Paste this into opencode.json mcp.servers:"
"$VENV_DIR/bin/python" -c "
from trnscrb.server_http import client_setup_snippet
print(client_setup_snippet('$SERVER_URL', '$TOKEN'))
"
echo "  ════════════════════════════════════════════════════════════"
echo ""