"""Headless-server import guard.

``trnscrb serve`` runs on Linux hosts that may have no audio stack at all
(no PulseAudio/ALSA host API). sounddevice initializes PortAudio at import
time and raises PortAudioError on such hosts — so the server import chain
(mcp_server → dictation → recorder) must not import sounddevice at module
level.

The guard runs in a subprocess so module state is clean, and with the
sounddevice import blocked to simulate a headless machine.
"""

import subprocess
import sys


def _check(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_server_import_chain_does_not_load_sounddevice():
    """Importing the server stack must not pull in sounddevice at all —
    that is what keeps ``trnscrb serve`` alive on a host whose PortAudio
    cannot initialize."""
    code = (
        "import sys, trnscrb.mcp_server, trnscrb.server_http; "
        "assert 'sounddevice' not in sys.modules, 'sounddevice loaded at import time'"
    )
    r = _check(code)
    assert r.returncode == 0, r.stderr


def test_server_builds_with_sounddevice_unavailable():
    """Simulate a headless host: block the sounddevice import, then the
    server app must still build with the full store toolset."""
    code = (
        "import sys; sys.modules['sounddevice'] = None; "
        "import trnscrb.mcp_server; "
        "from trnscrb import server_http; "
        "app = server_http.build_app(token='test'); "
        "assert len(server_http.STORE_TOOLS) == 11"
    )
    r = _check(code)
    assert r.returncode == 0, r.stderr
