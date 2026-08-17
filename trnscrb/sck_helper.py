"""Drive the bundled sck-capture helper for system-audio capture.

macOS attributes the Screen Recording permission to the process that asks
for it. When Python asks, that process is Homebrew's Python.app
(``org.python.python``) — an identity we cannot claim, so the grant can only
ever be given to "Python", never to Trnscrb. Re-signing the interpreter
inside the bundle was tried and is killed by code-signing enforcement under
launchd.

The helper sidesteps all of it: a small Swift binary inside Trnscrb.app that
does nothing but ScreenCaptureKit capture and streams raw 16 kHz mono
float32 PCM over a pipe. It carries the bundle's identity, so the permission
is scoped to Trnscrb alone — Python needs no screen-recording rights at all.
"""

import ctypes
import os
import signal
import struct
import threading
from pathlib import Path

import numpy as np

from trnscrb.log import get_logger

_log = get_logger("trnscrb.sck_helper")

_READ_FRAMES = 1600  # 100 ms at 16 kHz

# Each chunk arrives as [magic][age float32][sample count uint32][float32 PCM].
# The age is how long ago the audio was actually captured, which is the only
# way to lay system audio down beside the microphone at the moment it
# happened rather than the moment it arrived.
_MAGIC = b"TRNS"
_HEADER_BYTES = 12


def _spawn_disclaimed(args: list[str], stdout_fd: int, stderr_fd: int) -> int:
    """posix_spawn a child that is its own TCC responsible process.

    This is the crux of the whole design. A normally-spawned child inherits
    our responsibility, so macOS would evaluate Screen Recording against
    Homebrew's Python.app (org.python.python) and deny it — the exact failure
    that made every earlier approach useless. Disclaiming makes the helper
    answer for itself, i.e. as the Trnscrb-signed binary inside the bundle.
    """
    libc = ctypes.CDLL(None, use_errno=True)

    attr = ctypes.c_void_p()
    if libc.posix_spawnattr_init(ctypes.byref(attr)) != 0:
        raise OSError("posix_spawnattr_init failed")
    actions = ctypes.c_void_p()
    if libc.posix_spawn_file_actions_init(ctypes.byref(actions)) != 0:
        libc.posix_spawnattr_destroy(ctypes.byref(attr))
        raise OSError("posix_spawn_file_actions_init failed")

    try:
        if libc.responsibility_spawnattrs_setdisclaim(ctypes.byref(attr), ctypes.c_int(1)) != 0:
            raise OSError("responsibility_spawnattrs_setdisclaim failed")
        libc.posix_spawn_file_actions_adddup2(
            ctypes.byref(actions), ctypes.c_int(stdout_fd), ctypes.c_int(1)
        )
        libc.posix_spawn_file_actions_adddup2(
            ctypes.byref(actions), ctypes.c_int(stderr_fd), ctypes.c_int(2)
        )

        argv = (ctypes.c_char_p * (len(args) + 1))(*[a.encode() for a in args], None)
        env_items = [f"{k}={v}".encode() for k, v in os.environ.items()]
        envp = (ctypes.c_char_p * (len(env_items) + 1))(*env_items, None)

        pid = ctypes.c_int()
        rc = libc.posix_spawn(
            ctypes.byref(pid),
            args[0].encode(),
            ctypes.byref(actions),
            ctypes.byref(attr),
            argv,
            envp,
        )
        if rc != 0:
            raise OSError(rc, f"posix_spawn failed for {args[0]}")
        return pid.value
    finally:
        libc.posix_spawn_file_actions_destroy(ctypes.byref(actions))
        libc.posix_spawnattr_destroy(ctypes.byref(attr))


def helper_path() -> Path | None:
    """Locate the capture helper inside the installed or packaged bundle."""
    from trnscrb.app_bundle import HELPER_NAME, bundle_path

    candidates = [bundle_path() / "Contents" / "MacOS" / HELPER_NAME]
    try:
        import shutil

        binary = shutil.which("trnscrb")
        if binary:
            real = Path(binary).resolve()
            for parent in list(real.parents)[:5]:
                candidates.append(parent / "Trnscrb.app" / "Contents" / "MacOS" / HELPER_NAME)
    except Exception:
        pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def has_permission() -> bool | None:
    """True/False from the helper's own preflight, None if unavailable.

    Must run as a child of this process: TCC evaluates the *responsible*
    process, so asking from a terminal would answer for the terminal.
    """
    helper = helper_path()
    if helper is None:
        return None
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        pid = _spawn_disclaimed([str(helper), "--check"], devnull, devnull)
        _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status) == 0
    except Exception:
        _log.debug("Helper permission check failed", exc_info=True)
        return None
    finally:
        if devnull is not None:
            os.close(devnull)


class HelperCapture:
    """System-audio capture via the bundled helper process."""

    def __init__(self, on_chunk):
        self._on_chunk = on_chunk
        self._pid: int | None = None
        self._stdout: int | None = None
        self._stderr = None
        self._reader: threading.Thread | None = None
        self._running = False

    @staticmethod
    def available() -> bool:
        return helper_path() is not None

    def start(self, timeout: float = 10.0) -> None:
        helper = helper_path()
        if helper is None:
            raise RuntimeError("sck-capture helper not found in the app bundle")

        out_r, out_w = os.pipe()
        err_r, err_w = os.pipe()
        try:
            self._pid = _spawn_disclaimed([str(helper), "--sck-capture"], out_w, err_w)
        finally:
            os.close(out_w)
            os.close(err_w)

        self._stdout = out_r
        self._stderr = os.fdopen(err_r, "rb", buffering=0)

        # The helper prints READY once capture is live, or an ERROR line.
        line = self._stderr.readline().decode(errors="replace").strip()
        if not line.startswith("READY"):
            self.stop()
            raise RuntimeError(line or "helper exited before starting capture")

        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        _log.info("System audio capture started (bundled helper, pid %s)", self._pid)

    def _read_loop(self) -> None:
        chunk_bytes = _READ_FRAMES * 4  # float32
        pending = b""
        framed = None  # decided by the first bytes; see _take_frame
        while self._running and self._stdout is not None:
            try:
                data = os.read(self._stdout, chunk_bytes)
            except Exception:
                break
            if not data:
                break
            pending += data
            if framed is None and len(pending) >= _HEADER_BYTES:
                framed = pending[:4] == _MAGIC
                if not framed:
                    # An older bundle that streams bare PCM. Still usable, just
                    # unaligned — the bundle is replaced on the next install.
                    _log.warning(
                        "System audio helper predates capture timestamps; "
                        "audio cannot be aligned. Run `trnscrb install`."
                    )
            if framed is False:
                usable = len(pending) - (len(pending) % 4)
                if usable:
                    self._on_chunk(np.frombuffer(pending[:usable], dtype=np.float32).copy(), 0.0)
                    pending = pending[usable:]
                continue
            while framed:
                taken, pending = self._take_frame(pending)
                if not taken:
                    break
        if self._running:
            _log.warning("System audio helper stopped delivering audio")

    def _take_frame(self, pending: bytes) -> tuple[bool, bytes]:
        """Pull one [magic][age][count] frame off the buffer, if a whole one is there."""
        if len(pending) < _HEADER_BYTES:
            return False, pending
        if pending[:4] != _MAGIC:
            # Resync rather than give up: one torn frame should not end capture.
            index = pending.find(_MAGIC, 1)
            if index < 0:
                return False, pending[-3:]  # keep only what could start a magic
            _log.debug("Resynced system audio stream, dropped %d byte(s)", index)
            return True, pending[index:]
        age, count = struct.unpack_from("<fI", pending, 4)
        end = _HEADER_BYTES + count * 4
        if len(pending) < end:
            return False, pending
        samples = np.frombuffer(pending[_HEADER_BYTES:end], dtype=np.float32).copy()
        self._on_chunk(samples, float(age))
        return True, pending[end:]

    def stop(self) -> None:
        self._running = False
        pid, self._pid = self._pid, None
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):  # up to 5 s
                    if os.waitpid(pid, os.WNOHANG)[0]:
                        break
                    threading.Event().wait(0.1)
                else:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
            except Exception:
                _log.debug("Helper shutdown issue", exc_info=True)
        if self._stdout is not None:
            try:
                os.close(self._stdout)
            except OSError:
                pass
            self._stdout = None
        if self._stderr is not None:
            try:
                self._stderr.close()
            except Exception:
                pass
            self._stderr = None
        _log.info("System audio capture stopped (bundled helper)")
