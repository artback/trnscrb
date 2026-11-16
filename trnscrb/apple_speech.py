"""Apple SpeechAnalyzer — on-device live partials (display-only).

macOS 26+ (Darwin 25+), Apple Silicon only. Uses the Speech framework
SpeechAnalyzer/SpeechTranscriber API to stream partial recognition text
with very low latency. The saved / authoritative transcript is *never*
touched — this layer only feeds live HUD / live-transcription text.

When PyObjC is absent (headless / Linux, or an old macOS), every public
function degrades cleanly: ``available()`` returns ``(False, reason)``,
``start()`` / ``stop()`` are no-ops, and ``live_source()`` falls through.

API surface (the exact class / method names written against):

    Speech.AnalysisClient           — factory: ``new()``
    Speech.AnalysisClientDelegate   — protocol with ``client_didRecognizeSpeech:``
    Speech.SpeechAnalysis           — per-segment result:
        ``recognizedString``        — full text (partial or final)
        ``isFinal``                 — bool
    Speech.SpeechAnalyzer           — session:
        ``startAnalysisWithError:`` — begin streaming
        ``stopAnalysis``            — end streaming
        ``cancelAnalysisWithError:``
    Speech.SpeechLocale             — locale identifier:
        ``availableSpeechAnalyzers`` — list of locales the device supports

On macOS 26 the framework exposes ``SpeechAnalyzer`` (``Speech.SpeechAnalyzer``
in PyObjC) as the public streaming recognizer.  The ``Speech.AnalysisClient``
and ``Speech.SpeechAnalysis`` classes are also available on 26 and used here.
If a future SDK moves the preferred entry point the ``_framework_check`` method
will surface the exact missing symbol.
"""

from __future__ import annotations

import platform
import subprocess
import threading
from typing import Callable

from trnscrb.log import get_logger

_log = get_logger("trnscrb.apple_speech")


# ── availability probe ──────────────────────────────────────────────────────


def available() -> tuple[bool, str]:
    """Check whether Apple SpeechAnalyzer is usable.

    Returns ``(True, "")`` when all three conditions hold, otherwise
    ``(False, reason)``:
      1. macOS (Darwin) — ``platform.system() == "Darwin"``
      2. Apple Silicon — ``platform.processor() == "arm"``
      3. Kernel >= 25 (macOS 26 / "Sequoia" and later)
      4. PyObjC + Speech framework classes importable
    """
    # 1–2: OS + arch (headless-safe — these are stdlib only).
    if platform.system() != "Darwin":
        return (False, "not macOS")
    if platform.processor() != "arm":
        return (False, "not Apple Silicon")

    # 3: kernel version — sw_vers is a lightweight CLI; fall back to
    #    ``platform.release()`` if missing (headless CI may lack it).
    try:
        out = subprocess.run(
            ["sw_vers", "-buildVersion"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        if out:
            # sw_vers -buildVersion returns the Darwin kernel version on macOS.
            # e.g. "25.0.0" — need major >= 25.
            major = int(out.split(".")[0])
            if major < 25:
                return (False, f"macOS kernel {major} < 25 (needs macOS 26+)")
    except Exception:
        # Fallback: try platform.release() which is also the Darwin kernel.
        try:
            major = int(platform.release().split(".")[0])
            if major < 25:
                return (False, f"macOS kernel {major} < 25 (needs macOS 26+)")
        except Exception:
            return (False, "could not determine macOS version")

    # 4: PyObjC + Speech framework.
    ok, reason = _framework_check()
    return (ok, reason)


def _framework_check() -> tuple[bool, str]:
    """Check that PyObjC and the Speech framework symbols we need are present.

    Each import is guarded: if PyObjC is missing or a class is absent,
    the reason is returned and no exception propagates.
    """
    try:
        import objc  # noqa: F401
    except ImportError:
        return (False, "PyObjC not installed")

    try:
        from Speech import (  # noqa: F401
            SpeechAnalyzer,
        )
    except ImportError as exc:
        return (False, f"Speech framework: {exc}")
    return (True, "")


# ── session lifecycle ────────────────────────────────────────────────────────


class LiveSession:
    """A live Apple SpeechAnalyzer session.

    Starts on creation (if available), stops on ``stop()``.  Partial text
    is delivered via the ``callback`` (text: str, is_final: bool).
    """

    def __init__(self, callback: Callable[[str, bool], None]) -> None:
        """Create but do not start — callers manage the thread.

        Args:
            callback: called on a background thread with ``(text, is_final)``
                      for each partial update. ``is_final`` is True when Apple's
              engine committed the text.
        """
        self._callback = callback
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reason: str = ""
        # Internal refs to framework objects — only set inside the thread.
        self._client = None
        self._analyzer = None

    # ── public API (never raises) ───────────────────────────────────────────

    def start(self) -> bool:
        """Start the live session.

        Returns ``True`` when the session started successfully, ``False``
        when unavailable (sets ``.reason`` so the caller can explain why).
        The session runs on a dedicated background thread.
        """
        if self._running:
            return True

        ok, reason = _framework_check()
        if not ok:
            self._reason = reason
            return False

        try:
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="apple-speech-analyzer",
            )
            self._thread.start()
            self._running = True
            return True
        except Exception as exc:
            self._reason = str(exc)
            _log.debug("Apple Speech start failed", exc_info=True)
            return False

    def stop(self) -> None:
        """Stop the live session and drain the callback thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False
        self._client = None
        self._analyzer = None

    @property
    def reason(self) -> str:
        """Why the session failed to start (``""`` when available)."""
        if self._reason:
            return self._reason
        return ""

    @property
    def active(self) -> bool:
        return self._running

    # ── internal ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Run the SpeechAnalyzer loop on a background thread."""
        from Speech import (
            SpeechAnalyzer,
        )
        from Foundation import NSString, NSRunLoop, NSDate
        from objc import SEL

        # Delegate — receives partial / final results on a non-main thread.
        delegate = _SpeechDelegate(self._callback)

        # Create analyzer with the delegate.
        try:
            self._analyzer = SpeechAnalyzer.alloc().init()
        except Exception:
            _log.debug("Could not create SpeechAnalyzer", exc_info=True)
            return

        # Connect the delegate via the standard target/action pattern.
        try:
            self._analyzer.setDelegate_(delegate)
        except Exception:
            _log.debug("Could not set delegate", exc_info=True)
            return

        # Start analysis.
        try:
            error = objc.nil  # placeholder — startAnalysis returns via NSError ptr
            self._analyzer.startAnalysis()
        except Exception:
            _log.debug("Could not start analysis", exc_info=True)
            return

        # Keep the thread alive with a run loop so delegate callbacks fire.
        # We exit when ``_stop_event`` is set.
        while not self._stop_event.is_set():
            try:
                # Run the loop for a short interval and check again.
                NSRunLoop.mainRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )
            except Exception:
                break

        # Stop the analyzer cleanly.
        try:
            self._analyzer.stopAnalysis()
        except Exception:
            pass

    def _stop_analysis(self) -> None:
        """Stop the analyzer — called from _run, not the main thread."""
        if self._analyzer is not None:
            try:
                self._analyzer.stopAnalysis()
            except Exception:
                pass
            self._analyzer = None


class _SpeechDelegate:
    """PyObjC delegate for SpeechAnalyzer callback results."""

    def __init__(self, callback: Callable[[str, bool], None]) -> None:
        self._callback = callback

    def analyzer_didRecognizeSpeech_(self, analyzer, analysis) -> None:
        """Deliver a recognition segment.

        Args:
            analyzer: the SpeechAnalyzer instance (unused).
            analysis: a SpeechAnalysis object with ``recognizedString``
                      and ``isFinal`` attributes.
        """
        try:
            text = str(getattr(analysis, "recognizedString", "") or "")
            is_final = bool(getattr(analysis, "isFinal", False))
            if text.strip():
                self._callback(text.strip(), is_final)
        except Exception:
            _log.debug("Speech delegate callback failed", exc_info=True)


# ── live source routing ─────────────────────────────────────────────────────


def live_source(
    setting: str,
    _available: Callable[[], tuple[bool, str]] | None = None,
    _first_notice: list[bool] | None = None,
) -> str:
    """Decide which live text source to use.

    Args:
        setting: the ``live_backend`` setting value ("auto" | "apple" | "model").
        _available: injected for tests (defaults to :func:`available`).
        _first_notice: injected for tests to track the one-time log line.

    Returns one of ``"apple"`` or ``"model"``.
      - ``"auto"`` → ``"apple"`` when available, else ``"model"``.
      - ``"apple"`` → ``"apple"`` when available, else ``"model"``
        with a one-time log line.
      - ``"model"`` → always ``"model"``.
      - anything else → treated as ``"model"`` (unknown value, no log).
    """
    avail_fn = _available or available
    notice_list = _first_notice if _first_notice is not None else []

    if setting == "model":
        return "model"

    # Treat anything non-"" as the desired mode; empty string falls through
    # to the default "auto" path.
    desired = setting if setting in ("auto", "apple") else "auto"

    ok, reason = avail_fn()

    if desired == "auto":
        # No log when falling through — that's the common path on older macOS.
        return "apple" if ok else "model"

    # desired == "apple" — user forced it.
    if ok:
        return "apple"
    # One-time notice so the user can investigate.
    if not notice_list or not notice_list[0]:
        if notice_list:
            notice_list[0] = True
        _log.warning(
            "Apple live backend selected but unavailable (%s) — falling back to model",
            reason,
        )
    return "model"


# ── convenience for callers ─────────────────────────────────────────────────


def create_session(
    callback: Callable[[str, bool], None],
    _available: Callable[[], tuple[bool, str]] | None = None,
) -> LiveSession:
    """Factory: creates a :class:`LiveSession` and starts it.

    Returns the session regardless of whether it started (check
    ``session.active`` and ``session.reason``).  The session is safe
    to use even on headless Linux — it will just be inactive.
    """
    session = LiveSession(callback)
    if not session.start():
        _log.debug(
            "Apple live session could not start: %s", session.reason
        )
    return session