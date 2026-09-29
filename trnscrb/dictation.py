"""Dictation — mic-only capture for short voice notes.

Phase 1 of the dictation feature. Two presets, both deliberately outside the
meeting pipeline:

  message    — capture with the mic only, transcribe with the glossary applied,
               copy the verbatim text to the clipboard (pbcopy), and save a
               `message-<HHMM>` note. No AI summary, no filler removal.
  brain-dump — captured the same way and saved as a `brain-dump-<HHMM>` note;
               optionally turned into a draft afterwards with
               `trnscrb dictation draft <id>` (see prompts/draft.md).

Raw output is the point: a dictation keeps the speaker's exact phrasing, so
nothing is stripped or "readability"-processed. Command mode and the
voice-trained glossary are Phase 2, not here.

Every surface ends in one function — `finish()` — which transcribes, saves,
and copies to the clipboard. The menu bar app and MCP server own their
Recorders in-process; the CLI's `dictation start` spawns a detached child that
records until `dictation stop` signals it.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from trnscrb import settings, storage, transcriber
from trnscrb.log import get_logger
from trnscrb.recorder import SAMPLE_RATE, Recorder

_log = get_logger("trnscrb.dictation")

PRESETS = ("message", "brain-dump")
PRESET_LABELS = {"message": "Message", "brain-dump": "Brain dump"}

# How often the dictation live display emits newly transcribed words.
# Dictation is short and mic-only, so updates run far tighter than the
# meeting live loop (which ticks every 60s).
_LIVE_INTERVAL = 3.0

_CONTROL_DIR = Path.home() / ".config" / "trnscrb"
_PID_FILE = _CONTROL_DIR / "dictation.pid"
_RESULT_FILE = _CONTROL_DIR / "dictation_result.json"

# ── app-side control channel (SIGUSR2 + request file) ─────────────────────────
# The CLI command "trnscrb dictate <preset>" signals the menu-bar app via
# SIGUSR2. Because Unix signals can't carry a preset, a small JSON request file
# does the heavy lifting: the app reads it on receipt.

_START_REQUEST_FILE = _CONTROL_DIR / "dictation_request.json"


def write_start_request(preset: str, save_note: bool = True) -> None:
    """Write a dictation-start request for the menu-bar app to pick up."""
    _CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        payload = {"preset": preset, "save_note": bool(save_note)}
        _START_REQUEST_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        _log.debug("Could not write dictation request", exc_info=True)


def read_start_request() -> dict | None:
    """The latest request as ``{"preset": ..., "save_note": ...}``, or None.

    Legacy request files without the ``save_note`` key read as save — a
    missing flag must never silently turn saving off.
    """
    try:
        payload = json.loads(_START_REQUEST_FILE.read_text())
        preset = payload.get("preset") if isinstance(payload, dict) else None
        if not is_preset(preset):
            return None
        return {"preset": preset, "save_note": bool(payload.get("save_note", True))}
    except Exception:
        return None


def clear_start_request() -> None:
    """Remove the request file so the app never picks up a stale one."""
    try:
        _START_REQUEST_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ── command mode ──────────────────────────────────────────────────────────────
# Command-mode dictation maps spoken phrases to actions (stop meeting, bookmark,
# etc.). The matcher is pure data + a single function so it is easy to test and
# reuse across the menu bar, CLI, and MCP.

_COMMANDS: list[dict] = [
    {
        "name": "stop_meeting",
        "trigger": "stop the meeting",
        "phrases": [
            "stop meeting",
            "stop the meeting",
            "end meeting",
            "end the meeting",
            "stop recording",
            "stop the recording",
            "stop transcribing",
            "stop the transcribing",
            "stop",
        ],
        "label": "Stop meeting",
    },
    {
        "name": "start_meeting",
        "trigger": "start the meeting",
        "phrases": [
            "start meeting",
            "start the meeting",
            "begin meeting",
            "begin the meeting",
            "start recording",
            "start the recording",
            "start transcribing",
            "start the transcribing",
            "begin",
        ],
        "label": "Start meeting",
    },
    {
        "name": "bookmark",
        "trigger": "bookmark this",
        "phrases": [
            "bookmark",
            "bookmark this",
            "mark this",
            "add a bookmark",
            "place a bookmark",
        ],
        "label": "Bookmark",
    },
    {
        "name": "stop_dictation",
        "trigger": "stop dictation",
        "phrases": [
            "stop dictation",
            "stop the dictation",
            "stop message",
            "stop brain dump",
        ],
        "label": "Stop dictation",
    },
    {
        "name": "start_message",
        "trigger": "start a message",
        "phrases": [
            "start a message",
            "start message dictation",
            "dictation message",
        ],
        "label": "Start message dictation",
    },
    {
        "name": "start_brain_dump",
        "trigger": "start a brain dump",
        "phrases": [
            "start a brain dump",
            "start brain dump dictation",
            "brain dump dictation",
        ],
        "label": "Start brain-dump dictation",
    },
]

_COMMAND_PHRASES: dict[str, str] = {}
for _cmd in _COMMANDS:
    for _phrase in _cmd["phrases"]:
        _COMMAND_PHRASES[_phrase] = _cmd["name"]


def match_command(plain_text: str) -> dict | None:
    """Match a dictation transcript against known commands.

    Returns ``{"command": "stop_meeting", "matched": "stop meeting"}`` or
    ``None`` when no command matches.
    """
    if not plain_text:
        return None
    # Normalize: lowercase, strip punctuation, collapse whitespace.
    import re

    normalized = re.sub(r"[^\w\s]", "", plain_text.lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    # Try the phrase table.
    if normalized in _COMMAND_PHRASES:
        cmd = _COMMAND_PHRASES[normalized]
        for _c in _COMMANDS:
            if _c["name"] == cmd:
                return {"command": cmd, "matched": _c["trigger"], "label": _c["label"]}
    # Partial match: the transcript *starts with* a known command phrase.
    # Only match if the rest of the transcript is short (a hesitation filler).
    # This catches "stop the meeting please" without firing on "what does stop
    # mean" — only consider prefixes <= 5 words when the rest is short.
    for phrase, cmd in _COMMAND_PHRASES.items():
        if normalized.startswith(phrase):
            remainder = normalized[len(phrase) :].strip()
            if not remainder or len(remainder.split()) <= 2:
                for _c in _COMMANDS:
                    if _c["name"] == cmd:
                        return {"command": cmd, "matched": phrase, "label": _c["label"]}
    return None


# Matches storage._SEPARATOR: the line that splits a note's header from its
# spoken body. Kept in sync deliberately so note_body() can find it.
_BODY_SEP = "=" * 60


def preset_label(preset: str) -> str:
    """User-facing label for a preset ("Message", "Brain dump")."""
    return PRESET_LABELS.get(preset, preset.replace("-", " ").title())


def is_preset(value: str) -> bool:
    return value in PRESETS


def meeting_in_progress() -> bool:
    """True when a meeting is being recorded, so dictation must wait.

    The live-session file is written by the meeting recorder and validated
    against its PID, so a crashed session can never block dictation forever.
    """
    return storage.get_live_session_info() is not None


# Convenience alias — callers in tests expect the module-level function.
get_live_session_info = storage.get_live_session_info


def refusing_reason() -> str | None:
    """Why dictation cannot start right now, or None when it can."""
    info = storage.get_live_session_info()
    if info:
        # Only refuse while a meeting is actively recording.  During
        # transcription the recorder has already released the mic, so
        # dictation can safely run alongside a finishing meeting.
        app_state = storage.read_app_state()
        if app_state and app_state.get("state") == "recording":
            return "A meeting is actively recording. Stop the meeting first, then dictate."
    return None


def new_recorder() -> Recorder:
    """A mic-only recorder, started and ready to capture."""
    recorder = Recorder(system_audio=False)
    recorder.start()
    return recorder


def plain_text(segments: list[dict]) -> str:
    """The verbatim dictation: segment text joined into one flowing passage.

    The transcriber splits speech into segments at pauses, but a dictation is
    a single continuous utterance — joining with spaces keeps it reading
    exactly as spoken instead of breaking mid-sentence ("I / hope testing…").
    """
    return " ".join(
        str(seg.get("text") or "").strip() for seg in segments if str(seg.get("text") or "").strip()
    )


def _noop(_text: str) -> None:
    """No-op sink for live text when the caller does not display it."""


def live_transcribe(
    recorder: Recorder,
    stop_event: threading.Event,
    interval: float = _LIVE_INTERVAL,
    on_text: Callable[[str], None] | None = None,
) -> list[dict]:
    """Transcribe newly captured audio and hand each chunk to ``on_text``.

    A display-only loop for the foreground CLI: run it in its own thread
    while the user speaks, and each pass transcribes only the audio captured
    since the previous pass (the same incremental pattern as the meeting
    live loop), so the work per tick stays constant. ``on_text`` receives
    each new chunk's plain text; pass None to collect silently.

    The model is preloaded first, in parallel with the opening seconds of
    speech, so the first update is not gated on a model load. The saved note
    is still produced by ``finish()`` from the full recording, so a dropped
    or late pass can never affect the transcript. Returns the segments seen
    so far (timestamps relative to each snapshot).
    """
    if on_text is None:
        on_text = _noop

    try:
        transcriber.preload()
    except Exception:
        _log.debug("Live display could not preload the transcription model", exc_info=True)

    transcribed_frames = 0
    segments_acc: list[dict] = []

    while not stop_event.is_set():
        if stop_event.wait(interval):
            break
        result = recorder.snapshot_since(transcribed_frames)
        if not result:
            continue
        snap, end_frame = result
        try:
            new_segments = transcriber.transcribe(snap)
            segments_acc.extend(new_segments)
            transcribed_frames = end_frame
            plain = plain_text(new_segments)
            if plain:
                on_text(plain)
        except Exception:
            # A failed pass (model hiccup, busy GPU) must not kill the loop;
            # the same audio is retried next tick and the final note comes
            # from finish() regardless.
            _log.debug("Dictation live transcription update failed", exc_info=True)
        finally:
            try:
                snap.unlink(missing_ok=True)
            except OSError:
                pass

    return segments_acc


# ── auto-stop on silence ──────────────────────────────────────────────────────
# The "normal app" end of a dictation: you stop talking, the dictation stops.
#
# _DICTATION_MIN_SECS — a leading pause (hotkey pressed before starting to
#      speak) must not kill a dictation that hasn't started yet.
# _DICTATION_SILENCE_SECS — continuous quiet that counts as "done talking".
#      Short enough to feel instant (the HUD already shows your words), long
#      enough that a natural mid-sentence breath doesn't cut you off.
# _DICTATION_MAX_SECS — safety net so a forgotten dictation never records
#      the whole afternoon.
_DICTATION_MIN_SECS = 3.0
_DICTATION_SILENCE_SECS = 1.5
_DICTATION_MAX_SECS = 180.0
# _STALE_AUDIO_SECS — the mic stopped *delivering* audio (device swap, driver
#      wedge), which is different from the user being quiet. Deliberately
#      longer than the recorder's own ~3s stream-recovery window, so a
#      transient device swap that recovers does not end a dictation.
_STALE_AUDIO_SECS = 5.0

# Energy thresholds on the recorder's per-block mic mean-square timeline
# (one block every 1024 frames ≈ 64 ms at 16 kHz).
_SILENCE_ENERGY = 2e-5  # ≈ -47 dBFS: below this the mic is effectively silent
_SPEECH_PEAK = 1e-4  # ≈ -40 dBFS: typical speech level on a laptop mic


def silence_watchdog(
    recorder: Recorder,
    stop_event: threading.Event,
    on_stop,
    interval: float = 0.5,
    min_secs: float = _DICTATION_MIN_SECS,
    silence_secs: float = _DICTATION_SILENCE_SECS,
    max_secs: float = _DICTATION_MAX_SECS,
    stale_secs: float = _STALE_AUDIO_SECS,
) -> None:
    """End a dictation on its own when the speaker stops talking.

    Runs in its own thread while a dictation is active. Each pass reads the
    recorder's per-block mic-energy timeline and calls ``on_stop`` — exactly
    once — after ``silence_secs`` of continuous quiet (only once at least
    ``min_secs`` of audio exist) or after ``max_secs`` as a safety net.
    Silence is detected adaptively: below the absolute floor, or a 20 dB drop
    from the loudest speech heard so far, so quiet speakers and loud rooms
    both work without per-user tuning.

    A third stop path covers mic death: if the timeline stopped *growing*
    (audio no longer arriving) for ``stale_secs`` after audio had been
    flowing, the dictation is ended so a wedged mic cannot burn the whole
    ``max_secs`` budget on dead audio.
    """
    start = time.monotonic()
    silent_since: float | None = None
    stale_since: float | None = None
    last_block_count = 0
    stopped = False
    blocks_per_sec = max(1, int(SAMPLE_RATE / 1024))

    while not stop_event.is_set():
        if stopped or stop_event.wait(interval):
            break
        try:
            _offsets, mic, _sys = recorder.attribution_timeline()
        except Exception:
            _log.debug("Silence watchdog: could not read the energy timeline", exc_info=True)
            continue
        if len(mic) == 0:
            continue
        now = time.monotonic()
        if len(mic) > last_block_count:
            last_block_count = len(mic)
            stale_since = None
        elif last_block_count > 0 and stale_secs > 0:
            # Audio was flowing and then stopped arriving: the mic stream is
            # wedged (or the device is gone). The recorder's own watchdog
            # reopens a stalled stream after ~3s, so only treat it as a hard
            # stall past that recovery window.
            stale_since = stale_since or now
            if now - stale_since >= stale_secs:
                _log.warning(
                    "Dictation auto-stopped: mic stopped delivering audio for %.0fs",
                    stale_secs,
                )
                on_stop()
                stopped = True
                break
        elapsed = now - start
        if elapsed >= max_secs:
            _log.info("Dictation auto-stopped after %.0fs (max duration)", elapsed)
            on_stop()
            stopped = True
            break
        if elapsed < min_secs:
            silent_since = None
            continue
        recent = mic[-blocks_per_sec:]
        if len(recent) < blocks_per_sec // 2:
            continue  # not enough timeline yet to judge the last second
        recent_ms = sum(recent) / len(recent)
        peak_ms = max(mic)
        silent = recent_ms < _SILENCE_ENERGY or (
            peak_ms > _SPEECH_PEAK and recent_ms < 0.1 * peak_ms
        )
        if silent:
            silent_since = silent_since or now
            if now - silent_since >= silence_secs:
                # Log the measured energy at the decision point so the user
                # can diagnose auto-stop misfires (quiet rooms, loud mics)
                # without re-reading the source.
                _log.info(
                    "Dictation auto-stopped after %.1fs of silence "
                    "(recent_energy=%.6g peak=%.6g threshold=%.1fs)",
                    now - silent_since,
                    recent_ms,
                    peak_ms,
                    silence_secs,
                )
                on_stop()
                stopped = True
                break
        else:
            silent_since = None


def _start_dictation_watchdog(recorder: Recorder, stop_event: threading.Event) -> threading.Event:
    """Start the settings-driven silence watchdog for a dictation.

    ``stop_event`` is set when the watchdog decides the dictation has ended
    (continuous silence, mic stall, or the max-duration cap). Returns an
    event the caller must set when the dictation ends by other means (stop
    button, SIGUSR1) so the watchdog thread can shut down. No-op when the
    ``dictation_auto_stop`` setting is off.
    """
    watchdog_stop = threading.Event()
    if not settings.get("dictation_auto_stop"):
        watchdog_stop.set()
        return watchdog_stop
    silence_secs = float(
        settings.get("dictation_auto_stop_silence_secs") or _DICTATION_SILENCE_SECS
    )
    threading.Thread(
        target=silence_watchdog,
        args=(recorder, watchdog_stop),
        kwargs={"on_stop": stop_event.set, "silence_secs": silence_secs},
        daemon=True,
        name="dictation-silence-watchdog",
    ).start()
    return watchdog_stop


def note_body(note: str) -> str:
    """The spoken body of a saved dictation note (everything after the header)."""
    _, _, body = note.partition(f"\n{_BODY_SEP}\n")
    return body.strip() if body else note.strip()


def copy_to_clipboard(text: str) -> bool:
    """Best-effort pbcopy. True when the text landed on the clipboard."""
    if not text:
        return False
    try:
        proc = subprocess.run(["pbcopy"], input=text.encode("utf-8"))
        return proc.returncode == 0
    except Exception:
        _log.debug("pbcopy failed", exc_info=True)
        return False


def _post_keystroke(keycode: int, modifiers: int = 0) -> bool:
    """Post a physical keystroke via CoreGraphics (⌘V = keycode 9, cmd = 1<<20).

    Needs only the Accessibility grant — the same permission class trnscrb
    already holds — and nothing per-app: no Apple Events, no System Events,
    no automation prompts. A missing grant drops the event silently, which
    is why callers pre-check with ``_tcc_accessibility_allowed``.
    """
    import ctypes

    try:
        cg = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
        cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        for is_down in (True, False):
            event = cg.CGEventCreateKeyboardEvent(None, keycode, is_down)
            if not event:
                return False
            if modifiers:
                cg.CGEventSetFlags(event, modifiers)
            cg.CGEventPost(0, event)  # kCGHIDEventTap
            cf.CFRelease(event)
            time.sleep(0.05)
        return True
    except Exception:
        _log.debug("CoreGraphics keystroke failed", exc_info=True)
        return False


# ── Accessibility (AX) paste: the frontmost app performs its own Paste ──────
#
# On some macOS builds (seen on macOS 27.0) the synthetic-event injection
# path is dead: CGEventPost ⌘V returns success but the event is silently
# dropped, and AppleScript keystrokes fail the same way — while the
# Accessibility API keeps working. Pressing the frontmost app's own
# "Edit > Paste" menu item via AXUIElementPerformAction makes the app run
# its normal paste handler (identical to a real ⌘V) without injecting any
# key events. Verified working on the affected build.


def _ax_paste_frontmost() -> tuple[bool, int | None]:
    """Press Edit > Paste in the frontmost app via the Accessibility API.

    Returns (performed, ax_error): performed=True when the app ran its own
    paste action (err 0 / kAXSuccess). The error code is surfaced so callers
    can distinguish a missing Accessibility grant (kAXErrorAPIDisabled,
    -25211) from "this app has no Edit > Paste menu item".
    """
    import ctypes

    try:
        from AppKit import NSWorkspace
    except Exception:
        _log.debug("AppKit unavailable — Accessibility paste skipped", exc_info=True)
        return False, None

    try:
        as_ = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        # Pure ctypes for the CF/AX layer: PyObjC objects passed into ctypes
        # args crash (SIGBUS) — keep every CF value a raw pointer.
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
        as_.AXUIElementCreateApplication.restype = ctypes.c_void_p
        as_.AXUIElementCreateApplication.argtypes = [ctypes.c_uint32]
        as_.AXUIElementCopyAttributeValue.restype = ctypes.c_int
        as_.AXUIElementCopyAttributeValue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        as_.AXUIElementPerformAction.restype = ctypes.c_int
        as_.AXUIElementPerformAction.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        def cfstr(s: str) -> int:
            return cf.CFStringCreateWithCString(None, s.encode("utf-8"), 0x08000100)

        def cf_to_pystring(ref) -> str | None:
            if not ref:
                return None
            buf = ctypes.create_string_buffer(4096)
            if cf.CFStringGetCString(ref, buf, 4096, 0x08000100):
                return buf.value.decode("utf-8")
            return None

        def attr(el, name):
            out = ctypes.c_void_p()
            err = as_.AXUIElementCopyAttributeValue(el, cfstr(name), ctypes.byref(out))
            if err != 0:
                return None, err
            return out.value, err

        def title_of(el):
            ref, _err = attr(el, "AXTitle")
            return cf_to_pystring(ref)

        def children_of(el):
            ref, _err = attr(el, "AXChildren")
            if not ref:
                return []
            n = cf.CFArrayGetCount(ref)
            return [cf.CFArrayGetValueAtIndex(ref, i) for i in range(n)]

        def menu_items(el):
            """Items of a menu-bar item: direct children, one level deeper if
            the child is an untitled menu container, else via AXMenu."""
            kids = children_of(el)
            items = []
            for k in kids:
                if title_of(k) is not None:
                    items.append(k)
                else:
                    sub = children_of(k)
                    if sub:
                        items.extend(sub)
            if items:
                return items
            menu, _err = attr(el, "AXMenu")
            if menu:
                return children_of(menu)
            return []

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return False, None
        pid = app.processIdentifier()
        name = app.localizedName()

        app_el = as_.AXUIElementCreateApplication(pid)
        menu_bar, err = attr(app_el, "AXMenuBar")
        if not menu_bar:
            _log.info("AX paste: no menu bar on %s (pid %s), err=%s", name, pid, err)
            return False, err
        edit = None
        for el in children_of(menu_bar):
            if title_of(el) == "Edit":
                edit = el
                break
        if edit is None:
            _log.info("AX paste: no Edit menu in %s", name)
            return False, None
        paste_item = None
        paste_fallback = None
        for el in menu_items(edit):
            t = title_of(el)
            if t is None:
                continue
            if t == "Paste":
                paste_item = el
                break
            # Some apps title it "Paste as Plain Text" or "Paste and Match
            # Style" — an exact "Paste" always wins, but any title starting
            # with "Paste" beats having no paste at all.
            if paste_fallback is None and t.startswith("Paste"):
                paste_fallback = el
        if paste_item is None:
            paste_item = paste_fallback
        if paste_item is None:
            _log.info("AX paste: no Paste item in %s's Edit menu", name)
            return False, None
        err = as_.AXUIElementPerformAction(paste_item, cfstr("AXPress"))
        if err == 0:
            _log.info("AX paste: performed Edit>Paste on %s (pid %s)", name, pid)
            return True, err
        _log.info("AX paste: AXPress on %s failed with err=%s", name, err)
        return False, err
    except Exception:
        _log.debug("Accessibility paste failed", exc_info=True)
        return False, None


# ── Direct AX insertion: text goes straight to the focused field ────────────
#
# AXUIElementSetAttributeValue(focused element, AXSelectedText) inserts text
# at the caret — or replaces the selection, exactly like a real paste — with
# no clipboard and no synthetic events. It is the first-choice delivery path:
# whatever the user had on the clipboard is left untouched.


def _ax_insert_text(text: str) -> bool:
    """Insert ``text`` at the caret of the focused text field.

    Sets AXSelectedText on the frontmost app's focused UI element via the
    Accessibility API — no clipboard, no synthetic events. Works for
    standard text fields (NSTextView/NSTextField and similar); when the
    focused element does not support writing the attribute the call fails
    gracefully and callers fall back to the clipboard-based paths.

    The focused element is queried on the frontmost app's AXUIElement, not
    the system-wide element: on the affected macOS 27.0 build the
    system-wide AXFocusedUIElement query returns kAXErrorCannotComplete
    (-25204) while the per-app query works.
    """
    import ctypes

    try:
        from AppKit import NSWorkspace
    except Exception:
        _log.debug("AppKit unavailable — AX insert skipped", exc_info=True)
        return False

    try:
        as_ = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        # Pure ctypes for the CF/AX layer (PyObjC objects crash in ctypes
        # args — keep every CF value a raw pointer).
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        as_.AXUIElementCreateApplication.restype = ctypes.c_void_p
        as_.AXUIElementCreateApplication.argtypes = [ctypes.c_uint32]
        as_.AXUIElementCopyAttributeValue.restype = ctypes.c_int
        as_.AXUIElementCopyAttributeValue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        as_.AXUIElementSetAttributeValue.restype = ctypes.c_int
        as_.AXUIElementSetAttributeValue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            _log.info("AX insert: no frontmost app")
            return False
        pid = app.processIdentifier()
        name = app.localizedName()

        focused = ctypes.c_void_p()
        err = as_.AXUIElementCopyAttributeValue(
            as_.AXUIElementCreateApplication(pid),
            cf.CFStringCreateWithCString(None, b"AXFocusedUIElement", 0x08000100),
            ctypes.byref(focused),
        )
        if err != 0 or not focused.value:
            _log.info("AX insert: no focused UI element on %s (err=%s)", name, err)
            return False
        err = as_.AXUIElementSetAttributeValue(
            focused.value,
            cf.CFStringCreateWithCString(None, b"AXSelectedText", 0x08000100),
            cf.CFStringCreateWithCString(None, text.encode("utf-8"), 0x08000100),
        )
        if err == 0:
            _log.info(
                "AX insert: text set directly in %s's focused field (clipboard untouched)",
                name,
            )
            return True
        _log.info("AX insert: AXSelectedText write failed on %s (err=%s) — falling back", name, err)
        return False
    except Exception:
        _log.debug("AX insert failed", exc_info=True)
        return False


def _pasteboard_has_items() -> bool | None:
    """True when the pasteboard holds items, False when empty, None when the
    state could not be determined (AppKit unavailable)."""
    try:
        from AppKit import NSPasteboard

        return bool(NSPasteboard.generalPasteboard().pasteboardItems())
    except Exception:
        return None


def _restore_clipboard_if_ours(text: str, old: bytes) -> None:
    """Put the clipboard back to ``old`` after a clipboard-based paste.

    Only when the clipboard still holds exactly ``text`` — if the user
    copied something new while the paste was in flight, that must win.
    And only when ``old`` fully represents the previous contents: pbpaste
    is text-only, so a non-text clipboard (image, file) is left alone
    rather than clobbered with an empty string.
    """
    try:
        cur = subprocess.run(["pbpaste"], capture_output=True, timeout=2).stdout
    except Exception:
        return
    if cur != text.encode("utf-8"):
        return  # the user copied something new — keep it
    if old or _pasteboard_has_items() is False:
        try:
            subprocess.run(["pbcopy"], input=old, timeout=2)
        except Exception:
            _log.debug("clipboard restore failed", exc_info=True)
    else:
        _log.warning(
            "clipboard held non-text data — dictation text left on the clipboard "
            "instead of restoring the previous contents"
        )


# Throttle for the on-failure re-prompt: a burst of test presses must not
# stack system dialogs.
_last_ax_prompt = 0.0


def _prompt_accessibility_if_missing() -> None:
    """Re-show the system Accessibility dialog after a -25211 paste failure.

    The startup prompt is easy to miss (the app starts in the background
    before the user has even tried dictation); prompting again at the
    moment the paste actually fails puts the dialog on screen when it
    matters. Plain HIServices calls via loadBundleFunctions — no TCC API
    (TCCAccessRequest SIGTRAPs unsigned processes on this machine).
    """
    global _last_ax_prompt
    now = time.time()
    if now - _last_ax_prompt < 300:
        return
    _last_ax_prompt = now
    try:
        import objc
        from Foundation import NSBundle

        bundle = NSBundle.bundleWithPath_(
            "/System/Library/Frameworks/ApplicationServices.framework"
        )
        if bundle is None or not bundle.load():
            return
        g = {}
        objc.loadBundleFunctions(
            bundle, g, [("AXIsProcessTrustedWithOptions", b"i@")]
        )
        g["AXIsProcessTrustedWithOptions"]({"AXTrustedCheckOptionPrompt": True})
        _log.info("Accessibility: re-prompted system dialog after paste failure")
    except Exception:
        _log.debug("Accessibility re-prompt failed", exc_info=True)


# Terminal emulators: their text views accept an AXSelectedText write and
# report success while silently dropping the text (the terminal buffer is
# not a real editable field). They take the Edit > Paste path instead, which
# runs the terminal's own paste handler — exactly what ⌘V would do.
_TERMINAL_BUNDLE_IDS = {
    "com.mitchellh.ghostty",   # Ghostty
    "com.calyx.terminal",      # Calyx (where the OpenCode TUI runs)
    "com.apple.Terminal",      # Terminal.app
    "com.googlecode.iterm2",   # iTerm2
    "org.alacritty",           # Alacritty
    "net.kovidgoyal.kitty",    # kitty
    "com.github.wez.wezterm",  # WezTerm
    "co.zeit.hyper",           # Hyper
    "io.tabby",                # Tabby
    "com.waveterm.waveterm",   # Wave
}


def _frontmost_is_terminal() -> bool:
    """True when the frontmost app is a terminal emulator (see above)."""
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return False
        return (app.bundleIdentifier() or "") in _TERMINAL_BUNDLE_IDS
    except Exception:
        _log.debug("frontmost-terminal check failed", exc_info=True)
        return False


def paste_text_to_active_app(text: str) -> tuple[bool, str]:
    """Deliver text to the currently focused text field.

    Paths, in order:
    1. Direct Accessibility insertion — set AXSelectedText on the focused
       field. Inserts at the caret like a real paste and touches the
       clipboard not at all.
    2. Accessibility menu action — the clipboard holds the text while the
       frontmost app performs its own Edit > Paste (works while macOS's
       synthetic event injection is broken); the previous clipboard
       contents are restored afterwards.
    3. CoreGraphics ⌘V keystroke — works when event injection is healthy.
    4. AppleScript keystroke — last resort when Apple Events are granted.
    Paths 3-4 leave the text on the clipboard (their delivery cannot be
    verified, so the text is never lost). When every path fails the text
    is also left on the clipboard so a manual ⌘V still works.
    Returns (success, detail).
    """
    if not text:
        return False, "empty text"
    # 1) Direct insertion: no clipboard involved at all. Terminals are
    # skipped — their text views report success for an AXSelectedText write
    # while silently dropping the text, so they take the Edit > Paste path
    # below (the terminal's own paste handler, with the clipboard restored
    # straight after).
    if _frontmost_is_terminal():
        _log.info("paste: frontmost app is a terminal — using Edit > Paste path")
    elif _ax_insert_text(text):
        return True, "pasted directly into text field (clipboard untouched)"
    # The clipboard-based paths: snapshot the clipboard so it can be
    # restored after a verified paste.
    old_clip = b""
    try:
        old_clip = subprocess.run(["pbpaste"], capture_output=True, timeout=2).stdout
    except Exception:
        pass
    if not copy_to_clipboard(text):
        return False, "could not copy to clipboard"
    # 2) Accessibility menu action: the frontmost app performs its own
    # paste (synchronous — the clipboard is read within the action, so the
    # restore below is safe). Works even while synthetic injection is dead.
    performed, ax_err = _ax_paste_frontmost()
    if performed:
        _restore_clipboard_if_ours(text, old_clip)
        return True, "pasted into active text field (Accessibility)"
    if ax_err == -25211:  # kAXErrorAPIDisabled — no Accessibility grant
        # Both synthetic paths below need that same grant, and this macOS
        # build silently drops posted events without it — attempting would
        # only report a fake success. The text stays on the clipboard so a
        # manual ⌘V still works.
        _log.info(
            "paste: Accessibility grant missing (-25211) — skipping "
            "synthetic input paths"
        )
        _prompt_accessibility_if_missing()
        return (
            False,
            "Accessibility grant missing — enable it in System Settings → "
            "Privacy & Security → Accessibility (the entry may be named "
            "'Python' or 'Trnscrb'); text is on the clipboard for a manual ⌘V",
        )
    # 3) CoreGraphics keystroke: no Apple Events, no System Events, no
    # per-app automation prompts — just the Accessibility grant. (No TCC API
    # is called to pre-check it: on this machine TCCAccessRequest SIGTRAPs
    # unsigned processes, and the grant itself was verified in the system
    # log — io.trnscrb.app, authValue=allowed.) The post is async and can be
    # silently dropped (macOS 27 input-stack bug), so the text is left on
    # the clipboard rather than restored — worst case the user pastes
    # manually.
    if _post_keystroke(9, 1 << 20):  # kVK_ANSI_V with the command flag
        return True, "pasted into active text field (CoreGraphics)"
    _log.info("CoreGraphics paste unavailable — falling back to AppleScript")
    # Fallback: paste via AppleScript (System Events → keystroke v with command).
    try:
        script = """
            tell application "System Events"
                tell (first process whose frontmost is true)
                    keystroke "v" using command down
                end tell
            end tell
        """
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        if proc.returncode == 0:
            return True, "pasted into active text field"
        _log.info(
            "AppleScript paste failed (rc=%s): %s",
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        # Fallback: try pasting via target app.
        try:
            script2 = f"""
                set the clipboard to "{text}"
                tell application "System Events"
                    tell (first process whose frontmost is true)
                        keystroke "v" using command down
                    end tell
                end tell
            """
            proc2 = subprocess.run(["osascript", "-e", script2], capture_output=True, timeout=5)
            if proc2.returncode == 0:
                return True, "pasted into active text field"
            _log.info(
                "AppleScript paste fallback failed (rc=%s): %s",
                proc2.returncode,
                (proc2.stderr or proc2.stdout or "").strip()[:200],
            )
        except Exception:
            pass
        grant_hint = (
            " (Accessibility grant missing — enable Trnscrb in System Settings → "
            "Privacy & Security → Accessibility)"
            if ax_err == -25211  # kAXErrorAPIDisabled
            else ""
        )
        return False, f"paste failed{grant_hint} — text is on the clipboard"
    except Exception as e:
        _log.debug("paste_text_to_active_app failed: %s", e, exc_info=True)
        return False, "paste failed — clipboard still has text"


# ── meeting-aware dictation ───────────────────────────────────────────────────


def get_meeting_context() -> dict | None:
    """Return a dict with the current meeting context, or None.

    The context contains:
      ``meeting`` — the meeting name (slugified)
      ``path``    — the live transcript path
      ``started_at`` — ISO datetime when the meeting started
    """
    info = storage.get_live_session_info()
    if not info:
        return None
    try:
        path = info.get("path")
        if isinstance(path, Path):
            path = str(path)
        return {
            "meeting": info.get("meeting", "unknown") or "unknown",
            "path": path,
            "started_at": info.get("started_at"),
        }
    except Exception:
        return None


def _meeting_slug(meeting_name: str) -> str:
    """Turn a meeting name into a safe directory slug."""
    import re

    return re.sub(r"[^A-Za-z0-9_-]", "-", meeting_name)[:50]


def save_note(
    preset: str,
    started_at: datetime,
    segments: list[dict],
    meeting_name: str | None = None,
) -> tuple[Path | None, str]:
    """Format and save the dictation note. Returns (path, text).

    When ``meeting_name`` is provided the note is saved into the meeting's
    subfolder (``~/meeting-notes/<meeting>/``) so all dictations for a
    meeting are grouped together.
    """
    name = f"{preset}-{started_at.strftime('%H%M')}"
    if not any(str(seg.get("text") or "").strip() for seg in segments):
        return None, ""
    text = storage.format_transcript(segments, started_at, name, kind=preset)
    if not text.strip():
        return None, ""

    if meeting_name:
        # Save into the meeting's subfolder.
        slug = _meeting_slug(meeting_name)
        notes_dir = storage.NOTES_DIR / slug
        notes_dir.mkdir(parents=True, exist_ok=True)
        path = notes_dir / f"{name}.txt"
    else:
        path = storage.get_transcript_path(name, started_at)

    storage.save_transcript(path, text)
    return path, text


def _save_and_mirror(
    preset: str,
    started_at: datetime,
    segments: list[dict],
    meeting_name: str | None = None,
) -> tuple[Path | None, str]:
    """Save the dictation note, then mirror it into the Obsidian vault."""
    path, text = save_note(preset, started_at, segments, meeting_name=meeting_name)
    if path:
        _mirror_note(preset, started_at, text)
    return path, text


def _mirror_note(preset: str, started_at: datetime, text: str) -> None:
    """Mirror a saved dictation note into the Obsidian vault (best-effort)."""
    try:
        from trnscrb import obsidian

        obsidian.mirror_dictation_note(preset, started_at, text)
    except Exception:
        _log.warning("Could not mirror dictation note into Obsidian", exc_info=True)


def inject_into_meeting_transcript(text: str, meeting_path: Path) -> bool:
    """Append dictated text to a live meeting transcript.

    Inserts a ``[Dictation]`` marker so the appended text is visually
    separate.  Returns True when the write succeeded.
    """
    try:
        marker = "\n[Dictation — added while recording]\n"
        existing = meeting_path.read_text(encoding="utf-8")
        merged = existing + marker + text + "\n"
        meeting_path.write_text(merged, encoding="utf-8")
        return True
    except Exception:
        _log.warning("Could not inject into meeting transcript: %s", meeting_path, exc_info=True)
        return False


# ── voice symbols: spoken word → typed character ─────────────────────────────
# The ASR writes down what you say; this pass turns the words you say for
# symbols into the symbols themselves ("at sign" → @, "new line" → a real
# newline in the pasted text). Matches are word-boundary anchored so ordinary
# prose is left alone ("the period between shots" keeps its period word).
# The built-in map is extended or overridden by the ``voice_symbol_map``
# setting (an empty value removes a built-in phrase); ``voice_symbols`` set
# to false disables the pass entirely.
_VOICE_SYMBOL_MAP: dict[str, str] = {
    # symbols
    "at sign": "@",
    "hash": "#",
    "number sign": "#",
    "ampersand": "&",
    "percent": "%",
    "per cent": "%",
    "dollar sign": "$",
    "euro sign": "€",
    "yen sign": "¥",
    "open quote": '"',
    "close quote": '"',
    "double quote": '"',
    "open single quote": "'",
    "close single quote": "'",
    "apostrophe": "'",
    "open paren": "(",
    "close paren": ")",
    "closing paren": ")",
    "open bracket": "[",
    "close bracket": "]",
    "open brace": "{",
    "close brace": "}",
    "slash": "/",
    "forward slash": "/",
    "backslash": "\\",
    "underscore": "_",
    "asterisk": "*",
    "star": "*",
    "caret": "^",
    "tilde": "~",
    "plus sign": "+",
    "minus sign": "-",
    "em dash": "—",
    "en dash": "–",
    "equals sign": "=",
    "equal sign": "=",
    "less than": "<",
    "greater than": ">",
    "exclamation point": "!",
    "exclamation mark": "!",
    "question mark": "?",
    "colon": ":",
    "semicolon": ";",
    "comma": ",",
    "period": ".",
    "full stop": ".",
    "dot": ".",
    "pipe": "|",
    "vertical bar": "|",
    # layout: real characters in the pasted text. A pasted newline is a line
    # break in a chat composer (it does not send the message); literal key
    # presses (Return, Tab) are deliberately never performed.
    "new line": "\n",
    "line break": "\n",
    "new paragraph": "\n",
    "carriage return": "\n",
    "tab": "\t",
}


def _voice_symbol_map() -> dict[str, str]:
    """The built-in map merged with the user's ``voice_symbol_map`` setting."""
    if not settings.get("voice_symbols"):
        return {}
    merged = dict(_VOICE_SYMBOL_MAP)
    custom = settings.get("voice_symbol_map")
    if isinstance(custom, dict):
        for phrase, value in custom.items():
            phrase = str(phrase).strip().lower()
            if not phrase:
                continue
            if value in (None, ""):
                merged.pop(phrase, None)
            else:
                merged[phrase] = str(value)
    return merged


def apply_voice_symbols(segments: list[dict]) -> int:
    """Convert spoken symbol words to characters in the segments, in place.

    Returns how many words were converted (0 when the pass is disabled or
    found nothing). Longest phrases win, so "forward slash" beats "slash".
    """
    mapping = _voice_symbol_map()
    if not mapping:
        return 0
    phrases = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE
    )
    total = 0
    for seg in segments:
        text = str(seg.get("text") or "")
        if not text.strip():
            continue
        converted, count = pattern.subn(lambda m: mapping[m.group(0).lower()], text)
        if count:
            # Tidy the spacing the ASR put around the mapped tokens: "john @
            # example . com" reads as an address, not a sentence.
            converted = converted.replace(" @ ", "@").replace(" @", "@").replace(" . ", ".").replace(" .", ".")
            seg["text"] = converted
            total += count
    return total


def finish(
    preset: str,
    started_at: datetime,
    audio_path: Path,
    inject_meeting: bool = False,
    save_note: bool = True,
) -> dict:
    """Transcribe, save the note, and copy the text for a message preset.

    The audio file is cleaned up after a successful transcription — the note
    and clipboard are the product, and a few seconds of dictation is cheap to
    redo. If the transcription itself fails, the audio is preserved in the
    notes folder (like a failed meeting) rather than thrown away. When
    ``save_note`` is False no note file is written and nothing is mirrored
    into the Obsidian vault — the text only reaches the clipboard / pasted
    field. Returns:
      {preset, path, text, plain, on_clipboard, duration_secs, injected}
    """
    _log.info("Dictation finishing (preset=%s, audio=%s)", preset, audio_path)
    try:
        segments = transcriber.transcribe(audio_path)
    except Exception as e:
        _log.error("Dictation transcription failed: %s", e, exc_info=True)
        # A transient model failure must not cost the user their words:
        # preserve the audio the way a failed meeting does, then re-raise.
        preserved = storage.preserve_audio(
            audio_path,
            f"dictation-{preset}",
            started_at,
            reason="Dictation transcription failed — audio preserved",
        )
        if preserved:
            _log.error("Dictation audio preserved for retry: %s", preserved)
        else:
            _cleanup_audio(audio_path)
        raise
    _cleanup_audio(audio_path)

    converted = apply_voice_symbols(segments)
    if converted:
        _log.info("Voice symbols: converted %d spoken word(s)", converted)

    plain = plain_text(segments)
    duration_secs = float(segments[-1]["end"]) if segments else 0.0

    path, text = None, ""
    injected = False
    if plain:
        meeting_ctx = get_meeting_context() if inject_meeting else None
        meeting_name = meeting_ctx["meeting"] if meeting_ctx else None
        if save_note:
            path, text = _save_and_mirror(preset, started_at, segments, meeting_name=meeting_name)

        # Inject into the meeting transcript if requested and a meeting is live.
        if inject_meeting and meeting_ctx and meeting_ctx.get("path"):
            target = Path(meeting_ctx["path"])
            if target.exists():
                injected = inject_into_meeting_transcript(plain, target)
                _log.info(
                    "Dictation %s into meeting transcript: %s",
                    "injected" if injected else "skipped",
                    injected,
                )

    on_clipboard = False
    # Only message preset copies to clipboard (brain-dump is saved, not typed
    # elsewhere).
    paste_ok = False
    paste_detail = ""
    if plain and preset == "message":
        if settings.get("paste_on_dictation"):
            # paste_text_to_active_app handles copy + paste in one step.
            paste_ok, paste_detail = paste_text_to_active_app(plain)
            on_clipboard = True  # it copies first, so clipboard landed.
            _log.info("Dictation paste: %s", "ok" if paste_ok else f"FAILED ({paste_detail})")
        else:
            on_clipboard = copy_to_clipboard(plain)

    return {
        "preset": preset,
        "path": str(path) if path else None,
        "text": text,
        "plain": plain,
        "on_clipboard": on_clipboard,
        "duration_secs": duration_secs,
        "injected": injected,
        "pasted": paste_ok,
        "paste_detail": paste_detail,
    }


def resolve_note(short_id: str) -> Path | None:
    """Resolve a dictation note id to its file.

    Accepts a full filename stem or a short suffix such as `message-0941`;
    a suffix resolves to the newest note whose stem ends with it.
    """
    target = str(short_id).strip()
    if not target:
        return None
    try:
        notes = storage.ensure_notes_dir()
        candidates = [path for path in notes.glob("*.txt") if path.stem.endswith(target)]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── background CLI child ─────────────────────────────────────────────────────


def _cleanup_audio(audio_path: Path) -> None:
    try:
        audio_path.unlink(missing_ok=True)
    except OSError:
        _log.debug("Could not remove dictation audio %s", audio_path, exc_info=True)


def running_pid() -> int | None:
    """PID of the background dictation process, or None."""
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # raises if the process is gone
        return pid
    except Exception:
        return None


def write_pid(pid: int) -> None:
    _CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(pid))


def remove_pid() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def write_result(payload: dict) -> None:
    try:
        _CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        _RESULT_FILE.write_text(json.dumps(payload))
    except Exception:
        _log.debug("Could not write dictation result", exc_info=True)


def read_result() -> dict | None:
    """The last background dictation's result, or None."""
    try:
        payload = json.loads(_RESULT_FILE.read_text())
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def clear_result() -> None:
    """Drop any previous result, so a stale one is never mistaken for fresh."""
    try:
        _RESULT_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def wait_for_pid(timeout: float = 5.0) -> int | None:
    """Wait (briefly) for the background child to publish its pid."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = running_pid()
        if pid is not None:
            return pid
        time.sleep(0.1)
    return None


def wait_for_stop(timeout: float = 180.0) -> dict | None:
    """After signalling the child, wait for its result. None on timeout.

    The child removes its pid file before transcribing, so this polls in two
    phases: for the pid to disappear, then for the result file to appear —
    which `starts` guarantees is fresh by clearing it first.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if running_pid() is None:
            break
        time.sleep(0.2)
    while time.monotonic() < deadline:
        result = read_result()
        if result is not None:
            return result
        time.sleep(0.2)
    return None  # still finishing — the caller can check `status` again


def run_background(preset: str, save_note: bool = True) -> None:
    """Child entry point: record until SIGUSR1 or the silence auto-stop, then finish and report.

    Launched detached by `trnscrb dictation start` as
    `python -m trnscrb.dictation <preset>`. stdout is /dev/null, so the
    outcome lands in dictation_result.json for `trnscrb dictation stop` to
    read (and the app log carries any failure). The silence auto-stop runs
    in this child too (honoring the ``dictation_auto_stop`` settings), so
    the flow is the same with or without the menu-bar app.
    """
    if not is_preset(preset):
        sys.exit(2)

    recorder = new_recorder()
    started_at = datetime.now()
    write_pid(os.getpid())

    stop_event = threading.Event()

    def _stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGUSR1, _stop)
    signal.signal(signal.SIGINT, _stop)
    # The child gets the same auto-stop as the menu-bar app: when the user
    # stops talking the dictation finishes itself, so the fallback path
    # (menu-bar app not running) is still the normal flow.
    watchdog_stop = _start_dictation_watchdog(recorder, stop_event)
    try:
        while not stop_event.wait(1.0):
            pass
    finally:
        watchdog_stop.set()
        remove_pid()

    _log.info("Dictation child finishing (preset=%s)", preset)
    audio_path = None
    try:
        audio_path = recorder.stop()
        if not audio_path:
            write_result({"preset": preset, "error": "No audio captured."})
            return
        result = finish(preset, started_at, audio_path, save_note=save_note)
        result["preset"] = preset
        write_result(result)
    except Exception as e:
        _log.error("Dictation child failed", exc_info=True)
        write_result({"preset": preset, "error": str(e)})
    finally:
        if audio_path:
            _cleanup_audio(audio_path)


def _main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="trnscrb.dictation")
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not write a note file — copy/paste the spoken text only.",
    )
    args = parser.parse_args(argv)
    run_background(args.preset, save_note=not args.no_save)


if __name__ == "__main__":
    _main()
