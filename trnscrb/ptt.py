"""Push-to-talk: hold a key to dictate, release to stop.

A *listen-only* CGEvent tap watches the HID keyboard stream for the
configured combo. Key-down starts a dictation (the menu-bar app's own
in-process start, with the silence auto-stop disabled for the session so a
mid-sentence pause cannot cut it off); key-up stops it, which transcribes
and pastes into the focused field.

The state machine (``handle_key_down`` / ``handle_key_up``) is pure and
unit-testable. ``EventTap`` owns the CoreFoundation plumbing; ``PTTMonitor``
and ``PTTCapture`` are the two users of it (dictation, and the menu bar's
"Record PTT key…" mode).

Permission: a listen-only HID tap requires the Input Monitoring grant
(System Settings → Privacy & Security → Input Monitoring) — the same grant
class trnscrb already asks for with the auto-paste. When it is not granted
yet, ``start()`` returns False and the caller retries; nothing else in the
app is affected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from trnscrb.hotkey import PTTKey, ptt_mods

# Sentinel event types the tap delivers instead of a keyboard event when the
# system disables it (callback stall, or the user revoked the grant).
_TAP_DISABLED_TYPES = (4294967294, 4294967295)  # kCGEventTapDisabledByTimeout/UserInput


# ── state machine (pure) ─────────────────────────────────────────────────────


@dataclass
class PTTState:
    """One push-to-talk press: key-down opens it, key-up closes it.

    ``holding`` is set on the matching key-down; ``started`` is set only
    after the caller confirms the dictation actually began. A release
    therefore never stops a dictation it did not start (e.g. one started
    from the menu bar while the user's PTT tap was refused).
    """

    key: PTTKey
    holding: bool = False
    started: bool = False

    def reset(self) -> None:
        self.holding = False
        self.started = False


def handle_key_down(state: PTTState, code: int, flags: int, autorepeat: bool = False) -> bool:
    """A key-down arrived. True when the caller should start a dictation.

    False for: other keys, key auto-repeat, a second press of an open press,
    and a press missing any required modifier (extra modifiers are ignored —
    a stray shift must not defeat the combo).
    """
    if code != state.key.key_code or autorepeat or state.holding:
        return False
    if (flags & state.key.flags) != state.key.flags:
        return False
    state.holding = True
    return True


def confirm_start(state: PTTState, ok: bool) -> None:
    """The start attempt finished; remember whether a dictation is live."""
    state.started = bool(ok)


def handle_key_up(state: PTTState, code: int) -> bool:
    """A key-up arrived. True when the caller should stop the dictation.

    Only releases of the PTT key that actually started a dictation stop
    anything; the release of an ignored or refused press is a no-op.
    """
    if code != state.key.key_code:
        return False
    was_started = state.started
    state.holding = False
    state.started = False
    return was_started


# ── event tap plumbing ───────────────────────────────────────────────────────


class EventTap:
    """A listen-only CGEvent tap delivering keyboard events to a callback.

    The callback runs on the main run loop (the tap source is installed on
    it), so callers may touch UI state directly. It must not block: heavy
    work belongs in a worker thread.
    """

    def __init__(self, callback: Callable[..., None]) -> None:
        self._callback = callback
        self._tap = None
        self._source = None
        self.active = False
        # Set when the system disables the tap; the re-arm loop can react.
        self.disabled_reason: str = ""

    def start(self) -> bool:
        """Create and arm the tap. False when it cannot be created
        (PyObjC missing, or the Input Monitoring grant not yet given)."""
        if self.active:
            return True
        try:
            import Quartz
        except ImportError:
            return False
        key_mask = int(Quartz.kCGEventKeyDown) | int(Quartz.kCGEventKeyUp)
        try:
            self._tap = Quartz.CGEventTapCreate(
                int(Quartz.kCGHIDEventTap),
                int(Quartz.kCGHeadInsertEventTap),
                int(Quartz.kCGEventTapOptionListenOnly),
                key_mask,
                self._c_callback,
                None,
            )
            if self._tap is None:
                return False  # grant missing (macOS asks on first tap)
            self._source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetMain(), self._source, int(Quartz.kCFRunLoopCommonModes)
            )
            Quartz.CGEventTapEnable(self._tap, True)
            self.active = True
            return True
        except Exception:
            self._tap = None
            self._source = None
            return False

    def _c_callback(self, _refcon, event, _user_info) -> None:
        """CFEventCallback shim (main run loop); never raises across the bridge."""
        try:
            import Quartz

            etype = int(Quartz.CGEventGetType(event))
            if etype in _TAP_DISABLED_TYPES:
                # The system disabled the tap (stall or revoked grant).
                # Re-enable best-effort; the owner's re-arm loop notices via
                # `disabled_reason` / a failed restart.
                self.disabled_reason = "timeout" if etype == 4294967294 else "user-input"
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                return
            code = int(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode))
            flags = int(Quartz.CGEventGetFlags(event))
            if etype == int(Quartz.kCGEventKeyDown):
                autorepeat = bool(
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
                )
                self._callback("down", code, flags, autorepeat)
            else:
                self._callback("up", code, flags, autorepeat=False)
        except Exception:
            # A crashing callback would kill the tap for good; the dictation
            # itself must survive any monitor hiccup.
            pass

    def stop(self) -> None:
        """Tear the tap down. Idempotent."""
        if not self.active:
            return
        try:
            import Quartz

            if self._tap is not None:
                Quartz.CGEventTapEnable(self._tap, False)
            if self._source is not None:
                Quartz.CFRunLoopRemoveSource(
                    Quartz.CFRunLoopGetMain(), self._source, int(Quartz.kCFRunLoopCommonModes)
                )
        except Exception:
            pass
        self._tap = None
        self._source = None
        self.active = False
        self.disabled_reason = ""


# ── push-to-talk monitor ─────────────────────────────────────────────────────


class PTTMonitor:
    """Hold the configured combo to dictate; release to stop and paste.

    ``on_start`` must return True when the dictation actually began (the
    release then stops it) and False when it was refused (dictation or a
    meeting already running), in which case the release stops nothing.
    """

    def __init__(self, key: PTTKey, on_start: Callable[[], bool], on_stop: Callable[[], None]):
        self.key = key
        self.state = PTTState(key)
        self._on_start = on_start
        self._on_stop = on_stop
        self._tap = EventTap(self._handle)

    @property
    def active(self) -> bool:
        return self._tap.active

    def start(self) -> bool:
        return self._tap.start()

    def stop(self) -> None:
        self._tap.stop()

    def _handle(self, kind: str, code: int, flags: int, autorepeat: bool) -> None:
        if kind == "down":
            if handle_key_down(self.state, code, flags, autorepeat):
                confirm_start(self.state, bool(self._on_start()))
        else:
            if handle_key_up(self.state, code):
                self._on_stop()


# ── record mode ──────────────────────────────────────────────────────────────


class PTTCapture:
    """One-shot capture of the next key press (the menu bar's record mode).

    Any non-modifier key with optional modifiers counts — modifier-only
    presses are ignored because they arrive as flags-changed events, which
    this tap does not listen for. ``on_capture`` receives (key_code,
    ptt_modifiers) on the main run loop and is fired exactly once.
    """

    def __init__(self, on_capture: Callable[[int, int], None]):
        self._on_capture = on_capture
        self._fired = threading.Event()
        self._tap = EventTap(self._handle)

    @property
    def active(self) -> bool:
        return self._tap.active

    def start(self) -> bool:
        return self._tap.start()

    def stop(self) -> None:
        self._tap.stop()

    def _handle(self, kind: str, code: int, flags: int, _autorepeat: bool) -> None:
        if kind != "down" or self._fired.is_set():
            return
        self._fired.set()
        self._tap.stop()
        self._on_capture(code, ptt_mods(flags))
