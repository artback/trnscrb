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
import time
from dataclasses import dataclass
from typing import Callable

from trnscrb.hotkey import CAPS_LOCK_CODE, FLAG_CAPS_LOCK, PTTKey, ptt_mods
from trnscrb.log import get_logger

_log = get_logger("trnscrb.ptt")

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

    ``caps_held`` is True while the finger is physically on caps lock
    (keycode ``CAPS_LOCK_CODE``). Caps lock is a toggle key — the only
    "modifier" that emits a real key-down/key-up pair — and its
    alpha-shift flag bit latches rather than following the finger, so a
    combo that requires caps lock matches this flag, never the event bit.

    ``pending`` is True while the PTT key is down but a required modifier
    has not arrived yet: the physical order can put the key down before
    its modifiers (a fast press, or a stale event flag snapshot), so the
    press is remembered and re-matched as the modifier state changes
    (``PTTMonitor`` re-runs the match on every flagsChanged).
    """

    key: PTTKey
    holding: bool = False
    started: bool = False
    caps_held: bool = False
    pending: bool = False

    def reset(self) -> None:
        self.holding = False
        self.started = False
        self.caps_held = False
        self.pending = False


def effective_flags(event_flags: int, tracked_flags: int) -> int:
    """Modifier bits to match a key-down against.

    A key-down's own flag snapshot can be stale: a fast modifier-then-key
    press (or a coalesced session event) can arrive before the window
    server merges the held modifiers into the event, so the event shows
    no modifier bits at all while the modifiers are physically held —
    the terminal still reports the combo, the tap does not see it.
    The tap's flagsChanged tracking is the live state; OR the two so the
    match sees the modifiers. A genuinely bare press stays bare: no
    flagsChanged, nothing to OR in.
    """
    return event_flags | tracked_flags


def handle_key_down(state: PTTState, code: int, flags: int, autorepeat: bool = False) -> bool:
    """A key-down arrived. True when the caller should start a dictation.

    False for: other keys, a second press of an open press, and a press
    missing any required modifier (extra modifiers are ignored — a stray
    shift must not defeat the combo). Auto-repeat events are accepted for
    the initial trigger (the user may already be holding the key when the
    tap catches up); ``state.holding`` prevents re-triggering.

    A key-down of the caps-lock key never starts anything: it only opens
    the ``caps_held`` window during which a combo requiring ``FLAG_CAPS_LOCK``
    will match. Note the match deliberately ignores the event's alpha-shift
    bit — while the finger is on caps lock that bit may actually be *off*
    (the press toggles the lock), so the down/up pair is the only honest
    signal.
    """
    if code == CAPS_LOCK_CODE:
        state.caps_held = True
        return False
    if state.holding:
        return False
    if code != state.key.key_code:
        # A different key press breaks a press waiting on its modifiers.
        state.pending = False
        return False
    regular = state.key.flags & ~FLAG_CAPS_LOCK
    if (flags & regular) != regular or (
        state.key.flags & FLAG_CAPS_LOCK and not state.caps_held
    ):
        # Key is down, a required modifier is not (yet): the physical order
        # can put the key first, or the event's flag snapshot can lag the
        # held modifiers. Remember the press; the live modifier state
        # (flagsChanged) completes it — see PTTMonitor._on_flags.
        state.pending = True
        return False
    state.holding = True
    state.pending = False
    return True


def confirm_start(state: PTTState, ok: bool) -> None:
    """The start attempt finished; remember whether a dictation is live."""
    state.started = bool(ok)


def handle_key_up(state: PTTState, code: int) -> bool:
    """A key-up arrived. True when the caller should stop the dictation.

    Only releases of the PTT key that actually started a dictation stop
    anything; the release of an ignored or refused press is a no-op.

    A key-up of the caps-lock key only closes the ``caps_held`` window —
    it never stops a dictation (caps lock is rejected as a main key by
    ``hotkey.parse``), so releasing caps early while the PTT key is still
    held behaves like releasing shift early: the dictation runs on until
    the PTT key goes up.
    """
    if code == CAPS_LOCK_CODE:
        state.caps_held = False
        return False
    if code != state.key.key_code:
        return False
    was_started = state.started
    state.holding = False
    state.started = False
    state.pending = False
    return was_started


# ── event tap plumbing ───────────────────────────────────────────────────────


class EventTap:
    """A listen-only CGEvent tap delivering keyboard events to a callback.

    The callback runs on the main run loop (the tap source is installed on
    it), so callers may touch UI state directly. It must not block: heavy
    work belongs in a worker thread.

    Also tracks the live modifier state from ``kCGEventFlagsChanged``
    (stored in ``_modifier_flags``): a key-down's own flag snapshot can be
    stale — a fast modifier-then-key press can land before the window
    server merges the held modifiers into the event, arriving with no
    modifier bits at all while the modifiers are physically held. Key
    downs are delivered to the callback with the tracked state ORed in
    (see :func:`effective_flags`).

    ``on_flags_changed``, when set, is called on the main run loop with the
    full live flag state on every ``kCGEventFlagsChanged`` — consumers that
    must react to the modifiers arriving *after* a key-down (the late-
    modifier completion in ``PTTMonitor``) hang off it.
    """

    def __init__(
        self,
        callback: Callable[..., None],
        on_flags_changed: Callable[[int], None] | None = None,
    ) -> None:
        self._callback = callback
        self._on_flags_changed = on_flags_changed
        self._tap = None
        self._source = None
        self.active = False
        # Set when the system disables the tap; the re-arm loop can react.
        self.disabled_reason: str = ""
        # Live modifier state from flagsChanged events; reset on stop so a
        # re-armed tap starts from a clean slate.
        self._modifier_flags: int = 0

    @property
    def modifier_flags(self) -> int:
        """Live modifier state tracked from flagsChanged events."""
        return self._modifier_flags

    def start(self) -> bool:
        """Create and arm the tap. False when it cannot be created
        (PyObjC missing, or the Input Monitoring grant not yet given)."""
        if self.active:
            return True
        try:
            import Quartz
        except ImportError:
            return False
        _log.debug("PTT tap start: creating CGEventTapCreate (session)")
        try:
            key_mask = (
                (1 << int(Quartz.kCGEventKeyDown))
                | (1 << int(Quartz.kCGEventKeyUp))
                | (1 << int(Quartz.kCGEventFlagsChanged))
            )
            self._tap = Quartz.CGEventTapCreate(
                int(Quartz.kCGSessionEventTap),
                int(Quartz.kCGHeadInsertEventTap),
                int(Quartz.kCGEventTapOptionListenOnly),
                key_mask,
                self._c_callback,
                None,
            )
            _log.debug("PTT CGEventTapCreate returned: %s", self._tap)
            if self._tap is None:
                return False  # grant missing (macOS asks on first tap)
            _log.debug("PTT creating run loop source")
            self._source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
            _log.debug("PTT adding to run loop (mode=common)")
            # CFRunLoopMode is a CFString, not a number: int() raised and the
            # except below swallowed it, so start() returned False and PTT
            # could never arm even with the grant in place.
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetMain(), self._source, Quartz.kCFRunLoopCommonModes
            )
            _log.debug("PTT tap enabled")
            Quartz.CGEventTapEnable(self._tap, True)
            self.active = True
            _log.debug("PTP tap created successfully: %s", type(self._tap))
            return True
        except Exception:
            _log.debug("PTT tap start failed with exception", exc_info=True)
            self._tap = None
            self._source = None
            return False

    def _c_callback(self, _proxy, event_type, event, _user_data) -> None:
        """CFEventCallback shim (main run loop); never raises across the bridge.

        PyObjC's CGEventCallback trampoline for ``kCGHIDEventTap`` passes
        *(proxy, event_type, event, user_data)* — ``event_type`` is an int,
        ``event`` is a CGEvent.  This is different from the
        ``(refcon, event, *args)`` layout some taps use, so we match the
        actual signature rather than relying on *args.
        """
        try:
            import Quartz

            if event_type in _TAP_DISABLED_TYPES:
                _log.debug(
                    "PTT tap disabled: %s, re-enabling",
                    "timeout" if event_type == 4294967294 else "user-input",
                )
                # The system disabled the tap (stall or revoked grant).
                # Re-enable best-effort; the owner's re-arm loop notices via
                # `disabled_reason` / a failed restart.
                self.disabled_reason = "timeout" if event_type == 4294967294 else "user-input"
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                    _log.debug("PTT tap re-enabled")
                return
            code = int(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode))
            flags = int(Quartz.CGEventGetFlags(event))
            _log.debug("PTT key event: code=%d flags=%d event_type=%d", code, flags, event_type)
            if event_type == int(Quartz.kCGEventFlagsChanged):
                # Not a key press — the authoritative live modifier state.
                # Key events that follow may carry a stale snapshot; match
                # against this instead (see effective_flags). A press may
                # also be *waiting* on these modifiers (late-modifier
                # completion); hand the new state to the consumer.
                self._modifier_flags = flags
                if self._on_flags_changed is not None:
                    self._on_flags_changed(flags)
                return
            if event_type == int(Quartz.kCGEventKeyDown):
                autorepeat = bool(
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
                )
                _log.debug(
                    "PTT calling _callback(down, code=%d, flags=%d, autorepeat=%s)",
                    code,
                    flags,
                    autorepeat,
                )
                self._callback(
                    "down", code, effective_flags(flags, self._modifier_flags), autorepeat
                )
                _log.debug("PTT _callback(down) returned OK")
            else:
                _log.debug("PTT calling _callback(up, code=%d, flags=%d)", code, flags)
                self._callback("up", code, flags, autorepeat=False)
                _log.debug("PTT _callback(up) returned OK")
        except Exception as e:
            # A crashing callback would kill the tap for good; the dictation
            # itself must survive any monitor hiccup.
            _log.debug("PTT callback exception: %s", e, exc_info=True)
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
                    Quartz.CFRunLoopGetMain(), self._source, Quartz.kCFRunLoopCommonModes
                )
        except Exception:
            pass
        self._tap = None
        self._source = None
        self.active = False
        self.disabled_reason = ""
        self._modifier_flags = 0


@dataclass
class PTTSelfTest:
    """One-shot verification that the tap and the combo actually fire.

    Armed by the menu bar when the CLI asks for a check (``trnscrb ptt
    check``); the next press of the PTT key is verified, never dictated.
    ``result`` is ``"pass"`` (the combo completed — the tap is alive and
    sees the modifiers), ``"modifiers_missing"`` (the tap saw the key but
    the modifiers never completed the combo), ``"no_event"`` (nothing
    reached the tap by the deadline), or ``None`` while in flight.
    ``nonce`` ties the result to the request that armed it.
    """

    armed: bool = False
    deadline: float = 0.0
    nonce: str = ""
    result: str | None = None
    detail: str = ""
    t: float = 0.0


# ── push-to-talk monitor ─────────────────────────────────────────────────────


class PTTMonitor:
    """Hold the configured combo to dictate; release to stop and paste.

    ``on_start`` must return True when the dictation actually began (the
    release then stops it) and False when it was refused (dictation or a
    meeting already running), in which case the release stops nothing.
    """

    def __init__(
        self,
        key: PTTKey,
        on_start: Callable[[], bool],
        on_stop: Callable[[], None],
        on_selftest: Callable[[], None] | None = None,
    ):
        self.key = key
        self.state = PTTState(key)
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_selftest = on_selftest
        self._tap = EventTap(self._handle, on_flags_changed=self._on_flags)
        self.selftest = PTTSelfTest()
        self._last_event: dict | None = None

    @property
    def active(self) -> bool:
        return self._tap.active

    @property
    def disabled_reason(self) -> str:
        """Why the system disabled the tap (``""`` while it is enabled).

        Non-empty means the tap object is alive but no events are being
        delivered — the owner should tear it down and re-arm rather than
        trust :attr:`active`, which stays True until ``stop()``.
        """
        return self._tap.disabled_reason

    @property
    def last_event(self) -> dict | None:
        """The last PTT key press the tap saw: ``{"t": …, "matched": …}``.

        ``matched`` is the verdict at the moment the key went down (a press
        completed late by its modifiers records ``matched`` True on release).
        """
        return self._last_event

    def arm_selftest(self, timeout: float = 15.0, nonce: str = "") -> None:
        """Arm the one-shot self-check. The next press is verified, not dictated.

        A full match reports ``pass``; a key-down that is released before the
        modifiers complete reports ``modifiers_missing``; the deadline with
        nothing at all reports ``no_event``. Never starts a dictation — the
        press that completes the combo is closed without side effects.
        """
        st = PTTSelfTest(armed=True, deadline=time.monotonic() + timeout, nonce=nonce)
        self.selftest = st
        timer = threading.Timer(timeout + 0.5, self._expire_selftest, args=(st,))
        timer.daemon = True
        timer.start()
        _log.info("PTT self-test armed (nonce=%s, %.0fs window)", nonce or "-", timeout)
        self._notify_selftest()

    def selftest_info(self) -> dict:
        """The current self-test for publishing (``trnscrb status``/``check``)."""
        st = self.selftest
        return {
            "nonce": st.nonce,
            "armed": st.armed and st.result is None,
            "result": st.result,
            "detail": st.detail,
            "t": st.t,
        }

    def _expire_selftest(self, st: PTTSelfTest) -> None:
        """The deadline passed without a press: nothing reached the tap."""
        if self.selftest is st and st.armed and st.result is None:
            st.result = "no_event"
            st.detail = "no key event reached the tap before the deadline"
            st.t = time.time()
            st.armed = False
            _log.info("PTT self-test: no_event")
            self._notify_selftest()

    def _finish_selftest(self, result: str, detail: str) -> bool:
        """Record a verdict (once per armed test). True when it was recorded."""
        st = self.selftest
        if not st.armed or st.result is not None:
            return False
        st.result = result
        st.detail = detail
        st.t = time.time()
        st.armed = False
        _log.info("PTT self-test: %s — %s", result, detail)
        self._notify_selftest()
        return True

    def _notify_selftest(self) -> None:
        if self._on_selftest is None:
            return
        try:
            self._on_selftest()
        except Exception:
            _log.debug("PTT self-test notify failed", exc_info=True)

    def start(self) -> bool:
        return self._tap.start()

    def stop(self) -> None:
        self._tap.stop()

    def _handle(self, kind: str, code: int, flags: int, autorepeat: bool) -> None:
        if kind == "down":
            matched = handle_key_down(self.state, code, flags, autorepeat)
            if code == self.key.key_code:
                self._last_event = {"t": time.time(), "matched": bool(matched)}
            if matched and self._finish_selftest(
                "pass", "combo matched — tap alive, modifiers seen"
            ):
                # Self-test press: close it without dictating; the key-up
                # clears the press state.
                self.state.started = False
                return
            if matched:
                ok = bool(self._on_start())
                confirm_start(self.state, ok)
                _log.info("PTT combo: dictation start ok=%s", ok)
            elif code == self.key.key_code:
                _log.debug(
                    "PTT combo: down not triggered (holding=%s started=%s "
                    "pending=%s event flags=%d required flags=%d)",
                    self.state.holding,
                    self.state.started,
                    self.state.pending,
                    flags,
                    self.key.flags,
                )
        else:
            if code == self.key.key_code:
                started = self.state.started
                was_pending = self.state.pending
                if handle_key_up(self.state, code):
                    self._on_stop()
                self._last_event = {"t": time.time(), "matched": bool(started)}
                if was_pending and self.selftest.armed and self.selftest.result is None:
                    self._finish_selftest(
                        "modifiers_missing",
                        "the tap saw the key but the modifiers never completed "
                        "the combo — hold the full combo as one press",
                    )

    def _on_flags(self, flags: int) -> None:
        """The modifier state changed while a press was waiting on it.

        The physical order can put the PTT key down before its modifiers
        (a fast press, or a stale event flag snapshot), so the key-down's
        first match saw no modifiers and left the press pending. Re-match
        with the live state: the same press now completes the combo and
        starts. A bare press with no modifier arriving completes nothing.
        """
        if not self.state.pending or self.state.holding:
            return
        if handle_key_down(self.state, self.key.key_code, flags):
            if self._finish_selftest(
                "pass", "combo completed late — modifiers arrived after the key"
            ):
                self.state.started = False
                return
            ok = bool(self._on_start())
            confirm_start(self.state, ok)
            _log.info("PTT combo (late modifiers): dictation start ok=%s", ok)


# ── record mode ──────────────────────────────────────────────────────────────


class PTTCapture:
    """One-shot capture of the next key press (the menu bar's record mode).

    Any non-modifier key with optional modifiers counts — modifier-only
    presses are ignored (the flagsChanged events used to track the live
    modifier state never fire a capture). ``on_capture`` receives
    (key_code, ptt_modifiers) on the main run loop and is fired exactly
    once. The tracked modifier state also repairs a stale key-down
    snapshot here, so a recorded combo keeps the modifiers that were
    physically held.

    Caps lock is the one "modifier" that does emit a key press (it is a
    toggle key), so the caps key-down itself never fires the capture — it
    only opens the window during which a captured combo carries
    ``FLAG_CAPS_LOCK``. The captured bit reflects the physical hold, not
    the latched alpha-shift flag, for the same reason as the monitor.
    """

    def __init__(self, on_capture: Callable[[int, int], None]):
        self._on_capture = on_capture
        self._fired = threading.Event()
        self._tap = EventTap(self._handle)
        # Finger physically on caps lock (see the class docstring).
        self._caps_held = False

    @property
    def active(self) -> bool:
        return self._tap.active

    def start(self) -> bool:
        return self._tap.start()

    def stop(self) -> None:
        self._tap.stop()

    def _handle(self, kind: str, code: int, flags: int, _autorepeat: bool) -> None:
        if code == CAPS_LOCK_CODE:
            # The caps press is a modifier in capture mode, never the combo.
            self._caps_held = kind == "down"
            return
        if kind != "down" or self._fired.is_set():
            return
        self._fired.set()
        self._tap.stop()
        mods = ptt_mods(flags) & ~FLAG_CAPS_LOCK
        if self._caps_held:
            mods |= FLAG_CAPS_LOCK
        self._on_capture(code, mods)
