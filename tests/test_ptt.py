"""Tests for the push-to-talk state machine (the pure part of trnscrb.ptt).

The CGEvent tap plumbing is exercised on a real Mac only; the press/stop
semantics — the part that must never stop a dictation it did not start —
are covered here, on any platform.
"""

import unittest

from trnscrb import hotkey, ptt

KEY = hotkey.PTTKey(
    key_code=99,  # F8
    flags=hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION,
    spec="ctrl+alt+f8",
)

CTRL_ALT = hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION


def _state() -> ptt.PTTState:
    return ptt.PTTState(KEY)


class TestStateMachine(unittest.TestCase):
    def test_matching_press_starts(self):
        s = _state()
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT))
        ptt.confirm_start(s, True)
        self.assertTrue(s.started)

    def test_missing_modifier_does_not_start(self):
        s = _state()
        self.assertFalse(ptt.handle_key_down(s, 99, hotkey.FLAG_CONTROL))
        self.assertFalse(s.holding)

    def test_extra_modifier_is_fine(self):
        # A stray shift must not defeat the combo.
        s = _state()
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT | hotkey.FLAG_SHIFT))

    def test_other_key_ignored(self):
        s = _state()
        self.assertFalse(ptt.handle_key_down(s, 0, CTRL_ALT))

    def test_autorepeat_ignored(self):
        s = _state()
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT, autorepeat=False))
        self.assertFalse(ptt.handle_key_down(s, 99, CTRL_ALT, autorepeat=True))

    def test_second_press_while_holding_ignored(self):
        s = _state()
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT))
        self.assertFalse(ptt.handle_key_down(s, 99, CTRL_ALT))

    def test_release_stops_only_when_started(self):
        s = _state()
        ptt.handle_key_down(s, 99, CTRL_ALT)
        ptt.confirm_start(s, True)
        self.assertTrue(ptt.handle_key_up(s, 99))
        # The press is closed; a stray release is a no-op.
        self.assertFalse(ptt.handle_key_up(s, 99))

    def test_release_without_press_is_noop(self):
        s = _state()
        self.assertFalse(ptt.handle_key_up(s, 99))

    def test_release_of_other_key_is_noop(self):
        s = _state()
        ptt.handle_key_down(s, 99, CTRL_ALT)
        ptt.confirm_start(s, True)
        self.assertFalse(ptt.handle_key_up(s, 0))

    def test_refused_start_release_does_not_stop(self):
        # e.g. a meeting is recording: the start is refused, so the release
        # must not stop the meeting recording that owns the mic.
        s = _state()
        ptt.handle_key_down(s, 99, CTRL_ALT)
        ptt.confirm_start(s, False)
        self.assertFalse(ptt.handle_key_up(s, 99))

    def test_refused_press_allows_next_press(self):
        s = _state()
        ptt.handle_key_down(s, 99, CTRL_ALT)
        ptt.confirm_start(s, False)
        ptt.handle_key_up(s, 99)
        # holding cleared → a fresh press works again
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT))

    def test_release_closes_press_even_if_session_was_cut(self):
        # The max-duration cap can stop the dictation while the user still
        # holds the key; the later release then stops nothing (no-op) rather
        # than double-stopping, because it still "was started".
        s = _state()
        ptt.handle_key_down(s, 99, CTRL_ALT)
        ptt.confirm_start(s, True)
        self.assertTrue(ptt.handle_key_up(s, 99))
        self.assertFalse(s.holding)
        self.assertFalse(s.started)

    def test_lost_key_up_leaves_stuck_press_reset_heals(self):
        # The key-up never arrives (the system disabled the tap mid-press,
        # or the dictation ended via the safety net first): ``holding``
        # latches and the next press is swallowed as a "second press".
        # ``reset()`` — what the 5s menu-bar check calls when no dictation
        # is live — heals the stuck state.
        s = _state()
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT))
        ptt.confirm_start(s, True)
        # key-up lost
        self.assertFalse(ptt.handle_key_down(s, 99, CTRL_ALT))  # swallowed
        s.reset()
        self.assertTrue(ptt.handle_key_down(s, 99, CTRL_ALT))


# ── caps lock as a hold modifier ──────────────────────────────────────────────
#
# Caps lock is a toggle key: the only "modifier" that emits a real
# key-down/key-up pair, and its alpha-shift flag bit latches (it says
# "caps lock is on", not "the finger is on caps lock"). The state machine
# therefore tracks the physical hold of CAPS_LOCK_CODE and matches
# ``FLAG_CAPS_LOCK`` against that — never against the event's flag bit.

CAPS_KEY = hotkey.PTTKey(
    key_code=17,  # T
    flags=hotkey.FLAG_CAPS_LOCK,
    spec="caps+t",
)


def _caps_state() -> ptt.PTTState:
    return ptt.PTTState(CAPS_KEY)


class TestCapsLockHold(unittest.TestCase):
    def test_caps_keydown_never_starts_and_tracks_hold(self):
        s = _caps_state()
        self.assertFalse(ptt.handle_key_down(s, hotkey.CAPS_LOCK_CODE, 0))
        self.assertFalse(s.holding)
        self.assertTrue(s.caps_held)

    def test_latched_flag_bit_alone_does_not_match(self):
        # The alpha-shift bit is set whenever the lock happens to be on,
        # finger or no finger. That is not a press of the combo.
        s = _caps_state()
        self.assertFalse(ptt.handle_key_down(s, 17, hotkey.FLAG_CAPS_LOCK))
        self.assertFalse(s.holding)

    def test_physical_hold_matches_regardless_of_flag_bit(self):
        # While the finger is on caps lock the latched bit may be on or off
        # (the press toggles the lock), so the match must not look at it at
        # all. The caps key-down itself never starts a dictation — it only
        # opens the hold window.
        s = _caps_state()
        self.assertFalse(ptt.handle_key_down(s, hotkey.CAPS_LOCK_CODE, 0))
        self.assertTrue(s.caps_held)
        self.assertTrue(ptt.handle_key_down(s, 17, 0))
        s2 = _caps_state()
        self.assertFalse(ptt.handle_key_down(s2, hotkey.CAPS_LOCK_CODE, hotkey.FLAG_CAPS_LOCK))
        self.assertTrue(ptt.handle_key_down(s2, 17, hotkey.FLAG_CAPS_LOCK))

    def test_release_of_caps_does_not_stop_dictation(self):
        s = _caps_state()
        ptt.handle_key_down(s, hotkey.CAPS_LOCK_CODE, 0)
        ptt.handle_key_down(s, 17, 0)
        ptt.confirm_start(s, True)
        # Releasing caps first (like releasing shift early) keeps the
        # dictation running until the PTT key goes up.
        self.assertFalse(ptt.handle_key_up(s, hotkey.CAPS_LOCK_CODE))
        self.assertTrue(s.holding)
        self.assertTrue(ptt.handle_key_up(s, 17))

    def test_caps_release_clears_hold_window(self):
        s = _caps_state()
        ptt.handle_key_down(s, hotkey.CAPS_LOCK_CODE, 0)
        ptt.handle_key_up(s, hotkey.CAPS_LOCK_CODE)
        self.assertFalse(s.caps_held)
        # Without the finger on caps the combo no longer matches — even
        # with the latched flag bit set.
        self.assertFalse(ptt.handle_key_down(s, 17, hotkey.FLAG_CAPS_LOCK))

    def test_reset_clears_caps_hold(self):
        s = _caps_state()
        ptt.handle_key_down(s, hotkey.CAPS_LOCK_CODE, 0)
        s.reset()
        self.assertFalse(s.caps_held)


class TestPTTCaptureCapsLock(unittest.TestCase):
    """Record mode captures the physical caps hold, not the latched flag."""

    def _capture(self):
        captured = []
        capture = ptt.PTTCapture(lambda code, mods: captured.append((code, mods)))
        return capture, captured

    def test_caps_keydown_never_fires_capture(self):
        capture, captured = self._capture()
        capture._handle("down", hotkey.CAPS_LOCK_CODE, 0, False)
        self.assertEqual(captured, [])

    def test_hold_recorded_as_caps_modifier(self):
        capture, captured = self._capture()
        capture._handle("down", hotkey.CAPS_LOCK_CODE, 0, False)
        capture._handle("down", 17, 0, False)
        self.assertEqual(captured, [(17, hotkey.FLAG_CAPS_LOCK)])

    def test_latched_flag_without_hold_not_recorded(self):
        capture, captured = self._capture()
        # Caps lock happens to be on (bit set) but the finger is not on it:
        # the captured combo must not claim caps.
        capture._handle("down", 17, hotkey.FLAG_CAPS_LOCK, False)
        self.assertEqual(captured, [(17, 0)])

    def test_hold_dropped_on_release(self):
        capture, captured = self._capture()
        capture._handle("down", hotkey.CAPS_LOCK_CODE, 0, False)
        capture._handle("up", hotkey.CAPS_LOCK_CODE, 0, False)
        capture._handle("down", 17, hotkey.FLAG_CAPS_LOCK, False)
        self.assertEqual(captured, [(17, 0)])


class TestModifierTracking(unittest.TestCase):
    """A stale key-down snapshot must not defeat the combo.

    The 2026-11-18 field case: the user held ctrl+shift+k (the terminal
    reported the combo), but the key-down event delivered to the tap
    carried flags=256 (no modifier bits) — the window server had not yet
    merged the held modifiers into the event. The tap's flagsChanged
    tracking supplies the live state; effective_flags ORs it in.
    """

    CTRL_SHIFT = hotkey.FLAG_CONTROL | hotkey.FLAG_SHIFT
    CTRL_SHIFT_K = hotkey.PTTKey(
        key_code=7,  # K
        flags=CTRL_SHIFT,
        spec="ctrl+shift+k",
    )
    STALE_EVENT = 256  # 0x100, kCGEventFlagMaskNonCoalesced only

    def _state(self) -> ptt.PTTState:
        return ptt.PTTState(self.CTRL_SHIFT_K)

    def test_stale_event_recovered_from_tracking(self):
        """Event without modifier bits + tracked ctrl+shift → matches."""
        s = self._state()
        self.assertTrue(
            ptt.handle_key_down(
                s, 7, ptt.effective_flags(self.STALE_EVENT, self.CTRL_SHIFT)
            )
        )

    def test_bare_press_stays_bare(self):
        """No modifiers tracked, none in the event → no match."""
        s = self._state()
        self.assertFalse(
            ptt.handle_key_down(s, 7, ptt.effective_flags(self.STALE_EVENT, 0))
        )
        self.assertFalse(s.holding)

    def test_fresh_event_needs_no_tracking(self):
        """Event that carries the modifiers matches on its own (the old
        behaviour, untouched)."""
        s = self._state()
        self.assertTrue(
            ptt.handle_key_down(
                s, 7, ptt.effective_flags(self.CTRL_SHIFT | self.STALE_EVENT, 0)
            )
        )

    def test_partial_tracking_fills_the_gap(self):
        """Event carries ctrl, tracking carries shift → both present."""
        s = self._state()
        self.assertTrue(
            ptt.handle_key_down(
                s,
                7,
                ptt.effective_flags(
                    hotkey.FLAG_CONTROL | self.STALE_EVENT, hotkey.FLAG_SHIFT
                ),
            )
        )

    def test_extra_tracked_modifier_is_fine(self):
        s = self._state()
        self.assertTrue(
            ptt.handle_key_down(
                s,
                7,
                ptt.effective_flags(self.STALE_EVENT, self.CTRL_SHIFT | hotkey.FLAG_COMMAND),
            )
        )


class TestLateModifiers(unittest.TestCase):
    """The PTT key may come down before its modifiers.

    A fast press (or a stale event snapshot) can land the key-down before
    the modifiers are visible, so the first match sees a bare key. The
    press is kept ``pending`` instead of being dropped; when the modifier
    state changes (flagsChanged) the monitor re-matches the same press and
    the combo completes.
    """

    CTRL_SHIFT = hotkey.FLAG_CONTROL | hotkey.FLAG_SHIFT
    CTRL_SHIFT_K = hotkey.PTTKey(
        key_code=7,  # K
        flags=CTRL_SHIFT,
        spec="ctrl+shift+k",
    )

    def _state(self) -> ptt.PTTState:
        return ptt.PTTState(self.CTRL_SHIFT_K)

    def test_bare_press_is_pending_not_dead(self):
        s = self._state()
        self.assertFalse(ptt.handle_key_down(s, 7, 0))
        self.assertFalse(s.holding)
        self.assertTrue(s.pending)

    def test_late_modifiers_complete_the_press(self):
        s = self._state()
        self.assertFalse(ptt.handle_key_down(s, 7, 0))  # K first
        self.assertTrue(ptt.handle_key_down(s, 7, self.CTRL_SHIFT))
        self.assertTrue(s.holding)
        self.assertFalse(s.pending)

    def test_partial_then_full(self):
        s = self._state()
        self.assertFalse(ptt.handle_key_down(s, 7, hotkey.FLAG_CONTROL))
        self.assertTrue(s.pending)
        self.assertFalse(s.holding)
        self.assertTrue(ptt.handle_key_down(s, 7, self.CTRL_SHIFT))
        self.assertTrue(s.holding)

    def test_autorepeat_keeps_pending(self):
        s = self._state()
        ptt.handle_key_down(s, 7, 0, autorepeat=True)
        ptt.handle_key_down(s, 7, 0, autorepeat=True)
        self.assertTrue(s.pending)
        self.assertFalse(s.holding)
        self.assertTrue(ptt.handle_key_down(s, 7, self.CTRL_SHIFT))

    def test_release_clears_pending(self):
        s = self._state()
        ptt.handle_key_down(s, 7, 0)
        self.assertTrue(s.pending)
        self.assertFalse(ptt.handle_key_up(s, 7))  # nothing started, nothing stopped
        self.assertFalse(s.pending)
        self.assertFalse(s.holding)

    def test_other_key_breaks_pending(self):
        s = self._state()
        ptt.handle_key_down(s, 7, 0)
        self.assertTrue(s.pending)
        ptt.handle_key_down(s, 5, 0)
        self.assertFalse(s.pending)

    def test_bare_press_without_modifiers_never_starts(self):
        """A bare tap of the key, no modifiers arriving, starts nothing."""
        s = self._state()
        started = []
        mon = ptt.PTTMonitor(
            self.CTRL_SHIFT_K,
            on_start=lambda: started.append(True) or True,
            on_stop=lambda: None,
        )
        mon._handle("down", 7, 0, False)
        mon._on_flags(0)  # no modifier state change to complete it
        self.assertEqual(started, [])
        self.assertFalse(mon.state.started)

    def test_monitor_starts_when_modifiers_arrive_late(self):
        started = []
        mon = ptt.PTTMonitor(
            self.CTRL_SHIFT_K,
            on_start=lambda: started.append(True) or True,
            on_stop=lambda: None,
        )
        mon._handle("down", 7, 0, False)
        self.assertEqual(started, [])
        mon._on_flags(self.CTRL_SHIFT)
        self.assertEqual(started, [True])
        self.assertTrue(mon.state.started)
        # The release stops what this press started.
        self.assertTrue(ptt.handle_key_up(mon.state, 7))


if __name__ == "__main__":
    unittest.main()
