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


if __name__ == "__main__":
    unittest.main()
