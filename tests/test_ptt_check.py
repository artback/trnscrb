"""_ptt_check: self-heal of a stuck PTT press and re-arm of a disabled tap.

A lost key-up (system-disabled tap, dictation ended by the safety net)
leaves ``PTTState.holding`` latched, which swallows the next press. The 5s
check must clear that state when no dictation is live, and must re-arm a
tap the system disabled rather than trust ``active``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from trnscrb import hotkey, ptt
from trnscrb import menu_bar

KEY = hotkey.PTTKey(
    key_code=99,  # F8
    flags=hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION,
    spec="ctrl+alt+f8",
)

CTRL_ALT = hotkey.FLAG_CONTROL | hotkey.FLAG_OPTION


class _FakeMonitor:
    """Stand-in for ptt.PTTMonitor with a real state machine."""

    def __init__(self, key, active=True, disabled_reason="", **_kwargs):
        self.key = key
        self.state = ptt.PTTState(key)
        self._active = active
        self._disabled_reason = disabled_reason
        self.stopped = False

    @property
    def active(self):
        return self._active

    @property
    def disabled_reason(self):
        return self._disabled_reason

    def start(self):
        return self._active

    def stop(self):
        self.stopped = True
        self._active = False
        self._disabled_reason = ""


def _app(monitor, dict_preset=None):
    app = menu_bar.TrnscrbApp.__new__(menu_bar.TrnscrbApp)
    app._ptt_item = SimpleNamespace(title="PTT: off")
    app._ptt_monitor = monitor
    app._ptt_armed_spec = "ctrl+alt+f8"
    app._ptt_notified = False
    app._dict_preset = dict_preset
    app._ptt_start = lambda: True
    app._ptt_stop = lambda: None
    return app


def _run_check(app):
    with mock.patch.object(menu_bar, "get_setting", return_value="ctrl+alt+f8"):
        app._ptt_check()


class TestStuckPressSelfHeal(unittest.TestCase):
    def _stuck_monitor(self):
        """A monitor whose key-up was lost: holding latched, dictation gone."""
        monitor = _FakeMonitor(KEY)
        ptt.handle_key_down(monitor.state, 99, CTRL_ALT)
        ptt.confirm_start(monitor.state, True)
        self.assertTrue(monitor.state.holding)
        return monitor

    def test_stuck_press_reset_when_no_dictation_live(self):
        monitor = self._stuck_monitor()
        app = _app(monitor)
        _run_check(app)
        self.assertFalse(monitor.state.holding)
        # The healthy monitor is kept (no-op path), the title shows the combo.
        self.assertIs(app._ptt_monitor, monitor)
        self.assertEqual(app._ptt_item.title, "PTT: ⌃⌥F8")

    def test_press_state_kept_while_dictation_live(self):
        # A dictation running means the press may still be in flight
        # (user still holding the key); clearing it here would let
        # autorepeat restart the dictation.
        monitor = self._stuck_monitor()
        app = _app(monitor, dict_preset="message")
        _run_check(app)
        self.assertTrue(monitor.state.holding)

    def test_idle_monitor_not_touched(self):
        monitor = _FakeMonitor(KEY)
        app = _app(monitor)
        _run_check(app)
        self.assertFalse(monitor.stopped)
        self.assertIs(app._ptt_monitor, monitor)


class TestDisabledTapRearm(unittest.TestCase):
    def test_disabled_tap_forces_rearm(self):
        monitor = _FakeMonitor(KEY, disabled_reason="timeout")
        app = _app(monitor)
        fresh = _FakeMonitor(KEY)
        with (
            mock.patch.object(menu_bar, "get_setting", return_value="ctrl+alt+f8"),
            mock.patch.object(menu_bar.ptt, "PTTMonitor", return_value=fresh),
        ):
            app._ptt_check()
        self.assertTrue(monitor.stopped)
        self.assertIs(app._ptt_monitor, fresh)
        self.assertEqual(app._ptt_armed_spec, "ctrl+alt+f8")

    def test_enabled_tap_not_rearmed(self):
        monitor = _FakeMonitor(KEY)
        app = _app(monitor)
        with (
            mock.patch.object(menu_bar, "get_setting", return_value="ctrl+alt+f8"),
            mock.patch.object(menu_bar.ptt, "PTTMonitor") as ctor,
        ):
            app._ptt_check()
        ctor.assert_not_called()
        self.assertIs(app._ptt_monitor, monitor)


if __name__ == "__main__":
    unittest.main()