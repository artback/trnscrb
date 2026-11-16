"""Tests for the Apple SpeechAnalyzer live-word layer.

Runs on headless Linux: ``available()`` returns ``(False, "not macOS")``,
``LiveSession.start()`` is a no-op that returns ``False``, and
``live_source()`` routes to "model".
"""

import threading
import unittest
from pathlib import Path
from unittest import mock

from trnscrb import apple_speech


class AvailableTest(unittest.TestCase):
    """``available()`` decision table for the current host."""

    def test_returns_false_on_headless(self):
        """On Linux the OS check fails immediately."""
        ok, reason = apple_speech.available()
        self.assertIs(ok, False)
        self.assertIsInstance(reason, str)
        self.assertTrue(len(reason) > 0)

    def test_no_exception_on_headless(self):
        """Must never raise on headless — degradation must be silent."""
        try:
            ok, reason = apple_speech.available()
            self.assertIsInstance(ok, bool)
            self.assertIsInstance(reason, str)
        except Exception:
            self.fail("available() raised on headless host")

    def test_os_check_is_standalone(self):
        """The OS/arch check happens before any PyObjC import."""
        # Inject a fake non-Darwin to confirm the early-return path.
        with mock.patch("platform.system", return_value="Linux"):
            ok, reason = apple_speech.available()
            self.assertIs(ok, False)
            self.assertIn("not macOS", reason)

    def test_arch_check_is_standalone(self):
        """x86_64 on macOS is rejected."""
        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch("platform.processor", return_value="i386"):
                with mock.patch("platform.release", return_value="0.0.0"):
                    ok, reason = apple_speech.available()
                    self.assertIs(ok, False)
                    self.assertIn("not Apple Silicon", reason)


class LiveSessionTest(unittest.TestCase):
    """Session lifecycle: start, stop, callback."""

    def test_start_inactive_on_headless(self):
        """``start()`` must not raise; returns False with a reason."""
        session = apple_speech.LiveSession(lambda text, is_final: None)
        result = session.start()
        self.assertIs(result, False)
        self.assertFalse(session.active)
        self.assertIsInstance(session.reason, str)

    def test_stop_is_noop_when_inactive(self):
        """Stopping an unstopped session is safe."""
        session = apple_speech.LiveSession(lambda text, is_final: None)
        session.stop()  # must not raise
        self.assertFalse(session.active)

    def test_double_stop_is_safe(self):
        """Calling stop() twice is idempotent."""
        session = apple_speech.LiveSession(lambda text, is_final: None)
        session.stop()
        session.stop()  # must not raise

    def test_factory_returns_inactive_session(self):
        """``create_session()`` always returns an object; on headless it's
        inactive but the object is valid."""
        session = apple_speech.create_session(lambda text, is_final: None)
        self.assertIsInstance(session, apple_speech.LiveSession)
        self.assertFalse(session.active)
        self.assertIsInstance(session.reason, str)

    def test_callback_accumulates(self):
        """A fake session (injected callback) feeds text through _on_live_text
        like the real menu-bar accumulator does.  Both partial and final text
        flow through the callback; the consumer decides what to show."""
        accumulated: list[str] = []

        def on_live(text: str) -> None:
            accumulated.append(text)

        # Inject a mock session that calls the callback directly.
        session = apple_speech.LiveSession(
            lambda text, is_final: on_live(text)
        )
        # Simulate callback invocations (would come from the framework
        # on a real macOS host).
        session._callback("hello", False)
        session._callback("hello world", True)
        session._callback("hello world today", False)
        self.assertEqual(
            accumulated,
            ["hello", "hello world", "hello world today"],
        )


class LiveSourceTest(unittest.TestCase):
    """``live_source()`` decision table."""

    def test_auto_when_available(self):
        """auto + available → apple."""
        result = apple_speech.live_source(
            "auto", _available=lambda: (True, "")
        )
        self.assertEqual(result, "apple")

    def test_auto_when_unavailable(self):
        """auto + unavailable → model (no log line)."""
        result = apple_speech.live_source(
            "auto", _available=lambda: (False, "reason text")
        )
        self.assertEqual(result, "model")

    def test_force_model_always_model(self):
        """Force model → model regardless of availability."""
        result = apple_speech.live_source(
            "model", _available=lambda: (True, "")
        )
        self.assertEqual(result, "model")

        result = apple_speech.live_source(
            "model", _available=lambda: (False, "anything")
        )
        self.assertEqual(result, "model")

    def test_force_apple_available(self):
        """Force apple + available → apple."""
        result = apple_speech.live_source(
            "apple", _available=lambda: (True, "")
        )
        self.assertEqual(result, "apple")

    def test_force_apple_unavailable_logs_once(self):
        """Force apple + unavailable → model with one-time log line."""
        notice: list[bool] = [False]
        with mock.patch.object(apple_speech._log, "warning") as mock_warn:
            result = apple_speech.live_source(
                "apple",
                _available=lambda: (False, "reason text"),
                _first_notice=notice,
            )
            self.assertEqual(result, "model")
            mock_warn.assert_called_once()
            self.assertIn("reason text", mock_warn.call_args.args[1])

            # Second call should NOT log again.
            notice[0] = True  # signal first notice was sent
            with mock.patch.object(apple_speech._log, "warning") as mock_warn2:
                apple_speech.live_source(
                    "apple",
                    _available=lambda: (False, "reason text"),
                    _first_notice=notice,
                )
                # The _first_notice list is shared; the function checks
                # if notice[0] is already True to skip logging.
                mock_warn2.assert_not_called()

    def test_unknown_backend_defaults_to_auto(self):
        """An unknown setting value falls through to auto (safe default)."""
        result = apple_speech.live_source(
            "unknown", _available=lambda: (True, "")
        )
        self.assertEqual(result, "apple")  # auto path + available

    def test_empty_setting_defaults_to_auto(self):
        """Empty string defaults to auto, so available → apple."""
        result = apple_speech.live_source(
            "", _available=lambda: (True, "")
        )
        self.assertEqual(result, "apple")  # auto path + available

    def test_no_external_dependency_in_default(self):
        """Default ``live_source()`` uses ``available()`` — no mock needed."""
        result = apple_speech.live_source("auto")
        self.assertEqual(result, "model")  # headless → model


class SettingsDefaultTest(unittest.TestCase):
    """Verify ``live_backend`` default in settings."""

    def test_default_is_auto(self):
        from trnscrb import settings

        default = settings._DEFAULTS.get("live_backend")
        self.assertEqual(default, "auto")


class BatteryPauseTest(unittest.TestCase):
    """Battery-pause interaction mirrors the existing dictation pattern.

    The Apple layer respects the same battery policy as the model layer:
    ``_dict_live_loop`` checks ``_on_battery()`` and ``live_on_battery``
    *before* starting the Apple session.  The session itself does not
    implement its own power management.
    """

    def test_session_has_no_battery_logic(self):
        """The LiveSession is power-agnostic; callers handle it."""
        session = apple_speech.LiveSession(lambda text, is_final: None)
        # No public battery-related attributes or methods.
        self.assertFalse(hasattr(session, "_battery"))
        self.assertFalse(hasattr(session, "on_battery"))
        self.assertFalse(hasattr(session, "pause"))


class HealthStatusTest(unittest.TestCase):
    """``health.apple_status()`` on headless."""

    def test_returns_unavailable_line(self):
        from trnscrb import health

        status = health.apple_status()
        self.assertIsInstance(status, str)
        self.assertIn("Apple live:", status)
        self.assertIn("unavailable", status)