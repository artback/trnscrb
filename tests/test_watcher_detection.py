"""Tests for MicWatcher crash recovery and detect_meeting priority logic.

All tests mock subprocess, osascript, and CoreAudio — no real devices or
network access required.
"""
import time
import unittest
from unittest.mock import MagicMock, patch

from trnscrb import watcher
from trnscrb.watcher import MicWatcher, detect_meeting


class WatcherCrashRecoveryTest(unittest.TestCase):
    """Test that _loop catches exceptions from _loop_inner and restarts."""

    def test_loop_restarts_after_loop_inner_crash(self):
        """If _loop_inner raises, _loop should sleep then spawn a new thread."""
        w = MicWatcher(on_start=lambda n: None, on_stop=lambda: None)
        w._running = True

        import threading
        new_thread = MagicMock(spec=threading.Thread)

        with patch.object(w, "_loop_inner", side_effect=RuntimeError("crash")), \
             patch.object(watcher.time, "sleep") as mock_sleep, \
             patch("threading.Thread", return_value=new_thread) as mock_thread_cls:
            w._loop()

        # Should have slept 5s before restart
        mock_sleep.assert_called_with(5)
        # Should have spawned a new thread targeting _loop
        mock_thread_cls.assert_called_once()
        new_thread.start.assert_called_once()

    def test_loop_does_not_restart_when_stopped(self):
        """If _running is False when crash happens, no restart should occur."""
        w = MicWatcher(on_start=lambda n: None, on_stop=lambda: None)
        w._running = False

        def crashing_inner():
            raise RuntimeError("simulated crash")

        with patch.object(w, "_loop_inner", side_effect=crashing_inner), \
             patch.object(watcher.time, "sleep"):
            w._loop()

        # _running was False, so no new thread should be started
        # Verify by checking that _thread was not reassigned
        # (it stays None since we never called start())
        self.assertIsNone(w._thread)

    def test_loop_sleeps_before_restart(self):
        """Crash recovery should sleep 5s before restarting."""
        w = MicWatcher(on_start=lambda n: None, on_stop=lambda: None)
        w._running = True

        def crashing_inner():
            w._running = False  # prevent infinite restart loop
            raise RuntimeError("boom")

        with patch.object(w, "_loop_inner", side_effect=crashing_inner), \
             patch.object(watcher.time, "sleep") as mock_sleep:
            w._loop()

        mock_sleep.assert_called_once_with(5)


class DetectMeetingPriorityTest(unittest.TestCase):
    """Test that detect_meeting checks browser tabs before native apps."""

    def test_browser_tab_checked_before_native_apps(self):
        """If browser has a meeting tab, native app check should be skipped."""
        with patch.object(watcher, "_browser_has_meeting_tab", return_value="Google Meet") as mock_browser, \
             patch.object(watcher, "_pids_using_mic_input") as mock_pids, \
             patch.object(watcher, "_meeting_app_pids") as mock_app_pids:
            result = detect_meeting()

        self.assertEqual(result, "Google Meet")
        mock_browser.assert_called_once_with(return_name=True)
        # Native app PID checks should not have been reached
        mock_pids.assert_not_called()

    def test_native_app_used_when_no_browser_tab(self):
        """If no browser tab, should fall through to native app detection."""
        with patch.object(watcher, "_browser_has_meeting_tab", return_value=None), \
             patch.object(watcher, "_pids_using_mic_input", return_value={42}), \
             patch.object(watcher, "_meeting_app_pids", return_value={42}), \
             patch("subprocess.run") as mock_run:
            # ps output: PID 42 is zoom.us
            mock_run.return_value = MagicMock(
                stdout="   42 /Applications/zoom.us.app/Contents/Frameworks/zoom.us",
                returncode=0,
            )
            result = detect_meeting()

        self.assertEqual(result, "Zoom")

    def test_session_only_fallback_when_no_pid_match(self):
        """Fallback to session-only process list (CptHost for Zoom)."""
        with patch.object(watcher, "_browser_has_meeting_tab", return_value=None), \
             patch.object(watcher, "_pids_using_mic_input", return_value=set()), \
             patch.object(watcher, "_meeting_app_pids", return_value=set()), \
             patch("subprocess.run") as mock_run:
            # First call: native app PID check (empty intersection, falls through)
            # Second call: session-only process check
            mock_run.return_value = MagicMock(
                stdout="/some/path/CptHost\n/usr/bin/something",
                returncode=0,
            )
            result = detect_meeting()

        self.assertEqual(result, "Zoom")

    def test_facetime_background_not_matched_in_fallback(self):
        """FaceTime persists as a background process — it should NOT match
        in the session-only fallback (_SESSION_FALLBACK does not include it)."""
        with patch.object(watcher, "_browser_has_meeting_tab", return_value=None), \
             patch.object(watcher, "_pids_using_mic_input", return_value=set()), \
             patch.object(watcher, "_meeting_app_pids", return_value=set()), \
             patch("subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"trnscrb.calendar_integration": MagicMock()}):
            mock_run.return_value = MagicMock(
                stdout="/System/Applications/FaceTime.app/Contents/MacOS/FaceTime\n",
                returncode=0,
            )
            # Also mock the calendar import so it doesn't try real calendar access
            import sys
            cal_mod = sys.modules["trnscrb.calendar_integration"]
            cal_mod.get_current_or_upcoming_event.return_value = None

            result = detect_meeting()

        # FaceTime is NOT in _SESSION_FALLBACK, so it should fall through
        # to the calendar check (which returns None) then to the timestamp fallback
        self.assertTrue(result.startswith("meeting-"),
                        f"Expected timestamp fallback, got: {result}")

    def test_facetime_matched_only_when_using_mic(self):
        """FaceTime should be detected via native app path only when its PID
        is actively using the mic (cross-referenced with CoreAudio)."""
        with patch.object(watcher, "_browser_has_meeting_tab", return_value=None), \
             patch.object(watcher, "_pids_using_mic_input", return_value={99}), \
             patch.object(watcher, "_meeting_app_pids", return_value={99}), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="   99 /System/Applications/FaceTime.app/Contents/MacOS/FaceTime",
                returncode=0,
            )
            result = detect_meeting()

        self.assertEqual(result, "FaceTime")


class BrowserMeetingTabFilterTest(unittest.TestCase):
    """Test that the Chrome AppleScript filters out landing URLs and post-meeting pages."""

    def test_meet_landing_url_filtered_in_script(self):
        """The _CHROME_TAB_SCRIPT contains a 'does not contain /landing' clause."""
        self.assertIn('does not contain "/landing"', watcher._CHROME_TAB_SCRIPT)
        self.assertIn('does not contain "/landing"', watcher._SAFARI_TAB_SCRIPT)

    def test_post_meeting_pages_filtered_in_script(self):
        """Script should filter out 'ended' and 'left' title pages."""
        self.assertIn('does not contain "ended"', watcher._CHROME_TAB_SCRIPT)
        self.assertIn('does not contain "left"', watcher._CHROME_TAB_SCRIPT)

    def test_browser_has_meeting_tab_returns_name_or_bool(self):
        """_browser_has_meeting_tab returns str when return_name=True, bool otherwise."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Google Meet\n", returncode=0,
            )
            # return_name=True → str
            result = watcher._browser_has_meeting_tab(return_name=True)
            self.assertEqual(result, "Google Meet")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Google Meet\n", returncode=0,
            )
            # return_name=False → bool
            result = watcher._browser_has_meeting_tab(return_name=False)
            self.assertIs(result, True)

    def test_browser_has_meeting_tab_empty_returns_false(self):
        """No meeting tab found → False / None."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="\n", returncode=0)
            result = watcher._browser_has_meeting_tab(return_name=False)
            self.assertIs(result, False)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="\n", returncode=0)
            result = watcher._browser_has_meeting_tab(return_name=True)
            self.assertIsNone(result)

    def test_browser_timeout_does_not_raise(self):
        """osascript timeout should be handled gracefully."""
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="osascript", timeout=4)):
            result = watcher._browser_has_meeting_tab(return_name=False)
            self.assertIs(result, False)


class ActiveSessionProcsTest(unittest.TestCase):
    """Verify that _ACTIVE_SESSION_PROCS is narrow and excludes background apps."""

    def test_facetime_not_in_active_session_procs(self):
        """FaceTime persists in background — must NOT be in _ACTIVE_SESSION_PROCS."""
        for proc in watcher._ACTIVE_SESSION_PROCS:
            self.assertNotIn("FaceTime", proc)

    def test_helper_processes_not_in_active_session_procs(self):
        """Helper processes (Slack Helper, etc.) are always running — must not match."""
        for proc in watcher._ACTIVE_SESSION_PROCS:
            self.assertNotIn("Helper", proc)

    def test_cpthost_is_in_active_session_procs(self):
        """CptHost (Zoom meeting capture) should be in the narrow list."""
        self.assertIn("CptHost", watcher._ACTIVE_SESSION_PROCS)


if __name__ == "__main__":
    unittest.main()
