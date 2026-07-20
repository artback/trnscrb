"""Tests for MicWatcher crash recovery and detect_meeting priority logic.

All tests mock subprocess, osascript, and CoreAudio — no real devices or
network access required.
"""

import unittest
from datetime import datetime
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

        with (
            patch.object(w, "_loop_inner", side_effect=RuntimeError("crash")),
            patch.object(watcher.time, "sleep") as mock_sleep,
            patch("threading.Thread", return_value=new_thread) as mock_thread_cls,
        ):
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

        with (
            patch.object(w, "_loop_inner", side_effect=crashing_inner),
            patch.object(watcher.time, "sleep"),
        ):
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

        with (
            patch.object(w, "_loop_inner", side_effect=crashing_inner),
            patch.object(watcher.time, "sleep") as mock_sleep,
        ):
            w._loop()

        mock_sleep.assert_called_once_with(5)


class DetectMeetingPriorityTest(unittest.TestCase):
    """Test that detect_meeting checks browser tabs before native apps."""

    def test_browser_tab_checked_before_native_apps(self):
        """If browser has a meeting tab, native app check should be skipped."""
        with (
            patch.object(
                watcher, "_browser_has_meeting_tab", return_value="Google Meet"
            ) as mock_browser,
            patch.object(watcher, "_pids_using_mic_input") as mock_pids,
            patch.object(watcher, "_meeting_app_pids"),
        ):
            result = detect_meeting()

        self.assertEqual(result, "Google Meet")
        mock_browser.assert_called_once_with(return_name=True)
        # Native app PID checks should not have been reached
        mock_pids.assert_not_called()

    def test_native_app_used_when_no_browser_tab(self):
        """If no browser tab, should fall through to native app detection."""
        with (
            patch.object(watcher, "_browser_has_meeting_tab", return_value=None),
            patch.object(watcher, "_pids_using_mic_input", return_value={42}),
            patch.object(watcher, "_meeting_app_pids", return_value={42}),
            patch("subprocess.run") as mock_run,
        ):
            # ps output: PID 42 is zoom.us
            mock_run.return_value = MagicMock(
                stdout="   42 /Applications/zoom.us.app/Contents/Frameworks/zoom.us",
                returncode=0,
            )
            result = detect_meeting()

        self.assertEqual(result, "Zoom")

    def test_session_only_fallback_when_no_pid_match(self):
        """Fallback to session-only process list (CptHost for Zoom)."""
        with (
            patch.object(watcher, "_browser_has_meeting_tab", return_value=None),
            patch.object(watcher, "_pids_using_mic_input", return_value=set()),
            patch.object(watcher, "_meeting_app_pids", return_value=set()),
            patch("subprocess.run") as mock_run,
        ):
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
        with (
            patch.object(watcher, "_browser_has_meeting_tab", return_value=None),
            patch.object(watcher, "_pids_using_mic_input", return_value=set()),
            patch.object(watcher, "_meeting_app_pids", return_value=set()),
            patch("subprocess.run") as mock_run,
            patch.dict("sys.modules", {"trnscrb.calendar_integration": MagicMock()}),
        ):
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
        self.assertTrue(
            result.startswith("meeting-"), f"Expected timestamp fallback, got: {result}"
        )

    def test_facetime_matched_only_when_using_mic(self):
        """FaceTime should be detected via native app path only when its PID
        is actively using the mic (cross-referenced with CoreAudio)."""
        with (
            patch.object(watcher, "_browser_has_meeting_tab", return_value=None),
            patch.object(watcher, "_pids_using_mic_input", return_value={99}),
            patch.object(watcher, "_meeting_app_pids", return_value={99}),
            patch("subprocess.run") as mock_run,
        ):
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

    def test_firefox_script_matches_meeting_window_titles(self):
        """Firefox is checked via window titles (no AppleScript tab API)."""
        self.assertIn('"Meet -"', watcher._FIREFOX_WINDOW_SCRIPT)
        self.assertIn("Microsoft Teams", watcher._FIREFOX_WINDOW_SCRIPT)
        self.assertIn("Zoom Meeting", watcher._FIREFOX_WINDOW_SCRIPT)
        self.assertIn('does not contain "ended"', watcher._FIREFOX_WINDOW_SCRIPT)

    def test_browser_check_includes_firefox(self):
        """All three browser scripts must be queried by _browser_has_meeting_tab."""
        queried = []

        def fake_run(label, script):
            queried.append(label)
            return None

        with patch.object(watcher, "_run_osascript", side_effect=fake_run):
            watcher._browser_has_meeting_tab()
        self.assertEqual(sorted(queried), ["Chrome", "Firefox", "Safari"])

    def test_browser_has_meeting_tab_returns_name_or_bool(self):
        """_browser_has_meeting_tab returns str when return_name=True, bool otherwise."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Google Meet\n",
                returncode=0,
            )
            # return_name=True → str
            result = watcher._browser_has_meeting_tab(return_name=True)
            self.assertEqual(result, "Google Meet")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Google Meet\n",
                returncode=0,
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


class WatcherStopDuringTransitionsTest(unittest.TestCase):
    """Test that calling stop() during warming/recording/cooling states
    causes the loop to exit cleanly."""

    def _make_watcher(self):
        return MicWatcher(on_start=MagicMock(), on_stop=MagicMock())

    def test_stop_during_warming_exits_loop(self):
        """Setting _running=False while in 'warming' state should exit the loop."""
        w = self._make_watcher()
        w._running = True
        w._state = "idle"
        call_count = [0]

        def fake_mic_in_use():
            call_count[0] += 1
            # Always return True so we enter warming and stay there
            if call_count[0] >= 3:
                w._running = False  # simulate stop() after a few iterations
            return True

        with (
            patch("trnscrb.watcher.is_mic_in_use", side_effect=fake_mic_in_use),
            patch("trnscrb.watcher.time.sleep"),
        ):
            w._loop_inner()

        # Loop exited cleanly — state should be warming (never reached recording
        # because WARMUP_SECS hasn't elapsed with mocked datetime)
        self.assertFalse(w._running)

    def test_stop_during_recording_exits_loop(self):
        """Setting _running=False while in 'recording' should exit."""
        w = self._make_watcher()
        w._running = True
        w._state = "recording"
        w._since = datetime.now()
        w._rec_started = datetime.now()
        call_count = [0]

        def fake_mic_in_use():
            call_count[0] += 1
            if call_count[0] >= 2:
                w._running = False
            return True

        with (
            patch("trnscrb.watcher.is_mic_in_use", side_effect=fake_mic_in_use),
            patch("trnscrb.watcher.time.sleep"),
            patch("trnscrb.watcher.is_meeting_app_running", return_value=True),
        ):
            w._loop_inner()

        self.assertFalse(w._running)
        # on_stop should NOT have been called — we stopped mid-recording
        w.on_stop.assert_not_called()

    def test_stop_during_cooling_exits_loop(self):
        """Setting _running=False while in 'cooling' should exit."""
        w = self._make_watcher()
        w._running = True
        w._state = "cooling"
        w._since = datetime.now()
        w._rec_started = datetime.now()
        call_count = [0]

        def fake_mic_in_use():
            call_count[0] += 1
            if call_count[0] >= 2:
                w._running = False
            return False

        with (
            patch("trnscrb.watcher.is_mic_in_use", side_effect=fake_mic_in_use),
            patch("trnscrb.watcher.time.sleep"),
            patch("trnscrb.watcher.is_meeting_app_running", return_value=False),
        ):
            w._loop_inner()

        self.assertFalse(w._running)


class MinSaveSecsTest(unittest.TestCase):
    """Test that recordings shorter than MIN_SAVE_SECS (30s) don't fire on_stop."""

    def test_short_recording_does_not_fire_on_stop(self):
        """A recording lasting < 30s should be discarded (on_stop not called)."""
        w = MicWatcher(on_start=MagicMock(), on_stop=MagicMock())
        w._running = True
        w._state = "cooling"
        # Recording started 10 seconds ago — shorter than MIN_SAVE_SECS
        now = datetime(2026, 3, 29, 12, 0, 30)
        w._rec_started = datetime(2026, 3, 29, 12, 0, 20)  # 10s duration
        w._since = datetime(2026, 3, 29, 12, 0, 20)  # cooling started 10s ago (> GRACE_SECS)

        call_count = [0]

        def fake_mic_in_use():
            call_count[0] += 1
            if call_count[0] >= 2:
                w._running = False
            return False

        with (
            patch("trnscrb.watcher.is_mic_in_use", side_effect=fake_mic_in_use),
            patch("trnscrb.watcher.time.sleep"),
            patch("trnscrb.watcher.is_meeting_app_running", return_value=False),
            patch("trnscrb.watcher.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            w._loop_inner()

        # on_stop should NOT have been called because duration < MIN_SAVE_SECS
        w.on_stop.assert_not_called()

    def test_long_recording_fires_on_stop(self):
        """A recording lasting >= 30s should fire on_stop."""
        w = MicWatcher(on_start=MagicMock(), on_stop=MagicMock())
        w._running = True
        w._state = "cooling"
        now = datetime(2026, 3, 29, 12, 5, 0)
        w._rec_started = datetime(2026, 3, 29, 12, 0, 0)  # 5 minutes = 300s
        w._since = datetime(2026, 3, 29, 12, 4, 50)  # cooling started 10s ago (> GRACE_SECS)

        call_count = [0]

        def fake_mic_in_use():
            call_count[0] += 1
            if call_count[0] >= 2:
                w._running = False
            return False

        with (
            patch("trnscrb.watcher.is_mic_in_use", side_effect=fake_mic_in_use),
            patch("trnscrb.watcher.time.sleep"),
            patch("trnscrb.watcher.is_meeting_app_running", return_value=False),
            patch("trnscrb.watcher.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            w._loop_inner()

        w.on_stop.assert_called_once()


class SourceAwareBrowserCheckTest(unittest.TestCase):
    """Browser AppleScript sweeps run only for browsers that are running."""

    def test_running_browsers_filters_by_ps_output(self):
        out = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n/usr/libexec/foo\n"
        labels = [label for label, _ in watcher._running_browsers(out)]
        self.assertEqual(labels, ["Chrome"])

    def test_running_browsers_unknown_ps_checks_all(self):
        labels = [label for label, _ in watcher._running_browsers("")]
        self.assertEqual(labels, ["Chrome", "Safari", "Firefox"])

    def test_empty_browser_list_short_circuits(self):
        self.assertIs(watcher._browser_has_meeting_tab(browsers=[]), False)
        self.assertIsNone(watcher._browser_has_meeting_tab(return_name=True, browsers=[]))

    def test_no_browsers_no_teams_spawns_no_osascript(self):
        """Zoom-only machine: meeting check must not spawn any osascript."""
        ps = MagicMock(stdout="/usr/sbin/somedaemon\n/Applications/Zoom.us.app/nothing\n")
        with (
            patch.object(watcher, "_pids_using_mic_input", return_value=set()),
            patch.object(watcher.subprocess, "run", return_value=ps),
            patch.object(watcher, "_teams_call_active") as teams,
            patch.object(watcher, "_run_osascript") as osascript,
        ):
            result = watcher.is_meeting_app_running()

        self.assertFalse(result)
        teams.assert_not_called()
        osascript.assert_not_called()

    def test_only_running_browser_is_queried(self):
        ps = MagicMock(stdout="/Applications/Safari.app/Contents/MacOS/Safari\n")
        queried = []
        with (
            patch.object(watcher, "_pids_using_mic_input", return_value=set()),
            patch.object(watcher.subprocess, "run", return_value=ps),
            patch.object(watcher, "_teams_call_active", return_value=False),
            patch.object(
                watcher, "_run_osascript", side_effect=lambda label, s: queried.append(label)
            ),
        ):
            watcher.is_meeting_app_running()

        self.assertEqual(queried, ["Safari"])


class CoreAudioConstantsTest(unittest.TestCase):
    """The process-object fourccs were silently wrong for months — pin them.

    Values verified against AudioHardware.h (macOS 14+):
    'prs#' ProcessObjectList, 'piri' IsRunningInput, 'piro' IsRunningOutput.
    """

    def test_fourccs_match_audiohardware_header(self):
        self.assertEqual(watcher._kProcessObjectList, 0x70727323)  # 'prs#'
        self.assertEqual(watcher._kProcessIsRunningIn, 0x70697269)  # 'piri'
        self.assertEqual(watcher._kProcessIsRunningOut, 0x7069726F)  # 'piro'
        self.assertEqual(watcher._kDefaultOutputDevice, 0x644F7574)  # 'dOut'


class MutedCallDetectionTest(unittest.TestCase):
    """A live call with the mic muted must still count as call activity."""

    def _watcher(self):
        return MicWatcher(on_start=lambda name: None, on_stop=lambda: None)

    def test_mic_active_short_circuits(self):
        w = self._watcher()
        with (
            patch.object(watcher, "is_mic_in_use", return_value=True),
            patch.object(watcher, "meeting_audio_output_active") as output,
        ):
            self.assertTrue(w._call_activity())
        output.assert_not_called()

    def test_no_output_means_inactive_without_app_check(self):
        w = self._watcher()
        with (
            patch.object(watcher, "is_mic_in_use", return_value=False),
            patch.object(watcher, "meeting_audio_output_active", return_value=False),
            patch.object(watcher, "is_meeting_app_running") as app_check,
        ):
            self.assertFalse(w._call_activity())
        app_check.assert_not_called()

    def test_meeting_output_with_meeting_app_is_active(self):
        w = self._watcher()
        with (
            patch.object(watcher, "is_mic_in_use", return_value=False),
            patch.object(watcher, "meeting_audio_output_active", return_value=True),
            patch.object(watcher, "is_meeting_app_running", return_value=True) as app_check,
        ):
            self.assertTrue(w._call_activity())
            self.assertTrue(w._call_activity())  # cached — no second osascript
        self.assertEqual(app_check.call_count, 1)

    def test_meeting_output_without_meeting_app_is_inactive(self):
        w = self._watcher()
        with (
            patch.object(watcher, "is_mic_in_use", return_value=False),
            patch.object(watcher, "meeting_audio_output_active", return_value=True),
            patch.object(watcher, "is_meeting_app_running", return_value=False),
        ):
            self.assertFalse(w._call_activity())

    def test_meeting_output_matches_browser_process(self):
        ps_out = MagicMock(
            stdout="500 /Applications/Google Chrome.app/Contents/Frameworks/Helper\n"
            "600 /usr/libexec/somedaemon\n"
        )
        with (
            patch.object(watcher, "_pids_producing_output", return_value={500}),
            patch.object(watcher.subprocess, "run", return_value=ps_out),
        ):
            self.assertTrue(watcher.meeting_audio_output_active())

    def test_non_meeting_output_does_not_match(self):
        ps_out = MagicMock(stdout="700 /Applications/Spotify.app/Contents/MacOS/Spotify\n")
        with (
            patch.object(watcher, "_pids_producing_output", return_value={700}),
            patch.object(watcher.subprocess, "run", return_value=ps_out),
        ):
            self.assertFalse(watcher.meeting_audio_output_active())

    def test_no_output_pids_skips_ps(self):
        with (
            patch.object(watcher, "_pids_producing_output", return_value=set()),
            patch.object(watcher.subprocess, "run") as ps,
        ):
            self.assertFalse(watcher.meeting_audio_output_active())
        ps.assert_not_called()


class MutedCallStateMachineTest(unittest.TestCase):
    def test_muted_call_starts_recording(self):
        """Meeting audio playing + meeting tab present + mic off → record."""
        import time as _time

        started = []
        w = MicWatcher(on_start=lambda name: started.append(name), on_stop=lambda: None)
        with (
            patch.object(watcher, "is_mic_in_use", return_value=False),
            patch.object(watcher, "meeting_audio_output_active", return_value=True),
            patch.object(watcher, "is_meeting_app_running", return_value=True),
            patch.object(watcher, "detect_meeting", return_value="Google Meet"),
            patch.object(watcher._MicActivityListener, "start", return_value=False),
            patch.object(watcher, "POLL_SECS", 0.01),
            patch.object(watcher, "WARMUP_SECS", 0.05),
            patch.object(watcher, "OUTPUT_APP_CHECK_SECS", 0.0),
        ):
            w.start()
            _time.sleep(0.5)
            state = w.state
            w.stop()

        self.assertEqual(state, "recording")
        self.assertEqual(started, ["Google Meet"])


class MicActivityListenerTest(unittest.TestCase):
    """Event-driven idle watching with polling fallback."""

    def test_listener_lifecycle_does_not_crash(self):
        import threading

        listener = watcher._MicActivityListener(threading.Event())
        started = listener.start()
        self.assertIsInstance(started, bool)
        listener.refresh()
        listener.stop()

    def test_watcher_falls_back_to_polling_when_listener_fails(self):
        with patch.object(watcher._MicActivityListener, "start", return_value=False):
            w = MicWatcher(on_start=lambda name: None, on_stop=lambda: None)
            w.start()
            try:
                self.assertFalse(w._event_driven)
            finally:
                w.stop()

    def test_watcher_uses_events_when_listener_starts(self):
        with patch.object(watcher._MicActivityListener, "start", return_value=True):
            w = MicWatcher(on_start=lambda name: None, on_stop=lambda: None)
            w.start()
            try:
                self.assertTrue(w._event_driven)
            finally:
                w.stop()


if __name__ == "__main__":
    unittest.main()
