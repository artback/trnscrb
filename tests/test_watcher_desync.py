"""Auto-record must survive a stop that the watcher did not initiate.

Observed in production: a meeting was recorded and stopped from the app, and
every later meeting was detected (the log showed the Meet tab found every few
seconds) but never recorded — the watcher was still in `recording`, and
on_start only fires on the warming → recording transition.
"""

import time
import unittest
from unittest.mock import patch

from trnscrb import watcher
from trnscrb.watcher import MicWatcher


class WatcherResetTest(unittest.TestCase):
    def _run_watcher(self, script):
        """Drive the watcher through a scripted sequence of activity states.

        ``script`` is a list of (call_active, action) where action may be
        "stop_externally" to simulate the app stopping the recording.
        """
        started, stopped = [], []
        w = MicWatcher(
            on_start=lambda name: started.append(name), on_stop=lambda: stopped.append(1)
        )
        state = {"active": False}

        with (
            patch.object(watcher, "is_mic_in_use", side_effect=lambda: state["active"]),
            patch.object(watcher, "meeting_audio_output_active", return_value=False),
            patch.object(watcher, "is_meeting_app_running", return_value=True),
            patch.object(watcher, "detect_meeting", return_value="Google Meet"),
            patch.object(watcher._MicActivityListener, "start", return_value=False),
            patch.object(watcher, "POLL_SECS", 0.01),
            patch.object(watcher, "WARMUP_SECS", 0.05),
            patch.object(watcher, "GRACE_SECS", 0.05),
            patch.object(watcher, "MIN_SAVE_SECS", 0),
            patch.object(watcher, "APP_POLL_EVERY", 1),
        ):
            w.start()
            try:
                for active, action in script:
                    state["active"] = active
                    time.sleep(0.15)
                    if action == "stop_externally":
                        w.notify_recording_stopped()
                        time.sleep(0.1)
                time.sleep(0.1)
                return started, stopped, w.state
            finally:
                w.stop()

    def test_second_meeting_records_after_a_manual_stop(self):
        """The production failure: meeting 1 recorded, stopped from the app,
        meeting 2 detected but never recorded."""
        started, _stopped, _state = self._run_watcher(
            [
                (True, None),  # meeting 1 starts
                (True, "stop_externally"),  # user hits Stop in the menu bar
                (False, None),  # brief gap
                (True, None),  # meeting 2 starts
            ]
        )
        self.assertEqual(len(started), 2, f"second meeting never auto-started: {started}")

    def test_notify_returns_watcher_to_idle(self):
        _started, _stopped, state = self._run_watcher([(True, "stop_externally")])
        self.assertEqual(state, "idle")

    def test_notify_is_safe_before_any_recording(self):
        w = MicWatcher(on_start=lambda name: None, on_stop=lambda: None)
        w.notify_recording_stopped()  # must not raise when never started
        self.assertEqual(w.state, "idle")

    def test_stop_mid_call_is_not_undone_by_auto_record(self):
        """Hitting Stop during a meeting is deliberate — auto-record must not
        restart it while that same call is still going."""
        started, _stopped, _state = self._run_watcher(
            [
                (True, None),  # meeting starts, recording
                (True, "stop_externally"),  # user stops it on purpose
                (True, None),  # …call continues
                (True, None),
            ]
        )
        self.assertEqual(len(started), 1, "auto-record overrode a deliberate stop")

    def test_notify_does_not_fire_on_stop_callback(self):
        """The app already knows it stopped; a duplicate on_stop would try to
        transcribe a recording that is already being processed."""
        _started, stopped, _state = self._run_watcher([(True, "stop_externally")])
        self.assertEqual(stopped, [], "reset must not trigger the stop callback")


if __name__ == "__main__":
    unittest.main()
