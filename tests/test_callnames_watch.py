"""Tests for callnames.watch — the background AX sampling loop."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from trnscrb.callnames import watch as _watch_mod
from trnscrb.callnames import attribute as _attr


class TestCallNamesWatch(unittest.TestCase):
    """Watch state machine: start, sample collection, stop."""

    def test_start_stop(self):
        """Start creates a thread; stop joins it."""
        w = _watch_mod.CallNamesWatch(bundle_ids=["us.zoom.xos"])
        w.start()
        self.assertTrue(w._thread is not None)
        self.assertTrue(w._thread.is_alive())
        samples = w.stop()
        self.assertIsInstance(samples, list)

    def test_double_start_noop(self):
        """Start is idempotent — only one thread runs."""
        w = _watch_mod.CallNamesWatch(bundle_ids=["us.zoom.xos"])
        w.start()
        thread1 = w._thread
        w.start()
        self.assertIs(w._thread, thread1)
        w.stop()

    def test_noop_with_no_ax(self):
        """Without AX available, sample() returns ok=False → error samples."""
        with patch.object(_watch_mod._ax_mod, "AX_AVAILABLE", False):
            w = _watch_mod.CallNamesWatch(bundle_ids=["us.zoom.xos"])
            w.start()
            time.sleep(0.3)
            samples = w.stop()
            # Should have at least one sample (the error one)
            self.assertGreaterEqual(len(samples), 1)
            self.assertFalse(samples[0].ok)

    @patch.object(_watch_mod._ax_mod, "sample")
    def test_successful_samples(self, mock_sample):
        """When AX succeeds, samples contain participant data."""
        mock_sample.return_value = {
            "ts": time.time(),
            "app": "Zoom",
            "bundle_id": "us.zoom.xos",
            "tree": [
                {
                    "role": "AXTabGroup",
                    "description": "Ann Lee, Computer audio unmuted, Active speaker",
                    "title": "",
                    "value": "",
                    "children": [],
                },
            ],
            "ok": True,
        }
        w = _watch_mod.CallNamesWatch(bundle_ids=["us.zoom.xos"], app_name="Zoom")
        w.start()
        time.sleep(0.6)
        samples = w.stop()
        self.assertGreater(len(samples), 0)
        # At least one sample should be ok and have participants
        ok_samples = [s for s in samples if s.ok]
        self.assertGreater(len(ok_samples), 0)
        self.assertIn("Ann Lee", ok_samples[0].participants)

    def test_stop_returns_samples(self):
        """stop() returns the samples collected up to that point."""
        with patch.object(_watch_mod._ax_mod, "AX_AVAILABLE", False):
            w = _watch_mod.CallNamesWatch(bundle_ids=["us.zoom.xos"])
            w.start()
            time.sleep(0.2)
            samples = w.stop()
            self.assertIsInstance(samples, list)
            self.assertGreater(len(samples), 0)

    @patch.object(_watch_mod._ax_mod, "sample")
    @patch.object(_watch_mod._ax_mod, "pick_bundle")
    def test_re_resolves_bundle_each_tick(self, mock_pick, mock_sample):
        """The frontmost browser can change mid-call — sample the live one."""
        mock_pick.return_value = "com.apple.Safari"
        mock_sample.return_value = {"ts": time.time(), "ok": False, "tree": []}
        w = _watch_mod.CallNamesWatch(
            bundle_ids=["com.google.Chrome", "com.apple.Safari"],
            app_name="Google Meet",
        )
        w.start()
        time.sleep(0.6)
        w.stop()
        self.assertGreaterEqual(mock_pick.call_count, 2)
        for call in mock_sample.call_args_list:
            self.assertEqual(call.args[0] if call.args else call.kwargs.get("bundle_id"),
                             "com.apple.Safari")

    def test_set_offset(self):
        w = _watch_mod.CallNamesWatch(bundle_ids=["us.zoom.xos"])
        self.assertEqual(w._meeting_offset_ms, 0.0)
        w.set_offset(1234.5)
        self.assertEqual(w._meeting_offset_ms, 1234.5)


class TestWatchFor(unittest.TestCase):
    def test_known_app(self):
        w = _watch_mod.watch_for("Zoom")
        self.assertIsNotNone(w)
        self.assertIsInstance(w, _watch_mod.CallNamesWatch)
        self.assertIn("us.zoom.xos", w._bundle_ids)

    def test_slack_huddle(self):
        w = _watch_mod.watch_for("Slack Huddle")
        self.assertIsNotNone(w)
        self.assertEqual(w._bundle_ids, ["com.tinyspeck.slackmacgap"])
        self.assertEqual(w._app_name, "Slack Huddle")

    def test_unknown_app(self):
        w = _watch_mod.watch_for("UnknownApp")
        self.assertIsNone(w)

    def test_none_app(self):
        w = _watch_mod.watch_for(None)
        self.assertIsNone(w)


if __name__ == "__main__":
    unittest.main()