"""Tests for the restart-loop guard.

launchd restarts a failing job every 10 seconds with no backoff, so an app
that dies during startup loops until somebody notices — one such loop ran
~6,900 times over four days. The app itself is the only party that can tell
"started" from "started for the fifth time in two minutes".
"""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from trnscrb import health, icon


class NoteStartTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, health, "STORE", health.STORE)
        health.STORE = Path(tmp.name) / "health.json"

    def test_a_single_start_is_one(self):
        self.assertEqual(health.note_start(), 1)

    def test_starts_accumulate(self):
        self.assertEqual([health.note_start() for _ in range(4)], [1, 2, 3, 4])

    def test_old_starts_age_out_of_the_window(self):
        """A daily launch is not a loop, however many days it runs."""
        stale = (datetime.now() - timedelta(seconds=health.CRASH_LOOP_WINDOW_SECS + 60)).isoformat()
        data = health.load()
        data["starts"] = [stale] * 4
        health._save(data)
        self.assertEqual(health.note_start(), 1)

    def test_the_loop_threshold_is_reached(self):
        counts = [health.note_start() for _ in range(health.CRASH_LOOP_STARTS)]
        self.assertEqual(counts[-1], health.CRASH_LOOP_STARTS)

    def test_clearing_lets_the_next_launch_run(self):
        for _ in range(health.CRASH_LOOP_STARTS):
            health.note_start()
        health.clear_starts()
        self.assertEqual(health.note_start(), 1)

    def test_unparsable_timestamps_are_ignored(self):
        data = health.load()
        data["starts"] = ["not-a-date", "also-not"]
        health._save(data)
        self.assertEqual(health.note_start(), 1)

    def test_a_broken_store_never_blocks_startup(self):
        health.STORE = Path("/nonexistent-root/health.json")
        self.assertEqual(health.note_start(), 1)


class StartupGuardTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, health, "STORE", health.STORE)
        health.STORE = Path(tmp.name) / "health.json"

    def _guard(self):
        from trnscrb.menu_bar import _startup_is_sane

        with mock.patch("trnscrb.menu_bar._notify"):
            return _startup_is_sane()

    def test_a_normal_start_proceeds(self):
        self.assertTrue(self._guard())

    def test_the_fifth_start_in_the_window_stops(self):
        results = [self._guard() for _ in range(health.CRASH_LOOP_STARTS)]
        self.assertEqual(results, [True] * (health.CRASH_LOOP_STARTS - 1) + [False])

    def test_the_loop_is_recorded_for_status_to_report(self):
        for _ in range(health.CRASH_LOOP_STARTS):
            self._guard()
        entry = health.get(health.APP_START)
        self.assertFalse(entry["ok"])
        self.assertIn("starts in under", entry["detail"])

    def test_the_user_can_start_it_again_by_hand(self):
        """The guard stops the loop; it must not lock the user out."""
        for _ in range(health.CRASH_LOOP_STARTS):
            self._guard()
        self.assertTrue(self._guard())

    def test_the_user_is_told(self):
        from trnscrb.menu_bar import _startup_is_sane

        for _ in range(health.CRASH_LOOP_STARTS - 1):
            self._guard()
        with mock.patch("trnscrb.menu_bar._notify") as notify:
            _startup_is_sane()
        notify.assert_called_once()
        self.assertIn("restart loop", notify.call_args[0][1].lower())


class IconIntegrityTest(unittest.TestCase):
    """A damaged icon must not be handed to ImageIO — it SIGBUSes the process."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for name, attr in (("mic.png", "ICON_IDLE"), ("mic_active.png", "ICON_RECORDING")):
            patcher = mock.patch.object(icon, attr, self.dir / name)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(icon, "ICON_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_generated_icon_is_used(self):
        icon.generate_icons()
        self.assertEqual(icon.icon_path(), str(icon.ICON_IDLE))

    def test_a_missing_icon_falls_back_to_the_emoji(self):
        self.assertIsNone(icon.icon_path())

    def test_a_truncated_icon_is_regenerated(self):
        icon.generate_icons()
        data = icon.ICON_IDLE.read_bytes()
        icon.ICON_IDLE.write_bytes(data[: len(data) // 2])
        self.assertEqual(icon.icon_path(), str(icon.ICON_IDLE))
        self.assertTrue(icon._is_readable_png(icon.ICON_IDLE))

    def test_an_unrepairable_icon_falls_back_instead_of_crashing(self):
        icon.ICON_IDLE.write_bytes(b"not a png at all")
        with mock.patch.object(icon, "generate_icons", side_effect=OSError("read-only")):
            self.assertIsNone(icon.icon_path())

    def test_garbage_is_not_a_readable_png(self):
        icon.ICON_IDLE.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
        self.assertFalse(icon._is_readable_png(icon.ICON_IDLE))


if __name__ == "__main__":
    unittest.main()
