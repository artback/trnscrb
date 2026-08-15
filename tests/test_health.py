"""Tests for the persistent component-health record."""

import tempfile
import unittest
from pathlib import Path

from trnscrb import health


class HealthStoreTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp()) / "health.json"
        self._real_store = health.STORE
        health.STORE = tmp
        self.addCleanup(setattr, health, "STORE", self._real_store)

    def test_nothing_recorded_yet(self):
        self.assertIsNone(health.get(health.DIARIZATION))
        self.assertEqual(health.unhealthy(), [])
        self.assertEqual(health.describe(health.DIARIZATION), "never run")

    def test_success_is_remembered(self):
        health.record_ok(health.DIARIZATION, "3 speaker(s)", "Standup")
        entry = health.get(health.DIARIZATION)
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["detail"], "3 speaker(s)")
        self.assertEqual(entry["failures"], 0)
        self.assertEqual(health.unhealthy(), [])

    def test_failures_accumulate_into_a_streak(self):
        for _ in range(3):
            entry = health.record_failure(health.DIARIZATION, RuntimeError("torchcodec"))
        self.assertEqual(entry["failures"], 3)
        self.assertTrue(entry["failing_since"])
        self.assertEqual([name for name, _ in health.unhealthy()], [health.DIARIZATION])
        self.assertIn("3 meetings", health.describe(health.DIARIZATION))

    def test_a_success_clears_the_streak(self):
        health.record_failure(health.DIARIZATION, "boom")
        first_failure = health.get(health.DIARIZATION)["failing_since"]
        self.assertTrue(first_failure)
        health.record_ok(health.DIARIZATION, "2 speaker(s)")
        entry = health.get(health.DIARIZATION)
        self.assertEqual(entry["failures"], 0)
        self.assertEqual(entry["failing_since"], "")
        self.assertTrue(entry["last_ok_at"])

    def test_failing_since_survives_later_failures(self):
        """When it broke matters more than when it last broke."""
        health.record_failure(health.DIARIZATION, "boom")
        since = health.get(health.DIARIZATION)["failing_since"]
        health.record_failure(health.DIARIZATION, "boom again")
        self.assertEqual(health.get(health.DIARIZATION)["failing_since"], since)

    def test_components_are_independent(self):
        health.record_ok(health.TRANSCRIPTION, "42 segments")
        health.record_failure(health.DIARIZATION, "boom")
        self.assertEqual([name for name, _ in health.unhealthy()], [health.DIARIZATION])

    def test_long_errors_are_truncated(self):
        health.record_failure(health.DIARIZATION, "x" * 5000)
        self.assertLessEqual(len(health.get(health.DIARIZATION)["detail"]), 300)

    def test_notifies_on_the_first_failure_then_every_fifth(self):
        notified = [
            health.should_notify(health.record_failure(health.DIARIZATION, "boom"))
            for _ in range(11)
        ]
        self.assertEqual([i + 1 for i, n in enumerate(notified) if n], [1, 5, 10])

    def test_an_unreadable_store_is_not_fatal(self):
        health.STORE.parent.mkdir(parents=True, exist_ok=True)
        health.STORE.write_text("{ not json", encoding="utf-8")
        self.assertIsNone(health.get(health.DIARIZATION))
        health.record_ok(health.DIARIZATION, "recovered")
        self.assertTrue(health.get(health.DIARIZATION)["ok"])

    def test_a_store_from_the_future_is_ignored(self):
        health.STORE.parent.mkdir(parents=True, exist_ok=True)
        health.STORE.write_text('{"version": 99, "components": {}}', encoding="utf-8")
        self.assertEqual(health.load()["version"], health._VERSION)

    def test_an_unwritable_store_never_raises(self):
        """Reporting health must not be the thing that breaks the meeting."""
        health.STORE = Path("/nonexistent-root/health.json")
        health.record_ok(health.DIARIZATION, "ok")
        # The streak still drives the notification even when nothing persists.
        self.assertEqual(health.record_failure(health.DIARIZATION, "boom")["failures"], 1)
        self.assertIsNone(health.get(health.DIARIZATION))

    def test_clear_forgets_one_component(self):
        health.record_failure(health.DIARIZATION, "boom")
        health.record_ok(health.TRANSCRIPTION, "ok")
        health.clear(health.DIARIZATION)
        self.assertIsNone(health.get(health.DIARIZATION))
        self.assertIsNotNone(health.get(health.TRANSCRIPTION))


if __name__ == "__main__":
    unittest.main()
