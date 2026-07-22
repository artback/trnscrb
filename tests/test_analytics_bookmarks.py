"""Tests for talk-time statistics and meeting bookmarks."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from trnscrb import analytics, storage


def _seg(start, end, speaker, text="hello"):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


class TalkTimeTest(unittest.TestCase):
    def test_splits_time_between_speakers(self):
        stats = analytics.talk_time([_seg(0, 10, "Me"), _seg(10, 30, "Them"), _seg(30, 40, "Me")])
        self.assertAlmostEqual(stats["totals"]["Me"], 20.0)
        self.assertAlmostEqual(stats["totals"]["Them"], 20.0)
        self.assertAlmostEqual(stats["total"], 40.0)

    def test_counts_turns_not_segments(self):
        stats = analytics.talk_time(
            [_seg(0, 2, "Me"), _seg(2, 4, "Me"), _seg(4, 6, "Them"), _seg(6, 8, "Me")]
        )
        self.assertEqual(stats["turns"], 3)

    def test_longest_monologue_merges_short_pauses(self):
        stats = analytics.talk_time([_seg(0, 20, "Them"), _seg(21, 40, "Them"), _seg(60, 65, "Me")])
        self.assertAlmostEqual(stats["longest_monologue"], 40.0)
        self.assertEqual(stats["longest_monologue_speaker"], "Them")

    def test_long_gap_breaks_the_monologue(self):
        stats = analytics.talk_time([_seg(0, 20, "Them"), _seg(120, 130, "Them")])
        self.assertAlmostEqual(stats["longest_monologue"], 20.0)

    def test_empty_and_malformed_segments(self):
        self.assertEqual(analytics.talk_time([]), {})
        self.assertEqual(analytics.talk_time([{"text": "no timings"}]), {})
        self.assertEqual(analytics.talk_time([_seg(5, 5, "Me")]), {}, "zero-length only")

    def test_format_reports_shares(self):
        text = analytics.format_talk_time(
            analytics.talk_time([_seg(0, 25, "Me"), _seg(25, 100, "Them")])
        )
        self.assertIn("Me", text)
        self.assertIn("25%", text)
        self.assertIn("75%", text)

    def test_format_skipped_without_attribution(self):
        """Unlabelled transcripts must not render a meaningless summary."""
        stats = analytics.talk_time([_seg(0, 30, "Unknown")])
        self.assertEqual(analytics.format_talk_time(stats), "")


class BookmarkStorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        for name, attr in (("live.json", "_LIVE_SESSION_FILE"), ("marks.json", "_BOOKMARKS_FILE")):
            patcher = patch.object(storage, attr, root / name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.notes = root / "notes"
        self.notes.mkdir()

    def _start_session(self, minutes_ago=2):
        started = datetime.now() - timedelta(minutes=minutes_ago)
        live = self.notes / "meeting.txt"
        live.write_text("x")
        storage.set_live_session(live, "Standup", started)
        return started

    def test_bookmark_records_offset_from_start(self):
        self._start_session(minutes_ago=2)
        offset = storage.add_bookmark("decision")
        self.assertIsNotNone(offset)
        self.assertGreater(offset, 110)
        marks = storage.read_bookmarks()
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["label"], "decision")

    def test_bookmark_without_recording_returns_none(self):
        self.assertIsNone(storage.add_bookmark("nope"))
        self.assertEqual(storage.read_bookmarks(), [])

    def test_multiple_bookmarks_accumulate(self):
        self._start_session()
        storage.add_bookmark("one")
        storage.add_bookmark("two")
        self.assertEqual([m["label"] for m in storage.read_bookmarks()], ["one", "two"])

    def test_clearing_the_session_clears_bookmarks(self):
        self._start_session()
        storage.add_bookmark("gone")
        storage.clear_live_session()
        self.assertEqual(storage.read_bookmarks(), [])


class TranscriptRenderingTest(unittest.TestCase):
    def test_talk_time_and_bookmarks_appear_in_transcript(self):
        text = storage.format_transcript(
            [_seg(0, 10, "Me", "hi"), _seg(10, 30, "Them", "hello")],
            datetime(2026, 7, 22, 9, 0),
            "Standup",
            bookmarks=[{"at": 12.0, "label": "the decision"}],
        )
        self.assertIn("Talk time:", text)
        self.assertIn("⭐", text)
        self.assertIn("the decision", text)

    def test_bookmark_is_placed_in_context(self):
        text = storage.format_transcript(
            [_seg(0, 10, "Me", "before"), _seg(20, 30, "Me", "after")],
            datetime(2026, 7, 22, 9, 0),
            "Standup",
            bookmarks=[{"at": 15.0, "label": "mark"}],
        )
        body = text.split("=" * 60)[1]
        self.assertLess(body.index("before"), body.index("mark"))
        self.assertLess(body.index("mark"), body.index("after"))

    def test_transcript_without_bookmarks_is_unchanged_in_shape(self):
        text = storage.format_transcript(
            [_seg(0, 10, "Me", "hi")], datetime(2026, 7, 22, 9, 0), "Standup"
        )
        self.assertNotIn("⭐", text)
        self.assertIn("Meeting: Standup", text)


if __name__ == "__main__":
    unittest.main()
