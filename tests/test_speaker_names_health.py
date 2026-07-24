"""Tests for calendar-based speaker naming and capture health reporting."""

import unittest
from datetime import datetime
from unittest.mock import patch

from trnscrb import analytics, attribution, storage


def _seg(start, end, speaker, text="hi"):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


class NameFromCalendarTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(attribution, "_looks_like_self", lambda name: name == "Jonathan")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_one_to_one_names_the_other_attendee(self):
        segments = [_seg(0, 5, "Me"), _seg(5, 10, "Them"), _seg(10, 15, "Them")]
        name = attribution.name_from_calendar(segments, {"attendees": ["Jonathan", "Anna Berg"]})
        self.assertEqual(name, "Anna Berg")
        self.assertEqual([s["speaker"] for s in segments], ["Me", "Anna Berg", "Anna Berg"])

    def test_group_meeting_keeps_generic_label(self):
        """With several attendees we cannot tell voices apart — a wrong name
        is worse than an honest 'Them'."""
        segments = [_seg(0, 5, "Them")]
        name = attribution.name_from_calendar(
            segments, {"attendees": ["Jonathan", "Anna", "Bo", "Cleo"]}
        )
        self.assertIsNone(name)
        self.assertEqual(segments[0]["speaker"], "Them")

    def test_me_segments_are_never_renamed(self):
        segments = [_seg(0, 5, "Me")]
        attribution.name_from_calendar(segments, {"attendees": ["Jonathan", "Anna"]})
        self.assertEqual(segments[0]["speaker"], "Me")

    def test_no_event_or_no_attendees(self):
        segments = [_seg(0, 5, "Them")]
        self.assertIsNone(attribution.name_from_calendar(segments, None))
        self.assertIsNone(attribution.name_from_calendar(segments, {"attendees": []}))
        self.assertEqual(segments[0]["speaker"], "Them")

    def test_solo_event_with_only_self(self):
        segments = [_seg(0, 5, "Them")]
        self.assertIsNone(attribution.name_from_calendar(segments, {"attendees": ["Jonathan"]}))


class SelfDetectionTest(unittest.TestCase):
    def test_matches_the_current_user(self):
        with patch.dict("os.environ", {"USER": "jonathan"}):
            self.assertTrue(attribution._looks_like_self("Jonathan"))
            self.assertTrue(attribution._looks_like_self("Jonathan Artback"))
            self.assertFalse(attribution._looks_like_self("Anna Berg"))

    def test_honours_explicit_override(self):
        with patch.dict("os.environ", {"TRNSCRB_USER_NAME": "Jo Svensson"}):
            self.assertTrue(attribution._looks_like_self("Jo Svensson"))


class CaptureHealthTest(unittest.TestCase):
    def test_reports_speech_ratio_and_source(self):
        health = analytics.capture_health(
            [_seg(0, 30, "Me"), _seg(30, 90, "Them")], recorded_secs=120, system_audio=True
        )
        self.assertAlmostEqual(health["speech"], 90.0)
        self.assertAlmostEqual(health["speech_ratio"], 0.75)
        self.assertTrue(health["system_audio"])
        self.assertFalse(health["mostly_silent"])

    def test_flags_mostly_silent_long_recording(self):
        """The 3.6-hour ghost recording case: a tab left open after the call."""
        health = analytics.capture_health(
            [_seg(0, 300, "Me")], recorded_secs=3600 * 3, system_audio=True
        )
        self.assertTrue(health["mostly_silent"])
        self.assertIn("Mostly silence", analytics.format_capture_health(health))

    def test_short_quiet_recording_is_not_flagged(self):
        health = analytics.capture_health([_seg(0, 5, "Me")], recorded_secs=60, system_audio=True)
        self.assertFalse(health["mostly_silent"])

    def test_warns_when_system_audio_missing(self):
        text = analytics.format_capture_health(
            analytics.capture_health([_seg(0, 60, "Me")], recorded_secs=120, system_audio=False)
        )
        self.assertIn("microphone only", text)
        self.assertIn("Screen Recording", text)

    def test_system_audio_present_has_no_warning(self):
        text = analytics.format_capture_health(
            analytics.capture_health([_seg(0, 60, "Me")], recorded_secs=120, system_audio=True)
        )
        self.assertIn("mic + system audio", text)
        self.assertNotIn("⚠︎", text)

    def test_empty_recording_renders_nothing(self):
        self.assertEqual(analytics.format_capture_health({}), "")
        self.assertEqual(analytics.format_capture_health(analytics.capture_health([], 0, True)), "")


class TranscriptHealthRenderingTest(unittest.TestCase):
    def test_health_line_appears_in_transcript(self):
        health = analytics.capture_health(
            [_seg(0, 60, "Me")], recorded_secs=120, system_audio=False
        )
        text = storage.format_transcript(
            [_seg(0, 60, "Me", "hello")], datetime(2026, 7, 22, 9, 0), "Standup", health=health
        )
        self.assertIn("Captured:", text)
        self.assertIn("microphone only", text)

    def test_transcript_without_health_is_unchanged(self):
        text = storage.format_transcript(
            [_seg(0, 60, "Me", "hello")], datetime(2026, 7, 22, 9, 0), "Standup"
        )
        self.assertNotIn("Captured:", text)


if __name__ == "__main__":
    unittest.main()
