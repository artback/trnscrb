"""Tests for retrying preserved audio that never got transcribed.

A failed transcription keeps the WAV, but nothing used to revisit it — the
recording sat there until retention deleted it.
"""

import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from trnscrb import backlog, storage


def _transcript(segments):
    return storage.format_transcript(segments, datetime(2026, 7, 20, 9, 52), "standup")


class HasTranscriptTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)

    def _pair(self, stem, transcript_text=None):
        wav = self.notes / f"{stem}.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        if transcript_text is not None:
            (self.notes / f"{stem}.txt").write_text(transcript_text, encoding="utf-8")
        return wav

    def test_real_transcript_counts(self):
        wav = self._pair("m", _transcript([{"start": 0, "end": 1, "text": "hello"}]))
        self.assertTrue(storage.has_transcript(wav))

    def test_missing_transcript_does_not_count(self):
        self.assertFalse(storage.has_transcript(self._pair("m")))

    def test_battery_placeholder_does_not_count(self):
        text = _transcript([]) + "\n[Recording in progress — 57 min captured and saved]\n"
        self.assertFalse(storage.has_transcript(self._pair("m", text)))

    def test_interrupted_note_does_not_count(self):
        text = _transcript([]) + "\n[Recording was interrupted]\n"
        self.assertFalse(storage.has_transcript(self._pair("m", text)))

    def test_silent_recording_counts_as_transcribed(self):
        """Zero segments is a result, not missing work — never retry it forever."""
        self.assertTrue(storage.has_transcript(self._pair("m", _transcript([]))))

    def test_pending_audio_lists_only_untranscribed(self):
        self._pair("done", _transcript([{"start": 0, "end": 1, "text": "hi"}]))
        stuck = self._pair("stuck")
        self.assertEqual(storage.pending_audio(self.notes), [stuck])


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)
        patcher = patch.object(storage, "NOTES_DIR", self.notes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _aged_wav(self, stem, days, transcript_text=None):
        wav = self.notes / f"{stem}.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        if transcript_text is not None:
            (self.notes / f"{stem}.txt").write_text(transcript_text, encoding="utf-8")
        old = time.time() - days * 86400
        os.utime(wav, (old, old))
        return wav

    def _run(self):
        with patch("trnscrb.settings.get", side_effect={"retention_audio_days": 30}.get):
            storage.apply_retention()

    def test_old_transcribed_audio_is_deleted(self):
        wav = self._aged_wav("done", 45, _transcript([{"start": 0, "end": 1, "text": "hi"}]))
        self._run()
        self.assertFalse(wav.exists())

    def test_old_untranscribed_audio_is_kept(self):
        """Deleting this would turn a failed transcription into data loss."""
        wav = self._aged_wav("stuck", 45)
        self._run()
        self.assertTrue(wav.exists())

    def test_old_placeholder_audio_is_kept(self):
        text = _transcript([]) + "\n[Recording was interrupted]\n"
        wav = self._aged_wav("interrupted", 45, text)
        self._run()
        self.assertTrue(wav.exists())


class MeetingFromFilenameTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)

    def _wav(self, name):
        f = self.notes / name
        f.write_bytes(b"RIFF" + b"\x00" * 40)
        return f

    def test_seconds_precision_stem(self):
        name, started = backlog.meeting_from_filename(
            self._wav("2026-08-05_10-02-56_Google-Meet.wav")
        )
        self.assertEqual(name, "Google-Meet")
        self.assertEqual(started, datetime(2026, 8, 5, 10, 2, 56))

    def test_minute_precision_and_recovered_suffix(self):
        name, started = backlog.meeting_from_filename(
            self._wav("2026-07-20_09-52_meeting-0952-recovered.wav")
        )
        self.assertEqual(name, "meeting-0952")
        self.assertEqual(started, datetime(2026, 7, 20, 9, 52))

    def test_unrecognised_name_falls_back_to_mtime(self):
        name, started = backlog.meeting_from_filename(self._wav("whatever.wav"))
        self.assertEqual(name, "whatever")
        self.assertIsInstance(started, datetime)


class ProcessPendingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)

    def _wav(self, name):
        f = self.notes / name
        f.write_bytes(b"RIFF" + b"\x00" * 40)
        return f

    def test_pending_files_are_transcribed(self):
        self._wav("2026-08-05_10-02-56_Google-Meet.wav")
        self._wav("2026-08-05_12-29-43_Google-Meet.wav")
        with (
            patch.object(storage, "get_live_session", return_value=None),
            patch.object(
                backlog, "transcribe_file", side_effect=lambda p, n="": (Path(f"{p.stem}.txt"), 3)
            ) as transcribe,
        ):
            written = backlog.process_pending(self.notes)
        self.assertEqual(len(written), 2)
        self.assertEqual(transcribe.call_count, 2)

    def test_stops_when_a_recording_starts(self):
        """A live meeting must not queue behind hours of catch-up work."""
        self._wav("2026-08-05_10-02-56_a.wav")
        self._wav("2026-08-05_12-29-43_b.wav")
        sessions = [None, Path("live.txt")]
        with (
            patch.object(storage, "get_live_session", side_effect=sessions),
            patch.object(backlog, "transcribe_file", return_value=(Path("a.txt"), 1)) as transcribe,
        ):
            written = backlog.process_pending(self.notes)
        self.assertEqual(len(written), 1)
        self.assertEqual(transcribe.call_count, 1)

    def test_one_failure_does_not_stop_the_rest(self):
        self._wav("2026-08-05_10-02-56_a.wav")
        self._wav("2026-08-05_12-29-43_b.wav")
        results = [RuntimeError("backend down"), (Path("b.txt"), 2)]

        def _transcribe(path, name=""):
            outcome = results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with (
            patch.object(storage, "get_live_session", return_value=None),
            patch.object(backlog, "transcribe_file", side_effect=_transcribe),
        ):
            written = backlog.process_pending(self.notes)
        self.assertEqual(written, [Path("b.txt")])

    def test_failed_audio_is_left_in_place_for_the_next_run(self):
        wav = self._wav("2026-08-05_10-02-56_a.wav")
        with (
            patch.object(storage, "get_live_session", return_value=None),
            patch.object(backlog, "transcribe_file", side_effect=RuntimeError("nope")),
        ):
            backlog.process_pending(self.notes)
        self.assertTrue(wav.exists())


if __name__ == "__main__":
    unittest.main()
