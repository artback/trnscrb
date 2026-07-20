"""Tests for PATH bootstrapping and failed-transcription audio preservation."""

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from trnscrb import storage
from trnscrb.cli import _ensure_tool_path


class EnsureToolPathTest(unittest.TestCase):
    def test_appends_homebrew_dirs_to_bare_path(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            _ensure_tool_path()
            parts = os.environ["PATH"].split(os.pathsep)
            self.assertEqual(parts[:2], ["/usr/bin", "/bin"], "existing entries keep priority")
            for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
                if Path(extra).is_dir():
                    self.assertIn(extra, parts)

    def test_idempotent(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            _ensure_tool_path()
            once = os.environ["PATH"]
            _ensure_tool_path()
            self.assertEqual(os.environ["PATH"], once)


class PreserveAudioTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name) / "notes"
        self.notes.mkdir()
        patcher = patch.object(storage, "NOTES_DIR", self.notes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_moves_audio_into_notes_dir(self):
        audio = Path(self._tmp.name) / "tmpabc.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        saved = storage.preserve_audio(audio, "standup", datetime(2026, 7, 20, 9, 52, 3))

        self.assertIsNotNone(saved)
        self.assertEqual(saved.parent, self.notes)
        self.assertEqual(saved.suffix, ".wav")
        self.assertIn("standup", saved.name)
        self.assertTrue(saved.exists())
        self.assertFalse(audio.exists(), "original temp file must be moved, not copied")

    def test_returns_none_when_source_missing(self):
        missing = Path(self._tmp.name) / "gone.wav"
        saved = storage.preserve_audio(missing, "x", datetime.now())
        self.assertIsNone(saved)


class LiveSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        session_file = Path(self._tmp.name) / "live_session.json"
        patcher = patch.object(storage, "_LIVE_SESSION_FILE", session_file)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(storage.clear_live_session)

    def test_set_get_clear_roundtrip(self):
        live = Path(self._tmp.name) / "meeting.txt"
        live.write_text("x")
        storage.set_live_session(live)
        self.assertEqual(storage.get_live_session(), live)
        storage.clear_live_session()
        self.assertIsNone(storage.get_live_session())

    def test_dead_recorder_pid_invalidates_session(self):
        import json

        live = Path(self._tmp.name) / "meeting.txt"
        live.write_text("x")
        storage._LIVE_SESSION_FILE.write_text(json.dumps({"path": str(live), "pid": 99999999}))
        self.assertIsNone(storage.get_live_session())


class OrphanedLiveMarkerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)
        patcher = patch.object(storage, "NOTES_DIR", self.notes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, name, marker, age_secs):
        f = self.notes / name
        f.write_text(f"Meeting: x\n\n{marker}\n")
        old = __import__("time").time() - age_secs
        os.utime(f, (old, old))
        return f

    def test_old_marker_files_are_finalized(self):
        f = self._write("old.txt", "[Live — recording in progress…]", 40 * 3600)
        storage.finalize_orphaned_live_markers()
        text = f.read_text()
        self.assertNotIn("recording in progress", text)
        self.assertIn("[Recording was interrupted]", text)

    def test_fresh_marker_files_are_left_alone(self):
        f = self._write("fresh.txt", "[Recording in progress — live updates every 60s]", 60)
        storage.finalize_orphaned_live_markers()
        self.assertIn("Recording in progress", f.read_text())


class FindLiveFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)

    def _write(self, name, text, age_secs=0):
        f = self.notes / name
        f.write_text(text)
        if age_secs:
            old = __import__("time").time() - age_secs
            os.utime(f, (old, old))
        return f

    def test_stale_marker_file_is_not_live(self):
        """A months-old orphan must not be presented as a live recording."""
        from trnscrb.cli import _find_live_file

        self._write("april.txt", "[Live — recording in progress…]", age_secs=90 * 24 * 3600)
        with patch.object(storage, "get_live_session", return_value=None):
            self.assertIsNone(_find_live_file(self.notes))

    def test_fresh_marker_file_is_live(self):
        from trnscrb.cli import _find_live_file

        f = self._write("now.txt", "[Live — recording in progress…]")
        with patch.object(storage, "get_live_session", return_value=None):
            self.assertEqual(_find_live_file(self.notes), f)

    def test_registered_session_wins_even_with_stale_file(self):
        """Battery mode: file mtime goes stale but the session is authoritative."""
        from trnscrb.cli import _find_live_file

        f = self._write("battery.txt", "[Live — recording in progress…]", age_secs=2 * 3600)
        with patch.object(storage, "get_live_session", return_value=f):
            self.assertEqual(_find_live_file(self.notes), f)


if __name__ == "__main__":
    unittest.main()
