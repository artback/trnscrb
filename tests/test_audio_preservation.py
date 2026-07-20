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


if __name__ == "__main__":
    unittest.main()
