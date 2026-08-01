"""Tests for Obsidian vault mirroring."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from trnscrb import obsidian


class VaultDetectionTest(unittest.TestCase):
    def test_configured_vault_wins(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(obsidian.settings, "get", return_value=d):
                self.assertEqual(obsidian.vault_path(), Path(d))

    def test_none_when_configured_path_missing(self):
        with patch.object(obsidian.settings, "get", return_value="/no/such/vault"):
            # Falls through to auto-detect; force that to None too.
            with patch.object(obsidian, "_detect_vault", return_value=None):
                self.assertIsNone(obsidian.vault_path())


class NoteNameTest(unittest.TestCase):
    def test_prefixes_date_and_sanitizes(self):
        name = obsidian.note_name("1:1 Jonathan / David [urgent]", datetime(2026, 7, 30, 11, 0))
        self.assertTrue(name.startswith("2026-07-30 "))
        for bad in '\\/:*?"<>|#^[]':
            self.assertNotIn(bad, name)


class MirrorTest(unittest.TestCase):
    def test_writes_note_and_returns_name(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(obsidian, "meetings_dir", return_value=Path(d)):
                name = obsidian.mirror_transcript("Standup", datetime(2026, 7, 30, 9, 0), "body")
        self.assertEqual(name, "2026-07-30 Standup")

    def test_noop_without_vault(self):
        with patch.object(obsidian, "meetings_dir", return_value=None):
            self.assertIsNone(
                obsidian.mirror_transcript("Standup", datetime(2026, 7, 30, 9, 0), "body")
            )

    def test_written_file_is_complete(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(obsidian, "meetings_dir", return_value=Path(d)):
                obsidian.mirror_transcript("Sync", datetime(2026, 7, 30, 9, 0), "hello world")
                written = (Path(d) / "2026-07-30 Sync.md").read_text()
        self.assertEqual(written, "hello world")


if __name__ == "__main__":
    unittest.main()
