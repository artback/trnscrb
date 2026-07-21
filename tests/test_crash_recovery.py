"""Tests for recovering recordings interrupted by a crash, kill, or restart."""

import struct
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from trnscrb import recorder
from trnscrb.recorder import SAMPLE_RATE


def _write_partial_recording(path: Path, seconds: float) -> None:
    """A recording as a killed process leaves it: placeholder header + audio."""
    frames = int(SAMPLE_RATE * seconds)
    path.write_bytes(b"\x00" * 44 + b"\x01\x00" * frames)


class FinalizeWavHeaderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_repairs_placeholder_header(self):
        wav = self.root / "partial.wav"
        _write_partial_recording(wav, 1.5)

        frames = recorder.finalize_wav_header(wav)

        self.assertEqual(frames, int(SAMPLE_RATE * 1.5))
        with wave.open(str(wav), "rb") as wf:  # unreadable before the repair
            self.assertEqual(wf.getframerate(), SAMPLE_RATE)
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getnframes(), frames)

    def test_header_reports_riff_and_data_sizes(self):
        wav = self.root / "partial.wav"
        _write_partial_recording(wav, 0.5)
        recorder.finalize_wav_header(wav)

        head = wav.read_bytes()[:44]
        data_size = struct.unpack_from("<I", head, 40)[0]
        self.assertEqual(data_size, wav.stat().st_size - 44)


class RecoverOrphanedRecordingsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tmpdir = self.root / "tmp"
        self.notes = self.root / "notes"
        self.tmpdir.mkdir()
        self.notes.mkdir()
        patcher = patch.object(recorder.tempfile, "gettempdir", lambda: str(self.tmpdir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _orphan(self, name, seconds, age_secs=3600):
        p = self.tmpdir / name
        _write_partial_recording(p, seconds)
        old = time.time() - age_secs
        import os

        os.utime(p, (old, old))
        return p

    def test_recovers_interrupted_recording_into_notes(self):
        orphan = self._orphan("tmpabc.wav", 90)

        recovered = recorder.recover_orphaned_recordings(self.notes)

        self.assertEqual(len(recovered), 1)
        self.assertFalse(orphan.exists(), "temp file should be moved, not copied")
        saved = recovered[0]
        self.assertEqual(saved.parent, self.notes)
        self.assertIn("recovered-recording", saved.name)
        with wave.open(str(saved), "rb") as wf:  # readable, i.e. header repaired
            self.assertEqual(wf.getnframes(), int(SAMPLE_RATE * 90))

    def test_short_fragments_are_discarded(self):
        orphan = self._orphan("tmpshort.wav", 3)
        self.assertEqual(recorder.recover_orphaned_recordings(self.notes), [])
        self.assertFalse(orphan.exists())

    def test_recent_files_are_left_alone(self):
        """A file still being written by an active recording must be untouched."""
        orphan = self._orphan("tmplive.wav", 120, age_secs=5)
        self.assertEqual(recorder.recover_orphaned_recordings(self.notes), [])
        self.assertTrue(orphan.exists())

    def test_cleanup_alias_recovers_instead_of_deleting(self):
        """The old name deleted audio after an hour; it must now rescue it."""
        orphan = self._orphan("tmpold.wav", 300)
        with patch("trnscrb.storage.ensure_notes_dir", return_value=self.notes):
            recorder.cleanup_stale_temp_files()
        self.assertFalse(orphan.exists())
        self.assertEqual(len(list(self.notes.glob("*recovered-recording.wav"))), 1)

    def test_multiple_orphans_all_recovered(self):
        self._orphan("tmp1.wav", 60)
        self._orphan("tmp2.wav", 45)
        self.assertEqual(len(recorder.recover_orphaned_recordings(self.notes)), 2)


if __name__ == "__main__":
    unittest.main()
