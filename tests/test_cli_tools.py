"""Tests for the `trnscrb transcribe` and `trnscrb status` commands and retention."""

import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from trnscrb import storage
from trnscrb.cli import _finalize_wav_header, cli


def _wav_bytes(n_samples=1600):
    data_size = n_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        16000,
        32000,
        2,
        16,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


class TranscribeCommandTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _run(self, filename, *args):
        wav = self.dir / filename
        wav.write_bytes(_wav_bytes())
        captured = {}

        def fake_save(path, content):
            captured["path"] = path
            captured["content"] = content

        with (
            mock.patch(
                "trnscrb.transcriber.transcribe",
                return_value=[{"start": 0.0, "end": 1.0, "text": "hello", "speaker": None}],
            ),
            mock.patch("trnscrb.settings.read_hf_token", return_value=None),
            mock.patch.object(storage, "NOTES_DIR", self.dir),
            mock.patch.object(storage, "save_transcript", side_effect=fake_save),
        ):
            result = CliRunner().invoke(cli, ["transcribe", str(wav), *args])
        return result, captured

    def test_transcribes_preserved_audio_with_inferred_name(self):
        result, captured = self._run("2026-07-20_09-52-03_meeting-0952.wav")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("1 segments", result.output)
        self.assertIn("meeting-0952", str(captured["path"]))
        self.assertIn("2026-07-20 09:52", captured["content"])

    def test_recovered_suffix_and_minute_precision(self):
        result, captured = self._run("2026-07-20_09-52_meeting-0952-recovered.wav")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("meeting-0952", str(captured["path"]))
        self.assertNotIn("recovered", str(captured["path"]))

    def test_name_override(self):
        result, captured = self._run("2026-07-20_09-52-03_meeting-0952.wav", "--name", "Board sync")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Board sync", captured["content"])

    def test_missing_file_fails_cleanly(self):
        result = CliRunner().invoke(cli, ["transcribe", str(self.dir / "nope.wav")])
        self.assertNotEqual(result.exit_code, 0)


class FinalizeWavHeaderTest(unittest.TestCase):
    def test_placeholder_header_is_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "interrupted.wav"
            wav.write_bytes(b"\x00" * 44 + b"\x01\x02" * 800)
            _finalize_wav_header(wav)
            self.assertEqual(wav.read_bytes()[:4], b"RIFF")
            data_size = struct.unpack_from("<I", wav.read_bytes(), 40)[0]
            self.assertEqual(data_size, 1600)

    def test_valid_header_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "fine.wav"
            original = _wav_bytes()
            wav.write_bytes(original)
            _finalize_wav_header(wav)
            self.assertEqual(wav.read_bytes(), original)


class StatusCommandTest(unittest.TestCase):
    def test_status_reports_running_and_recording(self):
        fake_lock = mock.MagicMock()
        fake_lock.acquire.return_value = False  # app is running
        fake_lock.holder_pid.return_value = 4242
        with (
            mock.patch("trnscrb.single_instance.SingleInstance", return_value=fake_lock),
            mock.patch.object(
                storage,
                "get_live_session_info",
                return_value={"path": Path("/tmp/x.txt"), "pid": 4242, "meeting": "sync"},
            ),
        ):
            result = CliRunner().invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("4242", result.output)
        self.assertIn("x.txt", result.output)

    def test_status_when_idle_and_stopped(self):
        fake_lock = mock.MagicMock()
        fake_lock.acquire.return_value = True  # nothing running
        with (
            mock.patch("trnscrb.single_instance.SingleInstance", return_value=fake_lock),
            mock.patch.object(storage, "get_live_session_info", return_value=None),
        ):
            result = CliRunner().invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("trnscrb start", result.output)
        self.assertIn("idle", result.output)


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name)
        patcher = mock.patch.object(storage, "NOTES_DIR", self.notes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, name, age_days):
        f = self.notes / name
        f.write_bytes(b"x")
        old = time.time() - age_days * 86400
        os.utime(f, (old, old))
        return f

    def _settings(self, audio_days, transcript_days):
        values = {
            "retention_audio_days": audio_days,
            "retention_transcript_days": transcript_days,
        }
        return mock.patch("trnscrb.settings.get", side_effect=values.get)

    def test_old_audio_deleted_fresh_kept(self):
        old = self._write("old.wav", 40)
        fresh = self._write("fresh.wav", 5)
        with self._settings(30, 0):
            storage.apply_retention()
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())

    def test_transcripts_kept_forever_by_default(self):
        transcript = self._write("ancient.txt", 400)
        with self._settings(30, 0):
            storage.apply_retention()
        self.assertTrue(transcript.exists())

    def test_transcript_retention_when_enabled(self):
        old = self._write("old.txt", 100)
        with self._settings(30, 90):
            storage.apply_retention()
        self.assertFalse(old.exists())

    def test_zero_disables_audio_retention(self):
        old = self._write("old.wav", 400)
        with self._settings(0, 0):
            storage.apply_retention()
        self.assertTrue(old.exists())


if __name__ == "__main__":
    unittest.main()
