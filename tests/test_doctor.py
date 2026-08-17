"""Tests for `trnscrb doctor` — the probe that runs the stack instead of inspecting it."""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from trnscrb import diarizer, health
from trnscrb.cli import _synthetic_clip, _voiceprint_store_state, doctor


class SyntheticClipTest(unittest.TestCase):
    def test_makes_a_readable_16k_mono_wav(self):
        path = _synthetic_clip(2.0)
        self.addCleanup(path.unlink, True)
        with wave.open(str(path), "rb") as w:
            self.assertEqual(w.getframerate(), 16_000)
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getnframes(), 32_000)

    def test_the_decoder_under_test_can_read_it(self):
        path = _synthetic_clip(1.0)
        self.addCleanup(path.unlink, True)
        audio = diarizer._audio_input(path)
        self.assertIsInstance(audio, dict)
        self.assertEqual(audio["sample_rate"], 16_000)


class DoctorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, health, "STORE", health.STORE)
        health.STORE = Path(tmp.name) / "health.json"

    def _run(self, args=(), **patches):
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value=patches.get("token", "hf_x")),
            mock.patch.object(
                diarizer, "is_downloaded", return_value=patches.get("downloaded", True)
            ),
        ):
            return CliRunner().invoke(doctor, list(args))

    def test_quick_skips_the_model_load(self):
        with mock.patch.object(diarizer, "_get_pipeline") as load:
            result = self._run(["--quick"])
        self.assertEqual(result.exit_code, 0)
        load.assert_not_called()
        self.assertIn("HuggingFace token", result.output)
        self.assertIn("audio decoding", result.output)

    def test_a_missing_token_is_named_with_its_fix(self):
        result = self._run(["--quick"], token=None)
        self.assertIn("no token", result.output)
        self.assertIn("hf.co/settings/tokens", result.output)

    def test_an_undownloaded_pipeline_is_named_with_its_fix(self):
        result = self._run(["--quick"], downloaded=False)
        self.assertIn("accept the model terms", result.output)

    def test_a_green_probe_clears_a_stale_failure(self):
        """Fix the machine and status should go green without waiting for a meeting."""
        health.record_failure(health.DIARIZATION, "torchcodec is not available")

        turns = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
        with (
            mock.patch.object(diarizer, "_get_pipeline", return_value=object()),
            mock.patch.object(diarizer, "pipeline_id", return_value="pyannote/community-1"),
            mock.patch.object(diarizer, "embedding_space", return_value="plda"),
            mock.patch.object(
                diarizer,
                "diarize_with_embeddings",
                return_value=(turns, {"SPEAKER_00": [0.1] * 128}),
            ),
        ):
            result = self._run()

        self.assertIn("work end to end", result.output)
        entry = health.get(health.DIARIZATION)
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["failures"], 0)

    def test_a_failing_pipeline_is_reported_and_not_cleared(self):
        health.record_failure(health.DIARIZATION, "torchcodec is not available")
        with mock.patch.object(diarizer, "_get_pipeline", side_effect=RuntimeError("gated")):
            result = self._run()
        self.assertIn("Broken", result.output)
        self.assertFalse(health.get(health.DIARIZATION)["ok"])


class VoiceprintStoreStateTest(unittest.TestCase):
    def _store(self, data):
        return mock.patch("trnscrb.voiceprints.load", return_value=data)

    def test_empty_store(self):
        with self._store({"voices": {}}):
            state, comparable = _voiceprint_store_state("plda")
        self.assertIn("empty", state)
        self.assertTrue(comparable)

    def test_counts_named_voices(self):
        data = {"space": "plda", "voices": {"voice-1": {"name": "Me"}, "voice-2": {"name": ""}}}
        with self._store(data):
            state, comparable = _voiceprint_store_state("plda")
        self.assertIn("2 voice(s), 1 named", state)
        self.assertTrue(comparable)

    def test_warns_when_the_space_no_longer_matches(self):
        """The silent wipe: stored vectors stop being comparable and vanish."""
        data = {"space": "embedding", "voices": {"voice-1": {"name": "Me"}}}
        with self._store(data):
            state, comparable = _voiceprint_store_state("plda")
        self.assertFalse(comparable)
        self.assertIn("discarded", state)


if __name__ == "__main__":
    unittest.main()


class SystemAudioChecksTest(unittest.TestCase):
    """Doctor has to name the failure the checkbox lies about."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, health, "STORE", health.STORE)
        health.STORE = Path(tmp.name) / "health.json"

    def _run(self, granted):
        from trnscrb import sck_helper
        from trnscrb.cli import doctor

        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", return_value=True),
            mock.patch.object(sck_helper, "has_permission", return_value=granted),
            mock.patch.object(sck_helper, "helper_path", return_value=Path("/Apps/T.app/x")),
        ):
            return CliRunner().invoke(doctor, ["--quick"])

    def test_a_denied_grant_is_reported(self):
        out = self._run(False).output
        self.assertIn("screen recording", out)
        self.assertIn("only your microphone", out)

    def test_the_stale_entry_is_explained_with_its_fix(self):
        """The checkbox says yes and the capture says no; say why."""
        out = self._run(False).output
        self.assertIn("stale", out)
        self.assertIn("tccutil reset ScreenCapture io.trnscrb.app", out)

    def test_a_granted_permission_says_so(self):
        out = self._run(True).output
        self.assertIn("granted", out)
        self.assertNotIn("tccutil", out)

    def test_an_unanswerable_check_is_not_a_silent_pass(self):
        out = self._run(None).output
        self.assertIn("could not ask the helper", out)


class LogIsolationTest(unittest.TestCase):
    """Test runs must not write into the log a user debugs from."""

    def test_the_log_file_is_redirected(self):
        from trnscrb import log

        self.assertNotEqual(log._LOG_DIR, Path.home() / "Library" / "Logs")

    def test_the_redirect_is_honoured_from_the_environment(self):
        import os

        self.assertIn(os.environ["TRNSCRB_LOG_DIR"], str(log_file()))


def log_file():
    from trnscrb import log

    return log._LOG_FILE


class VerdictSeparationTest(unittest.TestCase):
    """Speaker labels and system audio are separate verdicts.

    A machine with no Screen Recording grant can diarize perfectly. Letting
    one fail the other reports the wrong thing in both directions — and hid
    the "works end to end" line on CI, which has no app bundle at all.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, health, "STORE", health.STORE)
        health.STORE = Path(tmp.name) / "health.json"

    def _run(self):
        from trnscrb import sck_helper
        from trnscrb.cli import doctor

        turns = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", return_value=True),
            mock.patch.object(diarizer, "_get_pipeline", return_value=object()),
            mock.patch.object(diarizer, "pipeline_id", return_value="pyannote/community-1"),
            mock.patch.object(diarizer, "embedding_space", return_value="plda"),
            mock.patch.object(
                diarizer, "diarize_with_embeddings", return_value=(turns, {"SPEAKER_00": [0.1]})
            ),
            mock.patch.object(sck_helper, "helper_path", return_value=None),
            mock.patch.object(sck_helper, "has_permission", return_value=None),
        ):
            return CliRunner().invoke(doctor, [])

    def test_missing_system_audio_does_not_fail_speaker_labels(self):
        out = self._run().output
        self.assertIn("Speaker labels work end to end", out)
        self.assertIn("System audio unavailable", out)

    def test_missing_system_audio_still_clears_the_diarization_failure(self):
        health.record_failure(health.DIARIZATION, "torchcodec is not available")
        self._run()
        self.assertTrue(health.get(health.DIARIZATION)["ok"])

    def test_the_consequence_is_spelled_out(self):
        self.assertIn("microphone only", self._run().output)
