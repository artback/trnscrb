"""Tests for diarization pipeline selection (community-1 with 3.1 fallback)."""

import shutil
import types
import unittest
from unittest.mock import patch

from trnscrb import diarizer


class _FakePipeline:
    def __init__(self, name):
        self.name = name

    def to(self, _device):
        return self


class PipelineSelectionTest(unittest.TestCase):
    def setUp(self):
        diarizer._pipeline = None
        self.addCleanup(setattr, diarizer, "_pipeline", None)

    def _get(self, load_side_effect, setting=None):
        with (
            patch.object(diarizer, "_load_pipeline", side_effect=load_side_effect) as load,
            patch("trnscrb.settings.get", return_value=setting),
        ):
            pipeline = diarizer._get_pipeline("hf_token")
        return pipeline, load

    def test_prefers_community_1(self):
        pipeline, load = self._get(lambda model_id, token: _FakePipeline(model_id))
        self.assertEqual(pipeline.name, "pyannote/speaker-diarization-community-1")
        self.assertEqual(load.call_count, 1)

    def test_falls_back_to_3_1_when_community_unavailable(self):
        def load(model_id, token):
            if "community" in model_id:
                raise RuntimeError("gated")
            return _FakePipeline(model_id)

        pipeline, mock_load = self._get(load)
        self.assertEqual(pipeline.name, "pyannote/speaker-diarization-3.1")
        self.assertEqual(mock_load.call_count, 2)

    def test_raises_when_no_pipeline_loads(self):
        with (
            patch.object(diarizer, "_load_pipeline", side_effect=RuntimeError("gated")),
            patch("trnscrb.settings.get", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "No diarization pipeline"):
                diarizer._get_pipeline("hf_token")

    def test_setting_overrides_preferred_pipeline(self):
        pipeline, load = self._get(
            lambda model_id, token: _FakePipeline(model_id),
            setting="my-org/custom-diarizer",
        )
        self.assertEqual(pipeline.name, "my-org/custom-diarizer")


if __name__ == "__main__":
    unittest.main()


class SpeakerTimelineTest(unittest.TestCase):
    """pyannote 4 wraps the timeline in a DiarizeOutput; 3.x returned it bare."""

    def test_prefers_the_exclusive_timeline(self):
        result = types.SimpleNamespace(
            exclusive_speaker_diarization="exclusive",
            speaker_diarization="overlapping",
        )
        self.assertEqual(diarizer._speaker_timeline(result), "exclusive")

    def test_falls_back_to_the_plain_timeline(self):
        result = types.SimpleNamespace(speaker_diarization="overlapping")
        self.assertEqual(diarizer._speaker_timeline(result), "overlapping")

    def test_bare_annotation_is_returned_as_is(self):
        annotation = object()
        self.assertIs(diarizer._speaker_timeline(annotation), annotation)


class AudioInputTest(unittest.TestCase):
    """The pipeline is fed a waveform, so a broken torchcodec can't kill it."""

    def _wav(self, seconds=1.0, rate=16_000):
        import struct
        import tempfile
        import wave
        from pathlib import Path

        path = Path(tempfile.mkstemp(suffix=".wav")[1])
        self.addCleanup(path.unlink, missing_ok=True)
        frames = int(seconds * rate)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"".join(struct.pack("<h", (i % 1000) - 500) for i in range(frames)))
        return path

    def test_decodes_a_wav_into_a_waveform(self):
        audio = diarizer._audio_input(self._wav(seconds=0.5))
        self.assertIsInstance(audio, dict)
        self.assertEqual(audio["sample_rate"], 16_000)
        channels, frames = audio["waveform"].shape
        self.assertEqual(channels, 1)
        self.assertEqual(frames, 8_000)

    def test_unreadable_audio_falls_back_to_the_path(self):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkstemp(suffix=".wav")[1])
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(b"not a wav")
        self.assertEqual(diarizer._audio_input(path), str(path))

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_audio_soundfile_cannot_open_goes_through_ffmpeg(self):
        """An mp3 is decoded by the CLI rather than handed to the pipeline."""
        import subprocess
        import tempfile
        from pathlib import Path

        source = self._wav(seconds=0.5)
        mp3 = Path(tempfile.mkdtemp()) / "clip.mp3"
        self.addCleanup(shutil.rmtree, mp3.parent, ignore_errors=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source), str(mp3)],
            check=True,
            capture_output=True,
        )

        audio = diarizer._audio_input(mp3)
        self.assertIsInstance(audio, dict, "mp3 should have been decoded, not passed as a path")
        self.assertEqual(audio["sample_rate"], 16_000)

    def test_diarize_hands_the_waveform_to_the_pipeline(self):
        seen = {}

        empty = types.SimpleNamespace(itertracks=lambda yield_label: iter(()))

        def pipeline(audio):
            seen["audio"] = audio
            return types.SimpleNamespace(speaker_diarization=empty, speaker_embeddings=None)

        with patch.object(diarizer, "_get_pipeline", return_value=pipeline):
            diarizer.diarize_with_embeddings(self._wav(), "hf_token")
        self.assertIsInstance(seen["audio"], dict)
        self.assertIn("waveform", seen["audio"])


class TorchcodecBlockTest(unittest.TestCase):
    """Only one FFmpeg may be loaded, so pyannote must not import torchcodec."""

    def test_import_is_refused_inside_the_context(self):
        import sys

        with diarizer._without_torchcodec():
            with self.assertRaises(ImportError):
                import torchcodec  # noqa: F401
        self.assertFalse(
            any(isinstance(f, diarizer._TorchcodecBlocked) for f in sys.meta_path),
            "the blocker should be removed again on exit",
        )

    def test_blocker_is_removed_even_when_the_body_raises(self):
        import sys

        with self.assertRaises(RuntimeError), diarizer._without_torchcodec():
            raise RuntimeError("boom")
        self.assertFalse(any(isinstance(f, diarizer._TorchcodecBlocked) for f in sys.meta_path))

    def test_other_imports_are_untouched(self):
        with diarizer._without_torchcodec():
            import json

            self.assertTrue(hasattr(json, "dumps"))
