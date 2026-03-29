import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _settings_getter(backend: str, parakeet_model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"):
    mapping = {
        "transcription_backend": backend,
        "parakeet_model_id": parakeet_model_id,
        "model_size": "small",
    }
    return lambda key: mapping.get(key)


def _fake_stat():
    """Return a stat result with non-zero size."""
    return types.SimpleNamespace(st_size=1024)


class TranscriberTests(unittest.TestCase):
    def _reload_transcriber(self):
        import trnscrb.transcriber as transcriber

        return importlib.reload(transcriber)

    def test_uses_parakeet_backend_and_normalizes_segments(self):
        class Sentence:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class ParakeetModel:
            def transcribe(self, _audio_path):
                return types.SimpleNamespace(
                    sentences=[
                        Sentence(0.0, 1.5, " hello "),
                        Sentence(1.5, 2.0, " "),
                        Sentence(2.0, 3.0, "world"),
                    ]
                )

        fake_parakeet = types.SimpleNamespace(from_pretrained=lambda _model_id: ParakeetModel())

        class WhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **_kwargs):
                seg = types.SimpleNamespace(start=0.0, end=1.0, text="whisper")
                return [seg], None

        fake_whisper = types.SimpleNamespace(WhisperModel=WhisperModel)

        with mock.patch.dict(
            sys.modules,
            {"parakeet_mlx": fake_parakeet, "faster_whisper": fake_whisper},
            clear=False,
        ):
            with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("parakeet")):
                transcriber = self._reload_transcriber()
                with mock.patch.object(Path, "exists", return_value=True), \
                     mock.patch.object(Path, "stat", return_value=_fake_stat()):
                    segments = transcriber.transcribe(Path("audio.wav"))

        self.assertEqual(
            segments,
            [
                {"start": 0.0, "end": 1.5, "text": "hello", "speaker": None},
                {"start": 2.0, "end": 3.0, "text": "world", "speaker": None},
            ],
        )

    def test_uses_whisper_backend_when_configured(self):
        class WhisperModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **_kwargs):
                seg = types.SimpleNamespace(start=0.0, end=1.0, text=" whisper ")
                return [seg], None

        fake_whisper = types.SimpleNamespace(WhisperModel=WhisperModel)

        with mock.patch.dict(sys.modules, {"faster_whisper": fake_whisper}, clear=False):
            with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("whisper")):
                transcriber = self._reload_transcriber()
                with mock.patch.object(Path, "exists", return_value=True), \
                     mock.patch.object(Path, "stat", return_value=_fake_stat()):
                    segments = transcriber.transcribe(Path("audio.wav"))

        self.assertEqual(
            segments,
            [{"start": 0.0, "end": 1.0, "text": "whisper", "speaker": None}],
        )

    def test_fails_fast_for_unknown_backend(self):
        with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("unknown")):
            transcriber = self._reload_transcriber()
            with mock.patch.object(Path, "exists", return_value=True), \
                 mock.patch.object(Path, "stat", return_value=_fake_stat()):
                with self.assertRaisesRegex(RuntimeError, "Unsupported transcription backend"):
                    transcriber.transcribe(Path("audio.wav"))

    def test_fails_fast_when_parakeet_dependency_missing(self):
        with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("parakeet")):
            with mock.patch.dict(sys.modules, {"parakeet_mlx": None}, clear=False):
                transcriber = self._reload_transcriber()
                with mock.patch.object(Path, "exists", return_value=True), \
                     mock.patch.object(Path, "stat", return_value=_fake_stat()):
                    with self.assertRaisesRegex(RuntimeError, "uv add parakeet-mlx"):
                        transcriber.transcribe(Path("audio.wav"))


if __name__ == "__main__":
    unittest.main()
