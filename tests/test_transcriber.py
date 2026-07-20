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
        "qwen3_model_id": "Qwen/Qwen3-ASR-0.6B",
        "model_size": "small",
    }
    return lambda key: mapping.get(key)


def _fake_qwen3_module(text, words):
    """A stand-in mlx_qwen3_asr module returning a fixed transcription."""

    def transcribe(_audio, **_kwargs):
        return types.SimpleNamespace(text=text, segments=words)

    return types.SimpleNamespace(
        load_model=lambda _model_id: (object(), object()),
        ForcedAligner=lambda: object(),
        transcribe=transcribe,
    )


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

        seen_kwargs = {}

        class ParakeetModel:
            def transcribe(self, _audio_path, **kwargs):
                seen_kwargs.update(kwargs)
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
                with (
                    mock.patch.object(Path, "exists", return_value=True),
                    mock.patch.object(Path, "stat", return_value=_fake_stat()),
                ):
                    segments = transcriber.transcribe(Path("audio.wav"))

        self.assertEqual(
            segments,
            [
                {"start": 0.0, "end": 1.5, "text": "hello", "speaker": None},
                {"start": 2.0, "end": 3.0, "text": "world", "speaker": None},
            ],
        )
        # Long recordings must be chunked or Metal OOMs on multi-hour meetings.
        self.assertGreater(seen_kwargs.get("chunk_duration") or 0, 0)

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
                with (
                    mock.patch.object(Path, "exists", return_value=True),
                    mock.patch.object(Path, "stat", return_value=_fake_stat()),
                ):
                    segments = transcriber.transcribe(Path("audio.wav"))

        self.assertEqual(
            segments,
            [{"start": 0.0, "end": 1.0, "text": "whisper", "speaker": None}],
        )

    def test_fails_fast_for_unknown_backend(self):
        with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("unknown")):
            transcriber = self._reload_transcriber()
            with (
                mock.patch.object(Path, "exists", return_value=True),
                mock.patch.object(Path, "stat", return_value=_fake_stat()),
            ):
                with self.assertRaisesRegex(RuntimeError, "Unsupported transcription backend"):
                    transcriber.transcribe(Path("audio.wav"))

    def test_fails_fast_when_parakeet_dependency_missing(self):
        with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("parakeet")):
            with mock.patch.dict(sys.modules, {"parakeet_mlx": None}, clear=False):
                transcriber = self._reload_transcriber()
                with (
                    mock.patch.object(Path, "exists", return_value=True),
                    mock.patch.object(Path, "stat", return_value=_fake_stat()),
                ):
                    with self.assertRaisesRegex(RuntimeError, "uv add parakeet-mlx"):
                        transcriber.transcribe(Path("audio.wav"))

    def test_uses_qwen3_backend_when_configured(self):
        words = [
            {"text": "Hello", "start": 0.5, "end": 0.9},
            {"text": "there", "start": 0.9, "end": 1.2},
            {"text": "Goodbye", "start": 2.0, "end": 2.5},
            {"text": "now", "start": 2.5, "end": 2.8},
        ]
        fake = _fake_qwen3_module("Hello there. Goodbye now.", words)

        with mock.patch.dict(sys.modules, {"mlx_qwen3_asr": fake}, clear=False):
            with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("qwen3")):
                transcriber = self._reload_transcriber()
                with (
                    mock.patch.object(Path, "exists", return_value=True),
                    mock.patch.object(Path, "stat", return_value=_fake_stat()),
                ):
                    segments = transcriber.transcribe(Path("audio.wav"))

        self.assertEqual(
            segments,
            [
                {"start": 0.5, "end": 1.2, "text": "Hello there.", "speaker": None},
                {"start": 2.0, "end": 2.8, "text": "Goodbye now.", "speaker": None},
            ],
        )

    def test_auto_routes_non_english_to_qwen3(self):
        with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("auto")):
            transcriber = self._reload_transcriber()
            with (
                mock.patch.object(Path, "exists", return_value=True),
                mock.patch.object(Path, "stat", return_value=_fake_stat()),
                mock.patch.object(transcriber, "_detect_language", return_value="sv"),
                mock.patch.object(
                    transcriber, "_transcribe_qwen3", return_value=[{"text": "hej"}]
                ) as qwen3,
                mock.patch.object(transcriber, "_transcribe_whisper") as whisper,
            ):
                segments = transcriber.transcribe(Path("audio.wav"))

        qwen3.assert_called_once()
        whisper.assert_not_called()
        self.assertEqual(segments, [{"text": "hej"}])

    def test_auto_falls_back_to_whisper_when_qwen3_fails(self):
        with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("auto")):
            transcriber = self._reload_transcriber()
            with (
                mock.patch.object(Path, "exists", return_value=True),
                mock.patch.object(Path, "stat", return_value=_fake_stat()),
                mock.patch.object(transcriber, "_detect_language", return_value="de"),
                mock.patch.object(
                    transcriber, "_transcribe_qwen3", side_effect=RuntimeError("no model")
                ),
                mock.patch.object(
                    transcriber, "_transcribe_whisper", return_value=[{"text": "hallo"}]
                ) as whisper,
            ):
                segments = transcriber.transcribe(Path("audio.wav"))

        whisper.assert_called_once()
        self.assertEqual(segments, [{"text": "hallo"}])


class InferenceThreadTests(unittest.TestCase):
    """MLX binds arrays to their creating thread — every load and inference
    must run on the transcriber's single dedicated worker thread."""

    def test_transcribe_runs_on_inference_thread_from_any_caller(self):
        import threading

        seen_threads = []

        class ParakeetModel:
            def transcribe(self, _audio_path, **kwargs):
                seen_threads.append(threading.current_thread().name)
                return types.SimpleNamespace(sentences=[])

        fake_parakeet = types.SimpleNamespace(from_pretrained=lambda _id: ParakeetModel())

        with mock.patch.dict(sys.modules, {"parakeet_mlx": fake_parakeet}, clear=False):
            with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("parakeet")):
                transcriber = importlib.reload(__import__("trnscrb.transcriber", fromlist=["x"]))
                with (
                    mock.patch.object(Path, "exists", return_value=True),
                    mock.patch.object(Path, "stat", return_value=_fake_stat()),
                ):
                    transcriber.transcribe(Path("audio.wav"))  # from main thread
                    worker = threading.Thread(target=transcriber.transcribe, args=(Path("a.wav"),))
                    worker.start()
                    worker.join()

        self.assertEqual(len(seen_threads), 2)
        for name in seen_threads:
            self.assertTrue(
                name.startswith("trnscrb-inference"),
                f"inference ran on wrong thread: {name}",
            )

    def test_preload_runs_on_inference_thread(self):
        import threading

        load_threads = []

        def fake_from_pretrained(_model_id):
            load_threads.append(threading.current_thread().name)
            return object()

        fake_parakeet = types.SimpleNamespace(from_pretrained=fake_from_pretrained)

        with mock.patch.dict(sys.modules, {"parakeet_mlx": fake_parakeet}, clear=False):
            with mock.patch("trnscrb.settings.get", side_effect=_settings_getter("parakeet")):
                transcriber = importlib.reload(__import__("trnscrb.transcriber", fromlist=["x"]))
                transcriber.preload("parakeet")

        self.assertEqual(len(load_threads), 1)
        self.assertTrue(load_threads[0].startswith("trnscrb-inference"))


class WordsToSegmentsTests(unittest.TestCase):
    def _fn(self):
        import trnscrb.transcriber as transcriber

        return transcriber._words_to_segments

    def test_splits_sentences_with_word_timings(self):
        words = [
            {"text": "One", "start": 0.0, "end": 0.4},
            {"text": "two", "start": 0.4, "end": 0.8},
            {"text": "Three", "start": 1.5, "end": 2.0},
        ]
        segments = self._fn()("One two. Three!", words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "One two.")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 0.8)
        self.assertEqual(segments[1]["text"], "Three!")
        self.assertEqual(segments[1]["start"], 1.5)

    def test_no_words_yields_single_untimed_segment(self):
        segments = self._fn()("Just text.", [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "Just text.")

    def test_word_list_shorter_than_text_pins_to_tail(self):
        words = [{"text": "One", "start": 0.0, "end": 0.4}]
        segments = self._fn()("One. Two three.", words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1]["start"], 0.4)

    def test_empty_text_returns_no_segments(self):
        self.assertEqual(self._fn()("", []), [])


if __name__ == "__main__":
    unittest.main()
