"""Tests for the noise-reduction front-end (denoise setting).

Covers the denoiser engine itself, the preprocess routing, the transcribe()
wiring (toggle on/off, temp-file cleanup on success and failure), the settings
default, and the CLI surface (config set + status line).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from click.testing import CliRunner

from trnscrb import cli, denoise, settings, transcriber


def _make_wav(path: Path, seconds: float = 1.0, channels: int = 1, sr: int = 16000) -> None:
    """Tone buried in white noise — the kind of audio the filter should help."""
    import soundfile as sf

    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    if channels == 1:
        data = tone + 0.15 * rng.standard_normal(len(t))
    else:
        data = np.stack(
            [tone + 0.15 * rng.standard_normal(len(t)) for _ in range(channels)], axis=1
        )
    sf.write(str(path), data, sr)


def _tiny_wav(path: Path) -> None:
    """Sub-half-second clip: under the _MIN_AUDIO_SECS threshold."""
    import soundfile as sf

    sf.write(str(path), np.zeros(1600, dtype=np.float32), 16000)


class DenoiseEngineTest(unittest.TestCase):
    def test_available_with_engine_installed(self):
        self.assertTrue(denoise.available())

    def test_reduce_noise_wav_mono(self):
        src = Path(tempfile.mkdtemp()) / "in.wav"
        dst = src.with_name("out.wav")
        _make_wav(src)
        import soundfile as sf

        frames_before = sf.info(str(src)).frames
        self.assertTrue(denoise.reduce_noise_wav(src, dst))
        self.assertTrue(dst.exists())
        self.assertEqual(sf.info(str(dst)).frames, frames_before)

    def test_reduce_noise_wav_stereo(self):
        src = Path(tempfile.mkdtemp()) / "in.wav"
        dst = src.with_name("out.wav")
        _make_wav(src, channels=2)
        import soundfile as sf

        data, sr = sf.read(str(src))
        self.assertTrue(denoise.reduce_noise_wav(src, dst))
        out, out_sr = sf.read(str(dst))
        self.assertEqual(out_sr, sr)
        self.assertEqual(out.shape, data.shape)

    def test_reduce_noise_wav_silence_copies_through(self):
        src = Path(tempfile.mkdtemp()) / "silence.wav"
        dst = src.with_name("out.wav")
        import soundfile as sf

        sf.write(str(src), np.zeros(16000, dtype=np.float32), 16000)
        self.assertTrue(denoise.reduce_noise_wav(src, dst))
        self.assertTrue(dst.exists())

    def test_reduce_noise_wav_without_engine_returns_false(self):
        src = Path(tempfile.mkdtemp()) / "in.wav"
        dst = src.with_name("out.wav")
        _make_wav(src)
        with mock.patch("builtins.__import__", side_effect=ImportError("no engine")):
            self.assertFalse(denoise.reduce_noise_wav(src, dst))
        self.assertFalse(dst.exists())


class PreprocessTest(unittest.TestCase):
    def test_tiny_clip_skipped(self):
        src = Path(tempfile.mkdtemp()) / "tiny.wav"
        _tiny_wav(src)
        path, tmp = denoise.preprocess(src)
        self.assertEqual(path, src)
        self.assertIsNone(tmp)

    @mock.patch.object(denoise, "available", return_value=False)
    def test_missing_engine_returns_original(self, _avail):
        src = Path(tempfile.mkdtemp()) / "in.wav"
        _make_wav(src)
        path, tmp = denoise.preprocess(src)
        self.assertEqual(path, src)
        self.assertIsNone(tmp)

    def test_missing_file_returns_original(self):
        src = Path(tempfile.mkdtemp()) / "ghost.wav"
        path, tmp = denoise.preprocess(src)
        self.assertEqual(path, src)
        self.assertIsNone(tmp)

    def test_normal_clip_denoised_to_temp(self):
        tmpdir = Path(tempfile.mkdtemp())
        src = tmpdir / "in.wav"
        _make_wav(src, seconds=0.5)
        path, tmp = denoise.preprocess(src)
        self.assertIsNotNone(tmp)
        self.assertEqual(path, tmp)
        self.assertTrue(tmp.exists())
        self.assertNotEqual(tmp, src)
        # Caller owns cleanup exactly as transcribe() does.
        tmp.unlink(missing_ok=True)


class TranscribeWiringTest(unittest.TestCase):
    def _patch_pipeline(self, denoise_on: bool, worker_error=None):
        """Patch everything below the denoise stage in transcribe().

        Patchers are stored (not the started mocks): calling ``.stop()`` on a
        MagicMock is a silent no-op, so teardown must stop the patchers.
        """
        fake_settings = mock.patch.object(
            transcriber.settings,
            "get",
            side_effect=lambda key, *a, **k: {"denoise": denoise_on}.get(key, False),
        )

        def _worker(audio_path, backend):
            if worker_error is not None:
                raise worker_error
            return [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": None}]

        patchers = [
            fake_settings,
            mock.patch.object(transcriber, "_backend", return_value="whisper"),
            mock.patch.object(transcriber, "_transcribe_on_worker", side_effect=_worker),
            mock.patch.object(transcriber, "trim_mlx_cache", return_value=None),
            mock.patch.object(transcriber, "_apply_glossary", return_value=None),
        ]
        for p in patchers:
            p.start()
        return patchers

    def _teardown(self, patchers):
        for p in patchers:
            p.stop()

    def test_denoise_off_skips_filter(self):
        src = Path(tempfile.mkdtemp()) / "in.wav"
        _make_wav(src)
        patches = self._patch_pipeline(denoise_on=False)
        try:
            with mock.patch.object(transcriber, "_denoise_audio") as denoise_audio:
                segments = transcriber.transcribe(src)
        finally:
            self._teardown(patches)
        self.assertEqual(segments[0]["text"], "hello")
        denoise_audio.assert_not_called()

    def test_denoise_on_routes_and_cleans_tmp(self):
        tmpdir = Path(tempfile.mkdtemp())
        src = tmpdir / "in.wav"
        _make_wav(src)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        patches = self._patch_pipeline(denoise_on=True)
        try:
            with mock.patch.object(
                transcriber, "_denoise_audio", return_value=(tmp_path, tmp_path)
            ) as da:
                segments = transcriber.transcribe(src)
        finally:
            self._teardown(patches)
        da.assert_called_once_with(src)
        self.assertEqual(segments[0]["text"], "hello")
        self.assertFalse(tmp_path.exists(), "denoised temp file must be cleaned up")

    def test_denoise_tmp_cleaned_on_failure(self):
        tmpdir = Path(tempfile.mkdtemp())
        src = tmpdir / "in.wav"
        _make_wav(src)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        patches = self._patch_pipeline(denoise_on=True, worker_error=RuntimeError("boom"))
        try:
            with mock.patch.object(
                transcriber, "_denoise_audio", return_value=(tmp_path, tmp_path)
            ):
                with self.assertRaises(RuntimeError):
                    transcriber.transcribe(src)
        finally:
            self._teardown(patches)
        self.assertFalse(tmp_path.exists(), "temp must be cleaned up even on failure")

    def test_denoise_audio_routes_to_preprocess(self):
        tmpdir = Path(tempfile.mkdtemp())
        src = tmpdir / "in.wav"
        _make_wav(src)
        with mock.patch.object(
            denoise,
            "preprocess",
            return_value=(tmpdir / "cleaned.wav", tmpdir / "cleaned.wav"),
        ) as pre:
            path, tmp = transcriber._denoise_audio(src)
        pre.assert_called_once_with(src)
        self.assertEqual(path, tmpdir / "cleaned.wav")
        self.assertEqual(tmp, tmpdir / "cleaned.wav")


class DenoiseSettingsTest(unittest.TestCase):
    def test_default_off_and_settable(self):
        self.assertFalse(settings.get("denoise"))
        self.assertIn("denoise", settings.scalar_keys())

    def test_config_cli_roundtrip(self):
        runner = CliRunner()
        result = runner.invoke(cli.cli, ["config", "set", "denoise", "true"])
        self.assertEqual(result.exit_code, 0, result.output)
        result = runner.invoke(cli.cli, ["config", "get", "denoise"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "true")

    def test_status_shows_filter_state(self):
        runner = CliRunner()
        settings.put("denoise", False)
        off = runner.invoke(cli.cli, ["status"])
        self.assertEqual(off.exit_code, 0, off.output)
        self.assertIn("Noise filter", off.output)
        self.assertIn("off (opt-in)", off.output)
        settings.put("denoise", True)
        on = runner.invoke(cli.cli, ["status"])
        self.assertEqual(on.exit_code, 0, on.output)
        self.assertIn("Noise filter", on.output)
        self.assertIn("denoise local audio", on.output)
        settings.put("denoise", False)


class VADGateTest(unittest.TestCase):
    """VAD gating in denoise: skip denoise for near-silent audio."""

    def test_has_speech_returns_true_for_signal(self):
        import numpy as np

        # A strong sine wave
        y = np.sin(np.linspace(0, 4 * np.pi, 48000)).astype(np.float32) * 0.5
        self.assertTrue(denoise._has_speech(y, 16000))

    def test_has_speech_returns_false_for_silence(self):
        import numpy as np

        # Pure silence
        y = np.zeros(48000, dtype=np.float32)
        self.assertFalse(denoise._has_speech(y, 16000))

    def test_has_speech_returns_false_for_empty(self):
        import numpy as np

        self.assertFalse(denoise._has_speech(np.array([], dtype=np.float32), 16000))


if __name__ == "__main__":
    unittest.main()
