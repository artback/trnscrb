"""Tests for Recorder streaming-to-file, WAV header, callback, and edge cases.

All tests run without real audio devices — sounddevice.InputStream is mocked.
"""

import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from trnscrb.recorder import SAMPLE_RATE, Recorder, _wav_header


class WavHeaderTest(unittest.TestCase):
    """Verify _wav_header produces valid 44-byte PCM WAV headers."""

    def test_header_length_is_44_bytes(self):
        hdr = _wav_header(16000, 1, 0)
        self.assertEqual(len(hdr), 44)

    def test_header_riff_and_wave_tags(self):
        hdr = _wav_header(16000, 1, 1000)
        self.assertEqual(hdr[0:4], b"RIFF")
        self.assertEqual(hdr[8:12], b"WAVE")
        self.assertEqual(hdr[12:16], b"fmt ")
        self.assertEqual(hdr[36:40], b"data")

    def test_header_riff_size_field(self):
        data_size = 32000
        hdr = _wav_header(16000, 1, data_size)
        riff_size = struct.unpack_from("<I", hdr, 4)[0]
        self.assertEqual(riff_size, 36 + data_size)

    def test_header_data_size_field(self):
        data_size = 48000
        hdr = _wav_header(16000, 1, data_size)
        written_data_size = struct.unpack_from("<I", hdr, 40)[0]
        self.assertEqual(written_data_size, data_size)

    def test_header_pcm_format(self):
        hdr = _wav_header(16000, 1, 0)
        audio_format = struct.unpack_from("<H", hdr, 20)[0]
        self.assertEqual(audio_format, 1, "Should be PCM format (1)")

    def test_header_sample_rate(self):
        hdr = _wav_header(44100, 2, 0)
        sr = struct.unpack_from("<I", hdr, 24)[0]
        self.assertEqual(sr, 44100)

    def test_header_channels(self):
        hdr = _wav_header(16000, 2, 0)
        ch = struct.unpack_from("<H", hdr, 22)[0]
        self.assertEqual(ch, 2)

    def test_header_byte_rate(self):
        hdr = _wav_header(16000, 1, 0)
        byte_rate = struct.unpack_from("<I", hdr, 28)[0]
        # 16000 * 1 * 16/8 = 32000
        self.assertEqual(byte_rate, 32000)

    def test_header_block_align(self):
        hdr = _wav_header(16000, 2, 0)
        block_align = struct.unpack_from("<H", hdr, 32)[0]
        # 2 channels * 16 bits / 8 = 4
        self.assertEqual(block_align, 4)

    def test_header_bits_per_sample(self):
        hdr = _wav_header(16000, 1, 0)
        bits = struct.unpack_from("<H", hdr, 34)[0]
        self.assertEqual(bits, 16)


class CallbackTest(unittest.TestCase):
    """Test that _callback writes int16 data to the temp file correctly."""

    def _make_recorder(self):
        """Create a Recorder with a real temp file but no audio stream."""
        rec = Recorder(device=None)
        rec._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        rec._tmpfile.write(b"\x00" * 44)  # placeholder header
        rec._frame_count = 0
        rec._recording = True
        return rec

    def tearDown(self):
        # Clean up any leftover temp files
        if hasattr(self, "_paths"):
            for p in self._paths:
                Path(p).unlink(missing_ok=True)

    def test_callback_writes_int16_data(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]

        # Simulate a single-channel block of 10 samples, all 0.5
        indata = np.full((10, 1), 0.5, dtype=np.float32)
        rec._callback(indata, 10, None, None)

        self.assertEqual(rec._frame_count, 10)

        # Read back what was written (skip the 44-byte placeholder)
        rec._tmpfile.seek(44)
        raw = rec._tmpfile.read()
        rec._tmpfile.close()
        samples = np.frombuffer(raw, dtype=np.int16)
        expected = np.int16(0.5 * 32767)
        np.testing.assert_array_equal(samples, np.full(10, expected, dtype=np.int16))

    def test_callback_accumulates_frame_count(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]

        block1 = np.zeros((100, 1), dtype=np.float32)
        block2 = np.zeros((200, 1), dtype=np.float32)
        rec._callback(block1, 100, None, None)
        rec._callback(block2, 200, None, None)

        self.assertEqual(rec._frame_count, 300)
        rec._tmpfile.close()

    def test_callback_noop_when_not_recording(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        rec._recording = False

        indata = np.ones((50, 1), dtype=np.float32)
        rec._callback(indata, 50, None, None)

        self.assertEqual(rec._frame_count, 0)
        rec._tmpfile.close()

    def test_callback_clips_out_of_range_values(self):
        """Values > 1.0 and < -1.0 should be clipped before int16 conversion."""
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]

        indata = np.array([[2.0], [-3.0], [0.0]], dtype=np.float32)
        rec._callback(indata, 3, None, None)

        rec._tmpfile.seek(44)
        raw = rec._tmpfile.read()
        rec._tmpfile.close()
        samples = np.frombuffer(raw, dtype=np.int16)

        # 2.0 clipped to 1.0 -> 32767, -3.0 clipped to -1.0 -> -32767, 0.0 -> 0
        self.assertEqual(samples[0], 32767)
        self.assertEqual(samples[1], -32767)
        self.assertEqual(samples[2], 0)


class RecorderFullCycleTest(unittest.TestCase):
    """Test start/stop producing a valid WAV file readable by the wave module."""

    def _patch_stream(self):
        """Patch sd.InputStream so start() doesn't need a real device."""
        mock_stream = MagicMock()
        return patch("trnscrb.recorder.sd.InputStream", return_value=mock_stream)

    def test_start_stop_produces_valid_wav(self):
        with self._patch_stream():
            rec = Recorder(device=None, system_audio=False)
            rec.start()

            # Simulate audio data arriving via the callback
            for _ in range(10):
                block = np.random.uniform(-0.5, 0.5, (1024, 1)).astype(np.float32)
                rec._callback(block, 1024, None, None)

            path = rec.stop()

        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

        # Validate with the standard library wave module
        with wave.open(str(path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)  # 16-bit = 2 bytes
            self.assertEqual(wf.getframerate(), SAMPLE_RATE)
            self.assertEqual(wf.getnframes(), 10 * 1024)
            # Read all frames to confirm no corruption
            data = wf.readframes(wf.getnframes())
            self.assertEqual(len(data), 10 * 1024 * 2)

        path.unlink(missing_ok=True)

    def test_stop_with_no_frames_returns_none_and_cleans_up(self):
        with self._patch_stream():
            rec = Recorder(device=None, system_audio=False)
            rec.start()
            tmp_path = Path(rec._tmpfile.name)

            # Stop immediately — no callback data
            result = rec.stop()

        self.assertIsNone(result)
        self.assertFalse(tmp_path.exists(), "Temp file should be deleted on zero frames")

    def test_stop_when_never_started_returns_none(self):
        rec = Recorder(device=None)
        result = rec.stop()
        self.assertIsNone(result)

    def test_stop_clears_stream_reference(self):
        with self._patch_stream():
            rec = Recorder(device=None, system_audio=False)
            rec.start()
            self.assertIsNotNone(rec._stream)

            # Inject at least one frame so stop() writes the file
            block = np.zeros((100, 1), dtype=np.float32)
            rec._callback(block, 100, None, None)
            path = rec.stop()

        self.assertIsNone(rec._stream)
        self.assertIsNone(rec._tmpfile)
        if path:
            path.unlink(missing_ok=True)

    def test_stop_handles_stream_close_exception(self):
        """stop() should not raise even if stream.close() throws."""
        mock_stream = MagicMock()
        mock_stream.close.side_effect = RuntimeError("device error")
        with patch("trnscrb.recorder.sd.InputStream", return_value=mock_stream):
            rec = Recorder(device=None, system_audio=False)
            rec.start()

            block = np.ones((50, 1), dtype=np.float32)
            rec._callback(block, 50, None, None)

            # Should not raise
            path = rec.stop()

        self.assertIsNotNone(path)
        if path:
            path.unlink(missing_ok=True)


class NpClipEdgeCaseTest(unittest.TestCase):
    """Verify np.clip behavior with extreme float32 values matches expectations."""

    def test_clip_positive_overflow(self):
        arr = np.array([[100.0]], dtype=np.float32)
        clipped = np.clip(arr, -1.0, 1.0)
        result = (clipped * 32767).astype(np.int16)
        self.assertEqual(result[0, 0], 32767)

    def test_clip_negative_overflow(self):
        arr = np.array([[-999.0]], dtype=np.float32)
        clipped = np.clip(arr, -1.0, 1.0)
        result = (clipped * 32767).astype(np.int16)
        self.assertEqual(result[0, 0], -32767)

    def test_clip_nan_passthrough(self):
        """NaN clipped to [-1, 1] remains NaN; converting NaN to int16 is
        implementation-defined but should not raise."""
        arr = np.array([[float("nan")]], dtype=np.float32)
        clipped = np.clip(arr, -1.0, 1.0)
        # Just verify it doesn't crash
        _ = (clipped * 32767).astype(np.int16)

    def test_clip_inf(self):
        arr = np.array([[float("inf")], [float("-inf")]], dtype=np.float32)
        clipped = np.clip(arr, -1.0, 1.0)
        result = (clipped * 32767).astype(np.int16)
        self.assertEqual(result[0, 0], 32767)
        self.assertEqual(result[1, 0], -32767)


class CallbackExceptionTest(unittest.TestCase):
    """Test that exceptions in _callback are handled correctly."""

    def _make_recorder(self):
        rec = Recorder(device=None)
        rec._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        rec._tmpfile.write(b"\x00" * 44)
        rec._frame_count = 0
        rec._recording = True
        return rec

    def test_oserror_in_write_sets_recording_false(self):
        """OSError in _tmpfile.write() should set _recording=False."""
        rec = self._make_recorder()
        tmp_path = rec._tmpfile.name

        with patch.object(rec._tmpfile, "write", side_effect=OSError("disk full")):
            indata = np.ones((100, 1), dtype=np.float32)
            rec._callback(indata, 100, None, None)

        self.assertFalse(rec._recording)
        rec._tmpfile.close()
        Path(tmp_path).unlink(missing_ok=True)

    def test_other_exception_does_not_crash(self):
        """Non-OSError exceptions should be caught and not crash the callback."""
        rec = self._make_recorder()
        tmp_path = rec._tmpfile.name

        with patch.object(rec._tmpfile, "write", side_effect=ValueError("unexpected")):
            indata = np.ones((50, 1), dtype=np.float32)
            # Should not raise
            rec._callback(indata, 50, None, None)

        # _recording should still be True (only OSError sets it False)
        self.assertTrue(rec._recording)
        rec._tmpfile.close()
        Path(tmp_path).unlink(missing_ok=True)

    def test_oserror_preserves_frame_count(self):
        """After OSError, frame_count should not be incremented."""
        rec = self._make_recorder()
        tmp_path = rec._tmpfile.name

        # Write one good block
        indata = np.zeros((100, 1), dtype=np.float32)
        rec._callback(indata, 100, None, None)
        self.assertEqual(rec._frame_count, 100)

        # Now cause OSError — frame_count should stay at 100
        rec._recording = True  # re-enable
        with patch.object(rec._tmpfile, "write", side_effect=OSError("fail")):
            rec._callback(indata, 100, None, None)

        self.assertEqual(rec._frame_count, 100)
        rec._tmpfile.close()
        Path(tmp_path).unlink(missing_ok=True)


class SystemAudioMixTest(unittest.TestCase):
    """Test mixing of ScreenCaptureKit system-audio chunks into the mic stream."""

    def _make_recorder(self):
        rec = Recorder(device=None)
        rec._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        rec._tmpfile.write(b"\x00" * 44)
        rec._frame_count = 0
        rec._recording = True
        rec._system_audio_active = True
        return rec

    def tearDown(self):
        if hasattr(self, "_paths"):
            for p in self._paths:
                Path(p).unlink(missing_ok=True)

    def test_pull_returns_none_when_inactive(self):
        rec = Recorder(device=None)
        rec._on_system_chunk(np.ones(100, dtype=np.float32))
        self.assertIsNone(rec._pull_system_frames(100))

    def test_pull_returns_none_when_buffer_empty(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        self.assertIsNone(rec._pull_system_frames(100))
        rec._tmpfile.close()

    def test_pull_pads_with_zeros_when_short(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        rec._on_system_chunk(np.full(30, 0.25, dtype=np.float32))

        out = rec._pull_system_frames(100)
        self.assertEqual(len(out), 100)
        np.testing.assert_array_equal(out[:30], np.full(30, 0.25, dtype=np.float32))
        np.testing.assert_array_equal(out[30:], np.zeros(70, dtype=np.float32))
        rec._tmpfile.close()

    def test_pull_spans_chunks_and_keeps_remainder(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        rec._on_system_chunk(np.full(60, 0.1, dtype=np.float32))
        rec._on_system_chunk(np.full(60, 0.2, dtype=np.float32))

        out = rec._pull_system_frames(100)
        np.testing.assert_array_equal(out[:60], np.full(60, 0.1, dtype=np.float32))
        np.testing.assert_allclose(out[60:], np.full(40, 0.2, dtype=np.float32))
        self.assertEqual(rec._sys_frames, 20)

        rest = rec._pull_system_frames(100)
        np.testing.assert_allclose(rest[:20], np.full(20, 0.2, dtype=np.float32))
        self.assertEqual(rec._sys_frames, 0)
        rec._tmpfile.close()

    def test_buffer_cap_drops_oldest_chunks(self):
        from trnscrb.recorder import _SYS_BUFFER_MAX_FRAMES

        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second per chunk
        for _ in range(40):
            rec._on_system_chunk(chunk)

        self.assertLessEqual(rec._sys_frames, _SYS_BUFFER_MAX_FRAMES)
        rec._tmpfile.close()

    def test_callback_mixes_system_audio_into_mic(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        rec._on_system_chunk(np.full(10, 0.25, dtype=np.float32))

        indata = np.full((10, 1), 0.25, dtype=np.float32)
        rec._callback(indata, 10, None, None)

        rec._tmpfile.seek(44)
        raw = rec._tmpfile.read()
        rec._tmpfile.close()
        samples = np.frombuffer(raw, dtype=np.int16)
        expected = np.int16(0.5 * 32767)
        np.testing.assert_array_equal(samples, np.full(10, expected, dtype=np.int16))

    def test_mic_gain_boosts_quiet_mic(self):
        """Quiet mic vs loud system audio → mic boosted toward system level."""
        rec = Recorder(device=None)
        mic = np.full(1024, 0.05, dtype=np.float32)
        system = np.full(1024, 0.5, dtype=np.float32)
        for _ in range(100):  # let the loudness EMAs converge
            gain = rec._mic_gain(mic, system)
        self.assertGreater(gain, 1.0)

    def test_mic_gain_clamped_at_max(self):
        from trnscrb.recorder import _MIC_GAIN_MAX

        rec = Recorder(device=None)
        mic = np.full(1024, 0.01, dtype=np.float32)  # 50x quieter than system
        system = np.full(1024, 0.5, dtype=np.float32)
        for _ in range(100):
            gain = rec._mic_gain(mic, system)
        self.assertLessEqual(gain, _MIC_GAIN_MAX)

    def test_mic_gain_never_attenuates(self):
        """Loud mic vs quiet system audio → gain stays 1.0, never < 1."""
        rec = Recorder(device=None)
        mic = np.full(1024, 0.5, dtype=np.float32)
        system = np.full(1024, 0.05, dtype=np.float32)
        for _ in range(100):
            gain = rec._mic_gain(mic, system)
        self.assertEqual(gain, 1.0)

    def test_mic_gain_unity_on_silence(self):
        rec = Recorder(device=None)
        silence = np.zeros(1024, dtype=np.float32)
        self.assertEqual(rec._mic_gain(silence, silence), 1.0)

    def test_callback_clips_mixed_signal(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        rec._on_system_chunk(np.full(5, 0.8, dtype=np.float32))

        indata = np.full((5, 1), 0.8, dtype=np.float32)  # 1.6 combined → clipped
        rec._callback(indata, 5, None, None)

        rec._tmpfile.seek(44)
        raw = rec._tmpfile.read()
        rec._tmpfile.close()
        samples = np.frombuffer(raw, dtype=np.int16)
        np.testing.assert_array_equal(samples, np.full(5, 32767, dtype=np.int16))


class MicStreamRestartTest(unittest.TestCase):
    """Test mic-stream recovery after a device change (stalled callback)."""

    def test_restart_replaces_stream(self):
        old_stream = MagicMock()
        new_stream = MagicMock()
        with patch("trnscrb.recorder.sd.InputStream", return_value=new_stream):
            rec = Recorder(device=None)
            rec._recording = True
            rec._stream = old_stream
            rec._restart_mic_stream()

        old_stream.stop.assert_called_once()
        old_stream.close.assert_called_once()
        new_stream.start.assert_called_once()
        self.assertIs(rec._stream, new_stream)

    def test_restart_survives_old_stream_close_error(self):
        old_stream = MagicMock()
        old_stream.close.side_effect = RuntimeError("device gone")
        new_stream = MagicMock()
        with patch("trnscrb.recorder.sd.InputStream", return_value=new_stream):
            rec = Recorder(device=None)
            rec._recording = True
            rec._stream = old_stream
            rec._restart_mic_stream()

        self.assertIs(rec._stream, new_stream)

    def test_restart_after_stop_closes_new_stream(self):
        """If stop() raced the restart, the fresh stream must not leak."""
        new_stream = MagicMock()
        with patch("trnscrb.recorder.sd.InputStream", return_value=new_stream):
            rec = Recorder(device=None)
            rec._recording = False  # stop() already ran
            rec._stream = MagicMock()
            rec._restart_mic_stream()

        new_stream.stop.assert_called_once()
        new_stream.close.assert_called_once()
        self.assertIsNone(rec._stream)

    def test_restart_failure_leaves_no_stream_and_backs_off(self):
        with patch("trnscrb.recorder.sd.InputStream", side_effect=RuntimeError("no device")):
            rec = Recorder(device=None)
            rec._recording = True
            rec._stream = MagicMock()
            before = rec._last_mic_frame
            rec._restart_mic_stream()

        self.assertIsNone(rec._stream)
        self.assertGreater(rec._last_mic_frame, before)


if __name__ == "__main__":
    unittest.main()
