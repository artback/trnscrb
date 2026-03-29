"""Audio capture using sounddevice.

Supports mic-only or BlackHole 2ch (system audio) as input.
Records at 16 kHz mono — suitable for local ASR backends used by trnscrb.
"""
import os
import struct
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from trnscrb.log import get_logger

_log = get_logger("trnscrb.recorder")

SAMPLE_RATE = 16_000  # fixed capture rate for local transcription backends

_STALE_AGE_SECS = 3600  # 1 hour


def cleanup_stale_temp_files() -> None:
    """Remove orphaned trnscrb WAV files from previous crashes."""
    tmp_dir = Path(tempfile.gettempdir())
    now = time.time()
    for p in tmp_dir.glob("tmp*.wav"):
        try:
            if now - p.stat().st_mtime > _STALE_AGE_SECS:
                p.unlink()
                _log.debug("Deleted stale temp file: %s", p)
        except Exception:
            pass


def _wav_header(sample_rate: int, channels: int, data_size: int) -> bytes:
    """Build a 44-byte PCM WAV header."""
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,             # chunk size
        1,              # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )


class Recorder:
    def __init__(self, device: int | str | None = None):
        # device=None → system default input
        self.device = device
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._tmpfile = None
        self._frame_count = 0

    # ── public ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        # Write a placeholder WAV header (44 bytes); finalized in stop().
        self._tmpfile.write(b"\x00" * 44)
        self._frame_count = 0
        self._recording = True
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()
        _log.info("Recording started (device=%s)", self.device)

    def stop(self) -> Path | None:
        """Stop recording and return the path to a temporary WAV file."""
        self._recording = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                _log.warning("Stream cleanup failed", exc_info=True)
            finally:
                self._stream = None

        # Hold the lock for all file operations so no late callback
        # can write while we finalize the header.
        with self._lock:
            if not self._tmpfile:
                _log.warning("Recording stopped with no temp file")
                return None

            frame_count = self._frame_count

            if frame_count == 0:
                _log.warning("Recording stopped with no frames captured")
                self._tmpfile.close()
                Path(self._tmpfile.name).unlink(missing_ok=True)
                self._tmpfile = None
                return None

            # Finalize WAV header now that we know the data size.
            data_size = frame_count * 2  # int16 = 2 bytes per sample
            self._tmpfile.seek(0)
            self._tmpfile.write(_wav_header(SAMPLE_RATE, 1, data_size))
            self._tmpfile.close()

            out = Path(self._tmpfile.name)
            self._tmpfile = None

        _log.info("Recording stopped: %d samples, saved to %s", frame_count, out)
        return out

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── helpers ─────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        try:
            if self._recording and self._tmpfile:
                audio_int16 = np.clip(indata, -1.0, 1.0)
                audio_int16 = (audio_int16 * 32_767).astype(np.int16)
                raw = audio_int16.tobytes()
                with self._lock:
                    self._tmpfile.write(raw)
                    self._frame_count += len(audio_int16)
        except OSError as e:
            _log.error("Audio write failed (disk full?): %s", e)
            self._recording = False
        except Exception:
            _log.debug("Callback error", exc_info=True)

    # ── class-level utilities ────────────────────────────────────────────────

    @staticmethod
    def find_blackhole_device() -> int | None:
        for i, dev in enumerate(sd.query_devices()):
            if "BlackHole" in dev["name"] and dev["max_input_channels"] > 0:
                _log.debug("BlackHole device found at index %d", i)
                return i
        _log.debug("BlackHole device not found")
        return None

    @staticmethod
    def list_input_devices() -> list[dict]:
        return [
            {"index": i, "name": dev["name"], "channels": dev["max_input_channels"]}
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
