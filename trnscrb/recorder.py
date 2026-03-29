"""Audio capture using sounddevice.

Supports mic-only or BlackHole 2ch (system audio) as input.
Records at 16 kHz mono — suitable for local ASR backends used by trnscrb.
"""
import threading
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

from trnscrb.log import get_logger

_log = get_logger("trnscrb.recorder")

SAMPLE_RATE = 16_000  # fixed capture rate for local transcription backends


class Recorder:
    def __init__(self, device: int | str | None = None):
        # device=None → system default input
        self.device = device
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    # ── public ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._frames = []
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

        with self._lock:
            frames = list(self._frames)

        if not frames:
            _log.warning("Recording stopped with no frames captured")
            return None

        audio = np.concatenate(frames, axis=0).flatten()
        audio_int16 = (audio * 32_767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)
            out = Path(tmp.name)
            _log.info("Recording stopped: %d frames, saved to %s", len(frames), out)
            return out
        finally:
            tmp.close()

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── helpers ─────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        try:
            if self._recording:
                with self._lock:
                    self._frames.append(indata.copy())
        except Exception:
            pass

    # ── class-level utilities ────────────────────────────────────────────────

    @staticmethod
    def find_blackhole_device() -> int | None:
        for i, dev in enumerate(sd.query_devices()):
            if "BlackHole" in dev["name"] and dev["max_input_channels"] > 0:
                return i
        return None

    @staticmethod
    def list_input_devices() -> list[dict]:
        return [
            {"index": i, "name": dev["name"], "channels": dev["max_input_channels"]}
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
