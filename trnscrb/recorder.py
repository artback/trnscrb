"""Audio capture using sounddevice + ScreenCaptureKit.

The microphone is captured with sounddevice; system audio (the other meeting
participants) is captured natively via ScreenCaptureKit and mixed in — no
BlackHole driver or Multi-Output Device needed.
Records at 16 kHz mono — suitable for local ASR backends used by trnscrb.
"""

import shutil
import struct
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from trnscrb.log import get_logger

_log = get_logger("trnscrb.recorder")

SAMPLE_RATE = 16_000  # fixed capture rate for local transcription backends

_STALE_AGE_SECS = 3600  # 1 hour

# Bound on buffered system audio awaiting mix-in. The mic clock is the master;
# clock drift between the two capture paths is tiny (~ms/hour), so this cap
# only matters if the mic stream stalls.
_SYS_BUFFER_MAX_FRAMES = SAMPLE_RATE * 30

# Mic level balancing: the mic is often much quieter than system playback,
# drowning the user's voice in the mix. Track loudness of both sources and
# boost the mic toward the system-audio level, never above this factor.
_MIC_GAIN_MAX = 4.0
_LOUDNESS_ALPHA = 0.05  # EMA weight per 64 ms block ≈ few-second horizon
_NOISE_FLOOR = 1e-6  # mean-square ≈ -60 dBFS; blocks below this don't update

# If the mic stream delivers nothing for this long (default input device
# changed, e.g. AirPods connected mid-meeting), reopen it.
_MIC_STALL_SECS = 3.0


# A killed recording leaves a temp WAV with a placeholder header. Anything
# longer than this is worth rescuing; shorter is noise from a false start.
_MIN_RECOVERABLE_SECS = 20
# Don't touch a file that a recording might still be writing to.
_ORPHAN_QUIET_SECS = 120


def finalize_wav_header(path: Path) -> int:
    """Write a correct header on a WAV left with the placeholder. Returns frames.

    Recording streams to disk with a zeroed 44-byte header that is only
    filled in on stop, so a process killed mid-meeting leaves an unreadable
    file whose audio is nevertheless fully intact.
    """
    size = path.stat().st_size
    data_size = max(0, size - 44)
    with open(path, "r+b") as f:
        f.write(_wav_header(SAMPLE_RATE, 1, data_size))
    return data_size // 2


def recover_orphaned_recordings(notes_dir: Path | None = None) -> list[Path]:
    """Rescue recordings left behind by a crash, kill, or logout.

    Called at startup. Orphaned temp WAVs are repaired and moved into the
    notes folder rather than deleted — audio is the one thing that cannot be
    regenerated. Transcribe them later with `trnscrb transcribe <file>`.
    """
    from trnscrb import storage

    notes_dir = notes_dir or storage.ensure_notes_dir()
    recovered: list[Path] = []
    tmp_dir = Path(tempfile.gettempdir())
    now = time.time()
    min_bytes = 44 + _MIN_RECOVERABLE_SECS * SAMPLE_RATE * 2

    for p in sorted(tmp_dir.glob("tmp*.wav")):
        try:
            stat = p.stat()
            if now - stat.st_mtime < _ORPHAN_QUIET_SECS:
                continue  # possibly still being written
            if stat.st_size < min_bytes:
                p.unlink()  # too short to be a meeting
                continue
            frames = finalize_wav_header(p)
            started = datetime.fromtimestamp(stat.st_mtime)
            dest = notes_dir / f"{started:%Y-%m-%d_%H-%M-%S}_recovered-recording.wav"
            shutil.move(str(p), dest)
            recovered.append(dest)
            _log.warning(
                "Recovered interrupted recording: %s (%.1f min) — transcribe with "
                "`trnscrb transcribe %s`",
                dest.name,
                frames / SAMPLE_RATE / 60,
                dest,
            )
        except Exception:
            _log.debug("Could not recover %s", p, exc_info=True)
    return recovered


def cleanup_stale_temp_files() -> None:
    """Backwards-compatible alias — recovers rather than deletes."""
    recover_orphaned_recordings()


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
        16,  # chunk size
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )


class Recorder:
    def __init__(self, device: int | str | None = None, system_audio: bool = True):
        # device=None → system default input (microphone)
        self.device = device
        self.capture_system_audio = system_audio
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._tmpfile = None
        self._frame_count = 0
        self._system_capture = None
        self._system_audio_active = False
        self._sys_lock = threading.Lock()
        self._sys_chunks: deque = deque()  # float32 mono @16 kHz awaiting mix-in
        self._sys_frames = 0
        self._mic_loudness = 0.0  # EMA of mic mean-square (speech-gated)
        self._sys_loudness = 0.0  # EMA of system-audio mean-square
        self._last_mic_frame = 0.0  # time.monotonic() of last mic callback
        self._watchdog: threading.Thread | None = None
        # Per-block energy timeline of both source streams, recorded before
        # mixing — lets the transcriber attribute segments to "Me" (mic) vs
        # "Them" (system audio) without any diarization model.
        self._attr_lock = threading.Lock()
        self._attr_offsets: list[int] = []  # frame offset of each block
        self._attr_mic: list[float] = []  # mic mean-square per block
        self._attr_sys: list[float] = []  # system-audio mean-square per block

    # ── public ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        # Write a placeholder WAV header (44 bytes); finalized in stop().
        self._tmpfile.write(b"\x00" * 44)
        self._frame_count = 0
        with self._attr_lock:
            self._attr_offsets.clear()
            self._attr_mic.clear()
            self._attr_sys.clear()
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
        self._last_mic_frame = time.monotonic()
        self._watchdog = threading.Thread(target=self._watch_mic_stream, daemon=True)
        self._watchdog.start()
        self._start_system_audio()
        _log.info(
            "Recording started (device=%s, system_audio=%s)",
            self.device,
            self._system_audio_active,
        )

    def stop(self) -> Path | None:
        """Stop recording and return the path to a temporary WAV file."""
        self._recording = False
        if self._system_capture:
            try:
                self._system_capture.stop()
            except Exception:
                _log.warning("System audio cleanup failed", exc_info=True)
            finally:
                self._system_capture = None
                self._system_audio_active = False
        with self._sys_lock:
            self._sys_chunks.clear()
            self._sys_frames = 0
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

    def snapshot(self) -> Path | None:
        """Create a valid WAV copy of audio captured so far (non-destructive)."""
        result = self.snapshot_since(0)
        return result[0] if result else None

    def snapshot_since(self, start_frame: int) -> tuple[Path, int] | None:
        """WAV copy of audio from ``start_frame`` up to now (non-destructive).

        Returns ``(path, end_frame)`` so callers can transcribe incrementally —
        pass the returned ``end_frame`` as the next call's ``start_frame``.
        None if nothing new was captured.
        """
        with self._lock:
            if not self._tmpfile or self._frame_count <= start_frame:
                return None
            # Flush pending writes
            self._tmpfile.flush()
            src = self._tmpfile.name
            end_frame = self._frame_count

        n_bytes = (end_frame - start_frame) * 2  # int16 samples
        snap = Path(src).with_suffix(".snap.wav")
        with open(src, "rb") as f_in, open(snap, "wb") as f_out:
            f_out.write(_wav_header(SAMPLE_RATE, 1, n_bytes))
            f_in.seek(44 + start_frame * 2)
            remaining = n_bytes
            while remaining > 0:
                chunk = f_in.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f_out.write(chunk)
                remaining -= len(chunk)
        return snap, end_frame

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def system_audio_active(self) -> bool:
        """True while system audio is being captured alongside the mic."""
        return self._system_audio_active

    def attribution_timeline(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-block (frame_offset, mic_energy, system_energy) arrays.

        Snapshot of the dual-stream energy timeline used to attribute
        transcript segments to "Me" (mic) vs "Them" (system audio).
        """
        with self._attr_lock:
            return (
                np.array(self._attr_offsets, dtype=np.int64),
                np.array(self._attr_mic, dtype=np.float32),
                np.array(self._attr_sys, dtype=np.float32),
            )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _start_system_audio(self) -> None:
        """Try to add ScreenCaptureKit system audio; fall back to mic-only."""
        self._system_audio_active = False
        if not self.capture_system_audio:
            return
        from trnscrb.system_audio import SystemAudioCapture

        if not SystemAudioCapture.is_supported():
            _log.info("System audio capture not supported here; recording mic only")
            return
        capture = SystemAudioCapture(self._on_system_chunk)
        try:
            capture.start()
        except Exception as e:
            _log.warning("System audio capture unavailable (%s); recording mic only", e)
            return
        self._system_capture = capture
        self._system_audio_active = True

    def _on_system_chunk(self, chunk: np.ndarray) -> None:
        with self._sys_lock:
            self._sys_chunks.append(chunk)
            self._sys_frames += len(chunk)
            while self._sys_frames > _SYS_BUFFER_MAX_FRAMES and len(self._sys_chunks) > 1:
                dropped = self._sys_chunks.popleft()
                self._sys_frames -= len(dropped)

    def _pull_system_frames(self, n: int) -> np.ndarray | None:
        """Take up to n frames of buffered system audio; None if none buffered."""
        if not self._system_audio_active:
            return None
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._sys_lock:
            while filled < n and self._sys_chunks:
                chunk = self._sys_chunks[0]
                take = min(n - filled, len(chunk))
                out[filled : filled + take] = chunk[:take]
                if take == len(chunk):
                    self._sys_chunks.popleft()
                else:
                    self._sys_chunks[0] = chunk[take:]
                self._sys_frames -= take
                filled += take
        return out if filled else None

    def _mic_gain(self, mic: np.ndarray, system: np.ndarray) -> float:
        """Boost factor for the mic so it stays audible next to system audio."""
        ms_mic = float(np.mean(np.square(mic)))
        ms_sys = float(np.mean(np.square(system)))
        if ms_mic > _NOISE_FLOOR:
            self._mic_loudness += _LOUDNESS_ALPHA * (ms_mic - self._mic_loudness)
        if ms_sys > _NOISE_FLOOR:
            self._sys_loudness += _LOUDNESS_ALPHA * (ms_sys - self._sys_loudness)
        if self._mic_loudness > _NOISE_FLOOR and self._sys_loudness > _NOISE_FLOOR:
            gain = (self._sys_loudness / self._mic_loudness) ** 0.5
            return min(max(gain, 1.0), _MIC_GAIN_MAX)
        return 1.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            _log.warning("Audio stream status: %s", status)
        self._last_mic_frame = time.monotonic()
        try:
            if self._recording and self._tmpfile:
                block_offset = self._frame_count  # only this thread mutates it
                mic = indata[:, 0]
                system = self._pull_system_frames(len(mic))
                with self._attr_lock:
                    self._attr_offsets.append(block_offset)
                    self._attr_mic.append(float(np.mean(np.square(mic))))
                    self._attr_sys.append(
                        float(np.mean(np.square(system))) if system is not None else 0.0
                    )
                if system is not None:
                    mixed = mic * self._mic_gain(mic, system) + system
                else:
                    mixed = mic
                audio_int16 = np.clip(mixed, -1.0, 1.0)
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

    def _watch_mic_stream(self) -> None:
        """Reopen the mic stream if it stops delivering (input device changed).

        When the default input switches mid-recording (AirPods connect or
        disconnect), CoreAudio keeps the old stream alive but the callback
        simply stops firing — no error is reported, so a stall timer is the
        only reliable signal.
        """
        while self._recording:
            time.sleep(1.0)
            if not self._recording:
                return
            if time.monotonic() - self._last_mic_frame > _MIC_STALL_SECS:
                self._restart_mic_stream()

    def _restart_mic_stream(self) -> None:
        _log.warning("Mic stream stalled — reopening input stream (device change?)")
        old, self._stream = self._stream, None
        if old is not None:
            try:
                old.stop()
                old.close()
            except Exception:
                _log.debug("Stalled stream cleanup failed", exc_info=True)
        try:
            stream = sd.InputStream(
                device=self.device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=1024,
            )
            stream.start()
            if not self._recording:  # stop() raced us — don't leak the stream
                stream.stop()
                stream.close()
                return
            self._stream = stream
            _log.info("Mic stream reopened")
        except Exception as e:
            _log.warning("Mic stream reopen failed (retrying in %ss): %s", _MIC_STALL_SECS, e)
        finally:
            # Reset the stall timer so failed reopens back off instead of spinning.
            self._last_mic_frame = time.monotonic()

    # ── class-level utilities ────────────────────────────────────────────────

    @staticmethod
    def system_audio_available() -> bool:
        """True if ScreenCaptureKit system audio capture can run right now."""
        from trnscrb.system_audio import SystemAudioCapture

        return SystemAudioCapture.is_supported() and SystemAudioCapture.has_permission()

    @staticmethod
    def list_input_devices() -> list[dict]:
        return [
            {"index": i, "name": dev["name"], "channels": dev["max_input_channels"]}
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
