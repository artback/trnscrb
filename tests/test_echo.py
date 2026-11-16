"""Tests for echo detection: the mic picking up the call audio.

Synthetic signals only: an identical tone in both streams must score high
after warmup, independent noise in each must not, and a delayed copy of the
system track (up to the recorder's 0.5 s alignment delay) must be found at
its lag. Recorder-level tests drive the low-rate tick directly, mirroring
tests/test_recorder_streaming.py's mocking style (no real audio devices).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from trnscrb.echo import EchoScorer, confident
from trnscrb.recorder import SAMPLE_RATE, Recorder

BLOCK = 1024  # the recorder's callback blocksize


def _tone(n: int, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    """A pure tone at the capture rate; the echo scorer's worst case for
    phase (a real room smears it, a grid lag error cannot)."""
    t = np.arange(n) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _feed(scorer: EchoScorer, mic: np.ndarray, system: np.ndarray) -> float:
    """Feed paired streams in BLOCK-sized pieces; return the final score."""
    last = 0.0
    for i in range(0, len(mic), BLOCK):
        last = scorer.update(mic[i : i + BLOCK], system[i : i + BLOCK])
    return last


class ScorerTest(unittest.TestCase):
    """The sliding-window cross-correlation over synthetic signals."""

    def test_identical_tone_scores_high_after_warmup(self):
        tone = _tone(int(3 * SAMPLE_RATE))
        self.assertGreater(_feed(EchoScorer(), tone, tone.copy()), 0.7)

    def test_independent_noise_scores_low(self):
        rng = np.random.default_rng(42)
        n = int(3 * SAMPLE_RATE)
        mic = rng.normal(0, 0.1, n).astype(np.float32)
        system = rng.normal(0, 0.1, n).astype(np.float32)
        self.assertLess(_feed(EchoScorer(), mic, system), 0.2)

    def test_silent_mic_with_speech_in_system_scores_low(self):
        n = int(3 * SAMPLE_RATE)
        silence = np.zeros(n, dtype=np.float32)
        self.assertLess(_feed(EchoScorer(), silence, _tone(n)), 0.1)

    def test_speech_in_mic_with_silent_system_scores_low(self):
        n = int(3 * SAMPLE_RATE)
        silence = np.zeros(n, dtype=np.float32)
        self.assertLess(_feed(EchoScorer(), _tone(n), silence), 0.1)

    def test_delayed_copy_scores_high(self):
        """A 0.5 s delayed copy of the system track — the recorder's max
        alignment delay, the case an untimestamped helper presents — is
        found at its lag."""
        total = int(5 * SAMPLE_RATE)
        source = _tone(total)
        system = source[total - int(3 * SAMPLE_RATE) :]
        # The mic carries what the system stream played 0.5 s (8000 samples)
        # ago: mic[i] = system[i - 8000].
        mic = source[total - int(3 * SAMPLE_RATE) - 8000 : total - 8000]
        self.assertGreater(_feed(EchoScorer(), mic, system), 0.7)

    def test_warmup_returns_zero(self):
        scorer = EchoScorer()
        self.assertEqual(scorer.update(_tone(BLOCK), _tone(BLOCK).copy()), 0.0)
        self.assertEqual(scorer.score, 0.0)

    def test_reset_clears_buffers_and_score(self):
        tone = _tone(int(3 * SAMPLE_RATE))
        scorer = EchoScorer()
        self.assertGreater(_feed(scorer, tone, tone.copy()), 0.7)
        scorer.reset()
        self.assertEqual(scorer.score, 0.0)
        self.assertEqual(scorer.update(_tone(BLOCK), _tone(BLOCK).copy()), 0.0)


class ConfidentTest(unittest.TestCase):
    """The hysteresis gate: arm at the threshold, disarm only below it."""

    def test_arms_at_threshold(self):
        self.assertTrue(confident([0.5], 0.5))
        self.assertTrue(confident([0.4, 0.55], 0.5))

    def test_never_arms_below_threshold(self):
        self.assertFalse(confident([0.49, 0.45, 0.48], 0.5))

    def test_stays_armed_on_a_dip_above_the_floor(self):
        # 0.45 is below the threshold but above floor 0.4 — still an echo.
        self.assertTrue(confident([0.6, 0.45], 0.5))

    def test_disarms_at_or_below_the_floor(self):
        self.assertFalse(confident([0.6, 0.4], 0.5))
        self.assertFalse(confident([0.6, 0.3], 0.5))

    def test_rearms_after_disarm(self):
        self.assertTrue(confident([0.6, 0.3, 0.9], 0.5))

    def test_wider_hysteresis_holds_longer(self):
        # 0.39 disarms under the default 0.10 floor (0.40) but holds under 0.15.
        self.assertFalse(confident([0.6, 0.39], 0.5, hysteresis=0.10))
        self.assertTrue(confident([0.6, 0.39], 0.5, hysteresis=0.15))


class RecorderEchoTest(unittest.TestCase):
    """The recorder fires exactly one echo warning per recording session.

    The worker thread is not started: the tick it would run is driven
    directly, the same way test_recorder_streaming.py drives _callback.
    """

    def _make_recorder(self) -> Recorder:
        rec = Recorder(device=None, system_audio=True)
        rec._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        rec._tmpfile.write(b"\x00" * 44)
        rec._frame_count = 0
        rec._recording = True
        rec._system_audio_active = True
        rec._echo_enabled = True
        rec._echo_scorer = EchoScorer()
        return rec

    def tearDown(self):
        if hasattr(self, "_paths"):
            for p in self._paths:
                Path(p).unlink(missing_ok=True)

    def _queue_block(self, rec: Recorder, block: np.ndarray) -> None:
        with rec._echo_lock:
            rec._echo_mic.append(block.copy())
            rec._echo_sys.append(block.copy())

    def test_forced_hot_scorer_notifies_exactly_once(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        calls = []
        rec._notify_echo = lambda: calls.append(1)
        rec._echo_scorer.update = lambda mic, system: 0.95  # force hot

        for _ in range(12):
            self._queue_block(rec, np.ones(BLOCK, dtype=np.float32))
            rec._echo_tick()

        self.assertEqual(calls, [1], "the warning must fire exactly once")
        self.assertTrue(rec.echo_detected)
        rec._echo_tick()  # still hot — no second warning
        self.assertEqual(calls, [1])
        rec._tmpfile.close()

    def test_matching_tones_fire_the_warning_once(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        calls = []
        rec._notify_echo = lambda: calls.append(1)

        # ~8 s of audio through the real callback and scorer: warmup plus
        # the echo_window consecutive hot ticks the warning requires.
        tone = _tone(int(8 * SAMPLE_RATE))
        for i in range(0, len(tone), BLOCK * 8):
            for j in range(0, BLOCK * 8, BLOCK):
                block = tone[j : j + BLOCK]
                rec._on_system_chunk(block)
                rec._callback(block.reshape(-1, 1), BLOCK, None, None)
            rec._echo_tick()

        self.assertEqual(calls, [1])
        self.assertTrue(rec.echo_detected)
        rec._tmpfile.close()

    def test_independent_noise_does_not_notify(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        calls = []
        rec._notify_echo = lambda: calls.append(1)

        rng = np.random.default_rng(7)
        for _ in range(12):  # 12 ticks of 0.5 s
            for _ in range(BLOCK * 8 // BLOCK):
                mic = rng.normal(0, 0.1, BLOCK).astype(np.float32)
                system = rng.normal(0, 0.1, BLOCK).astype(np.float32)
                rec._on_system_chunk(system)
                rec._callback(mic.reshape(-1, 1), BLOCK, None, None)
            rec._echo_tick()

        self.assertEqual(calls, [])
        self.assertFalse(rec.echo_detected)
        rec._tmpfile.close()

    def test_echo_disabled_by_setting_starts_no_worker(self):
        with mock.patch("trnscrb.settings.get", return_value=False):
            rec = Recorder(device=None, system_audio=True)
            rec._reset_echo()
        self.assertFalse(rec._echo_enabled)
        self.assertIsNone(rec._echo_scorer)
        self.assertIsNone(rec._echo_worker)

    def test_mic_only_recording_starts_no_worker(self):
        """Dictation has no system track to echo, whatever the setting says."""
        rec = Recorder(device=None, system_audio=False)
        rec._reset_echo()
        self.assertFalse(rec._echo_enabled)
        self.assertIsNone(rec._echo_worker)

    def test_new_session_resets_the_warning(self):
        rec = self._make_recorder()
        self._paths = [rec._tmpfile.name]
        rec._echo_notified = True
        rec.echo_detected = True
        rec._echo_streak = 3
        rec._reset_echo()  # starts the new session's worker
        self.assertFalse(rec._echo_notified)
        self.assertFalse(rec.echo_detected)
        self.assertEqual(rec._echo_streak, 0)
        self.assertIsNotNone(rec._echo_worker)
        rec._stop_echo_worker()
        self.assertIsNone(rec._echo_worker)
        rec._tmpfile.close()


if __name__ == "__main__":
    unittest.main()
