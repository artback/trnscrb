"""Echo detection: the mic picking up the call audio from the speakers.

Without headphones the other side's voice reaches the microphone twice —
cleanly through system audio, and as a faint copy through the room. A
sliding-window normalized cross-correlation between the two streams sees
that copy: where the mic follows the system track at a small lag the
correlation is high, where the user is simply talking over the call it is
not.

The module only scores; the caller decides. ``EchoScorer.update`` returns a
running, EMA-smoothed score in 0..1, and ``confident`` turns a history of
scores into an arm/disarm decision with hysteresis. Pure numpy, no audio
I/O — trivially unit-testable with synthetic arrays.
"""

import numpy as np

# Both streams are decimated to this rate before scoring. Decimation is a
# plain every-Nth sample (no low-pass): cheap, and the correlation only
# cares about the envelope's shape, not about fidelity.
_TARGET_RATE = 8_000

# The recorder holds the mic back by up to half a second to meet system
# audio (recorder._align_frames), and an untimestamped helper has no delay
# at all — so a copy can sit up to this far behind the reference stream.
# The candidate-lag grid divides this evenly, so the boundary itself is a
# grid point (a pure tone would otherwise alias between grid lags).
_MAX_LAG = 4_000  # samples at _TARGET_RATE (500 ms)
_LAG_STEP = 100  # 12.5 ms between candidate lags

# How much recent audio each stream keeps, in samples at _TARGET_RATE.
_WINDOW = 8_000  # 1 s

# Windows quieter than this carry no evidence either way: a meeting where
# nobody is talking must not read as echoing. Mean amplitude, per window.
_SYS_FLOOR = 0.004  # ~-48 dBFS: the other side is actually talking
_MIC_FLOOR = 0.001  # ~-60 dBFS: something is on the mic at all

_EPS = 1e-10


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


class EchoScorer:
    """Running echo score between a mic stream and a system-audio stream.

    Feed it paired blocks from the same moment — the recorder's callback
    pairs them, the mic already delayed to meet system audio — and it
    returns a smoothed 0..1 score: how strongly the mic's recent window
    follows the system track at a small lag. The score is 0 until both
    streams have a full window of history (warmup), and it decays toward 0
    while either stream is silent.
    """

    def __init__(self, sample_rate: int = 16_000, alpha: float = 0.35):
        # Blocks arrive at the capture rate; decimate to the scoring rate.
        self._factor = max(1, round(sample_rate / _TARGET_RATE))
        # The buffers hold the window plus the whole lag range, so the lag
        # search never reaches past what is buffered.
        self._cap = _WINDOW + _MAX_LAG
        self._mic = np.zeros(0, dtype=np.float32)
        self._sys = np.zeros(0, dtype=np.float32)
        self._alpha = alpha
        self._score = 0.0
        self._lags = np.arange(0, _MAX_LAG + 1, _LAG_STEP)

    def reset(self) -> None:
        """Drop the buffers and the running score (a new recording)."""
        self._mic = np.zeros(0, dtype=np.float32)
        self._sys = np.zeros(0, dtype=np.float32)
        self._score = 0.0

    @property
    def score(self) -> float:
        """The current running score (0..1); 0 during warmup."""
        return self._score

    def update(self, mic: np.ndarray, system: np.ndarray) -> float:
        """Score one pair of new blocks; return the running score (0..1).

        ``mic`` and ``system`` are mono float arrays at the capture sample
        rate, from the same moment. Returns 0 until both streams hold a
        full window of history.
        """
        mic = np.asarray(mic, dtype=np.float32).ravel()
        system = np.asarray(system, dtype=np.float32).ravel()
        if len(mic) < self._factor or len(system) < self._factor:
            return self._score
        self._mic = self._append(self._mic, mic[:: self._factor])
        self._sys = self._append(self._sys, system[:: self._factor])
        if len(self._mic) < self._cap or len(self._sys) < self._cap:
            return 0.0  # still warming up

        # The mic's newest window against the system track's window ending
        # `lag` samples earlier, for every candidate lag.
        a = self._mic[_MAX_LAG:]
        if _rms(a) < _MIC_FLOOR or _rms(self._sys[-_WINDOW:]) < _SYS_FLOOR:
            raw = 0.0  # no evidence on one side; let the score decay
        else:
            raw = max(0.0, self._best_ncc(a, self._sys))
        self._score = min(1.0, (1.0 - self._alpha) * self._score + self._alpha * raw)
        return self._score

    def _append(self, buf: np.ndarray, new: np.ndarray) -> np.ndarray:
        out = np.append(buf, new)
        return out if len(out) <= self._cap else out[-self._cap :]

    def _best_ncc(self, a: np.ndarray, sys_buf: np.ndarray) -> float:
        """Peak normalized cross-correlation of ``a`` against the system
        buffer at every candidate lag (0 = no correlation, 1 = a copy)."""
        starts = (_MAX_LAG - self._lags)[:, None] + np.arange(_WINDOW)
        b = sys_buf[starts]  # (lags, window)
        an = a - a.mean()
        bn = b - b.mean(axis=1, keepdims=True)
        norm_a = np.linalg.norm(an)
        if norm_a < _EPS:
            return 0.0
        norm_b = np.linalg.norm(bn, axis=1)
        corr = (bn @ an) / (norm_b * norm_a + _EPS)
        return float(np.max(corr))


def confident(history, threshold: float, hysteresis: float = 0.10) -> bool:
    """Whether the recent scores say the echo is real, with hysteresis.

    ``history`` is the running scores, oldest first. A score at or above
    ``threshold`` arms the gate; it only disarms at or below
    ``threshold - hysteresis``, so a dip back toward — but not through —
    the threshold keeps a real echo armed. The caller owns the history
    (its length is the gate's memory) and what being armed means.
    """
    armed = False
    floor = threshold - hysteresis
    for s in history:
        armed = armed or s >= threshold
        if s <= floor:
            armed = False
    return armed
