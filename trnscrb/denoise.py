"""Optional noise-reduction front-end run before transcription.

A learned model (DeepFilterNet-class) would beat this, but on Python 3.14 /
macOS ARM no wheel exists for deepfilter yet, so this ships noisereduce's
classical spectral gating instead: cheap, fully local, and a solid
improvement for fan/AC/hiss and steady background noise — exactly the
conditions that break dictation.

Controlled by the ``denoise`` setting (off by default). Both meetings and
dictation funnel through ``transcriber.transcribe()``, which routes the audio
here first when the toggle is on.
"""

import tempfile
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.denoise")

# Aggressive enough to cut real background hiss, gentle enough that the
# speech's own harmonics survive. Noisereduce's default is 1.0 (full
# reduction), which can hollow out speech in a quiet room.
_DEFAULT_PROP_DECREASE = 0.5
# Anything this short has no noise profile worth fitting, and the spectral
# gate can misbehave on a handful of frames.
_MIN_AUDIO_SECS = 0.2


def available() -> bool:
    """True when the noisereduce engine is importable."""
    try:
        import noisereduce  # noqa: F401
        import soundfile  # noqa: F401

        return True
    except Exception:
        return False


def reduce_noise_wav(src: Path, dst: Path, prop_decrease: float = _DEFAULT_PROP_DECREASE) -> bool:
    """Read SRC, apply spectral-gate denoising, write DST. False on failure."""
    try:
        import noisereduce
        import numpy as np
        import soundfile as sf
    except Exception as e:
        _log.warning("Noise reduction unavailable (%s)", e)
        return False

    try:
        data, sr = sf.read(str(src), dtype="float32")
    except Exception as e:
        _log.warning("Could not read %s for denoising: %s", src, e)
        return False
    if data is None or len(data) == 0:
        return False
    if not bool(data.max()):  # pure silence: copy through, nothing to reduce
        try:
            sf.write(str(dst), data, sr)
            return True
        except Exception as e:
            _log.warning("Failed writing denoised copy: %s", e)
            return False

    # Stationary mode fits one noise profile for the whole clip — the right
    # model for fan/AC/traffic hiss, and it never needs a noise-only sample.
    try:
        if data.ndim == 2 and data.shape[1] > 1:  # (frames, channels)
            channels = [
                noisereduce.reduce_noise(
                    y=data[:, ch],
                    sr=sr,
                    stationary=True,
                    prop_decrease=prop_decrease,
                    use_tqdm=False,
                )
                for ch in range(data.shape[1])
            ]
            reduced = np.stack(channels, axis=1)
        else:
            reduced = noisereduce.reduce_noise(
                y=data,
                sr=sr,
                stationary=True,
                prop_decrease=prop_decrease,
                use_tqdm=False,
            )
    except Exception as e:
        _log.warning("Noise reduction failed on %s: %s", src, e)
        return False

    try:
        sf.write(str(dst), reduced, sr)
    except Exception as e:
        _log.warning("Failed writing denoised copy: %s", e)
        return False
    return True


def preprocess(audio_path: Path) -> tuple[Path, Path | None]:
    """Denoise a recording when possible.

    Returns ``(path_to_transcribe, tmp_to_delete)``. When no denoising
    happened (engine missing, audio too short, read failure), the original
    path is returned untouched and nothing needs cleaning up.
    """
    path = Path(audio_path)
    if not available():
        _log.warning(
            "Noise filter is on but noisereduce is not installed — run `uv add noisereduce`"
        )
        return path, None

    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames <= 0 or info.frames / info.samplerate < _MIN_AUDIO_SECS:
            return path, None
    except Exception as e:
        _log.warning("Skipping noise filter for %s (%s)", path, e)
        return path, None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="trnscrb-denoised-", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    started = _monotonic()
    ok = reduce_noise_wav(path, tmp_path)
    if not ok:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return path, None
    _log.info(
        "Denoised %s in %.1fs (%.1f seconds of audio)",
        path.name,
        _monotonic() - started,
        info.frames / info.samplerate,
    )
    return tmp_path, tmp_path


def _monotonic() -> float:
    import time

    return time.monotonic()
