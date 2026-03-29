"""Local transcription with configurable backend (Parakeet or Whisper)."""

import threading
from pathlib import Path

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.transcriber")

_SUPPORTED_BACKENDS = {"parakeet", "whisper"}

_whisper_model = None
_whisper_model_lock = threading.Lock()
_whisper_model_size = "small"

_parakeet_model = None
_parakeet_model_id = None
_parakeet_model_lock = threading.Lock()


def set_model_size(size: str) -> None:
    global _whisper_model_size, _whisper_model
    _whisper_model_size = size
    _whisper_model = None  # force reload on next call


def _backend() -> str:
    backend = str(settings.get("transcription_backend") or "parakeet").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise RuntimeError(
            f"Unsupported transcription backend '{backend}'. "
            f"Set transcription_backend to one of: {allowed}."
        )
    return backend


def _get_whisper_model():
    global _whisper_model
    size = str(settings.get("model_size") or _whisper_model_size)
    with _whisper_model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Whisper backend selected but faster-whisper is not installed. "
                    "Install it with `uv add faster-whisper`."
                ) from e
            try:
                _whisper_model = WhisperModel(size, device="auto", compute_type="auto")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Whisper model '{size}'. "
                    "Check local model cache and backend dependencies."
                ) from e
        return _whisper_model


def _get_parakeet_model():
    global _parakeet_model, _parakeet_model_id
    model_id = str(settings.get("parakeet_model_id") or "").strip()
    if not model_id:
        raise RuntimeError(
            "Parakeet backend selected but no model id is configured. "
            "Set `parakeet_model_id` in ~/.config/trnscrb/settings.json."
        )

    with _parakeet_model_lock:
        if _parakeet_model is None or _parakeet_model_id != model_id:
            try:
                from parakeet_mlx import from_pretrained
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Parakeet backend selected but parakeet-mlx is not installed. "
                    "Install it with `uv add parakeet-mlx`."
                ) from e
            try:
                _parakeet_model = from_pretrained(model_id)
                _parakeet_model_id = model_id
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Parakeet model '{model_id}'. "
                    "Verify network/cache access for first-time model download."
                ) from e
        return _parakeet_model


def _transcribe_whisper(audio_path: Path) -> list[dict]:
    model = _get_whisper_model()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,  # skip silent gaps automatically
        language=None,  # auto-detect
    )
    results = []
    for seg in segments:
        text = getattr(seg, "text", None)
        if not text or not text.strip():
            continue
        results.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": text.strip(),
                "speaker": None,
            }
        )
    return results


def _transcribe_parakeet(audio_path: Path) -> list[dict]:
    model = _get_parakeet_model()
    result = model.transcribe(str(audio_path))
    sentences = getattr(result, "sentences", None)
    if sentences is None:
        raise RuntimeError("Parakeet transcription did not return aligned sentences output.")

    normalized = []
    for sentence in sentences:
        text = str(getattr(sentence, "text", "")).strip()
        if not text:
            continue
        try:
            start = float(getattr(sentence, "start", 0.0))
        except (TypeError, ValueError):
            _log.warning(
                "Could not parse start timestamp %r, defaulting to 0.0",
                getattr(sentence, "start", None),
            )
            start = 0.0
        try:
            end = float(getattr(sentence, "end", 0.0))
        except (TypeError, ValueError):
            _log.warning(
                "Could not parse end timestamp %r, defaulting to 0.0",
                getattr(sentence, "end", None),
            )
            end = 0.0
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": None,
            }
        )
    return normalized


def transcribe(audio_path: Path) -> list[dict]:
    """Return segments: [{start, end, text, speaker}] — speaker filled later by diarizer."""
    audio_path = Path(audio_path)
    file_size = audio_path.stat().st_size if audio_path.exists() else 0
    _log.info("Transcribing %s (%d bytes)", audio_path, file_size)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if file_size == 0:
        raise FileNotFoundError(f"Audio file is empty: {audio_path}")
    backend = _backend()
    _log.debug("Using backend: %s", backend)
    if backend == "parakeet":
        segments = _transcribe_parakeet(audio_path)
    else:
        segments = _transcribe_whisper(audio_path)
    _log.info("Transcription complete: %d segments", len(segments))
    return segments
