"""Local transcription with configurable backend (Parakeet or Whisper)."""
import threading
from pathlib import Path

from trnscrb import settings

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
    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": None,
        }
        for seg in segments
        if seg.text.strip()
    ]


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
        normalized.append(
            {
                "start": float(getattr(sentence, "start", 0.0)),
                "end": float(getattr(sentence, "end", 0.0)),
                "text": text,
                "speaker": None,
            }
        )
    return normalized


def transcribe(audio_path: Path) -> list[dict]:
    """Return segments: [{start, end, text, speaker}] — speaker filled later by diarizer."""
    backend = _backend()
    if backend == "parakeet":
        return _transcribe_parakeet(audio_path)
    return _transcribe_whisper(audio_path)
