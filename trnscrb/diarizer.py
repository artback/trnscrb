"""Speaker diarization via pyannote.audio.

Prefers pyannote/speaker-diarization-community-1 (better speaker assignment
and counting, the native pipeline of pyannote.audio 4.x) and falls back to the
legacy speaker-diarization-3.1. Both are gated on HuggingFace — accept the
model's conditions at hf.co once, then it works offline. The pipeline can be
overridden with the `diarization_pipeline` setting.
"""

import threading
from pathlib import Path

from trnscrb import settings
from trnscrb.log import get_logger

log = get_logger(__name__)

_FALLBACK_PIPELINE = "pyannote/speaker-diarization-3.1"

_pipeline = None
# Serializes MPS/GPU inference when transcription jobs overlap.
_diarize_lock = threading.Lock()


def _load_pipeline(model_id: str, hf_token: str):
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(model_id, token=hf_token)  # pyannote.audio >= 4
    except TypeError:
        return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)  # 3.x


def _get_pipeline(hf_token: str):
    """Return the cached pyannote pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        import torch

        preferred = str(
            settings.get("diarization_pipeline") or "pyannote/speaker-diarization-community-1"
        )
        candidates = [preferred]
        if preferred != _FALLBACK_PIPELINE:
            candidates.append(_FALLBACK_PIPELINE)

        last_error: Exception | None = None
        for model_id in candidates:
            log.info("Loading diarization pipeline %s …", model_id)
            try:
                _pipeline = _load_pipeline(model_id, hf_token)
                break
            except Exception as e:
                last_error = e
                log.warning("Could not load %s: %s", model_id, e)
        if _pipeline is None:
            raise RuntimeError(
                f"No diarization pipeline could be loaded (tried {', '.join(candidates)}). "
                "Accept the model conditions on hf.co and check your HuggingFace token."
            ) from last_error

        # Prefer Apple Silicon Metal, fallback to CPU
        if torch.backends.mps.is_available():
            _pipeline = _pipeline.to(torch.device("mps"))

        log.info("Diarization pipeline ready")
    return _pipeline


def unload_pipeline() -> None:
    """Release the diarization pipeline to free memory after a long idle period."""
    global _pipeline
    import gc

    with _diarize_lock:
        _pipeline = None
    gc.collect()
    log.info("Diarization pipeline unloaded")


def diarize(audio_path: Path, hf_token: str) -> list[dict]:
    """Return [{start, end, speaker}] segments.

    Serialized with a lock so concurrent jobs don't overlap on the GPU.
    """
    with _diarize_lock:
        pipeline = _get_pipeline(hf_token)
        diarization = pipeline(str(audio_path))

    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


def merge(transcript: list[dict], diarization: list[dict]) -> list[dict]:
    """Assign the best-matching speaker label to each transcript segment."""
    for seg in transcript:
        best_speaker = None
        best_overlap = 0.0
        for d in diarization:
            overlap = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]
        seg["speaker"] = best_speaker or "Unknown"
    return transcript
