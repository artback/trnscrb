"""Speaker diarization via pyannote.audio.

Prefers pyannote/speaker-diarization-community-1 (better speaker assignment
and counting, the native pipeline of pyannote.audio 4.x) and falls back to the
legacy speaker-diarization-3.1. Both are gated on HuggingFace — accept the
model's conditions at hf.co once, then it works offline. The pipeline can be
overridden with the `diarization_pipeline` setting.
"""

import threading
from pathlib import Path

import numpy as np

from trnscrb import settings
from trnscrb.log import get_logger

log = get_logger(__name__)

_FALLBACK_PIPELINE = "pyannote/speaker-diarization-3.1"

_pipeline = None
_loaded_pipeline_id = ""
# Serializes MPS/GPU inference when transcription jobs overlap.
_diarize_lock = threading.Lock()


def _load_pipeline(model_id: str, hf_token: str):
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(model_id, token=hf_token)  # pyannote.audio >= 4
    except TypeError:
        return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)  # 3.x


def pipeline_candidates() -> list[str]:
    """Pipelines tried in order, preferred first."""
    preferred = str(
        settings.get("diarization_pipeline") or "pyannote/speaker-diarization-community-1"
    )
    return [preferred] if preferred == _FALLBACK_PIPELINE else [preferred, _FALLBACK_PIPELINE]


def is_downloaded(model_id: str) -> bool:
    """True when this pipeline is already in the local HuggingFace cache.

    pyannote's repos are gated: a valid token is not enough, the model
    conditions have to be accepted on hf.co before anything downloads. This
    answers the question offline, so `trnscrb status` can tell "token set" and
    "speaker labels will actually work" apart.
    """
    import os

    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    root = Path(cache) if cache else Path.home() / ".cache" / "huggingface"
    if root.name != "hub":
        root = root / "hub"
    snapshots = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _get_pipeline(hf_token: str):
    """Return the cached pyannote pipeline, loading it on first call."""
    global _pipeline, _loaded_pipeline_id
    if _pipeline is None:
        import torch

        candidates = pipeline_candidates()
        last_error: Exception | None = None
        for model_id in candidates:
            log.info("Loading diarization pipeline %s …", model_id)
            try:
                _pipeline = _load_pipeline(model_id, hf_token)
                _loaded_pipeline_id = model_id
                break
            except Exception as e:
                last_error = e
                log.warning("Could not load %s: %s", model_id, e)
        if _pipeline is None:
            # These repos are gated, so the usual cause is unaccepted model
            # conditions rather than a bad token — name the pages to visit.
            pages = " and ".join(f"hf.co/{m}" for m in candidates)
            raise RuntimeError(
                f"No diarization pipeline could be loaded (tried {', '.join(candidates)}). "
                f"Accept the model conditions at {pages} with the same account as your "
                "HuggingFace token, then retry."
            ) from last_error

        # Prefer Apple Silicon Metal, fallback to CPU
        if torch.backends.mps.is_available():
            _pipeline = _pipeline.to(torch.device("mps"))

        log.info("Diarization pipeline ready")
    return _pipeline


def preload(hf_token: str) -> bool:
    """Load the pipeline now rather than at the end of the meeting.

    torch and pyannote are imported lazily, at stop. That leaves a window: a
    `brew upgrade` during the meeting deletes the tree they would be imported
    from, and the recording then finishes with

        Diarization skipped: No module named 'torch'

    — losing the speaker labels, voiceprints and voice clips for a meeting
    that recorded perfectly. Loading at the start puts them in memory before
    an upgrade can take them away, which is what makes deferring a restart
    until the meeting ends actually sufficient.

    Skipped when the pipeline is not already downloaded, so starting a
    recording never waits on the network. Returns True when it is resident.
    """
    if not hf_token:
        return False
    if not any(is_downloaded(model_id) for model_id in pipeline_candidates()):
        log.debug("Diarization preload skipped: no pipeline in the local cache")
        return False
    try:
        with _diarize_lock:
            _get_pipeline(hf_token)
    except Exception as e:
        log.debug("Diarization preload failed (%s); it will load on demand", e)
        return False
    return True


def unload_pipeline() -> None:
    """Release the diarization pipeline to free memory after a long idle period."""
    global _pipeline, _loaded_pipeline_id
    import gc

    with _diarize_lock:
        _pipeline = None
        _loaded_pipeline_id = ""
    gc.collect()
    log.info("Diarization pipeline unloaded")


def _speaker_timeline(result):
    """The speaker timeline from a pipeline result, across pyannote versions.

    pyannote.audio 4 returns a DiarizeOutput dataclass; 3.x returned the
    Annotation itself. Prefer the exclusive timeline — it drops overlapping
    speech turns, which is what assigning one speaker per transcript segment
    wants.
    """
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(result, attr, None)
        if annotation is not None:
            return annotation
    return result


RAW_SPACE = "embedding"
PLDA_SPACE = "plda"

_embedding_space = RAW_SPACE


def embedding_space() -> str:
    """Which space the embeddings last handed out live in.

    Vectors are only comparable within one space, so a store has to record
    this alongside the pipeline id.
    """
    return _embedding_space


def _project(vectors, pipeline):
    """Map raw embeddings into the pipeline's PLDA space when it has one.

    community-1 ships a PLDA model beside its embedding model and clusters
    with cosine *in that projected space*, not on the raw embeddings. Raw
    cosine is measurably worse at telling speakers apart, so anything we
    store for later comparison should live in the same space the pipeline
    itself trusts.
    """
    global _embedding_space

    plda = getattr(pipeline, "_plda", None)
    if plda is None:
        _embedding_space = RAW_SPACE
        return vectors
    try:
        projected = np.asarray(plda(np.asarray(vectors)))
    except Exception:
        log.debug("PLDA projection failed; keeping raw embeddings", exc_info=True)
        _embedding_space = RAW_SPACE
        return vectors
    if projected.shape[0] != len(vectors):
        log.debug(
            "PLDA returned %d rows for %d speakers; keeping raw", len(projected), len(vectors)
        )
        _embedding_space = RAW_SPACE
        return vectors
    _embedding_space = PLDA_SPACE
    return projected


def _embeddings_by_speaker(result, pipeline=None) -> dict:
    """Map each speaker label to its centroid embedding, when available.

    pyannote computes these as part of clustering and returns them ordered to
    match `speaker_diarization.labels()` — note that is the plain timeline,
    not the exclusive one we read turns from. Returns {} on any mismatch
    rather than risking a fingerprint attached to the wrong voice.
    """
    vectors = getattr(result, "speaker_embeddings", None)
    annotation = getattr(result, "speaker_diarization", None)
    if vectors is None or annotation is None:
        return {}
    labels = list(annotation.labels())
    if len(labels) != len(vectors):
        log.debug("Embedding count %d != %d speakers; ignoring", len(vectors), len(labels))
        return {}
    if pipeline is not None:
        vectors = _project(vectors, pipeline)
    return dict(zip(labels, vectors, strict=True))


def diarize_with_embeddings(audio_path: Path, hf_token: str) -> tuple[list[dict], dict]:
    """Speaker turns plus each speaker's centroid embedding.

    Serialized with a lock so concurrent jobs don't overlap on the GPU.
    """
    with _diarize_lock:
        pipeline = _get_pipeline(hf_token)
        diarization = pipeline(str(audio_path))

    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in _speaker_timeline(diarization).itertracks(yield_label=True)
    ]
    return turns, _embeddings_by_speaker(diarization, pipeline)


def diarize(audio_path: Path, hf_token: str) -> list[dict]:
    """Return [{start, end, speaker}] segments."""
    turns, _ = diarize_with_embeddings(audio_path, hf_token)
    return turns


def pipeline_id() -> str:
    """The pipeline that actually loaded — which may be the fallback.

    Embeddings only compare within the model that produced them, so this is
    what a stored fingerprint has to be tagged with.
    """
    return _loaded_pipeline_id


# A speaker run shorter than this is treated as noise in the diarization and
# folded into its neighbour, rather than chopping a sentence into confetti.
_MIN_RUN_SECS = 0.9


def _best_speaker(start: float, end: float, diarization: list[dict]) -> str | None:
    """The diarized speaker overlapping [start, end] the most."""
    best_speaker = None
    best_overlap = 0.0
    for d in diarization:
        overlap = min(end, d["end"]) - max(start, d["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = d["speaker"]
    return best_speaker


def _split_by_speaker(seg: dict, diarization: list[dict]) -> list[dict]:
    """Cut one transcript segment where its words change speaker."""
    runs: list[dict] = []
    for word in seg["words"]:
        speaker = _best_speaker(word["start"], word["end"], diarization) or "Unknown"
        if runs and runs[-1]["speaker"] == speaker:
            runs[-1]["words"].append(word)
        else:
            runs.append({"speaker": speaker, "words": [word]})

    # Fold away runs too short to be a real turn — a single word landing on a
    # neighbour's label is far more often a diarization boundary being a
    # fraction of a second out than someone actually interjecting one word.
    # Folding one away can leave its neighbours on the same speaker, so
    # matching runs are coalesced in the same pass.
    merged: list[dict] = []
    for run in runs:
        duration = run["words"][-1]["end"] - run["words"][0]["start"]
        same_speaker = bool(merged) and merged[-1]["speaker"] == run["speaker"]
        if merged and (same_speaker or duration < _MIN_RUN_SECS):
            merged[-1]["words"].extend(run["words"])
        else:
            merged.append(run)

    from trnscrb.transcriber import words_to_text

    out = []
    for run in merged:
        words = run["words"]
        out.append(
            {
                **seg,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": words_to_text(words),
                "speaker": run["speaker"],
                "words": words,
            }
        )
    return out


def merge(transcript: list[dict], diarization: list[dict]) -> list[dict]:
    """Attach speaker labels, splitting segments where the speaker changes.

    Labelling a segment as a whole is only correct while segments are short.
    A multi-minute one takes a single label and silently reassigns everyone
    who spoke inside it to that person — which then misattributes their
    action items too. When word timings are available the segment is cut at
    the speaker boundaries instead.
    """
    out: list[dict] = []
    for seg in transcript:
        if seg.get("words"):
            out.extend(_split_by_speaker(seg, diarization))
        else:
            seg["speaker"] = _best_speaker(seg["start"], seg["end"], diarization) or "Unknown"
            out.append(seg)
    return out
