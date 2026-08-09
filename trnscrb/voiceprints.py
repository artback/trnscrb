"""Persistent speaker fingerprints, learned from meetings you already record.

pyannote computes a centroid embedding per speaker as part of clustering and
hands it back in `DiarizeOutput.speaker_embeddings`. That vector is a voice
fingerprint; it was simply being discarded. This module keeps the ones we can
attach a confident identity to, so a future call can recognise the voice.

The first identity worth learning is the user's own. The recorder captures the
microphone separately from the system audio, and a conferencing app never
plays your own mic back to you — so a diarized speaker whose turns are
consistently mic-only is you, in *every* meeting, with no calendar lookup and
no guessing. That makes "Me" the fastest fingerprint to accumulate and the
only one enrolled today.

Fingerprints are local, tied to the pipeline that produced them, and
inspectable with `trnscrb voiceprints`. They are biometric data, so anything
beyond the user's own voice should stay a deliberate opt-in.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from trnscrb.log import get_logger

_log = get_logger("trnscrb.voiceprints")

STORE = Path.home() / ".config" / "trnscrb" / "voiceprints.json"
_VERSION = 1

# The label used for the user's own voice.
SELF = "Me"

# Enrolments below this much attributed speech are too thin to characterise a
# voice, and would drag the running average toward whatever the room sounded
# like that day.
MIN_ENROLL_SECS = 60.0


def _unit(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, so averaging compares directions rather than magnitudes."""
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def load() -> dict:
    """The store, or an empty one when absent or unreadable."""
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": _VERSION, "model": "", "prints": {}}
    if data.get("version") != _VERSION:
        _log.info("Ignoring voiceprint store from a different version")
        return {"version": _VERSION, "model": "", "prints": {}}
    return data


def _save(data: dict) -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        _log.warning("Could not write voiceprint store", exc_info=True)


def enroll(name: str, vector: np.ndarray, model: str, speech_secs: float) -> bool:
    """Fold one observation of `name`'s voice into the store.

    Returns True when the fingerprint was updated. Embeddings only mean
    anything relative to the model that produced them, so a change of pipeline
    discards the store rather than averaging vectors from different spaces.
    """
    if speech_secs < MIN_ENROLL_SECS:
        _log.debug("Skipping %s enrolment: only %.0fs of speech", name, speech_secs)
        return False

    vector = _unit(np.asarray(vector, dtype=np.float64).ravel())
    if vector.size == 0 or not np.isfinite(vector).all():
        _log.debug("Skipping %s enrolment: unusable embedding", name)
        return False

    data = load()
    if data.get("prints") and data.get("model") and data["model"] != model:
        _log.warning(
            "Diarization pipeline changed (%s -> %s); discarding %d stored voiceprint(s)",
            data["model"],
            model,
            len(data["prints"]),
        )
        data = {"version": _VERSION, "model": model, "prints": {}}
    data["model"] = model

    entry = data["prints"].get(name)
    if entry and len(entry.get("vector", [])) == vector.size:
        # Running mean over unit vectors: every meeting counts once, so a
        # single long call cannot dominate the fingerprint.
        n = int(entry.get("enrollments", 1))
        merged = _unit((np.asarray(entry["vector"], dtype=np.float64) * n + vector) / (n + 1))
        entry = {
            "vector": merged.tolist(),
            "enrollments": n + 1,
            "speech_secs": round(float(entry.get("speech_secs", 0.0)) + speech_secs, 1),
        }
    else:
        entry = {
            "vector": vector.tolist(),
            "enrollments": 1,
            "speech_secs": round(speech_secs, 1),
        }
    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    data["prints"][name] = entry
    _save(data)
    _log.info(
        "Voiceprint for %s updated (%d enrolment(s), %.0fs of speech total)",
        name,
        entry["enrollments"],
        entry["speech_secs"],
    )
    return True


def forget(name: str) -> bool:
    """Delete one fingerprint. Returns True when there was one to delete."""
    data = load()
    if name not in data.get("prints", {}):
        return False
    del data["prints"][name]
    _save(data)
    _log.info("Forgot voiceprint for %s", name)
    return True


def summary() -> list[dict]:
    """One row per stored fingerprint, without the vectors."""
    data = load()
    return [
        {
            "name": name,
            "enrollments": entry.get("enrollments", 0),
            "speech_secs": entry.get("speech_secs", 0.0),
            "updated_at": entry.get("updated_at", ""),
            "dimension": len(entry.get("vector", [])),
        }
        for name, entry in sorted(data.get("prints", {}).items())
    ]
