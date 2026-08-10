"""Persistent voice identities, learned from meetings you already record.

pyannote computes a centroid embedding per speaker as part of clustering and
hands it back in `DiarizeOutput.speaker_embeddings`. Within one meeting those
labels are arbitrary — SPEAKER_00 in March has nothing to do with SPEAKER_00
in April. This module carries them across meetings: each observed speaker is
matched against the voices already known, so the same person accumulates one
stable identity over time.

Two things follow from that. A voice can be recognised before anyone has said
who it is, and when a name does arrive — from the mic/system split for the
user, from a 1:1's calendar entry, or typed in — it applies to every past
meeting that voice appeared in, not just the one that named it.

Matching is deliberately reluctant. An unmatched voice becomes a new identity,
which is cheap to merge later; a wrong match silently fuses two people and
would put one person's name on another's words. So a match needs both a high
absolute similarity and a clear margin over the runner-up.

Fingerprints are local, tied to the pipeline that produced them, and
inspectable with `trnscrb voices`. Identities other than the user's own are
biometric data about people who have not consented, which is why clustering
them is opt-in (`cluster_voices`).
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.voiceprints")

STORE = Path.home() / ".config" / "trnscrb" / "voiceprints.json"
_VERSION = 2

# The name given to the user's own voice.
SELF = "Me"

# Observations below this much attributed speech are too thin to characterise
# a voice, and would drag a centroid toward whatever the room sounded like.
MIN_ENROLL_SECS = 60.0

# Keep the most recent occurrences per voice; enough to explain a match
# without growing without bound.
_MAX_OBSERVATIONS = 100


def _empty(model: str = "") -> dict:
    return {"version": _VERSION, "model": model, "voices": {}, "next_id": 1}


def _unit(vector) -> np.ndarray:
    """L2-normalise, so similarity compares direction rather than magnitude."""
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def _cosine(a, b) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _thresholds() -> tuple[float, float]:
    """(match, margin) — how similar, and how much clearer than the runner-up."""
    return (
        float(settings.get("voice_match_threshold") or 0.55),
        float(settings.get("voice_match_margin") or 0.10),
    )


def _migrate(data: dict) -> dict:
    """Carry a v1 store (named fingerprints, no clusters) into the v2 shape."""
    migrated = _empty(data.get("model", ""))
    for name, entry in (data.get("prints") or {}).items():
        vector = entry.get("vector") or []
        if not vector:
            continue
        migrated["voices"][f"voice-{migrated['next_id']}"] = {
            "vector": vector,
            "name": name,
            "observations": int(entry.get("enrollments", 1)),
            "speech_secs": float(entry.get("speech_secs", 0.0)),
            "seen": [],
            "updated_at": entry.get("updated_at", ""),
        }
        migrated["next_id"] += 1
    if migrated["voices"]:
        _log.info("Migrated %d voiceprint(s) to the clustered store", len(migrated["voices"]))
    return migrated


def load() -> dict:
    """The store, or an empty one when absent, unreadable, or from the future."""
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    version = data.get("version")
    if version == 1:
        return _migrate(data)
    if version != _VERSION:
        _log.info("Ignoring voiceprint store from an unknown version (%s)", version)
        return _empty()
    return data


def _save(data: dict) -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        _log.warning("Could not write voiceprint store", exc_info=True)


def match(vector, data: dict | None = None) -> tuple[str | None, float]:
    """The known voice this embedding belongs to, and its similarity.

    Returns (None, best_score) when nothing is similar enough or when two
    identities are too close to separate — an ambiguous match is refused
    rather than guessed, since fusing two people is not self-correcting.
    """
    data = load() if data is None else data
    voices = data.get("voices") or {}
    if not voices:
        return None, 0.0

    threshold, margin = _thresholds()
    scored = sorted(
        ((vid, _cosine(entry["vector"], vector)) for vid, entry in voices.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    best_id, best_score = scored[0]
    if best_score < threshold:
        return None, best_score
    if len(scored) > 1 and scored[1][1] > best_score - margin:
        _log.debug(
            "Ambiguous voice match: %s (%.3f) vs %s (%.3f)",
            best_id,
            best_score,
            scored[1][0],
            scored[1][1],
        )
        return None, best_score
    return best_id, best_score


def observe(
    vector, model: str, speech_secs: float, meeting: str = "", label: str = ""
) -> str | None:
    """Record one sighting of a voice, joining or creating an identity.

    Returns the voice id, or None when the observation was rejected.
    """
    if speech_secs < MIN_ENROLL_SECS:
        _log.debug("Ignoring voice observation: only %.0fs of speech", speech_secs)
        return None
    vector = _unit(vector)
    if vector.size == 0 or not np.isfinite(vector).all():
        _log.debug("Ignoring voice observation: unusable embedding")
        return None

    data = load()
    if data.get("voices") and data.get("model") and data["model"] != model:
        _log.warning(
            "Diarization pipeline changed (%s -> %s); discarding %d stored voice(s)",
            data["model"],
            model,
            len(data["voices"]),
        )
        data = _empty(model)
    data["model"] = model

    voice_id, score = match(vector, data)
    if voice_id is None:
        voice_id = f"voice-{data['next_id']}"
        data["next_id"] += 1
        entry = {
            "vector": vector.tolist(),
            "name": "",
            "observations": 0,
            "speech_secs": 0.0,
            "seen": [],
        }
        data["voices"][voice_id] = entry
        _log.info("New voice %s (best existing match %.3f)", voice_id, score)
    else:
        entry = data["voices"][voice_id]
        # Running mean per observation, so one long meeting cannot dominate.
        n = int(entry.get("observations", 1))
        entry["vector"] = _unit(
            (np.asarray(entry["vector"], dtype=np.float64) * n + vector) / (n + 1)
        ).tolist()
        _log.info(
            "Voice %s%s matched (%.3f)",
            voice_id,
            f" ({entry['name']})" if entry.get("name") else "",
            score,
        )

    entry["observations"] = int(entry.get("observations", 0)) + 1
    entry["speech_secs"] = round(float(entry.get("speech_secs", 0.0)) + speech_secs, 1)
    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    seen = list(entry.get("seen") or [])
    seen.append(
        {
            "meeting": meeting,
            "label": label,
            "secs": round(speech_secs, 1),
            "at": entry["updated_at"],
        }
    )
    entry["seen"] = seen[-_MAX_OBSERVATIONS:]
    _save(data)
    return voice_id


def name_voice(voice_id: str, name: str) -> bool:
    """Attach a name to an identity — and so to every meeting it appears in."""
    data = load()
    entry = (data.get("voices") or {}).get(voice_id)
    if entry is None:
        return False
    entry["name"] = name
    _save(data)
    _log.info(
        "Voice %s named %s (%d past meeting(s))", voice_id, name, len(entry.get("seen") or [])
    )
    return True


def find_by_name(name: str) -> str | None:
    for voice_id, entry in (load().get("voices") or {}).items():
        if entry.get("name") == name:
            return voice_id
    return None


def forget(voice_id: str) -> bool:
    """Delete one identity. Returns True when there was one to delete."""
    data = load()
    if voice_id not in (data.get("voices") or {}):
        return False
    del data["voices"][voice_id]
    _save(data)
    _log.info("Forgot voice %s", voice_id)
    return True


def summary() -> list[dict]:
    """One row per identity, without the vectors."""
    data = load()
    return [
        {
            "id": voice_id,
            "name": entry.get("name", ""),
            "observations": entry.get("observations", 0),
            "speech_secs": entry.get("speech_secs", 0.0),
            "updated_at": entry.get("updated_at", ""),
            "dimension": len(entry.get("vector", [])),
            "meetings": [
                s.get("meeting", "") for s in (entry.get("seen") or []) if s.get("meeting")
            ],
        }
        for voice_id, entry in sorted((data.get("voices") or {}).items())
    ]
