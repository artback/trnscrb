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

# Founding a *new* identity takes more than joining a known one. A short
# cameo still lands on the right person, because the centroid it is judged
# against was built from real speech; a centroid built from the cameo itself
# is half room-noise, and every later sighting would be judged against it.
# Below this an unknown voice is simply not remembered yet.
MIN_NEW_VOICE_SECS = 180.0

# An unnamed voice heard once and not again for this long is more likely a
# one-off — a guest, a mislabelled stretch — than someone worth keeping.
PRUNE_AFTER_DAYS = 30

# Keep the most recent occurrences per voice; enough to explain a match
# without growing without bound.
_MAX_OBSERVATIONS = 100

# Two *stored* centroids this alike are one person kept twice, not two
# people. A single observation joins a voice at 0.75; a pair of averages
# agreeing well beyond that is a split that happened, not a resemblance.
DUPLICATE_SIMILARITY = 0.90


def _empty(model: str = "", space: str = "") -> dict:
    return {"version": _VERSION, "model": model, "space": space, "voices": {}, "next_id": 1}


def _unit(vector) -> np.ndarray:
    """L2-normalise, so similarity compares direction rather than magnitude."""
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def _cosine(a, b) -> float:
    """Similarity, or -1 when the vectors are not comparable.

    Two spaces (raw embedding vs PLDA projection) have different lengths, so a
    store carried across a change would otherwise raise here — deep inside a
    best-effort path that swallows exceptions, silently ending enrolment.
    """
    a, b = _unit(a), _unit(b)
    if a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))


def _founding_secs() -> float:
    """Speech needed to create an identity; never below what joining one needs."""
    return max(MIN_ENROLL_SECS, MIN_NEW_VOICE_SECS)


def _thresholds() -> tuple[float, float]:
    """(match, margin) — how similar, and how much clearer than the runner-up."""
    return (
        float(settings.get("voice_match_threshold") or 0.75),
        float(settings.get("voice_match_margin") or 0.10),
    )


def _migrate(data: dict) -> dict:
    """Carry a v1 store (named fingerprints, no clusters) into the v2 shape.

    v1 predates PLDA projection, so its vectors are raw embeddings. Recording
    that is what lets the space check retire them cleanly on the first
    projected observation, instead of comparing 256 dimensions against 128.
    """
    # Matches diarizer.RAW_SPACE; not imported, to keep the store free of a
    # dependency on the model stack.
    migrated = _empty(data.get("model", ""), "embedding")
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
    for other_id, other_score in scored[1:]:
        if other_score <= best_score - margin:
            break
        # A runner-up that is itself a copy of the winner is not a second
        # person to confuse with the first. Refusing here is what bred the
        # copies: once a voice is stored twice, every later sighting of it
        # is "ambiguous", and each refusal stores it once more.
        if _cosine(voices[best_id]["vector"], voices[other_id]["vector"]) >= DUPLICATE_SIMILARITY:
            continue
        _log.debug(
            "Ambiguous voice match: %s (%.3f) vs %s (%.3f)",
            best_id,
            best_score,
            other_id,
            other_score,
        )
        return None, best_score
    return best_id, best_score


def observe(
    vector,
    model: str,
    speech_secs: float,
    meeting: str = "",
    label: str = "",
    space: str = "",
    *,
    founding_secs: float | None = None,
) -> str | None:
    """Record one sighting of a voice, joining or creating an identity.

    ``founding_secs`` is the speech needed to create a new identity when
    nothing matches; the default is the founding bar, and a caller that
    already knows who is speaking may lower it to the joining bar.

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
    # Vectors compare only within the model *and* the space that produced
    # them. A PLDA-projected vector and a raw one have different geometry and
    # different length, so mixing them would score noise as similarity.
    stored_model, stored_space = data.get("model"), data.get("space")
    if data.get("voices") and (
        (stored_model and stored_model != model) or (stored_space and stored_space != space)
    ):
        _log.warning(
            "Voice representation changed (%s/%s -> %s/%s); discarding %d stored voice(s)",
            stored_model,
            stored_space or "unknown",
            model,
            space or "unknown",
            len(data["voices"]),
        )
        data = _empty(model, space)
    data["model"] = model
    data["space"] = space

    voice_id, score = match(vector, data)
    if voice_id is None:
        bar = _founding_secs() if founding_secs is None else founding_secs
        if speech_secs < bar:
            _log.debug(
                "Not founding a voice on %.0fs of speech (best existing match %.3f)",
                speech_secs,
                score,
            )
            return None
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
    # A match that had to look past a duplicate leaves the duplicate behind;
    # fold it in now so the next sighting does not have to.
    for kept, absorbed, score in dedupe(data, only=voice_id):
        _log.info("Merged duplicate voice %s into %s (%.3f)", absorbed, kept, score)
        if kept != voice_id:
            voice_id = kept
    _save(data)
    return voice_id


def _survivor(data: dict, a: str, b: str) -> str:
    """Which of two duplicates keeps its id: the named one, else the better known."""
    va, vb = data["voices"][a], data["voices"][b]
    if bool(va.get("name")) != bool(vb.get("name")):
        return a if va.get("name") else b
    if va.get("observations", 0) != vb.get("observations", 0):
        return a if va.get("observations", 0) > vb.get("observations", 0) else b
    return min(a, b, key=lambda vid: (len(vid), vid))


def merge(data: dict, keep: str, absorb: str, *, move_samples: bool = True) -> bool:
    """Fold one identity into another, in place.

    Refused when both carry different names: that is two people, and no
    similarity score outranks what the user typed. The sample clip moves
    with the identity when the survivor has none of its own — unless
    ``move_samples`` is off, for a caller only asking what would happen.
    """
    voices = data.get("voices") or {}
    if keep == absorb or keep not in voices or absorb not in voices:
        return False
    kept, gone = voices[keep], voices[absorb]
    if kept.get("name") and gone.get("name") and kept["name"] != gone["name"]:
        return False

    n_kept = max(int(kept.get("observations", 1)), 1)
    n_gone = max(int(gone.get("observations", 1)), 1)
    kept["vector"] = _unit(
        np.asarray(kept["vector"], dtype=np.float64) * n_kept
        + np.asarray(gone["vector"], dtype=np.float64) * n_gone
    ).tolist()
    kept["name"] = kept.get("name") or gone.get("name", "")
    kept["observations"] = n_kept + n_gone
    kept["speech_secs"] = round(
        float(kept.get("speech_secs", 0.0)) + float(gone.get("speech_secs", 0.0)), 1
    )
    seen = list(kept.get("seen") or []) + list(gone.get("seen") or [])
    seen.sort(key=lambda s: s.get("at", ""))
    kept["seen"] = seen[-_MAX_OBSERVATIONS:]
    kept["updated_at"] = max(kept.get("updated_at", ""), gone.get("updated_at", ""))
    del voices[absorb]

    if not move_samples:
        return True
    try:
        if sample_path(absorb).is_file():
            if sample_path(keep).is_file():
                sample_path(absorb).unlink()
            else:
                sample_path(absorb).rename(sample_path(keep))
    except OSError:
        _log.debug("Could not move the sample of %s to %s", absorb, keep, exc_info=True)
    return True


def dedupe(
    data: dict,
    similarity: float = DUPLICATE_SIMILARITY,
    only: str | None = None,
    *,
    move_samples: bool = True,
) -> list[tuple[str, str, float]]:
    """Merge every pair of stored voices at least this alike, most alike first.

    Merging moves a centroid, so the pairs are re-scored after each one
    rather than planned up front. With ``only``, just the pairs involving
    that voice are considered. Returns (kept, absorbed, score) per merge.
    """
    merged: list[tuple[str, str, float]] = []
    refused: set[frozenset[str]] = set()
    while True:
        voices = data.get("voices") or {}
        ids = sorted(voices, key=lambda vid: (len(vid), vid))
        best: tuple[float, str, str] | None = None
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if only is not None and only not in (a, b):
                    continue
                if frozenset((a, b)) in refused:
                    continue
                score = _cosine(voices[a]["vector"], voices[b]["vector"])
                if score >= similarity and (best is None or score > best[0]):
                    best = (score, a, b)
        if best is None:
            return merged
        score, a, b = best
        keep = _survivor(data, a, b)
        absorb = b if keep == a else a
        if not merge(data, keep, absorb, move_samples=move_samples):
            refused.add(frozenset((a, b)))
            continue
        merged.append((keep, absorb, score))
        if only == absorb:
            only = keep


def merge_duplicates(
    similarity: float = DUPLICATE_SIMILARITY, dry_run: bool = False
) -> list[tuple[str, str, float]]:
    """Collapse duplicate identities in the store. Returns what was (or would be) merged."""
    data = load()
    # A dry run must leave the clips alone too: a clip is the only way a
    # voice can ever be identified by ear, and the audio it came from is
    # long gone.
    merged = dedupe(data, similarity, move_samples=not dry_run)
    if merged and not dry_run:
        _save(data)
        _log.info("Merged %d duplicate voice(s)", len(merged))
    return merged


SAMPLES_DIR = STORE.parent / "voice-samples"
# Long enough to recognise someone, short enough to be trivial to store.
SAMPLE_SECS = 6.0
# Skip the first moment of a turn: it often carries the tail of the previous
# speaker or the click of an unmute.
_SAMPLE_LEAD_IN = 0.5


def sample_path(voice_id: str) -> Path:
    """Where this voice's audio clip lives, whether or not it exists."""
    return SAMPLES_DIR / f"{voice_id}.wav"


def save_sample(voice_id: str, audio_path: Path, turns: list[dict]) -> Path | None:
    """Keep a few seconds of this voice so a person can identify it by ear.

    Captured during enrolment because that is the only moment the audio is
    still on disk — a meeting's recording is deleted as soon as its transcript
    is saved, so there is nothing to extract from afterwards.
    """
    import wave

    dest = sample_path(voice_id)
    if dest.exists():
        return dest
    longest = max(turns, key=lambda t: float(t["end"]) - float(t["start"]), default=None)
    if longest is None:
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate, channels, width = src.getframerate(), src.getnchannels(), src.getsampwidth()
            begin = float(longest["start"]) + _SAMPLE_LEAD_IN
            span = min(SAMPLE_SECS, float(longest["end"]) - begin)
            if span <= 0.5:
                return None
            start_frame = min(int(begin * rate), max(src.getnframes() - 1, 0))
            src.setpos(start_frame)
            frames = src.readframes(int(span * rate))
        if not frames:
            return None
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(frames)
    except (OSError, wave.Error, ValueError):
        _log.debug("Could not save a voice sample for %s", voice_id, exc_info=True)
        return None
    _log.info("Saved a %.0fs sample for %s", span, voice_id)
    return dest


def speech_by_speaker(turns: list[dict]) -> dict[str, float]:
    """Total speaking time per diarized label, for the enrolment quality gate."""
    totals: dict[str, float] = {}
    for turn in turns:
        speaker = turn.get("speaker")
        if not speaker:
            continue
        try:
            duration = float(turn["end"]) - float(turn["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if duration > 0:
            totals[speaker] = totals.get(speaker, 0.0) + duration
    return totals


def enrol(
    embeddings: dict,
    turns: list[dict],
    *,
    model: str,
    space: str,
    meeting: str = "",
    self_label: str | None = None,
    cluster_others: bool = False,
    audio_path=None,
) -> list[str]:
    """Carry one meeting's speakers into the persistent identities.

    Shared by the live recording path and by transcribing a file after the
    fact, which differ in exactly one thing: live capture keeps the mic and
    the system audio apart, so it can say which speaker is the user. A file
    on disk is already mixed, so ``self_label`` is None there and the user's
    own voice is enrolled only if it *matches* an identity already known —
    which is the honest answer, since nothing in a mixed recording can tell
    the user's voice from anyone else's.

    Returns the voice ids touched.
    """
    learned: list[str] = []
    speech = speech_by_speaker(turns)

    if self_label is not None and self_label in embeddings:
        # The mic/system split already says who this is, so the founding
        # bar buys nothing: a first sighting of the user is trustworthy at
        # the joining bar.
        voice_id = observe(
            embeddings[self_label],
            model,
            speech.get(self_label, 0.0),
            meeting,
            self_label,
            space,
            founding_secs=MIN_ENROLL_SECS,
        )
        # Naming is idempotent, and re-asserting it each meeting repairs the
        # case where the user's voice was first seen (unnamed) in a recording
        # without system audio.
        if voice_id:
            name_voice(voice_id, SELF)
            learned.append(voice_id)
            _keep_sample(voice_id, audio_path, turns, self_label)

    if cluster_others:
        for label, vector in embeddings.items():
            if label == self_label:
                continue
            other_id = observe(vector, model, speech.get(label, 0.0), meeting, label, space)
            if other_id:
                learned.append(other_id)
                _keep_sample(other_id, audio_path, turns, label)

    return learned


def _keep_sample(voice_id: str, audio_path, turns: list[dict], label: str) -> None:
    """Save a clip of this speaker while the recording still exists.

    A meeting's audio is deleted as soon as its transcript is saved, so this
    is the only chance — without it a voice can never be identified by ear.
    """
    if not audio_path:
        return
    try:
        save_sample(voice_id, audio_path, [t for t in turns if t.get("speaker") == label])
    except Exception:
        _log.debug("Could not keep a sample for %s", voice_id, exc_info=True)


def enrolment_health(learned: list[str], turns: list[dict]) -> tuple[bool, str]:
    """(ok, detail) describing what this meeting's enrolment achieved.

    "Ran and enrolled nobody" is the shape the silent failure takes: every
    meeting looks fine and `trnscrb voices` quietly never grows. A meeting
    where nobody spoke for long enough is not that — it is the enrolment bar
    doing its job, and flagging it would train the user to ignore the warning
    that matters.
    """
    if learned:
        return True, f"{len(learned)} voice(s)"
    speech = speech_by_speaker(turns)
    bar = _founding_secs()
    enrollable = [label for label, secs in speech.items() if secs >= bar]
    if not enrollable:
        return True, f"nobody spoke the {bar:.0f}s needed to enrol"
    return False, f"{len(enrollable)} speaker(s) spoke long enough but none was enrolled"


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
    sample_path(voice_id).unlink(missing_ok=True)  # forgetting includes the audio
    _log.info("Forgot voice %s", voice_id)
    return True


def prune(max_age_days: int = PRUNE_AFTER_DAYS, dry_run: bool = False) -> list[str]:
    """Forget unnamed voices heard exactly once and not since ``max_age_days``.

    A name is a promise to keep; a second sighting is evidence the person
    recurs. Lacking both, an old single sighting is noise the list is
    better without. Returns the ids forgotten (or, with ``dry_run``, that
    would be).
    """
    from datetime import timedelta

    data = load()
    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat(timespec="seconds")
    stale = [
        voice_id
        for voice_id, entry in sorted((data.get("voices") or {}).items())
        if not entry.get("name")
        and int(entry.get("observations", 0)) <= 1
        and (entry.get("updated_at") or "") < cutoff
    ]
    if dry_run or not stale:
        return stale
    for voice_id in stale:
        del data["voices"][voice_id]
        sample_path(voice_id).unlink(missing_ok=True)
    _save(data)
    _log.info("Pruned %d voice(s) heard once, over %d days ago", len(stale), max_age_days)
    return stale


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
            "clip": sample_path(voice_id).is_file(),
            "meetings": [
                s.get("meeting", "") for s in (entry.get("seen") or []) if s.get("meeting")
            ],
        }
        for voice_id, entry in sorted((data.get("voices") or {}).items())
    ]
