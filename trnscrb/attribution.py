"""Attribute transcript segments to "Me" vs "Them" — no diarization model.

The recorder captures the microphone and the system audio (other meeting
participants) as separate streams before mixing them, and keeps a per-block
energy timeline of each. Attribution turns that into speaker labels:

  system stream speaking -> "Them"   (keeps a diarizer sublabel if present)
  only the mic speaking  -> "Me"

Two properties make this robust for headphones *and* speakers:

1. Each stream is compared to *its own* speech level, not to the other
   stream's. The mic and the system-audio path are captured through entirely
   different gains, so their raw energies aren't comparable — an earlier
   "louder stream wins" rule mislabeled a stream simply because it was
   recorded hotter. Normalizing each stream against itself removes that bias.

2. The system stream is authoritative for "Them". A conferencing app never
   plays the user's own mic back to them, so speech-level energy there means
   *someone else* is talking — even when the mic is just as hot from acoustic
   bleed (the other participant coming out of laptop speakers and back into
   the mic). Bleed inflates the mic, never the system stream, so keying "Them"
   off the system stream sidesteps it.

Interplay with pyannote diarization: attribution runs after the diarizer
merge. Mic-only segments are always relabelled "Me" (the diarizer can't know
which voice is the user's); segments where the system stream is active keep
the diarizer's finer-grained speaker label when one exists, else become "Them".
"""

import math

import numpy as np

from trnscrb.log import get_logger
from trnscrb.recorder import SAMPLE_RATE

_log = get_logger("trnscrb.attribution")

# A block above this mean-square may carry speech; below it is (suppressed)
# silence or comfort noise. Conferencing apps aggressively gate silent
# participants, so real speech sits far above this and an idle stream far below.
_PRESENT = 3e-5
# A block counts as its stream speaking at this fraction of the stream's own
# typical loud level. Comparing each stream to itself makes the decision
# independent of the two capture paths' very different gains.
_ACTIVE_FRACTION = 0.15
# Percentile of a stream's above-silence blocks taken as its "loud" reference.
_ACTIVE_PCT = 85
# Fewer present blocks than this ⇒ the stream never really spoke (absent or
# failed capture); nothing in it counts as speech.
_MIN_ACTIVE_BLOCKS = 3

_DIARIZER_PLACEHOLDERS = (None, "", "Unknown")

THEM = "Them"


def _speech_floor(energy: np.ndarray) -> float:
    """Mean-square above which this stream counts as actively speaking.

    Derived from the stream's own loud level (a high percentile of its
    above-silence blocks), so a quietly-captured stream is measured against
    itself, not against the other, louder stream. Returns +inf when the stream
    never rises above the silence gate, so nothing in it is treated as speech.
    """
    active = energy[energy > _PRESENT]
    if active.size < _MIN_ACTIVE_BLOCKS:
        return math.inf
    return float(np.percentile(active, _ACTIVE_PCT)) * _ACTIVE_FRACTION


# A speaker must be this dominantly mic-only before we call them the user.
# Anything less means the diarizer's cluster mixes the user with someone else,
# and enrolling it would poison the fingerprint.
_SELF_PURITY = 0.8
# The runner-up must have at most this share of the winner's mic-only time,
# else which cluster is the user is genuinely ambiguous.
_SELF_MARGIN = 0.5


def self_speaker(turns: list[dict], timeline) -> tuple[str | None, float]:
    """Which diarized speaker is the user, and how much they spoke.

    The system stream cannot contain the user's own voice, so a speaker whose
    turns are consistently mic-only is the user — the same signal
    label_segments uses, aggregated per diarized speaker instead of per
    transcript segment.

    Returns (label, mic_only_secs), or (None, 0.0) when no speaker is clearly
    and unambiguously the user. Deliberately conservative: a wrong answer
    trains a fingerprint on someone else's voice.
    """
    offsets, mic_energy, sys_energy = timeline
    if len(offsets) == 0 or not turns:
        return None, 0.0

    mic_floor = _speech_floor(mic_energy)
    sys_floor = _speech_floor(sys_energy)
    if not math.isfinite(mic_floor):
        return None, 0.0  # the mic never spoke; nothing here is the user

    times = offsets.astype(np.float64) / SAMPLE_RATE
    mic_only: dict[str, float] = {}
    total: dict[str, float] = {}
    for turn in turns:
        speaker = turn.get("speaker")
        if not speaker:
            continue
        try:
            start = float(turn["start"])
            end = float(turn["end"])
        except (KeyError, TypeError, ValueError):
            continue
        window = (times >= start) & (times < max(end, start + 0.1))
        if not window.any():
            continue
        duration = max(end - start, 0.0)
        total[speaker] = total.get(speaker, 0.0) + duration
        # An infinite system floor means that stream never spoke, so nothing
        # here is "Them" — a mic-only recording is all user.
        if float(sys_energy[window].mean()) < sys_floor:
            if float(mic_energy[window].mean()) >= mic_floor:
                mic_only[speaker] = mic_only.get(speaker, 0.0) + duration

    if not mic_only:
        return None, 0.0

    ranked = sorted(mic_only.items(), key=lambda kv: kv[1], reverse=True)
    best, best_secs = ranked[0]
    if total.get(best, 0.0) <= 0 or best_secs / total[best] < _SELF_PURITY:
        _log.debug("No self speaker: %s is not dominantly mic-only", best)
        return None, 0.0
    if len(ranked) > 1 and ranked[1][1] > best_secs * _SELF_MARGIN:
        _log.debug("No self speaker: %s and %s both look mic-only", best, ranked[1][0])
        return None, 0.0
    return best, best_secs


def name_from_calendar(segments: list[dict], event: dict | None) -> str | None:
    """Rename "Them" to the other attendee when the meeting has exactly one.

    A 1:1 is the common case and the calendar already tells us who it is, so
    the transcript can say "Anna" instead of "Them" without any diarization
    model. With more attendees we cannot tell voices apart, so the generic
    label stays — a plausible-looking wrong name is worse than "Them".
    """
    if not event:
        return None
    attendees = [str(a).strip() for a in (event.get("attendees") or []) if str(a).strip()]
    # Calendar lists the organiser/self too; anything but one counterpart is
    # ambiguous.
    others = [a for a in attendees if not _looks_like_self(a)]
    if len(others) != 1:
        return None
    name = others[0]
    renamed = 0
    for seg in segments:
        if seg.get("speaker") == THEM:
            seg["speaker"] = name
            renamed += 1
    if renamed:
        _log.info("Named %d segments after the other attendee (%s)", renamed, name)
        return name
    return None


def _looks_like_self(attendee: str) -> bool:
    """True if this attendee is probably the user running trnscrb."""
    import getpass
    import os

    candidates = {
        os.environ.get("USER", ""),
        getpass.getuser(),
        os.environ.get("TRNSCRB_USER_NAME", ""),
    }
    name = attendee.casefold()
    for candidate in candidates:
        candidate = candidate.strip().casefold()
        if candidate and (candidate == name or candidate in name.split()):
            return True
    return False


def label_segments(segments: list[dict], timeline) -> list[dict]:
    """Assign "Me"/"Them" speaker labels in place from the energy timeline.

    ``timeline`` is Recorder.attribution_timeline(). Segments outside the
    timeline or with both streams silent are left untouched.
    """
    offsets, mic_energy, sys_energy = timeline
    if len(offsets) == 0:
        return segments

    mic_floor = _speech_floor(mic_energy)
    sys_floor = _speech_floor(sys_energy)
    times = offsets.astype(np.float64) / SAMPLE_RATE
    labelled = 0
    for seg in segments:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            continue
        window = (times >= start) & (times < max(end, start + 0.1))
        if not window.any():
            continue
        mic = float(mic_energy[window].mean())
        system = float(sys_energy[window].mean())

        if system >= sys_floor:
            # Someone else is transmitting speech. The system stream can't
            # contain the user's own voice, so this is "Them" even when the
            # mic is just as hot from speaker bleed. Keep a diarizer sublabel
            # if one exists, else the generic label.
            if seg.get("speaker") in _DIARIZER_PLACEHOLDERS:
                seg["speaker"] = THEM
            labelled += 1
        elif mic >= mic_floor:
            # Others silent, only the mic is active → the user. Override any
            # diarizer sublabel: it can't know which voice is the user's.
            seg["speaker"] = "Me"
            labelled += 1
        # else: both streams silent — leave the segment untouched.

    if labelled:
        _log.debug("Attributed %d/%d segments via stream energy", labelled, len(segments))
    return segments
