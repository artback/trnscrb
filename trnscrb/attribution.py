"""Attribute transcript segments to "Me" vs "Them" — no diarization model.

The recorder captures the microphone and the system audio (other meeting
participants) as separate streams before mixing them, and keeps a per-block
energy timeline of each. Whichever stream dominates during a transcript
segment tells us who was speaking: mic → "Me", system audio → "Them".

Interplay with pyannote diarization: attribution runs after the diarizer
merge. Mic-dominated segments are always relabelled "Me" (the diarizer can't
know which voice is the user's); system-dominated segments keep the
diarizer's finer-grained speaker label when one exists, else become "Them".
"""

import numpy as np

from trnscrb.log import get_logger
from trnscrb.recorder import SAMPLE_RATE

_log = get_logger("trnscrb.attribution")

_NOISE_FLOOR = 1e-7  # mean-square; blocks quieter than this are silence
# One stream must beat the other by this factor for a confident label —
# below it (crosstalk, echo without headphones) the louder stream wins only
# when the segment has no label yet.
_DOMINANCE = 2.0

_DIARIZER_PLACEHOLDERS = (None, "", "Unknown")

THEM = "Them"


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
        if mic < _NOISE_FLOOR and system < _NOISE_FLOOR:
            continue

        if mic >= system * _DOMINANCE:
            seg["speaker"] = "Me"
            labelled += 1
        elif system >= mic * _DOMINANCE:
            if seg.get("speaker") in _DIARIZER_PLACEHOLDERS:
                seg["speaker"] = "Them"
            labelled += 1
        elif seg.get("speaker") in _DIARIZER_PLACEHOLDERS:
            # Ambiguous (crosstalk/echo) — fall back to the louder stream.
            seg["speaker"] = "Me" if mic > system else "Them"
            labelled += 1

    if labelled:
        _log.debug("Attributed %d/%d segments via stream energy", labelled, len(segments))
    return segments
