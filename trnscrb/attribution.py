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
