"""Temporal attribution: assign names to segments from call samples.

Inputs = list of samples (ts, participants, speaking) + transcript segments
(start_ms, end_ms, speaker).  Output = {cluster_id: name} using the same
rules as yogurt:

- A name is the one participant whose speaking intervals overlap >50% of a
  segment's duration (``MOST``).
- A reading stands until the next reading arrives, but never longer than
  ``HOLD_MS`` — a stuck app never keeps naming someone.
- Two-person calls: when exactly two participants are seen and one is "me",
  assign the non-me name to the non-me speaker regardless of speaking markers.
- No sample coverage → no name (``None`` → ``"Them"`` fallback in caller).

Everything here is headless and testable with hand-computed fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from trnscrb.log import get_logger

_log = get_logger("trnscrb.callnames.attr")

# A reading stands until the next one, but never longer than this:
# a stuck app never keeps naming someone.
HOLD_MS = 2000

# A name needs more than this share of the segment duration.
MOST = 0.5


def _norm(s: str) -> str:
    """Normalized name: NFC, single-space, lowercase."""
    return s.strip().casefold()


# ── public types ─────────────────────────────────────────────────────────────

@dataclass
class Sample:
    """One reading of the call app."""
    ts: float  # wall-clock timestamp (seconds)
    speaking: list[str]  # who the app marks as talking
    participants: list[str]  # everyone in the call (incl. the user)
    me: str | None = None  # the user (from "(me)", "You")
    ok: bool = True  # False if the tree was unreadable


@dataclass
class Segment:
    """One transcript segment from diarization."""
    start_ms: float  # milliseconds
    end_ms: float  # milliseconds
    speaker: str  # diarized cluster label


@dataclass
class AttributionResult:
    result: dict[str, str] = None  # cluster_id -> name (None means "Them")
    participants: list[str] = None  # everyone seen (excluding "me")

    def __post_init__(self):
        if self.result is None:
            self.result = {}
        if self.participants is None:
            self.participants = []


# ── helpers ────────────────────────────────────────────────────────────────────

def _is_me(name: str, sample: Sample, my_names: list[str]) -> bool:
    n = _norm(name)
    return (sample.me is not None and _norm(sample.me) == n) or any(
        m and _norm(m) == n for m in my_names
    )


def _unique(names: list[str]) -> list[str]:
    """Case-insensitive unique, preserving first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        k = _norm(n)
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def _samples_timeline(
    samples: list[Sample],
    meeting_offset_ms: float = 0.0,
) -> Iterator[tuple[float, float, Sample]]:
    """Yield (start_ms, end_ms, sample) for each reading's active window.

    Each reading is active from its timestamp until the next reading (plus
    the hold threshold as a safety net).  Times are converted to meeting-
    relative milliseconds using ``meeting_offset_ms``.
    """
    if not samples:
        return
    t0_ms = samples[0].ts * 1000.0 - meeting_offset_ms
    sorted_samples = sorted(samples, key=lambda s: s.ts)
    for i, s in enumerate(sorted_samples):
        ms = s.ts * 1000.0 - meeting_offset_ms
        if i + 1 < len(sorted_samples):
            next_ms = sorted_samples[i + 1].ts * 1000.0 - meeting_offset_ms
        else:
            next_ms = ms + HOLD_MS
        yield ms, next_ms, s


# ── core logic ─────────────────────────────────────────────────────────────────

def name_for(
    span_start_ms: float,
    span_end_ms: float,
    samples: list[Sample],
    my_names: list[str] = (),
    meeting_offset_ms: float = 0.0,
) -> str | None:
    """The name for a line spoken over [start_ms, end_ms], or ``None``."""
    if not samples:
        return None

    span_dur = span_end_ms - span_start_ms

    # Build an overlap map: name -> total_ms of overlap
    time: dict[str, float] = {}
    name_map: dict[str, str] = {}
    for seg_start, seg_end, s in _samples_timeline(samples, meeting_offset_ms):
        overlap = min(seg_end, span_end_ms) - max(seg_start, span_start_ms)
        if overlap <= 0:
            continue
        who = _who_at(s, my_names)
        if who is None:
            continue
        k = _norm(who)
        time[k] = time.get(k, 0.0) + overlap
        name_map[k] = who

    if not time:
        return None

    best_k, best_ms = max(time.items(), key=lambda kv: kv[1])
    if best_ms > MOST * span_dur:
        return name_map[best_k]
    return None


def _who_at(sample: Sample, my_names: list[str]) -> str | None:
    """Who the other side was at one reading, or ``None``."""
    if not sample.ok:
        return None
    people = _unique(sample.participants)
    roster = {_norm(p): p for p in people}

    # Floating speaker tile naming someone not in the roster → not a 2-person
    # call signal; it happens when the roster hasn't refreshed yet.
    speaker_outside = any(
        not _is_me(p, sample, my_names) and _norm(p) not in roster
        for p in sample.speaking
    )

    # Two-person call: the non-me person gets the other side.
    if len(people) == 2 and not speaker_outside:
        me_people = [p for p in people if _is_me(p, sample, my_names)]
        if len(me_people) == 1:
            others = [p for p in people if not _is_me(p, sample, my_names)]
            if others:
                return others[0]

    # General case: exactly one non-me person marked speaking.
    others = [p for p in sample.speaking if not _is_me(p, sample, my_names)]
    others_uniq = _unique(others)
    if len(others_uniq) == 1:
        return others_uniq[0]
    return None


def others_in(
    samples: list[Sample],
    my_names: list[str] = (),
) -> list[str]:
    """Everyone seen in the call except the user, in first-seen order."""
    out: dict[str, str] = {}
    for s in samples:
        for p in s.participants:
            if _is_me(p, s, my_names):
                continue
            k = _norm(p)
            if k not in out:
                out[k] = p
    return list(out.values())


def attributed_meeting(
    samples: list[Sample],
    segments: list[Segment],
    my_names: list[str] = (),
    meeting_offset_ms: float = 0.0,
) -> AttributionResult:
    """Assign names to every segment and collect participants.

    Returns an ``AttributionResult`` with ``result`` (speaker_label → name)
    and ``participants`` (the list of non-me people seen during the call).
    """
    if not samples:
        return AttributionResult({}, [])

    participants = others_in(samples, my_names)

    # Special case: exactly two participants → assign non-me to the non-me
    # cluster unconditionally.
    two_person = len(participants) == 1 and samples[0].participants and (
        len(samples[0].participants) == 2
    )

    result: dict[str, str] = {}
    for seg in segments:
        if seg.speaker in result:
            continue

        span_start = seg.start_ms
        span_end = seg.end_ms
        name = name_for(span_start, span_end, samples, my_names, meeting_offset_ms)

        if name:
            result[seg.speaker] = name
        elif two_person and participants and seg.speaker.casefold() not in ("me", "you"):
            result[seg.speaker] = participants[0]

    return AttributionResult(result, participants)