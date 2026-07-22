"""Talk-time statistics derived from speaker-attributed transcript segments.

Dual-stream capture already tells us who spoke each segment ("Me" from the
microphone, everyone else from the system-audio stream — see
trnscrb.attribution), so the split costs nothing extra to compute and needs
no diarization model.
"""

from trnscrb.log import get_logger

_log = get_logger("trnscrb.analytics")

ME = "Me"
_UNKNOWN = ("Unknown", "", None)
# Gaps shorter than this don't break a monologue — normal breathing pauses
# between sentences by the same speaker.
_MONOLOGUE_GAP_SECS = 2.0


def talk_time(segments: list[dict]) -> dict:
    """Speaking time per speaker, turn count, and longest monologue.

    Returns ``{}`` when no segment carries usable timing, so callers can skip
    the section rather than render a meaningless one.
    """
    totals: dict[str, float] = {}
    turns = 0
    longest = 0.0
    longest_speaker = None

    previous_speaker = None
    run_start: float | None = None
    run_end: float | None = None

    def _close_run() -> None:
        nonlocal longest, longest_speaker
        if run_start is None or run_end is None:
            return
        length = run_end - run_start
        if length > longest:
            longest = length
            longest_speaker = previous_speaker

    for seg in segments:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        speaker = seg.get("speaker") or "Unknown"
        totals[speaker] = totals.get(speaker, 0.0) + (end - start)

        if speaker != previous_speaker:
            _close_run()
            turns += 1
            previous_speaker = speaker
            run_start, run_end = start, end
        elif run_end is not None and start - run_end > _MONOLOGUE_GAP_SECS:
            _close_run()  # same speaker, but after a long pause
            run_start, run_end = start, end
        else:
            run_end = end
    _close_run()

    total = sum(totals.values())
    if total <= 0:
        return {}
    return {
        "totals": totals,
        "total": total,
        "turns": turns,
        "longest_monologue": longest,
        "longest_monologue_speaker": longest_speaker,
    }


def format_talk_time(stats: dict) -> str:
    """One-line-per-speaker summary, or "" when there is nothing to say."""
    if not stats:
        return ""
    totals: dict[str, float] = stats["totals"]
    # Speaker labels are only meaningful if attribution actually ran.
    if all(name in _UNKNOWN for name in totals):
        return ""

    total = stats["total"]
    lines = ["Talk time:"]
    for name, secs in sorted(totals.items(), key=lambda kv: -kv[1]):
        share = 100 * secs / total
        lines.append(f"  {name:<12} {_fmt_duration(secs):>8}  {share:4.0f}%")
    lines.append(f"  {'turns':<12} {stats['turns']:>8}")
    longest = stats.get("longest_monologue") or 0
    if longest >= 30:
        who = stats.get("longest_monologue_speaker") or "?"
        lines.append(f"  {'longest':<12} {_fmt_duration(longest):>8}  ({who})")
    return "\n".join(lines)


# Below this share of speech a "meeting" is really an idle tab left open —
# the classic case is a call that ended while the browser kept the mic warm.
LOW_SPEECH_RATIO = 0.15


def capture_health(
    segments: list[dict],
    recorded_secs: float,
    system_audio: bool,
) -> dict:
    """Facts about how well a recording captured the meeting.

    Silent failures (mic-only when both sides were expected, or an hour of an
    empty tab) are indistinguishable from success until someone reads the
    transcript — this puts them in the transcript itself.
    """
    speech = 0.0
    for seg in segments:
        try:
            speech += max(0.0, float(seg["end"]) - float(seg["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    ratio = (speech / recorded_secs) if recorded_secs > 0 else 0.0
    return {
        "recorded": recorded_secs,
        "speech": speech,
        "speech_ratio": ratio,
        "system_audio": bool(system_audio),
        "mostly_silent": recorded_secs > 300 and ratio < LOW_SPEECH_RATIO,
    }


def format_capture_health(health: dict) -> str:
    """One-line capture summary, plus a warning when the recording looks idle."""
    if not health or not health.get("recorded"):
        return ""
    source = "mic + system audio" if health["system_audio"] else "microphone only"
    line = (
        f"Captured: {source} · {_fmt_duration(health['recorded'])} recorded · "
        f"{_fmt_duration(health['speech'])} speech ({health['speech_ratio'] * 100:.0f}%)"
    )
    if not health["system_audio"]:
        line += "\n  ⚠︎ Other participants were not captured — check Screen Recording permission."
    if health.get("mostly_silent"):
        line += "\n  ⚠︎ Mostly silence — the meeting may have ended while a tab stayed open."
    return line


def _fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"
