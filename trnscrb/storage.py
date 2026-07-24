"""Transcript file storage — saves and reads .txt files from ~/meeting-notes/."""

import re
from datetime import datetime
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.storage")

NOTES_DIR = Path.home() / "meeting-notes"


def ensure_notes_dir() -> Path:
    NOTES_DIR.mkdir(exist_ok=True)
    return NOTES_DIR


def get_transcript_path(meeting_name: str, started_at: datetime) -> Path:
    ensure_notes_dir()
    date_str = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "-", meeting_name)[:50]
    return NOTES_DIR / f"{date_str}_{safe_name}.txt"


_LIVE_SESSION_FILE = Path.home() / ".config" / "trnscrb" / "live_session.json"
_APP_STATE_FILE = Path.home() / ".config" / "trnscrb" / "app_state.json"


def write_app_state(**fields) -> None:
    """Publish state of the running app for CLI commands to read.

    TCC permissions are per process identity: a terminal running
    `trnscrb status` cannot answer questions about Trnscrb.app's own
    permissions — only the app process can, so it publishes them here.
    """
    import json
    import os

    try:
        _APP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = read_app_state() or {}
        state.update(fields)
        state["pid"] = os.getpid()
        _APP_STATE_FILE.write_text(json.dumps(state))
    except Exception:
        _log.debug("Could not write app state", exc_info=True)


def read_app_state() -> dict | None:
    """State published by a running app instance, or None if it isn't running."""
    import json
    import os

    try:
        data = json.loads(_APP_STATE_FILE.read_text())
        os.kill(int(data["pid"]), 0)  # raises if that process is gone
        return data
    except Exception:
        return None


# Substrings that mark a transcript as still being live-updated.
LIVE_MARKERS = (
    "[Recording in progress",
    "[Live — recording in progress",
)
_INTERRUPTED_NOTE = "[Recording was interrupted]"


def set_live_session(
    path: Path, meeting_name: str = "", started_at: datetime | None = None
) -> None:
    """Record which transcript is being live-updated (read by `trnscrb live`)."""
    import json
    import os

    try:
        payload: dict = {"path": str(path), "pid": os.getpid(), "meeting": meeting_name}
        if started_at is not None:
            payload["started_at"] = started_at.isoformat()
        _LIVE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LIVE_SESSION_FILE.write_text(json.dumps(payload))
    except Exception:
        _log.debug("Could not write live session file", exc_info=True)


_BOOKMARKS_FILE = Path.home() / ".config" / "trnscrb" / "bookmarks.json"


def clear_live_session() -> None:
    _LIVE_SESSION_FILE.unlink(missing_ok=True)
    _BOOKMARKS_FILE.unlink(missing_ok=True)


def add_bookmark(label: str = "") -> float | None:
    """Mark the current moment of the running recording. Returns its offset.

    None when nothing is recording. Deliberately keyed off the live session
    rather than a hotkey, so marking a moment needs no Accessibility grant —
    the menu item and `trnscrb bookmark` both land here.
    """
    import json

    info = get_live_session_info()
    if not info or not info.get("started_at"):
        return None
    try:
        started = datetime.fromisoformat(info["started_at"])
        offset = max(0.0, (datetime.now() - started).total_seconds())
        marks = read_bookmarks()
        marks.append({"at": offset, "label": label.strip()})
        _BOOKMARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BOOKMARKS_FILE.write_text(json.dumps(marks))
        _log.info("Bookmark at %.0fs%s", offset, f" — {label}" if label else "")
        return offset
    except Exception:
        _log.warning("Could not add bookmark", exc_info=True)
        return None


def read_bookmarks() -> list[dict]:
    import json

    try:
        marks = json.loads(_BOOKMARKS_FILE.read_text())
        return marks if isinstance(marks, list) else []
    except Exception:
        return []


def get_live_session_info() -> dict | None:
    """Details of the active recording session, or None.

    Authoritative: written by the recording process on start, cleared on stop,
    and validated against the recorder still being alive — so a crashed
    session can never present a stale file as live.
    """
    import json
    import os

    try:
        data = json.loads(_LIVE_SESSION_FILE.read_text())
        path = Path(data["path"])
        os.kill(int(data["pid"]), 0)  # raises if the recorder is gone
        if path.exists():
            data["path"] = path
            return data
    except Exception:
        pass
    return None


def get_live_session() -> Path | None:
    """Path of the actively updated live transcript, or None."""
    info = get_live_session_info()
    return info["path"] if info else None


def apply_retention() -> None:
    """Delete old files per the retention settings.

    Preserved audio is large (~230 MB/hour); default is to keep it 30 days.
    Transcripts are kept forever unless retention_transcript_days is set.
    A value of 0 disables deletion for that category.
    """
    import time

    from trnscrb import settings

    now = time.time()

    def _purge(pattern: str, days: int) -> None:
        if days <= 0:
            return
        cutoff = now - days * 86400
        for f in NOTES_DIR.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    _log.info("Retention: deleted %s (older than %d days)", f.name, days)
            except OSError:
                pass

    try:
        _purge("*.wav", int(settings.get("retention_audio_days") or 0))
        _purge("*.txt", int(settings.get("retention_transcript_days") or 0))
    except Exception:
        _log.debug("Retention pass failed", exc_info=True)


def finalize_orphaned_live_markers(max_age_secs: float = 24 * 3600) -> None:
    """Rewrite live markers left behind by crashed sessions.

    Live transcripts keep their in-progress marker forever if the recorder
    dies mid-meeting; months later `trnscrb live` would still present them
    as active. Any marker file untouched for max_age_secs gets its marker
    replaced with an interruption note.
    """
    import time

    now = time.time()
    try:
        for f in NOTES_DIR.glob("*.txt"):
            if now - f.stat().st_mtime < max_age_secs:
                continue
            text = f.read_text(encoding="utf-8")
            if not any(marker in text for marker in LIVE_MARKERS):
                continue
            lines = [
                _INTERRUPTED_NOTE if any(m in line for m in LIVE_MARKERS) else line
                for line in text.splitlines()
            ]
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _log.info("Finalized orphaned live transcript: %s", f.name)
    except Exception:
        _log.debug("Orphan live-marker cleanup failed", exc_info=True)


def preserve_audio(audio_path: Path, meeting_name: str, started_at: datetime) -> Path | None:
    """Move a recording into the notes folder instead of deleting it.

    Called when transcription fails — hours of meeting audio must never be
    thrown away because of a transient error. Returns the saved path.
    """
    import shutil

    try:
        dest = get_transcript_path(meeting_name, started_at).with_suffix(".wav")
        shutil.move(str(audio_path), dest)
        _log.warning("Transcription failed — audio preserved at %s", dest)
        return dest
    except Exception:
        _log.error("Could not preserve audio %s", audio_path, exc_info=True)
        return None


def save_transcript(path: Path, content: str) -> None:
    if not content or not content.strip():
        _log.warning("Skipping save_transcript: empty content")
        return
    _log.info("Saving transcript to %s", path)
    path.write_text(content, encoding="utf-8")


def list_transcripts() -> list[dict]:
    ensure_notes_dir()
    files = sorted(NOTES_DIR.glob("*.txt"), reverse=True)
    return [
        {
            "id": f.stem,
            "path": str(f),
            "name": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in files
    ]


def read_transcript(transcript_id: str) -> str | None:
    path = (NOTES_DIR / f"{transcript_id}.txt").resolve()
    if not path.is_relative_to(NOTES_DIR.resolve()):
        _log.warning("Path traversal blocked for transcript_id=%r", transcript_id)
        return None
    if not path.exists():
        _log.debug("Transcript not found: %s", path)
        return None
    return path.read_text(encoding="utf-8")


def format_transcript(
    segments: list[dict],
    started_at: datetime,
    meeting_name: str,
    bookmarks: list[dict] | None = None,
    health: dict | None = None,
) -> str:
    from trnscrb import analytics

    if segments:
        duration = _fmt_time(segments[-1]["end"])
    else:
        duration = "00:00"

    lines = [
        f"Meeting: {meeting_name}",
        f"Date:    {started_at.strftime('%Y-%m-%d %H:%M')}",
        f"Duration: {duration}",
    ]

    capture = analytics.format_capture_health(health or {})
    if capture:
        lines += ["", capture]

    summary = analytics.format_talk_time(analytics.talk_time(segments))
    if summary:
        lines += ["", summary]

    marks = sorted(bookmarks or [], key=lambda m: m.get("at", 0))
    if marks:
        lines += ["", "Bookmarks:"]
        for mark in marks:
            label = str(mark.get("label") or "").strip()
            lines.append(
                f"  ⭐ {_fmt_time(float(mark.get('at', 0)))}" + (f"  {label}" if label else "")
            )

    lines += ["", "=" * 60, ""]

    # Interleave bookmarks with the transcript so a marked moment is visible
    # in context, not just in the header.
    pending = list(marks)
    current_speaker = None
    for seg in segments:
        start = float(seg.get("start") or 0)
        while pending and float(pending[0].get("at", 0)) <= start:
            mark = pending.pop(0)
            label = str(mark.get("label") or "").strip()
            lines.append("")
            lines.append(
                f"  ⭐ {_fmt_time(float(mark.get('at', 0)))}" + (f"  {label}" if label else "")
            )
            lines.append("")
            current_speaker = None  # re-print the speaker after the marker
        speaker = seg.get("speaker") or "Unknown"
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"[{speaker}]")
            current_speaker = speaker
        lines.append(f"  {_fmt_time(seg['start'])}  {clean_filler_words(seg['text'])}")

    for mark in pending:  # marks after the last spoken segment
        label = str(mark.get("label") or "").strip()
        lines.append(
            f"  ⭐ {_fmt_time(float(mark.get('at', 0)))}" + (f"  {label}" if label else "")
        )

    return "\n".join(lines)


_FILLER_WORDS = {
    # Universal / cross-language hesitation sounds
    "um",
    "uh",
    "uhm",
    "umm",
    "erm",
    "ah",
    "eh",
    "hmm",
    "hm",
    "mm",
    "mhm",
    # English
    "like",
    "you know",
    "I mean",
    "sort of",
    "kind of",
    "basically",
    "actually",
    "right",
    "so yeah",
    # Swedish
    "liksom",
    "typ",
    "alltså",
    "asså",
    "ba",
    "öh",
    "äh",
    "ju",
    "va",
    "såhär",
    "eller hur",
    "på nåt sätt",
    # Spanish
    "pues",
    "bueno",
    "o sea",
    "este",
    "la verdad",
    "en plan",
    "digamos",
    # German
    "halt",
    "also",
    "quasi",
    "sozusagen",
    "eigentlich",
    "na ja",
    "genau",
    # French
    "genre",
    "en fait",
    "du coup",
    "voilà",
    "quoi",
    "bah",
    "ben",
    "euh",
}

_filler_alts = "|".join(re.escape(f) for f in sorted(_FILLER_WORDS, key=len, reverse=True))
_FILLER_PATTERN = re.compile(rf"\b(?:{_filler_alts})\b", re.IGNORECASE)


def clean_filler_words(text: str) -> str:
    """Remove common filler words/phrases that clutter transcripts.

    Supports English, Swedish, Spanish, German, and French fillers.
    """
    cleaned = _FILLER_PATTERN.sub("", text)
    # Collapse leftover punctuation artifacts and whitespace
    cleaned = re.sub(r"(,\s*){2,}", ", ", cleaned)  # repeated commas
    cleaned = re.sub(r"^\s*[,\s]+", "", cleaned)  # leading commas/space
    cleaned = re.sub(r"[,\s]+$", "", cleaned)  # trailing commas/space
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
