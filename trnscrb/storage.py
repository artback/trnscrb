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


def format_transcript(segments: list[dict], started_at: datetime, meeting_name: str) -> str:
    if segments:
        duration = _fmt_time(segments[-1]["end"])
    else:
        duration = "00:00"

    lines = [
        f"Meeting: {meeting_name}",
        f"Date:    {started_at.strftime('%Y-%m-%d %H:%M')}",
        f"Duration: {duration}",
        "",
        "=" * 60,
        "",
    ]
    current_speaker = None
    for seg in segments:
        speaker = seg.get("speaker") or "Unknown"
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"[{speaker}]")
            current_speaker = speaker
        lines.append(f"  {_fmt_time(seg['start'])}  {clean_filler_words(seg['text'])}")

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
