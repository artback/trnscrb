"""Write trnscrb content into the user's Obsidian vault.

Transcripts are mirrored as notes so action items can [[backlink]] to them and
everything is browsable in Obsidian. The vault is auto-detected from Obsidian's
own config, or set explicitly via the `obsidian_vault` setting. When no vault is
available every function is a no-op, so the rest of trnscrb keeps working.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.obsidian")

_OBSIDIAN_CONFIG = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"

# Characters Obsidian disallows (or mangles) in a note name / wikilink.
_UNSAFE = re.compile(r'[\\/:*?"<>|#^\[\]]')


def vault_path() -> Path | None:
    """The Obsidian vault to write into — the setting, else auto-detected."""
    configured = str(settings.get("obsidian_vault") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else None
    return _detect_vault()


def meetings_dir() -> Path | None:
    """Subfolder inside the vault for trnscrb notes (created on demand)."""
    vault = vault_path()
    if not vault:
        return None
    sub = str(settings.get("obsidian_subdir") or "Meetings").strip() or "Meetings"
    return vault / sub


def note_name(meeting_name: str, started_at: datetime) -> str:
    """Stable, link-safe note name: '<date> <title>'."""
    date = started_at.strftime("%Y-%m-%d")
    safe = _UNSAFE.sub("-", meeting_name).strip()
    safe = re.sub(r"\s+", " ", safe) or "meeting"
    return f"{date} {safe}"


# Speaker labels that name no one: linking them would fill the graph with a
# node per diarizer cluster, and one "Me" node joined to every meeting.
_NOT_A_PERSON = re.compile(r"^(SPEAKER_\d+|Them|Me|Unknown|Participant \d+)$", re.IGNORECASE)
# Trailing words that distinguish one occurrence of a recurring meeting from
# the next; stripping them leaves the series name.
_SERIES_NOISE = re.compile(
    r"\s*[-–—]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(/\d{2,4})?|week \d+|#\d+|\(\d+\))\s*$",
    re.IGNORECASE,
)


def speakers_in(text: str) -> list[str]:
    """Real names appearing as speaker headings, in order of first appearance."""
    found: list[str] = []
    for match in re.finditer(r"^\[([^\]\n]{1,60})\]\s*$", text, flags=re.MULTILINE):
        name = match.group(1).strip()
        if not name or _NOT_A_PERSON.match(name) or name in found:
            continue
        found.append(name)
    return found


def topics_in(text: str) -> list[str]:
    """Glossary terms actually mentioned, which become shared topic nodes.

    The glossary is already a curated list of what this person talks about,
    so it needs no separate configuration to be useful here.
    """
    from trnscrb import glossary

    body = text.lower()
    found = []
    for term in glossary.terms():
        if not term or len(term) < 3:
            continue
        if re.search(rf"\b{re.escape(term.lower())}\b", body):
            found.append(term)
    return found


def series_name(meeting_name: str) -> str:
    """The recurring series a meeting belongs to, or its own name.

    Daily standups are otherwise six unconnected islands in the graph.
    """
    return _SERIES_NOISE.sub("", meeting_name).strip() or meeting_name


def _quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def build_note(meeting_name: str, started_at: datetime, text: str, duration: str = "") -> str:
    """The transcript wrapped in the properties Obsidian builds a graph from.

    Links live in the frontmatter rather than sprinkled through the transcript:
    they connect the note to people and topics in the graph without turning the
    spoken text into link soup.
    """
    people = speakers_in(text)
    topics = topics_in(text)
    series = series_name(meeting_name)

    lines = ["---", f"date: {started_at.strftime('%Y-%m-%d')}"]
    lines.append(f"time: {_quote(started_at.strftime('%H:%M'))}")
    if duration:
        lines.append(f"duration: {_quote(duration)}")
    lines.append("tags:")
    lines.append("  - meeting")
    if series and series != meeting_name:
        lines.append(f"series: {_quote(f'[[{_UNSAFE.sub("-", series)}]]')}")
    if people:
        lines.append("attendees:")
        lines += [f"  - {_quote(f'[[{_UNSAFE.sub("-", p)}]]')}" for p in people]
    if topics:
        lines.append("topics:")
        lines += [f"  - {_quote(f'[[{_UNSAFE.sub("-", t)}]]')}" for t in topics]
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n" + text


def mirror_transcript(
    meeting_name: str, started_at: datetime, text: str, duration: str = ""
) -> str | None:
    """Write the transcript as a vault note. Returns its note name for backlinks."""
    directory = meetings_dir()
    if not directory:
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        name = note_name(meeting_name, started_at)
        note = build_note(meeting_name, started_at, text, duration)
        _atomic_write(directory / f"{name}.md", note)
        return name
    except OSError:
        _log.warning("Could not mirror transcript into the Obsidian vault", exc_info=True)
        return None


def write_note(filename: str, text: str) -> Path | None:
    """Write an arbitrary note (e.g. the action-items index) into the vault."""
    directory = meetings_dir()
    if not directory:
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        _atomic_write(path, text)
        return path
    except OSError:
        _log.warning("Could not write %s into the Obsidian vault", filename, exc_info=True)
        return None


def read_note(filename: str) -> str | None:
    directory = meetings_dir()
    if not directory:
        return None
    path = directory / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ── helpers ───────────────────────────────────────────────────────────────


def _detect_vault() -> Path | None:
    try:
        data = json.loads(_OBSIDIAN_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vaults = data.get("vaults", {})
    fallback = None
    for entry in vaults.values():
        path = entry.get("path")
        if not path or not Path(path).is_dir():
            continue
        if entry.get("open"):  # prefer the vault the user has open
            return Path(path)
        fallback = fallback or Path(path)
    return fallback


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".md")
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
