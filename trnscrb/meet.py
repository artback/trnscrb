"""Learn real speaker names from an exported Google Meet transcript.

Meet labels every turn with the speaker's Google account name. That is the one
thing local diarization can never produce: it can tell two voices apart, but
not who they are. Matching Meet's turns against our own supplies the missing
names — and because voiceprints persist, a name learned here attaches to the
voice itself and follows that person into meetings Google never transcribed.

Alignment is by *text*, not time. Meet starts its clock when someone presses
transcribe; we start ours on mic activity, usually earlier. Rather than trying
to reconcile two drifting clocks, each Meet turn is matched to our segments by
what was said, which is stable regardless of offset.

Meet writes its transcripts to Drive as Google Docs, which sync as URL stubs
rather than text, so nothing here can read Drive directly. Export the doc as
plain text (Docs → File → Download → Plain text) and this reads that.
"""

import difflib
import re
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.meet")

# "Jonathan Artback: we should split the network" — the shape every Meet
# export shares, whatever decoration surrounds it.
_TURN_RE = re.compile(r"^\s*(?P<speaker>[^:\n]{1,60}?)\s*:\s*(?P<text>\S.*)$")
# Standalone timestamps ("00:04:12", "4:12") that some exports interleave.
_TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")
# Lines a transcript header uses that would otherwise parse as "Name: value".
_HEADER_KEYS = frozenset(
    "attendees participants transcript date meeting notes summary duration organizer".split()
)

# A label must win by this margin of matched words to be renamed at all;
# a name attached to the wrong voice is worse than no name.
_MIN_VOTE_WORDS = 12
_MIN_VOTE_SHARE = 0.6
# How similar two turns must read before their words count as the same speech.
_MIN_SIMILARITY = 0.6


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def _looks_like_a_name(candidate: str) -> bool:
    """Reject header and title lines that happen to contain a colon.

    "Polycloud Redesign - 2026/08/10 15:01 CEST" splits at the time and would
    otherwise be read as a speaker called "…2026/08/10 15".
    """
    if any(ch.isdigit() for ch in candidate):
        return False
    return 1 <= len(candidate.split()) <= 5


def parse(path: Path) -> list[dict]:
    """Turns as [{speaker, text}] from an exported Meet transcript."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        _log.warning("Could not read Meet transcript %s", path)
        return []

    turns: list[dict] = []
    for line in raw.splitlines():
        if not line.strip() or _TIMESTAMP_RE.match(line):
            continue
        match = _TURN_RE.match(line)
        if not match:
            # A continuation of the previous turn, wrapped onto its own line.
            if turns and line.startswith((" ", "\t")):
                turns[-1]["text"] += " " + line.strip()
            continue
        speaker = match.group("speaker").strip()
        if speaker.lower().rstrip(":") in _HEADER_KEYS:
            continue
        # Timestamps sometimes prefix the name: "00:04:12 Jonathan Artback".
        speaker = re.sub(r"^\d{1,2}:\d{2}(:\d{2})?\s*", "", speaker).strip()
        if not speaker or not _looks_like_a_name(speaker):
            continue
        turns.append({"speaker": speaker, "text": match.group("text").strip()})
    _log.info("Parsed %d turn(s) from %s", len(turns), path.name)
    return turns


def parse_transcript(text: str) -> list[dict]:
    """Our own saved transcript, back into [{speaker, text}] blocks."""
    from trnscrb.storage import _SEPARATOR

    _, _, body = text.partition(_SEPARATOR)
    blocks: list[dict] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            blocks.append({"speaker": stripped[1:-1], "text": ""})
            continue
        if not blocks:
            continue
        # "  02:59  Hopefully, more people will see."
        spoken = re.sub(r"^\d{1,2}:\d{2}(:\d{2})?\s+", "", stripped)
        blocks[-1]["text"] = f"{blocks[-1]['text']} {spoken}".strip()
    return blocks


def map_speakers(ours: list[dict], theirs: list[dict]) -> dict[str, str]:
    """Which real name belongs to each of our diarized labels.

    Every Meet turn is matched to the most similar of our blocks, and the
    words it contributed are credited to that pairing. A label is only
    renamed when one name wins clearly — an ambiguous vote leaves the label
    alone rather than guessing.
    """
    if not ours or not theirs:
        return {}

    our_words = [_words(b["text"]) for b in ours]
    votes: dict[str, dict[str, int]] = {}
    for turn in theirs:
        turn_words = _words(turn["text"])
        if len(turn_words) < 3:
            continue  # too short to identify anyone
        best_index, best_score = -1, 0.0
        for index, words in enumerate(our_words):
            if not words:
                continue
            score = difflib.SequenceMatcher(None, turn_words, words).quick_ratio()
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0 or best_score < _MIN_SIMILARITY:
            continue
        label = ours[best_index]["speaker"]
        votes.setdefault(label, {})
        votes[label][turn["speaker"]] = votes[label].get(turn["speaker"], 0) + len(turn_words)

    mapping: dict[str, str] = {}
    for label, tally in votes.items():
        total = sum(tally.values())
        name, count = max(tally.items(), key=lambda kv: kv[1])
        if count < _MIN_VOTE_WORDS or count / total < _MIN_VOTE_SHARE:
            _log.debug("Leaving %s unnamed: best %s has %d/%d words", label, name, count, total)
            continue
        mapping[label] = name
    return mapping


def apply_names(text: str, mapping: dict[str, str]) -> str:
    """Rewrite our speaker headings with the real names."""
    if not mapping:
        return text
    for label, name in mapping.items():
        text = text.replace(f"[{label}]", f"[{name}]")
        text = re.sub(rf"^  {re.escape(label)}\b", f"  {name}", text, flags=re.MULTILINE)
    return text


def glossary_candidates(ours: list[dict], theirs: list[dict]) -> list[tuple[str, str]]:
    """(heard, actual) pairs where our text and Meet's disagree on one word.

    Suggestions only. Meet's recogniser is not ground truth either, so these
    are worth a human glance before they become rewrite rules.
    """
    ours_text = " ".join(b["text"] for b in ours)
    theirs_text = " ".join(t["text"] for t in theirs)
    a, b = _words(ours_text), _words(theirs_text)
    if not a or not b:
        return []

    seen: dict[tuple[str, str], int] = {}
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        ours_run, theirs_run = a[i1:i2], b[j1:j2]
        # One-for-one is the simple mishearing. Many-for-one is the compound
        # case and by far the most common for product names: a recogniser
        # that has never seen "Policloud" writes "poly cloud".
        if len(theirs_run) == 1 and 1 <= len(ours_run) <= 3:
            heard, actual = " ".join(ours_run), theirs_run[0]
        elif len(ours_run) == 1 and 1 < len(theirs_run) <= 3:
            heard, actual = ours_run[0], " ".join(theirs_run)
        else:
            continue
        if heard == actual or len(actual.replace(" ", "")) < 4:
            continue
        seen[(heard, actual)] = seen.get((heard, actual), 0) + 1
    return [pair for pair, count in sorted(seen.items(), key=lambda kv: -kv[1]) if count >= 2]
