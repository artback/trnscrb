"""Derive a human title for meetings that arrive without a calendar name.

Calendar meetings already have a good title; ad-hoc calls fall back to a
generic name ("Google Meet", "meeting-1549") that makes 100+ transcripts
impossible to skim. When the LLM summary runs it also emits a short TITLE we
reuse; otherwise a lightweight keyword heuristic gives something better than a
timestamp.
"""

import re
from collections import Counter

# Names that carry no meeting-specific information — worth replacing with a
# content title. Matched case-insensitively; "meeting-<time>" is matched by regex.
_GENERIC_NAMES = {
    "",
    "meeting",
    "google meet",
    "zoom",
    "microsoft teams",
    "teams",
    "slack huddle",
    "webex",
    "around",
    "tuple",
    "loom",
    "facetime",
    "discord",
}
_MEETING_STAMP = re.compile(r"^meeting-\d+$", re.IGNORECASE)

# Very common words that never make a useful title on their own.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "so",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "me",
    "them",
    "us",
    "my",
    "your",
    "our",
    "their",
    "do",
    "did",
    "does",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "just",
    "like",
    "yeah",
    "okay",
    "ok",
    "right",
    "know",
    "think",
    "going",
    "get",
    "got",
    "one",
    "really",
    "actually",
    "well",
    "there",
    "here",
    "what",
    "when",
    "where",
    "how",
    "why",
    "who",
    "not",
    "no",
    "yes",
    "about",
    "from",
    "then",
    "than",
    "some",
    "much",
    "more",
    "very",
    "good",
    "now",
}


def is_generic(name: str | None) -> bool:
    """True if the meeting name is a placeholder worth replacing with a title."""
    normalized = (name or "").strip().casefold()
    return normalized in _GENERIC_NAMES or bool(_MEETING_STAMP.match(normalized))


def from_enrichment(enrichment: str | None) -> str | None:
    """Pull the TITLE the summary model emitted, if any."""
    if not enrichment:
        return None
    lines = enrichment.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("TITLE:"):
            inline = line.split(":", 1)[1].strip()
            if inline:
                return _clean(inline)
            for nxt in lines[i + 1 :]:  # value on the following line
                if nxt.strip():
                    return _clean(nxt)
            return None
    return None


def local(segments: list[dict]) -> str | None:
    """Keyword-based fallback title when no LLM summary is available."""
    counts: Counter[str] = Counter()
    for seg in segments:
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(seg.get("text", ""))):
            low = word.casefold()
            if low not in _STOPWORDS:
                counts[low] += 1
    top = [w for w, n in counts.most_common(4) if n >= 2]
    if len(top) < 2:  # too little signal to name it
        return None
    return _clean(" ".join(w.capitalize() for w in top[:3]))


def _clean(title: str) -> str:
    title = re.sub(r"\s+", " ", title.strip().strip("\"'").strip())
    # Drop a trailing period; keep it short enough for a filename.
    return title.rstrip(".")[:60].strip()
