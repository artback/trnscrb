"""Custom work vocabulary applied during transcription.

Stored at ~/.config/trnscrb/glossary.json:

    {"terms": [
        {"term": "Hivenet", "aliases": ["high vnet", "hive net"]},
        {"term": "Kubernetes", "aliases": ["kubernets"]}
    ]}

The transcriber applies this to every segment as it is produced, so the saved
transcript already carries your terminology — it is not a post-hoc edit and
not part of any summary step. Two mechanisms:

  * Correction (all backends): alias phrases are rewritten to the canonical
    term, the canonical casing is normalised, and — optionally — single tokens
    that are near-spelling-matches of a term are nudged onto it.
  * Decode biasing (Whisper only): the terms are handed to faster-whisper as
    hotwords so the model itself leans toward them. Parakeet's MLX API has no
    such hook, which is why the correction pass exists.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.glossary")

_GLOSSARY_FILE = Path.home() / ".config" / "trnscrb" / "glossary.json"

# Fuzzy correction only touches reasonably long, distinctive tokens/terms and
# needs a high similarity, so everyday words are left alone.
_FUZZY_CUTOFF = 0.86
_FUZZY_MIN_LEN = 5


def load() -> list[dict]:
    """Return the glossary as a list of {"term", "aliases"} dicts."""
    if not _GLOSSARY_FILE.exists():
        return []
    try:
        raw = json.loads(_GLOSSARY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _log.warning("Glossary file unreadable; treating as empty", exc_info=True)
        return []
    entries = raw.get("terms", []) if isinstance(raw, dict) else []
    return [normalized for e in entries if (normalized := _normalize_entry(e))]


def save(terms: list[dict]) -> None:
    _GLOSSARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"terms": [_normalize_entry(t) for t in terms if _normalize_entry(t)]}
    _GLOSSARY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_terms(entries: list) -> list[dict]:
    """Merge entries (plain strings or {"term", "aliases"}) into the glossary.

    Existing terms keep their identity (matched case-insensitively) and gain any
    new aliases. Returns the full updated glossary.
    """
    current = load()
    by_key = {e["term"].casefold(): e for e in current}
    for entry in entries:
        parsed = _normalize_entry(entry)
        if not parsed:
            continue
        key = parsed["term"].casefold()
        if key in by_key:
            existing = by_key[key]
            merged = list(dict.fromkeys(existing["aliases"] + parsed["aliases"]))
            existing["aliases"] = merged
        else:
            by_key[key] = parsed
            current.append(parsed)
    save(current)
    _log.info("Glossary now has %d term(s)", len(current))
    return current


def remove_term(term: str) -> bool:
    """Drop a term (case-insensitive). Returns True if something was removed."""
    key = str(term).strip().casefold()
    current = load()
    kept = [e for e in current if e["term"].casefold() != key]
    if len(kept) == len(current):
        return False
    save(kept)
    return True


def terms() -> list[str]:
    """Canonical terms, for display or decode biasing."""
    return [e["term"] for e in load()]


def whisper_hotwords() -> str | None:
    """Terms joined for faster-whisper's `hotwords` argument, or None if empty."""
    names = terms()
    return ", ".join(names) if names else None


def correct(text: str, entries: list[dict] | None = None) -> str:
    """Rewrite a transcript segment to use the glossary's terminology."""
    if not text:
        return text
    entries = load() if entries is None else entries
    if not entries:
        return text

    result = text
    # 1. Alias phrases → canonical term. Longest aliases first so a multi-word
    #    alias wins over a shorter overlapping one.
    alias_pairs = sorted(
        ((alias, e["term"]) for e in entries for alias in e["aliases"]),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for alias, canonical in alias_pairs:
        pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
        result = pattern.sub(lambda _m, c=canonical: c, result)

    # 2. Normalise the casing of any canonical term already spelled correctly.
    for entry in entries:
        canonical = entry["term"]
        pattern = re.compile(rf"\b{re.escape(canonical)}\b", re.IGNORECASE)
        result = pattern.sub(lambda _m, c=canonical: c, result)

    # 3. Optional fuzzy nudge of single tokens onto a near-matching term.
    if settings.get("glossary_fuzzy"):
        result = _fuzzy_correct(result, entries)

    return result


def suggest_from_transcripts(sample: int = 20, limit: int = 30) -> list[dict]:
    """Propose candidate terms from recent transcripts for Claude to review.

    Surfaces capitalised mid-sentence words and ALL-CAPS acronyms that recur
    and aren't already in the glossary — the shapes domain jargon usually
    takes. Returns [{"candidate", "count"}] ranked by frequency; the caller
    decides what is actually worth adding.
    """
    from trnscrb import storage

    known = {t.casefold() for t in terms()}
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    files = sorted(
        storage.NOTES_DIR.glob("*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:sample]
    for path in files:
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for token in _candidate_tokens(body):
            key = token.casefold()
            if key in known or key in _COMMON_WORDS:
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, token)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [{"candidate": display[k], "count": n} for k, n in ranked[:limit] if n >= 2]


# ── helpers ───────────────────────────────────────────────────────────────


def _normalize_entry(entry) -> dict | None:
    if isinstance(entry, str):
        term = entry.strip()
        return {"term": term, "aliases": []} if term else None
    if isinstance(entry, dict):
        term = str(entry.get("term", "")).strip()
        if not term:
            return None
        aliases = [
            a.strip()
            for a in entry.get("aliases", [])
            if isinstance(a, str) and a.strip() and a.strip().casefold() != term.casefold()
        ]
        return {"term": term, "aliases": list(dict.fromkeys(aliases))}
    return None


def _fuzzy_correct(text: str, entries: list[dict]) -> str:
    canon = [
        e["term"] for e in entries if " " not in e["term"] and len(e["term"]) >= _FUZZY_MIN_LEN
    ]
    if not canon:
        return text
    lower_to_canon = {c.casefold(): c for c in canon}

    def replace(match: re.Match) -> str:
        token = match.group(0)
        low = token.casefold()
        if low in lower_to_canon or len(token) < _FUZZY_MIN_LEN:
            return token  # already correct or too short to risk
        hit = difflib.get_close_matches(low, lower_to_canon.keys(), n=1, cutoff=_FUZZY_CUTOFF)
        return lower_to_canon[hit[0]] if hit else token

    return re.sub(r"\b[\w'-]+\b", replace, text)


def _candidate_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        # Skip the header/analytics block lines that aren't spoken content.
        words = re.findall(r"[A-Za-z][A-Za-z'’-]+", line)
        for i, word in enumerate(words):
            if word.isupper() and len(word) >= 2:  # acronym: KPI, POC, SCK
                tokens.append(word)
            elif i > 0 and word[:1].isupper() and not word.isupper():  # mid-sentence Capitalised
                tokens.append(word)
    return tokens


# Frequent capitalised English words that are not jargon — kept out of
# suggestions so the candidate list stays signal-heavy.
_COMMON_WORDS = {
    "i",
    "the",
    "yeah",
    "okay",
    "so",
    "and",
    "but",
    "we",
    "they",
    "it",
    "im",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "me",
    "them",
    "meeting",
    "google",
    "meet",
    "zoom",
}
