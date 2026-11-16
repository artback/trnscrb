"""Hybrid keyword + vector transcript search with RRF fusion, citations, scoping.

Two rankers answer one query and are fused by reciprocal rank fusion (RRF,
k=60 — the same constant FrigadeHQ/yogurt uses for its FTS + embedding
fusion):

- ``keyword_rank`` — the literal, case-insensitive phrase search the MCP
  used before, moved here. Every hit carries line provenance, so a result
  can be opened at the right place.
- ``vector_rank`` — the on-device semantic passages from ``semantic_search``,
  re-located to line ranges by re-deriving the index's passages from the
  file (same chunker, so a passage matches its indexed copy).

A passage both rankers land on in the same region of the same file is one
result with source ``"both"``; its score is the sum of the two rankers' RRF
terms. When the embedding backend is unavailable (or fails at runtime) the
search degrades to keyword-only and says so — ``vector_available`` is False
on the result, and the MCP renders a one-line note.

Scoping filters ("chips") accepted by :func:`hybrid`:

- ``since`` / ``until`` — the transcript's date (header ``Date:`` line, else
  the filename's ``YYYY-MM-DD`` prefix; mirrors ``semantic_search._date_from``).
  Undated transcripts are kept, never provably out of range.
- ``speaker`` — only matches in regions spoken by that person (a case-
  insensitive substring of the declared speaker). A vector hit whose lines
  could not be located cannot be verified and does not pass the filter.
- ``path`` — only transcripts under a subfolder of the notes dir. Discovery
  walks subfolders recursively (every other consumer of the notes dir globs
  top level only), so a folder of transcripts is searchable; with no
  ``path`` filter nothing is scoped, so the default view stays flat.

Generated weekly/annual summaries are skipped, as is ``semantic_search``'s
index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from trnscrb import semantic_search, storage
from trnscrb.log import get_logger

_log = get_logger("trnscrb.search")

# RRF constant: large enough that a ranker's own ordering still dominates
# within it, small enough that the two rankers' lists weigh equally.
_RRF_K = 60
# Default size of a fused result set.
_LIMIT = 25
# Candidate vector passages pulled before scoping/trimming.
_VECTOR_CANDIDATES = 25
# Matching lines within this many lines of each other form one result region.
_CLUSTER_GAP = 2

# The chunker must stay in sync with semantic_search's (_LINE,
# _WORDS_PER_CHUNK): it re-derives the index's passages to recover their line
# ranges. A drift means vector citations degrade to timestamp-only, not
# wrong results.
_LINE = re.compile(r"^\s*(\d{1,2}:\d{2})\s+(.*\S)\s*$")
_WORDS_PER_CHUNK = 80

_TS = re.compile(r"^\s*(\d{1,2}:\d{2})\b")
_SPEAKER = re.compile(r"^\[([^\[\]\n]{1,60})\]\s*$")
_DATE_HEADER = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
_DATE_STEM = re.compile(r"(\d{4}-\d{2}-\d{2})")
_SUMMARY_PREFIXES = ("weekly-", "annual-")
_SUMMARY_SUFFIXES = ("_weekly", "_annual")


@dataclass
class Hit:
    """One ranked result with stable provenance.

    ``name`` is the transcript's path relative to the notes dir (the bare
    filename at the top level) so citations stay disambiguated when
    subfolders are searched. ``line_start``/``line_end`` are 1-based and
    inclusive; 0 means the ranker could not locate the lines (a vector
    passage whose file changed out of the index).
    """

    name: str
    text: str = ""
    line_start: int = 0
    line_end: int = 0
    source: str = "keyword"  # keyword | vector | both
    score: float = 0.0
    when: str = ""
    timestamp: str = ""

    @property
    def citation(self) -> str:
        """Stable anchor: ``name:142-148``, `` @ MM:SS`` when known."""
        where = self.name
        if self.line_start:
            span = (
                f"{self.line_start}"
                if self.line_end == self.line_start
                else f"{self.line_start}-{self.line_end}"
            )
            where = f"{where}:{span}"
        if self.timestamp:
            where = f"{where} @ {self.timestamp}"
        return where


@dataclass
class SearchResult:
    """Fused ranked results plus what the caller should say about them.

    ``vector_available`` is False when the embedding backend is missing or
    failed — the MCP renders a one-line note in that case. ``vector_used``
    is True when the vector ranker ran, even if it produced no hits.
    """

    query: str
    results: list[Hit]
    vector_available: bool
    vector_used: bool
    files_searched: int


# ── discovery ───────────────────────────────────────────────────────────────


def transcript_files(path: str = "", notes_dir: Path | None = None) -> list[Path]:
    """Every searchable transcript under notes_dir, walked recursively.

    Subfolders are included (dictation notes and older layouts live there),
    so a ``path`` chip can scope to one folder; with no ``path`` nothing is
    scoped. Generated weekly/annual summaries are skipped, mirroring
    ``semantic_search``'s index.
    """
    notes_dir = notes_dir or storage.NOTES_DIR
    if not notes_dir.is_dir():
        return []
    prefix = (path or "").strip().strip("/")
    files = []
    for f in sorted(notes_dir.rglob("*.txt")):
        if f.name.startswith(_SUMMARY_PREFIXES) or f.stem.endswith(_SUMMARY_SUFFIXES):
            continue
        rel = f.relative_to(notes_dir).as_posix()
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        files.append(f)
    return files


def _rel_name(f: Path) -> str:
    """Path relative to the notes dir — the citation's locator."""
    try:
        return f.relative_to(storage.NOTES_DIR).as_posix()
    except ValueError:
        return f.name


def _read_lines(f: Path) -> list[str] | None:
    try:
        return f.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None


# ── keyword ranker ──────────────────────────────────────────────────────────


def keyword_rank(query: str, files: list[Path], context: int = 1) -> list[Hit]:
    """Literal case-insensitive phrase hits with line provenance, ranked.

    Hits within ``_CLUSTER_GAP`` lines merge into one region; each region
    is the matching lines plus ``context`` lines each side. Rank order:
    files with more matches first, then bigger regions, then earlier in the
    file.
    """
    query = (query or "").strip()
    if not query:
        return []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    by_file: dict[Path, list[Hit]] = {}
    for f in files:
        lines = _read_lines(f)
        if lines is None:
            continue
        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        if not hits:
            continue
        regions = []
        for cluster in _clusters(hits, _CLUSTER_GAP):
            start = max(0, cluster[0] - context)
            end = min(len(lines), cluster[-1] + context + 1)
            regions.append(
                Hit(
                    name=_rel_name(f),
                    line_start=start + 1,
                    line_end=end,
                    text="\n".join(lines[start:end]),
                    timestamp=_first_ts(lines[start:end]),
                    when=_file_date_from(lines, f.stem),
                )
            )
        if regions:
            by_file[f] = regions
    ranked: list[Hit] = []
    for f, regions in sorted(by_file.items(), key=lambda kv: (-len(kv[1]), kv[0].name)):
        ranked.extend(sorted(regions, key=lambda h: (h.line_start - h.line_end, h.line_start)))
    return ranked


def _clusters(indices: list[int], gap: int) -> list[list[int]]:
    """Group sorted hit lines into runs no more than ``gap`` apart."""
    clusters: list[list[int]] = []
    for i in indices:
        if clusters and i - clusters[-1][-1] <= gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _ts_of(line: str) -> str:
    """The MM:SS a transcript line carries, if any."""
    match = _TS.match(line)
    return match.group(1) if match else ""


def _first_ts(lines: list[str]) -> str:
    """The earliest timestamp a region's lines carry — its citation anchor."""
    for line in lines:
        ts = _ts_of(line)
        if ts:
            return ts
    return ""


# ── vector ranker ───────────────────────────────────────────────────────────


def vector_rank(query: str, files: list[Path]) -> tuple[list[Hit], bool]:
    """Semantic ranker: ``semantic_search``'s top passages as Hits.

    Returns ``(hits, ok)`` — ok is False when the embedding backend is
    missing or the lookup failed; the caller degrades to keyword-only and
    says so. Hits are scoped to ``files`` (by stem) and their line ranges
    are recovered by re-deriving the index's passages from each file.
    """
    if not semantic_search.available():
        return [], False
    try:
        raw = semantic_search.search(query, k=_VECTOR_CANDIDATES)
    except Exception as e:
        _log.warning("Vector search failed, degrading to keyword-only: %s", e)
        return [], False
    by_stem = {}
    for f in files:
        by_stem.setdefault(f.stem, f)
    hits = []
    for record in raw:
        f = by_stem.get(record.get("transcript_id") or "")
        if f is None:
            continue
        lines = _read_lines(f)
        if lines is None:
            continue
        hit = Hit(
            name=_rel_name(f),
            text=record.get("text") or "",
            source="vector",
            when=record.get("when") or "",
            timestamp=record.get("timestamp") or "",
        )
        for _ts, first, last, text in _chunks_with_lines("\n".join(lines)):
            if text == hit.text:
                hit.line_start, hit.line_end = first, last
                break
        hits.append(hit)
    return hits, True


def _chunks_with_lines(body: str):
    """(timestamp, first_line, last_line, passage) per ~80-word passage.

    Mirrors ``semantic_search._chunks`` exactly (same line regex, same
    passage size) so a passage re-derived from a file equals the one stored
    in the index; the line range is extra provenance the index does not
    keep.
    """
    words: list[str] = []
    start_ts = ""
    first = last = 0
    for i, line in enumerate(body.splitlines()):
        match = _LINE.match(line)
        if not match:
            continue
        if not words:
            start_ts = match.group(1)
            first = i + 1
        words.extend(match.group(2).split())
        last = i + 1
        if len(words) >= _WORDS_PER_CHUNK:
            yield start_ts, first, last, " ".join(words)
            words = []
    if words:
        yield start_ts, first, last, " ".join(words)


# ── fusion ──────────────────────────────────────────────────────────────────


def rrf(rank_lists: list[list], k: int = _RRF_K) -> dict:
    """Fuse ranked lists by reciprocal rank fusion (Cormack et al., k=60).

    ``rank_lists`` holds one ranked list per ranker; item i of list j is
    that ranker's (i+1)-th best. Returns ``{item: (score, rankers)}`` where
    ``score = sum_j 1 / (k + rank_j)`` and ``rankers`` lists the 0-based
    ranker positions the item appeared in.
    """
    fused: dict = {}
    for pos, ranked in enumerate(rank_lists):
        for rank, item in enumerate(ranked, start=1):
            score, rankers = fused.get(item, (0.0, []))
            fused[item] = (score + 1.0 / (k + rank), rankers + [pos])
    return fused


def fuse(keyword_hits: list[Hit], vector_hits: list[Hit], k: int = _RRF_K) -> list[Hit]:
    """Fuse the two ranked hit lists into one RRF-ranked list.

    A vector passage whose lines overlap a keyword region is the same
    result: one hit with source ``"both"`` scoring both rankers' RRF terms
    (the keyword region keeps the display, since it carries line context).
    """
    kw_by_name: dict[str, list[Hit]] = {}
    for h in keyword_hits:
        kw_by_name.setdefault(h.name, []).append(h)

    # A vector passage that overlaps a keyword region is one result: merge it
    # into that region (which keeps the display, since it carries line
    # context) so both rankers score it.
    absorbed: list[tuple[Hit, Hit]] = []  # (vector hit, keyword region)
    for v in vector_hits:
        if not v.line_start:
            continue
        for h in kw_by_name.get(v.name, ()):
            if _overlap(h.line_start, h.line_end, v.line_start, v.line_end):
                absorbed.append((v, h))
                break
    absorbed_by_id = {id(v): h for v, h in absorbed}

    kw_list = [_key(h) for h in keyword_hits]
    vec_list = [
        _key(absorbed_by_id[id(v)]) if id(v) in absorbed_by_id else _key(v) for v in vector_hits
    ]
    fused = rrf([kw_list, vec_list], k)

    out: list[Hit] = []
    by_id: dict[int, Hit] = {}
    for h in keyword_hits:
        hit = replace(h)
        by_id[id(h)] = hit
        hit.score, rankers = fused[_key(h)]
        if len(rankers) >= 2:
            hit.source = "both"
        out.append(hit)
    for v, h in absorbed:
        # the absorbed passage may know where its lines start
        if not by_id[id(h)].timestamp and v.timestamp:
            by_id[id(h)].timestamp = v.timestamp
    for v in vector_hits:
        if id(v) in absorbed_by_id:
            continue  # merged into its keyword region, which carries the score
        hit = replace(v)
        hit.score, _ = fused[_key(v)]
        out.append(hit)
    out.sort(key=lambda h: (-h.score, h.name, h.line_start or 1))
    return out


def _key(hit: Hit):
    """Fusion identity: same transcript, same lines, same timestamp."""
    return (hit.name, hit.line_start, hit.line_end, hit.timestamp)


def _overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 <= e2 and e1 >= s2


# ── filters ─────────────────────────────────────────────────────────────────


def _file_date_from(lines: list[str], stem: str) -> str:
    """YYYY-MM-DD from the header Date line, else the filename prefix."""
    match = _DATE_HEADER.search("\n".join(lines[:40]))
    if match:
        return match.group(1)
    match = _DATE_STEM.match(stem)
    return match.group(1) if match else ""


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_filter(label: str, value: str) -> date:
    """A since/until chip: YYYY-MM-DD, else ValueError (MCP shows it)."""
    parsed = _parse_date(value.strip())
    if parsed is None:
        raise ValueError(f"invalid {label} date '{value}' — use YYYY-MM-DD")
    return parsed


def _file_date(f: Path) -> date | None:
    """The transcript's date; None when neither header nor name carries one."""
    lines = _read_lines(f)
    if lines is None:
        return None
    return _parse_date(_file_date_from(lines, f.stem))


def _line_speakers(lines: list[str]) -> list[str | None]:
    """Per line, the speaker declared by the nearest ``[Name]`` line above it."""
    out: list[str | None] = [None] * len(lines)
    current = None
    for i, line in enumerate(lines):
        match = _SPEAKER.match(line)
        if match:
            current = match.group(1).strip() or None
        out[i] = current
    return out


def _speaks(hit: Hit, want: str) -> bool:
    """True when ``want`` (a case-insensitive name substring) was spoken in
    the hit's region. Only spoken (timestamped) lines count — the region's
    context window may reach into the next speaker's marker line. A hit
    whose lines are not located cannot be verified and does not pass."""
    if not hit.line_start:
        return False
    lines = _read_lines(Path(storage.NOTES_DIR) / hit.name)
    if lines is None:
        return False
    want = want.casefold()
    speakers = _line_speakers(lines)
    start, end = hit.line_start - 1, min(hit.line_end, len(lines))
    names = {speakers[i] for i in range(start, end) if speakers[i] and _LINE.match(lines[i])}
    return any(want in name.casefold() for name in names)


# ── hybrid ──────────────────────────────────────────────────────────────────


def hybrid(
    query: str,
    since: str = "",
    until: str = "",
    speaker: str = "",
    path: str = "",
    limit: int = _LIMIT,
) -> SearchResult:
    """Rank transcripts for ``query`` by fusing the keyword and vector
    rankers (RRF, k=60) into one list of provenance-carrying hits.

    Filters: see the module docstring. Raises ValueError on an unparseable
    since/until. An empty query yields no results.
    """
    query = (query or "").strip()
    since_d = _parse_filter("since", since) if since else None
    until_d = _parse_filter("until", until) if until else None

    notes_dir = storage.NOTES_DIR
    files = transcript_files(path=path, notes_dir=notes_dir)
    if since_d is not None or until_d is not None:
        kept = []
        for f in files:
            d = _file_date(f)
            if d is None:  # undated: not provably out of range
                kept.append(f)
                continue
            if (since_d is not None and d < since_d) or (until_d is not None and d > until_d):
                continue
            kept.append(f)
        files = kept

    keyword_hits = keyword_rank(query, files) if query else []
    vector_hits, vector_ok = vector_rank(query, files) if query else ([], False)
    fused = fuse(keyword_hits, vector_hits)
    if speaker:
        fused = [h for h in fused if _speaks(h, speaker)]
    return SearchResult(
        query=query,
        results=fused[:limit],
        vector_available=vector_ok,
        vector_used=vector_ok,
        files_searched=len(files),
    )
