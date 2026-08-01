"""Offline semantic search across saved transcripts.

Keyword search misses "what did we decide about the cloud POC?" when the words
don't match. This embeds every transcript passage with a small local
sentence-transformer (all-MiniLM-L6-v2, ~90 MB, cached after first use) and
ranks passages by meaning. Everything runs on-device — no transcript leaves the
machine.

The index lives in ~/.config/trnscrb and is updated incrementally: only new or
changed transcripts are re-embedded, so adding a meeting costs one embed pass,
not a full rebuild.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.semantic_search")

_MODEL_ID = "all-MiniLM-L6-v2"
_INDEX_DIR = Path.home() / ".config" / "trnscrb"
_VECTORS_FILE = _INDEX_DIR / "search_vectors.npy"
_META_FILE = _INDEX_DIR / "search_meta.json"

# ~80 words per passage: long enough for a coherent thought, short enough to
# pinpoint where in a meeting something was said.
_WORDS_PER_CHUNK = 80

_model = None


def available() -> bool:
    """True if the embedding backend is installed."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def search(query: str, k: int = 8) -> list[dict]:
    """Return the top-k passages most semantically similar to ``query``.

    Each hit: {transcript_id, when, timestamp, text, score}. Refreshes the
    index first (cheap when nothing changed).
    """
    import numpy as np

    if not query.strip():
        return []
    build_index()
    if not _VECTORS_FILE.exists():
        return []
    vectors = np.load(_VECTORS_FILE)
    meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
    if vectors.shape[0] == 0 or not meta:
        return []

    q = _embed([query])[0]
    scores = vectors @ q  # vectors are L2-normalised → cosine similarity
    top = np.argsort(-scores)[:k]
    hits = []
    for i in top:
        record = dict(meta[int(i)])
        record["score"] = round(float(scores[int(i)]), 3)
        hits.append(record)
    return hits


def build_index(force: bool = False) -> int:
    """Ensure the index reflects the current transcripts. Returns passage count.

    Only transcripts whose file changed (or are new) are re-embedded; deleted
    transcripts are dropped. A no-op when nothing changed.
    """
    import numpy as np

    from trnscrb import storage

    files = _transcript_files(storage.NOTES_DIR)
    current = {f.stem: f.stat().st_mtime for f in files}

    meta: list[dict] = []
    vectors = None
    if not force and _META_FILE.exists() and _VECTORS_FILE.exists():
        try:
            meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
            vectors = np.load(_VECTORS_FILE)
        except (OSError, ValueError, json.JSONDecodeError):
            meta, vectors = [], None

    indexed = {}  # id -> mtime already represented in meta
    for record in meta:
        indexed[record["transcript_id"]] = record.get("mtime", 0)

    changed = {stem for stem, mtime in current.items() if indexed.get(stem) != mtime}
    removed = set(indexed) - set(current)
    if not force and not changed and not removed:
        return len(meta)

    # Keep rows from transcripts that are unchanged and still present.
    keep_rows = [
        idx
        for idx, record in enumerate(meta)
        if record["transcript_id"] not in changed and record["transcript_id"] not in removed
    ]
    kept_meta = [meta[i] for i in keep_rows]
    kept_vectors = vectors[keep_rows] if vectors is not None and len(keep_rows) else None

    # Embed passages from the new/changed transcripts.
    new_meta: list[dict] = []
    passages: list[str] = []
    id_by_stem = {f.stem: f for f in files}
    for stem in sorted(changed):
        path = id_by_stem[stem]
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        when = _date_from(body, stem)
        for timestamp, text in _chunks(body):
            new_meta.append(
                {
                    "transcript_id": stem,
                    "mtime": current[stem],
                    "when": when,
                    "timestamp": timestamp,
                    "text": text,
                }
            )
            passages.append(f"{when} {text}" if when else text)

    new_vectors = _embed(passages) if passages else None

    all_meta = kept_meta + new_meta
    all_vectors = _stack(kept_vectors, new_vectors)

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if all_vectors is None or len(all_meta) == 0:
        _META_FILE.write_text("[]", encoding="utf-8")
        np.save(_VECTORS_FILE, np.zeros((0, 384), dtype="float32"))
        return 0
    np.save(_VECTORS_FILE, all_vectors.astype("float32"))
    _META_FILE.write_text(json.dumps(all_meta), encoding="utf-8")
    _log.info("Semantic index: %d passages (%d re-embedded)", len(all_meta), len(new_meta))
    return len(all_meta)


# ── helpers ───────────────────────────────────────────────────────────────


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_ID)
    return _model


def _embed(texts: list[str]):
    return _get_model().encode(
        texts,
        normalize_embeddings=True,  # so a dot product is cosine similarity
        convert_to_numpy=True,
        batch_size=64,
    )


def _stack(a, b):
    import numpy as np

    parts = [x for x in (a, b) if x is not None and len(x)]
    if not parts:
        return None
    return np.vstack(parts)


def _transcript_files(notes_dir: Path) -> list[Path]:
    # Skip the generated weekly/annual summaries — search the meetings themselves.
    return [
        f
        for f in notes_dir.glob("*.txt")
        if not f.stem.endswith("_weekly") and not f.stem.endswith("_annual")
    ]


_LINE = re.compile(r"^\s*(\d{1,2}:\d{2})\s+(.*\S)\s*$")


def _chunks(body: str):
    """Yield (timestamp, passage) pairs of roughly _WORDS_PER_CHUNK words."""
    words: list[str] = []
    start_ts = ""
    for line in body.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        timestamp, text = match.group(1), match.group(2)
        if not words:
            start_ts = timestamp
        words.extend(text.split())
        if len(words) >= _WORDS_PER_CHUNK:
            yield start_ts, " ".join(words)
            words = []
    if words:
        yield start_ts, " ".join(words)


def _date_from(body: str, stem: str) -> str:
    match = re.search(r"^Date:\s*(.+)$", body, re.MULTILINE)
    if match:
        return match.group(1).strip()
    stamp = re.match(r"(\d{4}-\d{2}-\d{2})", stem)
    return stamp.group(1) if stamp else ""
