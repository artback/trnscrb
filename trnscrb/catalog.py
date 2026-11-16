"""The model catalog: every model trnscrb downloads, where it comes from,
and how it fares on the machine it runs on.

The pattern is FrigadeHQ/yogurt's: a declarative list of models carrying
size, hardware requirements and (optionally) a pinned checksum, plus a
scorer that picks the best model the machine can run. The entries here are
the models the transcription, diarization and search backends actually
load — see transcriber.py (parakeet/qwen3/whisper/voxtral), diarizer.py
(pyannote pipelines) and semantic_search.py (sentence-transformers) — so
the catalog can answer "what should this machine run?" and "does what is
on disk match what we expect?".

Pure module: stdlib only, no macOS-only imports, so the headless server
and the test suite import it on any platform.
"""

import hashlib
import os
import platform
import subprocess
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.catalog")

# Per-entry download states, returned by verify().
MISSING = "missing"
OK = "ok"
MISMATCH = "mismatch"
NOT_VERIFIABLE = "not-verifiable"

# The backends recommend() scores. Diarization and search models are
# hardware-independent and always run (when their dependencies are
# installed), so they are catalogued but not scored.
_TRANSCRIPTION_BACKENDS = ("parakeet", "qwen3", "whisper", "voxtral")

# Entry fields:
#   id                — short unique name, used in `trnscrb models`
#   backend           — which trnscrb component loads it
#   source            — HuggingFace repo id the download comes from
#   approx_size_gb    — what lands on disk (rounded, for display)
#   sha256            — pinned checksum of `file`, or None when the
#                       library manages the download and no pinned
#                       artifact exists (most entries)
#   file              — the artifact name sha256 refers to (only meaningful
#                       when sha256 is set)
#   requirements      — hardware gates: apple_silicon (bool), min_ram_gb
#   quality           — transcription quality 1-10; rank in recommend()
#   model_size        — the WhisperModel size key, whisper entries only
#   notes             — one line about what it is for
CATALOG: list[dict] = [
    {
        "id": "parakeet-tdt-0.6b-v3",
        "backend": "parakeet",
        "source": "mlx-community/parakeet-tdt-0.6b-v3",
        "approx_size_gb": 2.5,
        "sha256": None,
        "requirements": {"apple_silicon": True, "min_ram_gb": 8},
        "quality": 10,
        "notes": "best for English meetings; word timings feed the speaker labelling",
    },
    {
        "id": "qwen3-asr-0.6b",
        "backend": "qwen3",
        "source": "Qwen/Qwen3-ASR-0.6B",
        "approx_size_gb": 1.9,
        "sha256": None,
        "requirements": {"apple_silicon": True, "min_ram_gb": 8},
        "quality": 9,
        "notes": "multilingual; the auto backend routes non-English meetings here",
    },
    {
        "id": "qwen3-asr-1.7b",
        "backend": "qwen3",
        "source": "Qwen/Qwen3-ASR-1.7B",
        "approx_size_gb": 4.7,
        "sha256": None,
        "requirements": {"apple_silicon": True, "min_ram_gb": 16},
        "quality": 10,
        "notes": "more accurate qwen3 variant for bigger machines (opt in via qwen3_model_id)",
    },
    {
        "id": "whisper-tiny",
        "backend": "whisper",
        "model_size": "tiny",
        "source": "Systran/faster-whisper-tiny",
        "approx_size_gb": 0.08,
        "sha256": None,
        "requirements": {"min_ram_gb": 4},
        "quality": 4,
        "notes": "smallest Whisper; last resort on constrained machines",
    },
    {
        "id": "whisper-base",
        "backend": "whisper",
        "model_size": "base",
        "source": "Systran/faster-whisper-base",
        "approx_size_gb": 0.15,
        "sha256": None,
        "requirements": {"min_ram_gb": 4},
        "quality": 5,
        "notes": "light Whisper for constrained machines",
    },
    {
        "id": "whisper-small",
        "backend": "whisper",
        "model_size": "small",
        "source": "Systran/faster-whisper-small",
        "approx_size_gb": 0.49,
        "sha256": None,
        "requirements": {"min_ram_gb": 4},
        "quality": 6,
        "notes": "the default Whisper size and the fallback every backend falls back to",
    },
    {
        "id": "whisper-medium",
        "backend": "whisper",
        "model_size": "medium",
        "source": "Systran/faster-whisper-medium",
        "approx_size_gb": 1.53,
        "sha256": None,
        "requirements": {"min_ram_gb": 8},
        "quality": 7,
        "notes": "more accurate Whisper for machines with room",
    },
    {
        "id": "whisper-turbo",
        "backend": "whisper",
        "model_size": "turbo",
        "source": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "approx_size_gb": 1.62,
        "sha256": None,
        "requirements": {"min_ram_gb": 8},
        "quality": 8,
        "notes": "large-v3-turbo: near large-v3 accuracy at half the decode cost",
    },
    {
        "id": "whisper-large-v3",
        "backend": "whisper",
        "model_size": "large-v3",
        "source": "Systran/faster-whisper-large-v3",
        "approx_size_gb": 3.09,
        "sha256": None,
        "requirements": {"min_ram_gb": 16},
        "quality": 8,
        "notes": "most accurate standard Whisper",
    },
    {
        "id": "voxtral-mini-3b",
        "backend": "voxtral",
        "source": "mistralai/Voxtral-Mini-3B-2507",
        "approx_size_gb": 9.4,
        "sha256": None,
        "requirements": {"min_ram_gb": 16},
        "quality": 8,
        "notes": "Mistral's multilingual ASR (opt in via transcription_backend=voxtral; slow on CPU)",
    },
    {
        "id": "pyannote-community-1",
        "backend": "diarization",
        "source": "pyannote/speaker-diarization-community-1",
        "approx_size_gb": 1.0,
        "sha256": None,
        "requirements": {"min_ram_gb": 8},
        "notes": "preferred speaker-diarization pipeline; gated — accept the terms at hf.co first",
    },
    {
        "id": "pyannote-3.1",
        "backend": "diarization",
        "source": "pyannote/speaker-diarization-3.1",
        "approx_size_gb": 0.7,
        "sha256": None,
        "requirements": {"min_ram_gb": 8},
        "notes": "legacy pipeline, fallback when community-1 cannot load; also gated on hf.co",
    },
    {
        "id": "all-minilm-l6-v2",
        "backend": "semantic_search",
        "source": "sentence-transformers/all-MiniLM-L6-v2",
        "approx_size_gb": 0.1,
        "sha256": None,
        "requirements": {"min_ram_gb": 4},
        "notes": "embeds transcript passages for semantic search; ~90 MB after download",
    },
]


def _entry_by_id(model_id: str) -> dict | None:
    return next((e for e in CATALOG if e["id"] == model_id), None)


def machine_facts() -> dict:
    """The machine's silhouette for recommend(): {"apple_silicon", "ram_gb"}.

    Each fact falls back to the conservative side of the guess when it
    cannot be determined (headless host, unknown platform): Intel with
    16 GB, the middle of what this app targets. A wrong guess costs a
    slower recommendation, never a broken one.
    """
    facts = {"apple_silicon": False, "ram_gb": 16.0}
    if platform.system() == "Darwin":
        facts["apple_silicon"] = platform.processor().lower() in {"arm", "arm64"}
    ram = _total_ram_gb()
    if ram:
        facts["ram_gb"] = ram
    return facts


def _total_ram_gb() -> float | None:
    """Total physical RAM in GB, or None when it cannot be read."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
            )
            return int(out.stdout.strip()) / 1e9
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 * 1024)  # kB → GB
    except (OSError, ValueError, IndexError):
        pass
    return None


def recommend(facts: dict | None = None) -> dict:
    """The transcription backend + model this machine should run.

    Mirrors yogurt's picker: gate the catalog on what the hardware can
    run, rank the rest by quality, break ties toward the smaller download
    (the live-transcription loop is the one that cannot wait). Diarization
    and search models are not part of the pick — they are hardware-
    independent and catalogued for status/verification only.
    """
    facts = facts or machine_facts()
    apple = bool(facts.get("apple_silicon"))
    ram = float(facts.get("ram_gb") or 0.0)

    best = None  # ((quality, -size), entry)
    for entry in CATALOG:
        if entry["backend"] not in _TRANSCRIPTION_BACKENDS:
            continue
        req = entry.get("requirements") or {}
        if req.get("apple_silicon") and not apple:
            continue
        if ram < float(req.get("min_ram_gb") or 0):
            continue
        rank = (int(entry.get("quality") or 0), -float(entry.get("approx_size_gb") or 0.0))
        if best is None or rank > best[0]:
            best = (rank, entry)

    if best is None:
        # Nothing in the catalog fits — offer the smallest model anyway.
        entry = _entry_by_id("whisper-tiny")
        reason = "nothing else fits this machine; whisper tiny is the smallest that runs"
    else:
        entry = best[1]
        hw = "Apple Silicon" if apple else "Intel"
        reason = f"best quality that fits this machine ({hw}, {ram:.0f} GB RAM)"

    return {
        "entry_id": entry["id"],
        "backend": entry["backend"],
        "model": entry["source"],
        "model_size": entry.get("model_size"),
        "reason": reason,
    }


def configured_entries(settings: dict) -> list[dict]:
    """The transcription entries the configured backend actually loads.

    The fallback chain matters: any backend that fails falls back to
    Whisper (transcriber._transcribe_on_worker), so Whisper is part of
    every configuration, and the auto backend additionally needs qwen3
    for non-English meetings. Diarization and search models are left out
    — `status` reports them on their own lines.
    """
    backend = str(settings.get("transcription_backend") or "auto").strip().lower()
    picks = []
    if backend in ("auto", "parakeet"):
        picks.append("parakeet-tdt-0.6b-v3")
    if backend in ("auto", "qwen3"):
        picks.append("qwen3-asr-0.6b")
    if backend == "voxtral":
        picks.append("voxtral-mini-3b")
    picks.append("whisper-" + str(settings.get("model_size") or "small"))

    out = []
    for pid in picks:
        entry = _entry_by_id(pid)
        if entry is None:
            continue  # a custom model id not in the catalog
        if pid == "parakeet-tdt-0.6b-v3":
            custom = str(settings.get("parakeet_model_id") or "").strip()
            if custom:
                entry = {**entry, "source": custom}
        out.append(entry)
    return out


# ── verification ────────────────────────────────────────────────────────────


def _hf_hub_root() -> Path:
    """The HF hub cache root, honouring HF_HUB_CACHE/HF_HOME like the rest of the app."""
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    root = Path(cache) if cache else Path.home() / ".cache" / "huggingface"
    if root.name != "hub":
        root = root / "hub"
    return root


def _repo_dir(entry: dict, cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else _hf_hub_root()
    source = entry.get("source") or entry["id"]
    return root / f"models--{source.replace('/', '--')}"


def _snapshot_files(repo_dir: Path) -> list[Path]:
    """Every file under the repo's snapshots/ — the snapshot entries are
    symlinks into blobs/, so following them counts each artifact once."""
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return [f for f in snapshots.rglob("*") if f.is_file()]


def disk_size(entry: dict, cache_root: str | Path | None = None) -> int:
    """Bytes this entry occupies in the model cache, 0 when not downloaded."""
    return sum(_file_sizes(_snapshot_files(_repo_dir(entry, cache_root))))


def _file_sizes(files: list[Path]) -> list[int]:
    sizes = []
    for f in files:
        try:
            sizes.append(f.stat().st_size)
        except OSError:
            pass
    return sizes


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(entry: dict, cache_root: str | Path | None = None) -> dict:
    """The download state of one entry: missing / ok / mismatch / not-verifiable.

    missing           — nothing for this model is in the cache.
    not-verifiable    — downloaded, but the library manages the artifacts
                        and the catalog pins no checksum, so only presence
                        can be checked.
    ok / mismatch     — a checksum is pinned and the downloaded file
                        matches it / does not.
    """
    result = {"id": entry.get("id", ""), "status": MISSING, "size_bytes": 0, "detail": ""}

    files = _snapshot_files(_repo_dir(entry, cache_root))
    result["size_bytes"] = sum(_file_sizes(files))

    if not files:
        # faster-whisper can also keep its ct2 cache outside the hub, where
        # no checksum could ever be checked — presence there still counts as
        # downloaded (mirrors the ct2 fallback in cli._whisper_model_cached).
        if entry.get("backend") == "whisper":
            size = entry.get("model_size")
            ct2 = Path.home() / ".cache" / "faster_whisper"
            if size and ct2.is_dir() and any(ct2.glob(f"*{size}*")):
                result["status"] = NOT_VERIFIABLE
                result["detail"] = "downloaded to the ct2 local cache (no pinned checksum)"
                return result
        result["detail"] = f"not downloaded (expected {entry.get('source', entry.get('id'))})"
        return result

    sha = entry.get("sha256")
    if not sha:
        result["status"] = NOT_VERIFIABLE
        result["detail"] = "downloaded; no pinned checksum to verify against"
        return result

    target = entry.get("file")
    matches = [f for f in files if target is None or f.name == target]
    if not matches:
        result["detail"] = "downloaded but the pinned file is absent (partial download?)"
        return result

    digest = _hash_file(matches[0])
    if digest == sha.lower():
        result["status"] = OK
        result["detail"] = "checksum verified"
    else:
        result["status"] = MISMATCH
        result["detail"] = f"sha256 {digest} != pinned {sha.lower()}"
        _log.warning("Model %s failed checksum verification: %s", result["id"], result["detail"])
    return result


def verify_all(entries: list[dict] | None = None, cache_root: str | Path | None = None) -> list[dict]:
    """verify() for the whole catalog — one status, size and detail per entry."""
    return [verify(e, cache_root) for e in (entries or CATALOG)]