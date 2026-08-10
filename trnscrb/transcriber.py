"""Transcription with configurable backend (Parakeet, Qwen3, Whisper, or Voxtral)."""

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.transcriber")

_SUPPORTED_BACKENDS = {"auto", "parakeet", "whisper", "voxtral", "qwen3"}

_whisper_model = None
_whisper_model_lock = threading.Lock()
_whisper_model_size = "small"

_parakeet_model = None
_parakeet_model_id = None
_parakeet_model_lock = threading.Lock()


def set_model_size(size: str) -> None:
    global _whisper_model_size, _whisper_model
    _whisper_model_size = size
    _whisper_model = None  # force reload on next call


def _backend() -> str:
    backend = str(settings.get("transcription_backend") or "parakeet").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise RuntimeError(
            f"Unsupported transcription backend '{backend}'. "
            f"Set transcription_backend to one of: {allowed}."
        )
    return backend


def _get_whisper_model():
    global _whisper_model
    size = str(settings.get("model_size") or _whisper_model_size)
    with _whisper_model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Whisper backend selected but faster-whisper is not installed. "
                    "Install it with `uv add faster-whisper`."
                ) from e
            try:
                _whisper_model = WhisperModel(size, device="auto", compute_type="auto")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Whisper model '{size}'. "
                    "Check local model cache and backend dependencies."
                ) from e
        return _whisper_model


def _get_parakeet_model():
    global _parakeet_model, _parakeet_model_id
    model_id = str(settings.get("parakeet_model_id") or "").strip()
    if not model_id:
        raise RuntimeError(
            "Parakeet backend selected but no model id is configured. "
            "Set `parakeet_model_id` in ~/.config/trnscrb/settings.json."
        )

    with _parakeet_model_lock:
        if _parakeet_model is None or _parakeet_model_id != model_id:
            try:
                from parakeet_mlx import from_pretrained
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Parakeet backend selected but parakeet-mlx is not installed. "
                    "Install it with `uv add parakeet-mlx`."
                ) from e
            try:
                _parakeet_model = from_pretrained(model_id)
                _parakeet_model_id = model_id
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Parakeet model '{model_id}'. "
                    "Verify network/cache access for first-time model download."
                ) from e
        return _parakeet_model


def _transcribe_whisper(audio_path: Path) -> list[dict]:
    from trnscrb import glossary

    model = _get_whisper_model()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,  # skip silent gaps automatically
        language=None,  # auto-detect
        hotwords=glossary.whisper_hotwords(),  # bias toward the work glossary
    )
    results = []
    for seg in segments:
        text = getattr(seg, "text", None)
        if not text or not text.strip():
            continue
        results.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": text.strip(),
                "speaker": None,
            }
        )
    return results


# Long recordings must be chunked: unchunked, Parakeet builds attention
# buffers over the whole file and a multi-hour meeting exceeds Metal's
# maximum buffer size (observed: 43 GB requested for a 3.6 h recording).
_PARAKEET_CHUNK_SECS = 120.0
# Cap on one aligned "sentence". Long enough to keep a thought together,
# short enough that a segment rarely spans a speaker change.
_PARAKEET_SENTENCE_MAX_SECS = 30.0
# A pause this long ends a sentence — usually a speaker handing over.
_PARAKEET_SENTENCE_GAP_SECS = 0.8


def _parakeet_decoding_config():
    """Bound how long one "sentence" may run, or None if unsupported.

    parakeet-mlx leaves max_words, silence_gap and max_duration unset, so a
    speaker who is never interrupted becomes a single multi-minute sentence.
    Diarization then has nothing to attach to: one segment gets one speaker
    label, and everyone else who spoke inside it is silently relabelled as
    that person. Capping duration and splitting on pauses keeps segments
    close to speaker turns.
    """
    try:
        from parakeet_mlx.alignment import SentenceConfig
        from parakeet_mlx.parakeet import DecodingConfig
    except ImportError:
        _log.debug("parakeet-mlx has no DecodingConfig; using its defaults")
        return None
    try:
        return DecodingConfig(
            sentence=SentenceConfig(
                max_duration=_PARAKEET_SENTENCE_MAX_SECS,
                silence_gap=_PARAKEET_SENTENCE_GAP_SECS,
            )
        )
    except TypeError:
        _log.debug("Unexpected DecodingConfig signature; using parakeet defaults")
        return None


def _transcribe_parakeet(audio_path: Path) -> list[dict]:
    model = _get_parakeet_model()
    kwargs = {"chunk_duration": _PARAKEET_CHUNK_SECS}
    decoding = _parakeet_decoding_config()
    if decoding is not None:
        kwargs["decoding_config"] = decoding
    result = model.transcribe(str(audio_path), **kwargs)
    sentences = getattr(result, "sentences", None)
    if sentences is None:
        raise RuntimeError("Parakeet transcription did not return aligned sentences output.")

    normalized = []
    for sentence in sentences:
        text = str(getattr(sentence, "text", "")).strip()
        if not text:
            continue
        try:
            start = float(getattr(sentence, "start", 0.0))
        except (TypeError, ValueError):  # fmt: skip
            _log.warning(
                "Could not parse start timestamp %r, defaulting to 0.0",
                getattr(sentence, "start", None),
            )
            start = 0.0
        try:
            end = float(getattr(sentence, "end", 0.0))
        except (TypeError, ValueError):  # fmt: skip
            _log.warning(
                "Could not parse end timestamp %r, defaulting to 0.0",
                getattr(sentence, "end", None),
            )
            end = 0.0
        words = _parakeet_words(sentence)
        if words:
            words = _drop_repeats(words)
            text = words_to_text(words) or text
            start, end = words[0]["start"], words[-1]["end"]
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": None,
                # Word timings let the diarizer cut a segment where the
                # speaker actually changes, instead of labelling it whole.
                "words": words,
            }
        )
    return normalized


def _parakeet_words(sentence) -> list[dict]:
    """Per-word timings from an aligned sentence, or [] when unavailable.

    Tokens are sub-word pieces carrying their own leading space — ' Pri',
    'mar', 'y' — so they are concatenated, never joined with spaces, and a
    leading space is what starts a new word. Each word keeps its raw form so
    the sentence can be rebuilt with its original spacing and punctuation.
    """
    words: list[dict] = []
    for token in getattr(sentence, "tokens", None) or []:
        raw = str(getattr(token, "text", ""))
        if not raw.strip():
            continue
        try:
            start = float(getattr(token, "start", 0.0))
            end = float(getattr(token, "end", 0.0))
        except (TypeError, ValueError):
            continue
        if words and not raw[:1].isspace():
            word = words[-1]
            word["raw"] += raw
            word["text"] = word["raw"].strip()
            word["end"] = end
        else:
            words.append({"raw": raw, "text": raw.strip(), "start": start, "end": end})
    return words


def words_to_text(words: list[dict]) -> str:
    """Rebuild a passage from word dicts, preserving original spacing."""
    return "".join(w.get("raw") or f" {w['text']}" for w in words).strip()


# Words that people genuinely say twice in a row; never collapsed on their own.
_NATURAL_REPEATS = frozenset(
    "yeah yes no ok okay right sure sorry hi hey bye thanks well now so very"
    " really please good come go run stop wait".split()
)
# Longest repeated run we try to collapse.
_MAX_REPEAT_WORDS = 8


def _norm(word: dict) -> str:
    """Comparison form: lowercase, punctuation removed.

    "area" and "area." are the same word said twice; only the second carries
    the sentence's full stop.
    """
    return re.sub(r"[^\w']+", "", word["text"].lower())


def _drop_repeats(words: list[dict]) -> list[dict]:
    """Collapse a run of words immediately repeated verbatim.

    The decoder emits words twice at the sub-word level — ' Pri','mar','y',
    ' pri','mar','y' for a single "Primary" — which reads as stutter through
    a whole transcript. A genuine stutter is nearly always one short word
    ("yeah, yeah"), so single words are only collapsed when they carry
    content, while longer runs are collapsed whenever they repeat.
    """
    out: list[dict] = []
    i = 0
    while i < len(words):
        dropped = 0
        for n in range(min(_MAX_REPEAT_WORDS, (len(words) - i) // 2), 0, -1):
            first = [_norm(w) for w in words[i : i + n]]
            second = [_norm(w) for w in words[i + n : i + 2 * n]]
            if first != second or not all(first):
                continue
            if n == 1 and (first[0] in _NATURAL_REPEATS or len(first[0]) < 4):
                continue
            dropped = n
            break

        kept = [dict(w) for w in words[i : i + max(dropped, 1)]]
        if dropped:
            # The repeat usually carries the punctuation and the true end of
            # the utterance; keep the first spelling but not at their cost.
            last_dropped = words[i + 2 * dropped - 1]
            tail = re.search(r"[^\w'\s]+$", last_dropped["text"])
            if tail and not kept[-1]["text"].endswith(tail.group()):
                kept[-1]["text"] += tail.group()
                kept[-1]["raw"] = kept[-1].get("raw", "") + tail.group()
            kept[-1]["end"] = last_dropped["end"]
        out.extend(kept)
        i += (2 * dropped) if dropped else 1
    return out


_qwen3_model = None
_qwen3_aligner = None
_qwen3_model_id = None
_qwen3_lock = threading.Lock()


def _get_qwen3():
    """Return (model, forced_aligner), loading and caching them on first call."""
    global _qwen3_model, _qwen3_aligner, _qwen3_model_id
    model_id = str(settings.get("qwen3_model_id") or "Qwen/Qwen3-ASR-0.6B").strip()
    with _qwen3_lock:
        if _qwen3_model is None or _qwen3_model_id != model_id:
            try:
                from mlx_qwen3_asr import ForcedAligner, load_model
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Qwen3 backend selected but mlx-qwen3-asr is not installed. "
                    "Install it with `uv add mlx-qwen3-asr`."
                ) from e
            try:
                _log.info("Loading Qwen3-ASR model %s", model_id)
                model, _config = load_model(model_id)
                aligner = ForcedAligner()
                _qwen3_model = model
                _qwen3_aligner = aligner
                _qwen3_model_id = model_id
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Qwen3-ASR model '{model_id}'. "
                    "Verify network/cache access for first-time model download."
                ) from e
        return _qwen3_model, _qwen3_aligner


def _words_to_segments(text: str, words: list[dict]) -> list[dict]:
    """Combine Qwen3's punctuated text with word-level timings into sentences.

    The word list carries timings but no punctuation; the full text carries
    punctuation but no timings. Sentences are split from the text and mapped
    onto the word list by word count — small tokenization drift only shifts a
    segment boundary slightly, which the diarizer merge tolerates.
    """
    import re

    words = [w for w in words if str(w.get("text", "")).strip()]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if s.strip()]
    if not sentences:
        return []
    if not words:
        return [{"start": 0.0, "end": 0.0, "text": " ".join(sentences), "speaker": None}]

    segments = []
    word_idx = 0
    last_end = float(words[-1].get("end", 0.0) or 0.0)
    for sentence in sentences:
        chunk = words[word_idx : word_idx + len(sentence.split())]
        word_idx += len(sentence.split())
        if chunk:
            start = float(chunk[0].get("start", 0.0) or 0.0)
            end = float(chunk[-1].get("end", start) or start)
        else:
            start = end = last_end  # word list ran short — pin to the tail
        segments.append({"start": start, "end": end, "text": sentence, "speaker": None})
    return segments


def _transcribe_qwen3(audio_path: Path) -> list[dict]:
    model, aligner = _get_qwen3()
    from mlx_qwen3_asr import transcribe as qwen3_transcribe

    result = qwen3_transcribe(
        str(audio_path),
        model=model,
        forced_aligner=aligner,
        return_timestamps=True,
    )
    text = str(getattr(result, "text", "") or "").strip()
    if not text:
        return []
    words = list(getattr(result, "segments", None) or [])
    return _words_to_segments(text, words)


_voxtral_pipeline = None
_voxtral_pipeline_lock = threading.Lock()
_voxtral_model_id = None


def _get_voxtral_pipeline():
    global _voxtral_pipeline, _voxtral_model_id

    model_id = str(settings.get("voxtral_model_id") or "mistralai/Voxtral-Mini-3B-2507").strip()

    with _voxtral_pipeline_lock:
        if _voxtral_pipeline is None or _voxtral_model_id != model_id:
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Voxtral backend selected but transformers is not installed. "
                    "Install it with `uv add transformers torch`."
                ) from e

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            _log.info("Loading Voxtral model %s on %s", model_id, device)

            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device,
                )
                _voxtral_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype,
                    device=device,
                    return_timestamps=True,
                )
                _voxtral_model_id = model_id
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Voxtral model '{model_id}'. "
                    "Check network/cache access for first-time model download."
                ) from e
        return _voxtral_pipeline


def _transcribe_voxtral(audio_path: Path) -> list[dict]:
    pipe = _get_voxtral_pipeline()
    _log.info("Transcribing with Voxtral (model=%s)", _voxtral_model_id)
    result = pipe(str(audio_path), return_timestamps=True)

    segments = []
    for chunk in result.get("chunks", []):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        ts = chunk.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
        segments.append(
            {
                "start": float(ts[0] or 0.0),
                "end": float(ts[1] or 0.0),
                "text": text,
                "speaker": None,
            }
        )

    if not segments and result.get("text"):
        segments.append({"start": 0.0, "end": 0.0, "text": result["text"].strip(), "speaker": None})

    return segments


def _detect_language(audio_path: Path) -> str:
    """Detect language from the first 30s of audio using Whisper.

    Returns an ISO 639-1 code (e.g. 'en', 'sv', 'es').
    """
    model = _get_whisper_model()
    # Whisper's detect_language only needs the first 30s
    _log.debug("Detecting language from %s", audio_path)
    _segments, info = model.transcribe(
        str(audio_path),
        beam_size=1,
        vad_filter=False,
        # Only read enough to detect language — transcribe returns
        # info.language after processing the first segment.
    )
    # Force the generator to yield at least one segment so info is populated
    try:
        next(iter(_segments))
    except StopIteration:
        pass
    lang = getattr(info, "language", "en") or "en"
    prob = getattr(info, "language_probability", 0)
    _log.info("Detected language: %s (probability: %.2f)", lang, prob)
    return lang


# Serializes inference across threads: the live-transcription loop and the
# final post-meeting transcription may otherwise run on MPS concurrently.
_transcribe_lock = threading.Lock()

# All model loads and inference run on this single worker thread. MLX (0.32+)
# binds arrays to the thread that created them — loading a model in one
# thread and evaluating it in another raises
# "There is no Stream(cpu, N) in current thread".
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trnscrb-inference")


# MLX keeps every GPU buffer it allocates in a cache that never shrinks on its
# own. Unbounded, a background app ends up holding multiple GB of Metal memory
# between meetings (observed: 7.3 GB of IOAccelerator after a few passes).
_MLX_CACHE_LIMIT_MB = 512


def _mlx() -> object | None:
    """mlx.core if it is already imported — never import it just to trim."""
    import sys

    return sys.modules.get("mlx.core")


def _bound_mlx_cache() -> None:
    """Cap MLX's GPU buffer cache. Safe to call repeatedly."""
    mx = _mlx()
    if mx is None:
        return
    try:
        configured = settings.get("mlx_cache_limit_mb")
        # Explicit 0 means "leave MLX at its own default"; unset means ours.
        limit = _MLX_CACHE_LIMIT_MB if configured is None else int(configured)
        if limit > 0:
            mx.set_cache_limit(limit * 1024 * 1024)
    except Exception:
        _log.debug("Could not set MLX cache limit", exc_info=True)


def trim_mlx_cache() -> float:
    """Release cached MLX GPU buffers. Returns MB freed (0 if MLX unused)."""
    mx = _mlx()
    if mx is None:
        return 0.0
    try:
        before = mx.get_cache_memory()
        mx.clear_cache()
        freed = (before - mx.get_cache_memory()) / 1e6
        if freed > 1:
            _log.debug("Released %.0f MB of MLX GPU cache", freed)
        return freed
    except Exception:
        _log.debug("Could not clear MLX cache", exc_info=True)
        return 0.0


def preload(backend: str | None = None) -> None:
    """Warm the configured backend's models on the shared inference thread."""
    backend = backend or _backend()

    def _load():
        if backend in ("auto", "parakeet"):
            _get_parakeet_model()
        if backend in ("auto", "whisper"):
            _get_whisper_model()
        if backend == "qwen3":
            _get_qwen3()
        if backend == "voxtral":
            _get_voxtral_pipeline()

    with _transcribe_lock:
        _inference_executor.submit(_load).result()
        _inference_executor.submit(_bound_mlx_cache).result()
    _log.info("Transcription model preloaded (%s)", backend)


def transcribe(audio_path: Path) -> list[dict]:
    """Return segments: [{start, end, text, speaker}] — speaker filled later by diarizer.

    Serialized with a lock so concurrent calls don't overlap on the GPU, and
    executed on the dedicated inference thread (see _inference_executor).
    """
    audio_path = Path(audio_path)
    file_size = audio_path.stat().st_size if audio_path.exists() else 0
    _log.info("Transcribing %s (%d bytes)", audio_path, file_size)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if file_size == 0:
        raise FileNotFoundError(f"Audio file is empty: {audio_path}")

    backend = _backend()
    with _transcribe_lock:
        segments = _inference_executor.submit(_transcribe_on_worker, audio_path, backend).result()
        # Hand back the GPU buffers this pass allocated; the models themselves
        # stay loaded (they are released by unload_models when idle).
        _inference_executor.submit(trim_mlx_cache).result()
    _apply_glossary(segments)
    _log.info("Transcription complete: %d segments", len(segments))
    return segments


def _apply_glossary(segments: list[dict]) -> None:
    """Rewrite each segment in place to use the custom vocabulary, if any.

    Runs for every backend so the saved transcript carries your terminology
    directly. Whisper is additionally biased at decode time (see
    _transcribe_whisper); Parakeet has no such hook, so this is its only path.
    """
    from trnscrb import glossary

    entries = glossary.load()
    if not entries:
        return
    for seg in segments:
        seg["text"] = glossary.correct(seg.get("text", ""), entries)


def _transcribe_on_worker(audio_path: Path, backend: str) -> list[dict]:
    if backend == "auto":
        lang = _detect_language(audio_path)
        backend = "parakeet" if lang == "en" else "qwen3"
        _log.info("Auto-routing to %s (%s detected)", backend, lang)
    else:
        _log.debug("Using backend: %s", backend)

    if backend == "whisper":
        return _transcribe_whisper(audio_path)

    runners = {
        "parakeet": _transcribe_parakeet,
        "voxtral": _transcribe_voxtral,
        "qwen3": _transcribe_qwen3,
    }
    try:
        return runners[backend](audio_path)
    except Exception as e:
        # Any configured backend can be unavailable — an uninstalled
        # dependency, a model that was never downloaded. Whisper is bundled,
        # so it can always answer. A fallback transcript beats losing the
        # meeting to a preserved WAV nobody ever revisits.
        _log.warning("%s backend failed (%s); falling back to Whisper", backend, e)
        return _transcribe_whisper(audio_path)


def unload_models() -> None:
    """Release all loaded models to free memory after a long idle period.

    Safe to call at any time — takes the transcribe lock, so it waits for any
    in-flight transcription; models reload lazily on next use.
    """
    global _whisper_model, _parakeet_model, _parakeet_model_id
    global _voxtral_pipeline, _voxtral_model_id
    global _qwen3_model, _qwen3_aligner, _qwen3_model_id
    import gc

    with _transcribe_lock:
        with _whisper_model_lock:
            _whisper_model = None
        with _parakeet_model_lock:
            _parakeet_model = None
            _parakeet_model_id = None
        with _voxtral_pipeline_lock:
            _voxtral_pipeline = None
            _voxtral_model_id = None
        with _qwen3_lock:
            _qwen3_model = None
            _qwen3_aligner = None
            _qwen3_model_id = None
    gc.collect()
    # Model weights live in MLX's GPU cache too — drop them after the Python
    # references are gone, or the memory stays held despite the unload.
    freed = trim_mlx_cache()
    _log.info("Transcription models unloaded (%.0f MB GPU cache released)", freed)
