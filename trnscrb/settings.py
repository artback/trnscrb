"""Persistent user settings stored in ~/.config/trnscrb/settings.json."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.settings")

_SETTINGS_FILE = Path.home() / ".config" / "trnscrb" / "settings.json"

_DEFAULT_ENRICH_PROFILES = {
    "claude_code": {
        "endpoint": "",
        "api_key": "",
        "model": "sonnet",
        "models": ["sonnet", "opus", "haiku"],
    },
    "ollama": {
        "endpoint": "http://127.0.0.1:11434",
        "api_key": "",
        "model": "",
        "models": [],
    },
    "llama_cpp": {
        "endpoint": "http://127.0.0.1:8080",
        "api_key": "",
        "model": "",
        "models": [],
    },
    "lmstudio": {
        "endpoint": "http://127.0.0.1:1234",
        "api_key": "",
        "model": "",
        "models": [],
    },
    "anthropic": {
        "endpoint": "https://api.anthropic.com",
        "api_key": "",
        "model": "",
        "models": [],
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1",
        "api_key": "",
        "model": "",
        "models": [],
    },
}

_DEFAULT_INTEGRATE_PROMPT = (
    "Read the meeting transcript at {transcript_path} and integrate the key "
    "decisions and action items into my notes."
)

_DEFAULTS: dict = {
    "auto_record": True,  # start watching for mic activity on launch
    "auto_enrich": False,  # run LLM enrichment automatically after transcription
    "auto_integrate": False,  # push transcripts into notes via the Claude Code CLI
    # Prompt for note integration; {transcript_path} is replaced with the
    # absolute path of the saved transcript.
    "integrate_prompt": _DEFAULT_INTEGRATE_PROMPT,
    # Comma-separated list passed to `claude -p --allowedTools`.
    # Empty string omits the flag (all tools allowed).
    "integrate_allowed_tools": "Read,Write,Edit,Glob,Grep",
    "live_on_battery": False,  # keep the live-transcription loop running on battery
    "transcription_backend": "auto",  # auto | parakeet | whisper | voxtral | qwen3
    "parakeet_model_id": "mlx-community/parakeet-tdt-0.6b-v3",
    "qwen3_model_id": "Qwen/Qwen3-ASR-0.6B",  # ~1.2 GB; Qwen3-ASR-1.7B for more accuracy
    "model_size": "small",  # whisper model size (used when backend=whisper)
    # Diarization pipeline; falls back to pyannote/speaker-diarization-3.1
    "diarization_pipeline": "pyannote/speaker-diarization-community-1",
    "enrich": {
        "provider": "llama_cpp",
        "profiles": _DEFAULT_ENRICH_PROFILES,
        "last_test_status": {},
    },
}


def load() -> dict:
    loaded: dict = {}
    if _SETTINGS_FILE.exists():
        try:
            raw = json.loads(_SETTINGS_FILE.read_text())
            if isinstance(raw, dict):
                loaded = raw
        except json.JSONDecodeError:
            bak = _SETTINGS_FILE.with_suffix(".json.bak")
            shutil.copy2(_SETTINGS_FILE, bak)
            _log.warning("Corrupted settings backed up to %s", bak)
        except Exception:
            pass
    merged = _deep_merge(_DEFAULTS, loaded)
    _log.debug("Settings loaded successfully")
    return merged


def save(settings: dict) -> None:
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def get(key: str):
    return load().get(key, _DEFAULTS.get(key))


def put(key: str, value) -> None:
    s = load()
    s[key] = value
    save(s)


def read_hf_token() -> str | None:
    """Read HuggingFace token from env or ~/.cache/huggingface/token."""
    import os

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip() or None
    return None


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        default_value = merged.get(key)
        if isinstance(default_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(default_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged
