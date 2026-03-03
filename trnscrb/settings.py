"""Persistent user settings stored in ~/.config/trnscrb/settings.json."""
from copy import deepcopy
import json
from pathlib import Path

_SETTINGS_FILE = Path.home() / ".config" / "trnscrb" / "settings.json"

_DEFAULT_ENRICH_PROFILES = {
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

_DEFAULTS: dict = {
    "auto_record": True,  # start watching for mic activity on launch
    "transcription_backend": "parakeet",  # parakeet | whisper
    "parakeet_model_id": "mlx-community/parakeet-tdt-0.6b-v3",
    "model_size": "small",  # whisper model size (used when backend=whisper)
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
        except Exception:
            pass
    return _deep_merge(_DEFAULTS, loaded)


def save(settings: dict) -> None:
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def get(key: str):
    return load().get(key, _DEFAULTS.get(key))


def put(key: str, value) -> None:
    s = load()
    s[key] = value
    save(s)


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        default_value = merged.get(key)
        if isinstance(default_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(default_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged
