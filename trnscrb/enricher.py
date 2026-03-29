"""Post-call LLM enrichment via configurable local/cloud providers."""
from __future__ import annotations

import json
from typing import Optional
from urllib import request

from trnscrb import settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.enricher")

_PROMPT_TEMPLATE = """You are analyzing a meeting transcript.{context}

Transcript:
{transcript}

Provide:
1. A brief summary (2-3 sentences)
2. Action items with owner names if identifiable
3. Inferred speaker names — if speakers appear as SPEAKER_00, SPEAKER_01 etc., \
infer their names or roles from the conversation

Respond in exactly this format:

SUMMARY:
<summary here>

ACTION ITEMS:
- <item> (Owner: <name or Unknown>)

SPEAKER MAPPING:
- SPEAKER_00 → <inferred name or "Participant 1">
- SPEAKER_01 → <inferred name or "Participant 2">
"""

PROVIDER_ORDER = ("ollama", "llama_cpp", "lmstudio", "anthropic", "openai")
OPENAI_COMPATIBLE_PROVIDERS = {"llama_cpp", "lmstudio", "openai"}
DEFAULT_ENDPOINTS = {
    "ollama": "http://127.0.0.1:11434",
    "llama_cpp": "http://127.0.0.1:8080",
    "lmstudio": "http://127.0.0.1:1234",
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
}
PROVIDER_LABELS = {
    "ollama": "Ollama API",
    "llama_cpp": "llama.cpp",
    "lmstudio": "LM Studio",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
}


class OllamaAdapter:
    def test_connection(self, config: dict) -> tuple[bool, str]:
        try:
            models = self.list_models(config)
        except Exception as exc:
            _log.debug("Ollama connection failed: %s", exc)
            return False, str(exc)
        _log.debug("Ollama connection succeeded, %d model(s)", len(models))
        return True, f"Connected ({len(models)} model(s))"

    def list_models(self, config: dict) -> list[str]:
        payload = _json_request(config["endpoint"], "/api/tags", method="GET")
        models = payload.get("models", [])
        names = [str(model.get("name") or "").strip() for model in models]
        return [name for name in names if name]

    def enrich(self, prompt: str, config: dict) -> str:
        payload = {
            "model": config["model"],
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = _json_request(config["endpoint"], "/api/chat", method="POST", payload=payload)
        message = response.get("message", {})
        content = str(message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content


class OpenAICompatibleAdapter:
    def __init__(self, provider: str):
        self.provider = provider

    def _client(self, config: dict):
        endpoint = normalize_endpoint(self.provider, config["endpoint"])
        api_key = str(config.get("api_key") or "local")
        return _build_openai_client(base_url=endpoint, api_key=api_key)

    def test_connection(self, config: dict) -> tuple[bool, str]:
        try:
            models = self.list_models(config)
        except Exception as exc:
            _log.debug("%s connection failed: %s", self.provider, exc)
            return False, str(exc)
        _log.debug("%s connection succeeded, %d model(s)", self.provider, len(models))
        return True, f"Connected ({len(models)} model(s))"

    def list_models(self, config: dict) -> list[str]:
        client = self._client(config)
        response = client.models.list()
        models = [str(model.id) for model in getattr(response, "data", []) if getattr(model, "id", None)]
        return models

    def enrich(self, prompt: str, config: dict) -> str:
        client = self._client(config)
        response = client.chat.completions.create(
            model=config["model"],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("Provider returned no choices.")
        content = getattr(choices[0].message, "content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if text:
                    parts.append(str(text))
            content = "\n".join(parts)
        content = str(content or "").strip()
        if not content:
            raise RuntimeError("Provider returned an empty response.")
        return content


class AnthropicAdapter:
    def _client(self, config: dict):
        from anthropic import Anthropic

        api_key = str(config.get("api_key") or "")
        if not api_key:
            raise RuntimeError("Anthropic API key is required.")
        endpoint = str(config.get("endpoint") or DEFAULT_ENDPOINTS["anthropic"]).rstrip("/")
        kwargs = {"api_key": api_key}
        if endpoint and endpoint != DEFAULT_ENDPOINTS["anthropic"]:
            kwargs["base_url"] = endpoint
        return Anthropic(**kwargs)

    def test_connection(self, config: dict) -> tuple[bool, str]:
        try:
            models = self.list_models(config)
        except Exception as exc:
            _log.debug("Anthropic connection failed: %s", exc)
            return False, str(exc)
        _log.debug("Anthropic connection succeeded, %d model(s)", len(models))
        return True, f"Connected ({len(models)} model(s))"

    def list_models(self, config: dict) -> list[str]:
        client = self._client(config)
        response = client.models.list(limit=100)
        if hasattr(response, "data"):
            items = response.data
        else:
            items = list(response)
        models = [str(model.id) for model in items if getattr(model, "id", None)]
        return models

    def enrich(self, prompt: str, config: dict) -> str:
        client = self._client(config)
        response = client.messages.create(
            model=config["model"],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        blocks = getattr(response, "content", [])
        parts = [str(block.text).strip() for block in blocks if getattr(block, "text", None)]
        content = "\n".join(part for part in parts if part).strip()
        if not content:
            raise RuntimeError("Anthropic returned an empty response.")
        return content


_ADAPTERS = {
    "ollama": OllamaAdapter(),
    "llama_cpp": OpenAICompatibleAdapter("llama_cpp"),
    "lmstudio": OpenAICompatibleAdapter("lmstudio"),
    "anthropic": AnthropicAdapter(),
    "openai": OpenAICompatibleAdapter("openai"),
}


def provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def normalize_provider(provider: str | None) -> str:
    normalized = str(provider or "").strip().lower().replace(".", "_")
    if normalized in PROVIDER_ORDER:
        return normalized
    return "llama_cpp"


def normalize_endpoint(provider: str, endpoint: str | None) -> str:
    normalized_provider = normalize_provider(provider)
    value = str(endpoint or DEFAULT_ENDPOINTS[normalized_provider]).strip().rstrip("/")
    if normalized_provider in OPENAI_COMPATIBLE_PROVIDERS and not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def get_active_provider_config() -> tuple[str, dict]:
    loaded = settings.load()
    enrich = loaded.get("enrich", {}) if isinstance(loaded, dict) else {}
    provider = normalize_provider(enrich.get("provider"))
    profile = _get_provider_profile(provider, loaded)
    return provider, profile


def test_provider_connection(provider: str, endpoint: str, api_key: str = "") -> tuple[bool, str]:
    provider = normalize_provider(provider)
    config = _build_runtime_config(provider, endpoint=endpoint, api_key=api_key, model="")
    return _ADAPTERS[provider].test_connection(config)


def list_provider_models(provider: str, endpoint: str, api_key: str = "") -> list[str]:
    provider = normalize_provider(provider)
    config = _build_runtime_config(provider, endpoint=endpoint, api_key=api_key, model="")
    return _ADAPTERS[provider].list_models(config)


def enrich_transcript(
    transcript_text: str,
    calendar_event: Optional[dict] = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """Return {enrichment, speaker_map, enriched_transcript, provider, model}."""
    active_provider, active_profile = get_active_provider_config()
    if provider:
        active_provider = normalize_provider(provider)
        active_profile = _get_provider_profile(active_provider)

    selected_model = str(model or active_profile.get("model") or "").strip()
    if not selected_model:
        loaded_models = active_profile.get("models") or []
        if loaded_models:
            selected_model = str(loaded_models[0]).strip()
    if not selected_model:
        raise RuntimeError(
            f"No model selected for {provider_label(active_provider)}. "
            "Use Settings → Test Endpoint & Load Models and choose a model."
        )

    config = _build_runtime_config(
        active_provider,
        endpoint=active_profile.get("endpoint"),
        api_key=active_profile.get("api_key", ""),
        model=selected_model,
    )
    adapter = _ADAPTERS[active_provider]

    context = ""
    if calendar_event:
        context = f"\nMeeting: {calendar_event.get('title', '')}"
        if calendar_event.get("attendees"):
            context += f"\nKnown attendees: {', '.join(calendar_event['attendees'])}"

    prompt = _PROMPT_TEMPLATE.format(context=context, transcript=transcript_text)
    _log.info("Enriching transcript with provider=%s model=%s", active_provider, selected_model)
    try:
        enrichment = adapter.enrich(prompt, config)
    except Exception:
        _log.error("Enrichment failed for provider=%s model=%s", active_provider, selected_model)
        raise
    _log.info("Enrichment succeeded for provider=%s", active_provider)

    speaker_map = _parse_speaker_map(enrichment)
    enriched = _apply_speaker_map(transcript_text, speaker_map)

    return {
        "enrichment": enrichment,
        "speaker_map": speaker_map,
        "enriched_transcript": enriched,
        "provider": active_provider,
        "model": selected_model,
    }


def _parse_speaker_map(enrichment: str) -> dict:
    speaker_map = {}
    in_section = False
    for line in enrichment.splitlines():
        if line.strip().startswith("SPEAKER MAPPING:"):
            in_section = True
            continue
        if in_section:
            separator = None
            for sep in ("→", "->", ":"):
                if sep in line:
                    separator = sep
                    break
            if separator is not None:
                raw, _, name = line.partition(separator)
                raw = raw.strip().lstrip("- ").strip()
                name = name.strip().strip('"')
                if raw:
                    speaker_map[raw] = name
            elif line.strip() and not line.strip().startswith("-"):
                break  # end of section
    _log.debug("Parsed %d speaker mapping(s)", len(speaker_map))
    return speaker_map


def _apply_speaker_map(text: str, speaker_map: dict) -> str:
    for raw, name in speaker_map.items():
        text = text.replace(f"[{raw}]", f"[{name}]")
    return text


def _build_openai_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _get_provider_profile(provider: str, loaded_settings: dict | None = None) -> dict:
    loaded = loaded_settings if loaded_settings is not None else settings.load()
    enrich = loaded.get("enrich", {}) if isinstance(loaded, dict) else {}
    profiles = enrich.get("profiles", {}) if isinstance(enrich, dict) else {}
    raw_profile = profiles.get(provider, {}) if isinstance(profiles, dict) else {}
    if not isinstance(raw_profile, dict):
        raw_profile = {}

    merged = {
        "endpoint": raw_profile.get("endpoint") or DEFAULT_ENDPOINTS[provider],
        "api_key": str(raw_profile.get("api_key") or ""),
        "model": str(raw_profile.get("model") or ""),
        "models": raw_profile.get("models") if isinstance(raw_profile.get("models"), list) else [],
    }
    merged["endpoint"] = normalize_endpoint(provider, str(merged["endpoint"]))
    merged["models"] = [str(model).strip() for model in merged["models"] if str(model).strip()]
    return merged


def _build_runtime_config(provider: str, endpoint: str | None, api_key: str, model: str) -> dict:
    return {
        "provider": provider,
        "endpoint": normalize_endpoint(provider, endpoint),
        "api_key": str(api_key or ""),
        "model": str(model or "").strip(),
    }


def _json_request(base_endpoint: str, path: str, method: str = "GET", payload: dict | None = None) -> dict:
    url = base_endpoint.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    response = request.urlopen(req, timeout=10)
    try:
        raw = response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _log.error("Invalid JSON response from %s: %s", url, exc)
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
