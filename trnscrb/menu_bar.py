"""macOS menu bar app (rumps).

States:
  idle        — mic icon, Start enabled, Stop disabled
  watching    — mic icon (auto-record on, listening)
  recording   — red icon, Start disabled, Stop enabled
  transcribing— red icon, Start disabled, Stop shows "Transcribing…" (disabled)
"""
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import rumps

from trnscrb import recorder as rec_module, transcriber, diarizer, storage, enricher
from trnscrb.recorder import cleanup_stale_temp_files
from trnscrb.calendar_integration import get_current_or_upcoming_event
from trnscrb.icon import icon_path, generate_icons
from trnscrb.watcher import MicWatcher
from trnscrb.log import get_logger
from trnscrb.settings import (
    get as get_setting,
    put as put_setting,
    load as load_settings,
    save as save_settings,
)

_log = get_logger("trnscrb.menu_bar")

_EMOJI_IDLE      = "🎙"
_EMOJI_RECORDING = "🔴"


def _notify(title: str, subtitle: str, message: str) -> None:
    """Best-effort notification; some non-bundle launches lack Info.plist metadata."""
    try:
        rumps.notification(title, subtitle, message)
    except Exception:
        pass


class TrnscrbApp(rumps.App):
    def __init__(self):
        cleanup_stale_temp_files()

        try:
            generate_icons()
        except Exception:
            pass

        idle_icon = icon_path(recording=False)
        super().__init__(
            "Trnscrb",
            icon=idle_icon,
            title=None if idle_icon else _EMOJI_IDLE,
            quit_button=None,
            template=True,
        )

        # Keep direct references so we can retitle without re-lookup
        self._start_item = rumps.MenuItem("Start Transcribing", callback=self.start_recording)
        self._stop_item  = rumps.MenuItem("Stop Transcribing",  callback=None)
        self._auto_item  = rumps.MenuItem("Auto-transcribe: Off", callback=self.toggle_auto_record)
        self._settings_item = rumps.MenuItem("Settings")
        self._provider_item = rumps.MenuItem("Provider")
        self._endpoint_item = rumps.MenuItem("Endpoint…", callback=self.edit_enrich_endpoint)
        self._api_key_item = rumps.MenuItem("API Key…", callback=self.edit_enrich_api_key)
        self._test_endpoint_item = rumps.MenuItem(
            "Test Endpoint & Load Models",
            callback=self.test_enrich_endpoint,
        )
        self._model_item = rumps.MenuItem("Model")

        self._settings_item.add(self._provider_item)
        self._settings_item.add(self._endpoint_item)
        self._settings_item.add(self._api_key_item)
        self._settings_item.add(self._test_endpoint_item)
        self._settings_item.add(self._model_item)

        self.menu = [
            self._start_item,
            self._stop_item,
            None,
            self._auto_item,
            self._settings_item,
            None,
            rumps.MenuItem("Open Notes Folder", callback=self.open_folder),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        self._recorder:   rec_module.Recorder | None = None
        self._started_at: datetime | None = None
        self._watcher:    MicWatcher | None = None
        self._process_thread: threading.Thread | None = None
        self._rec_lock = threading.Lock()  # guards _do_start / _do_stop

        self._set_state("idle")

        if get_setting("auto_record"):
            self._start_watcher()
            self._auto_item.title = "Auto-transcribe: On ✓"
        self._refresh_enrich_settings_menu()

    # ── watcher ───────────────────────────────────────────────────────────────

    def _start_watcher(self):
        self._watcher = MicWatcher(on_start=self._auto_start, on_stop=self._auto_stop)
        self._watcher.start()
        if not (self._recorder and self._recorder.is_recording):
            self._set_icon_state("watching")

    # ── manual controls ───────────────────────────────────────────────────────

    def start_recording(self, _):
        if self._recorder and self._recorder.is_recording:
            return
        self._do_start()

    def stop_recording(self, _):
        if not self._recorder or not self._recorder.is_recording:
            return
        self._do_stop()

    def toggle_auto_record(self, sender):
        if self._watcher and self._watcher.is_watching:
            self._watcher.stop()
            self._watcher = None
            sender.title = "Auto-transcribe: Off"
            put_setting("auto_record", False)
            if not (self._recorder and self._recorder.is_recording):
                self._set_icon_state("idle")
            _notify("Trnscrb", "Auto-transcribe off", "")
        else:
            self._start_watcher()
            sender.title = "Auto-transcribe: On ✓"
            put_setting("auto_record", True)
            _notify("Trnscrb", "Auto-transcribe on", "Will start when mic is active for 5+ seconds")

    # ── enrichment settings ───────────────────────────────────────────────────

    def select_enrich_provider(self, sender):
        provider = getattr(sender, "_provider_key", "")
        if not provider:
            return
        settings = load_settings()
        enrich_cfg = settings.setdefault("enrich", {})
        enrich_cfg["provider"] = provider
        save_settings(settings)
        self._refresh_enrich_settings_menu()
        _notify("Trnscrb", "Enrich provider updated", enricher.provider_label(provider))

    def edit_enrich_endpoint(self, _):
        settings, provider, profile = self._active_enrich_profile()
        title = f"{enricher.provider_label(provider)} endpoint"
        window = rumps.Window(
            message="Base URL",
            title=title,
            default_text=profile["endpoint"],
            ok="Save",
            cancel="Cancel",
            dimensions=(440, 120),
        )
        result = window.run()
        if not result.clicked:
            return
        endpoint = result.text.strip()
        if not endpoint:
            return
        profile["endpoint"] = enricher.normalize_endpoint(provider, endpoint)
        self._save_enrich_profile(settings, provider, profile)
        self._refresh_enrich_settings_menu()
        _notify("Trnscrb", "Endpoint saved", profile["endpoint"])

    def edit_enrich_api_key(self, _):
        settings, provider, profile = self._active_enrich_profile()
        secure = provider in {"anthropic", "openai"}
        window = rumps.Window(
            message=f"{enricher.provider_label(provider)} API key",
            title="LLM API Key",
            default_text=profile["api_key"],
            ok="Save",
            cancel="Cancel",
            dimensions=(440, 120),
            secure=secure,
        )
        result = window.run()
        if not result.clicked:
            return
        profile["api_key"] = result.text.strip()
        self._save_enrich_profile(settings, provider, profile)
        self._refresh_enrich_settings_menu()
        state = "saved" if profile["api_key"] else "cleared"
        _notify("Trnscrb", f"API key {state}", enricher.provider_label(provider))

    def test_enrich_endpoint(self, _):
        threading.Thread(target=self._test_enrich_endpoint_worker, daemon=True).start()

    def _test_enrich_endpoint_worker(self):
        settings, provider, profile = self._active_enrich_profile()
        ok, message = enricher.test_provider_connection(
            provider,
            profile["endpoint"],
            profile["api_key"],
        )
        enrich_cfg = settings.setdefault("enrich", {})
        status = enrich_cfg.setdefault("last_test_status", {})
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        provider_name = enricher.provider_label(provider)
        if not ok:
            status[provider] = f"{stamp} FAIL: {message}"
            save_settings(settings)
            self._refresh_enrich_settings_menu()
            _notify("Trnscrb", f"{provider_name} test failed", str(message)[:180])
            return

        try:
            models = enricher.list_provider_models(
                provider,
                profile["endpoint"],
                profile["api_key"],
            )
        except Exception as exc:
            status[provider] = f"{stamp} FAIL: {exc}"
            save_settings(settings)
            self._refresh_enrich_settings_menu()
            _notify("Trnscrb", f"{provider_name} model load failed", str(exc)[:180])
            return

        profile["models"] = models
        if models and profile.get("model") not in models:
            profile["model"] = models[0]
        self._save_enrich_profile(settings, provider, profile)
        status = settings.setdefault("enrich", {}).setdefault("last_test_status", {})
        status[provider] = f"{stamp} OK: {len(models)} model(s)"
        save_settings(settings)
        self._refresh_enrich_settings_menu()
        _notify("Trnscrb", f"{provider_name} connected", f"{len(models)} model(s) loaded")

    def select_enrich_model(self, sender):
        model = getattr(sender, "_model_name", "").strip()
        if not model:
            return
        settings, provider, profile = self._active_enrich_profile()
        profile["model"] = model
        self._save_enrich_profile(settings, provider, profile)
        self._refresh_enrich_settings_menu()
        _notify("Trnscrb", "Enrich model selected", model)

    def _refresh_enrich_settings_menu(self):
        settings, provider, profile = self._active_enrich_profile()

        self._settings_item.title = f"Settings ({enricher.provider_label(provider)})"

        # Claude Code uses the local CLI — no endpoint or API key needed.
        if provider == "claude_code":
            self._endpoint_item.title = "Endpoint… (n/a)"
            self._endpoint_item.set_callback(None)
            self._api_key_item.title = "API Key… (n/a)"
            self._api_key_item.set_callback(None)
        else:
            endpoint_display = profile["endpoint"]
            if len(endpoint_display) > 36:
                endpoint_display = endpoint_display[:33] + "..."
            self._endpoint_item.title = f"Endpoint… ({endpoint_display})"
            self._endpoint_item.set_callback(self.edit_enrich_endpoint)
            key_state = "Set" if profile["api_key"] else "Not set"
            self._api_key_item.title = f"API Key… ({key_state})"
            self._api_key_item.set_callback(self.edit_enrich_api_key)

        self._clear_submenu_if_initialized(self._provider_item)
        for option in enricher.PROVIDER_ORDER:
            item = rumps.MenuItem(enricher.provider_label(option), callback=self.select_enrich_provider)
            item._provider_key = option
            item.state = 1 if option == provider else 0
            self._provider_item.add(item)

        self._clear_submenu_if_initialized(self._model_item)
        models = profile["models"]
        selected_model = str(profile.get("model") or "")
        if models:
            for model in models:
                item = rumps.MenuItem(model, callback=self.select_enrich_model)
                item._model_name = model
                item.state = 1 if model == selected_model else 0
                self._model_item.add(item)
            model_display = selected_model or "Select model"
        else:
            self._model_item.add(rumps.MenuItem("No models loaded"))
            model_display = "No models loaded"
        if len(model_display) > 32:
            model_display = model_display[:29] + "..."
        self._model_item.title = f"Model ({model_display})"

    def _active_enrich_profile(self) -> tuple[dict, str, dict]:
        settings = load_settings()
        enrich_cfg = settings.setdefault("enrich", {})
        provider = enricher.normalize_provider(enrich_cfg.get("provider"))
        profiles = enrich_cfg.setdefault("profiles", {})
        profile = profiles.setdefault(provider, {})
        endpoint = profile.get("endpoint") or enricher.DEFAULT_ENDPOINTS[provider]
        model_list = profile.get("models")
        profile["endpoint"] = enricher.normalize_endpoint(provider, endpoint)
        profile["api_key"] = str(profile.get("api_key") or "")
        profile["model"] = str(profile.get("model") or "")
        if isinstance(model_list, list):
            profile["models"] = [str(model) for model in model_list if str(model).strip()]
        else:
            profile["models"] = []
        return settings, provider, profile

    def _save_enrich_profile(self, settings: dict, provider: str, profile: dict):
        enrich_cfg = settings.setdefault("enrich", {})
        profiles = enrich_cfg.setdefault("profiles", {})
        profiles[provider] = profile
        save_settings(settings)

    def _clear_submenu_if_initialized(self, menu_item: rumps.MenuItem):
        # rumps only initializes MenuItem._menu after first submenu insertion.
        if getattr(menu_item, "_menu", None) is not None:
            menu_item.clear()

    def open_folder(self, _):
        subprocess.run(["open", str(storage.ensure_notes_dir())])

    def quit_app(self, _):
        if self._watcher:
            self._watcher.stop()

        # If a recording is in progress, stop it and save the WAV so it isn't lost.
        if self._recorder and self._recorder.is_recording:
            _log.info("Quit requested while recording; stopping recorder and saving audio")
            audio_path = self._recorder.stop()
            self._recorder = None
            if audio_path:
                notes_dir = storage.ensure_notes_dir()
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                saved = notes_dir / f"{stamp}_unsaved-recording.wav"
                try:
                    import shutil
                    shutil.move(str(audio_path), str(saved))
                    _log.info("Saved in-progress recording to %s", saved)
                except Exception:
                    _log.error("Failed to rescue recording from %s", audio_path, exc_info=True)

        # If a background transcription thread is running, give it a few seconds.
        if self._process_thread and self._process_thread.is_alive():
            _log.info("Waiting up to 5 s for transcription thread to finish")
            self._process_thread.join(timeout=5)
            if self._process_thread.is_alive():
                _log.warning("Transcription thread still running; quitting anyway")

        rumps.quit_application()

    # ── shared start / stop ───────────────────────────────────────────────────

    def _do_start(self, meeting_name: str = ""):
        with self._rec_lock:
            if self._recorder and self._recorder.is_recording:
                return
            if not meeting_name:
                evt = get_current_or_upcoming_event()
                meeting_name = evt["title"] if evt else ""

            device = rec_module.Recorder.find_blackhole_device()
            self._recorder   = rec_module.Recorder(device=device)
            self._started_at = datetime.now()
            self._recorder.start()
            self._set_state("recording")

        source = "BlackHole (system + mic)" if device is not None else "built-in mic"
        _log.info("Recording started: meeting=%s device=%s", meeting_name or "(unnamed)", source)
        label  = f" — {meeting_name}" if meeting_name else ""
        _notify("Trnscrb", f"Transcription started{label}", f"via {source}")

    def _do_stop(self):
        with self._rec_lock:
            if not self._recorder or not self._recorder.is_recording:
                return
            _log.info("Recording stopped, starting transcription")
            started_at     = self._started_at or datetime.now()
            recorder       = self._recorder
            self._recorder = None
            self._set_state("transcribing")

        self._process_thread = threading.Thread(
            target=self._process, args=(recorder, started_at), daemon=True
        )
        self._process_thread.start()

    # ── auto-record callbacks ─────────────────────────────────────────────────

    def _auto_start(self, meeting_name: str):
        if getattr(self, "_current_state", "idle") in ("recording", "transcribing"):
            return
        self._do_start(meeting_name=meeting_name)

    def _auto_stop(self):
        if self._recorder and self._recorder.is_recording:
            self._do_stop()

    # ── background transcription ──────────────────────────────────────────────

    def _process(self, recorder: rec_module.Recorder, started_at: datetime):
        audio_path = None
        try:
            audio_path = recorder.stop()
            if not audio_path:
                _notify("Trnscrb", "Error", "No audio captured.")
                return

            try:
                evt = get_current_or_upcoming_event()
                meeting_name = evt["title"] if evt else ""
            except Exception:
                meeting_name = ""
            if not meeting_name:
                meeting_name = f"meeting-{started_at.strftime('%H%M')}"

            _log.info("Transcription starting: %s", meeting_name)
            try:
                segments = transcriber.transcribe(audio_path)
            except Exception as e:
                _log.error("Transcription failed for %s: %s", meeting_name, e)
                _notify("Trnscrb", "Transcription failed", str(e))
                return

            hf_token = _read_hf_token()
            if hf_token and segments:
                try:
                    diar     = diarizer.diarize(audio_path, hf_token)
                    segments = diarizer.merge(segments, diar)
                except Exception as e:
                    _log.warning("Diarization skipped: %s", e)
                    _notify("Trnscrb", "Speaker labels skipped", str(e)[:180])

            text = storage.format_transcript(segments, started_at, meeting_name)
            path = storage.get_transcript_path(meeting_name, started_at)
            storage.save_transcript(path, text)
            _log.info("Transcription complete: %s -> %s", meeting_name, path.name)
            _notify("Trnscrb", f"Saved: {meeting_name}", f"~/meeting-notes/{path.name}")
        except Exception as e:
            _log.error("Unexpected error in _process: %s", e, exc_info=True)
            _notify("Trnscrb", "Error", str(e)[:180])
        finally:
            if audio_path:
                audio_path.unlink(missing_ok=True)
            self._restore_idle()

    def _restore_idle(self):
        """Called from background thread when transcription finishes."""
        state = "watching" if (self._watcher and self._watcher.is_watching) else "idle"
        self._set_state(state)

    # ── state / icon management ───────────────────────────────────────────────

    def _set_state(self, state: str):
        """state: idle | watching | recording | transcribing"""
        self._current_state = state
        if state in ("idle", "watching"):
            self._start_item.set_callback(self.start_recording)
            self._stop_item.title = "Stop Transcribing"
            self._stop_item.set_callback(None)
        elif state == "recording":
            self._start_item.set_callback(None)
            self._stop_item.title = "Stop Transcribing"
            self._stop_item.set_callback(self.stop_recording)
        elif state == "transcribing":
            self._start_item.set_callback(None)
            self._stop_item.title = "Transcribing…"
            self._stop_item.set_callback(None)

        self._set_icon_state(state)

    def _set_icon_state(self, state: str):
        rec_icon  = icon_path(recording=True)
        idle_icon = icon_path(recording=False)
        if state in ("recording", "transcribing"):
            self.icon, self.title = (rec_icon, None) if rec_icon else (None, _EMOJI_RECORDING)
        else:
            self.icon, self.title = (idle_icon, None) if idle_icon else (None, _EMOJI_IDLE)


def _read_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip() or None
    return None


def main():
    import AppKit
    app = TrnscrbApp()
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(
        AppKit.NSApplicationActivationPolicyAccessory
    )
    app.run()
