"""macOS menu bar app (rumps).

States:
  idle        — mic icon, Start enabled, Stop disabled
  watching    — mic icon (auto-record on, listening)
  recording   — red icon, Start disabled, Stop enabled
  transcribing— red icon, Start disabled, Stop shows "Transcribing…" (disabled)
"""

import subprocess
import threading
from datetime import datetime
from pathlib import Path

import rumps

from trnscrb import attribution, diarizer, enricher, storage, transcriber
from trnscrb import recorder as rec_module
from trnscrb.calendar_integration import get_current_or_upcoming_event
from trnscrb.enricher import enrich_transcript
from trnscrb.icon import generate_icons, icon_path
from trnscrb.log import get_logger
from trnscrb.recorder import cleanup_stale_temp_files
from trnscrb.settings import get as get_setting
from trnscrb.settings import load as load_settings
from trnscrb.settings import put as put_setting
from trnscrb.settings import read_hf_token
from trnscrb.settings import save as save_settings
from trnscrb.watcher import MicWatcher

_log = get_logger("trnscrb.menu_bar")

_EMOJI_IDLE = "🎙"
_EMOJI_RECORDING = "🔴"


def _notify(title: str, subtitle: str, message: str) -> None:
    """Best-effort notification; some non-bundle launches lack Info.plist metadata."""
    try:
        rumps.notification(title, subtitle, message)
    except Exception:
        pass


def _on_battery() -> bool:
    """True when running on battery power (best-effort; False on any error)."""
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3
        ).stdout
        return "Battery Power" in out
    except Exception:
        return False


def _find_claude_cli() -> str | None:
    """Locate the claude CLI — launchd runs with a bare PATH, so check common spots."""
    import os
    import shutil

    found = shutil.which("claude")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "claude",
        Path.home() / ".claude" / "local" / "claude",
        Path("/opt/homebrew/bin/claude"),
        Path("/usr/local/bin/claude"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _integrate_notes(transcript_path: Path) -> None:
    """Fire-and-forget: ask Claude Code to fold the transcript into the user's notes.

    Prompt and tool allowlist come from the `integrate_prompt` and
    `integrate_allowed_tools` settings.
    """
    claude = _find_claude_cli()
    if not claude:
        _log.warning("Auto-integrate skipped: claude CLI not found on PATH or common locations")
        _notify("Trnscrb", "Note integration skipped", "Claude CLI not found")
        return
    template = str(get_setting("integrate_prompt") or "")
    allowed = str(get_setting("integrate_allowed_tools") or "")
    try:
        prompt = template.format(transcript_path=transcript_path)
    except (KeyError, IndexError) as e:
        _log.error("Invalid integrate_prompt template (%s); skipping note integration", e)
        return
    cmd = [claude, "-p", prompt]
    if allowed:
        cmd += ["--allowedTools", allowed]
    _log.info("Note integration via Claude Code started for %s", transcript_path.name)
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path.home()),
        )
    except Exception as e:
        _log.error("Could not launch claude CLI for note integration: %s", e)


class TrnscrbApp(rumps.App):
    def __init__(self):
        cleanup_stale_temp_files()
        storage.clear_live_session()  # any previous session died with us
        storage.finalize_orphaned_live_markers()
        self._publish_app_state()
        storage.apply_retention()

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
        self._stop_item = rumps.MenuItem("Stop Transcribing", callback=None)
        self._auto_item = rumps.MenuItem("Auto-transcribe: Off", callback=self.toggle_auto_record)
        self._integrate_item = rumps.MenuItem(
            "Auto-integrate notes: Off", callback=self.toggle_auto_integrate
        )
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

        self._open_latest_item = rumps.MenuItem("Open Latest", callback=self.open_latest)

        self.menu = [
            self._start_item,
            self._stop_item,
            None,
            self._auto_item,
            self._integrate_item,
            self._open_latest_item,
            self._settings_item,
            None,
            rumps.MenuItem("Open Notes Folder", callback=self.open_folder),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        self._recorder: rec_module.Recorder | None = None
        self._started_at: datetime | None = None
        self._watcher: MicWatcher | None = None
        self._process_thread: threading.Thread | None = None
        self._rec_lock = threading.Lock()  # guards _do_start / _do_stop
        self._live_path: Path | None = None  # transcript written during recording
        self._live_thread: threading.Thread | None = None
        self._meeting_name: str = ""

        self._set_state("idle")
        self._install_signal_handlers()

        if get_setting("auto_record"):
            self._start_watcher()
            self._auto_item.title = "Auto-transcribe: On ✓"
        if get_setting("auto_integrate"):
            self._integrate_item.title = "Auto-integrate notes: On ✓"
        self._refresh_enrich_settings_menu()

        # Models load lazily when a recording starts (see _do_start) and are
        # released again after a long idle period to free ~1 GB of memory.
        self._unload_timer: threading.Timer | None = None

        # Elapsed recording time next to the menu bar icon (🔴 12:34).
        self._duration_shown = False
        self._duration_timer = rumps.Timer(self._update_duration_title, 15)
        self._duration_timer.start()

    _MODEL_IDLE_UNLOAD_SECS = 30 * 60

    def _cancel_model_unload(self):
        if self._unload_timer:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _schedule_model_unload(self):
        self._cancel_model_unload()
        timer = threading.Timer(self._MODEL_IDLE_UNLOAD_SECS, self._unload_idle_models)
        timer.daemon = True
        timer.start()
        self._unload_timer = timer

    def _unload_idle_models(self):
        if self._recorder and self._recorder.is_recording:
            return  # a new recording started; _do_start reschedules
        if self._process_thread and self._process_thread.is_alive():
            self._schedule_model_unload()  # transcription still running — try later
            return
        try:
            transcriber.unload_models()
            diarizer.unload_pipeline()
        except Exception:
            _log.debug("Idle model unload failed", exc_info=True)

    def _publish_app_state(self, **extra):
        """Publish permission/capability state for `trnscrb status`/`install`.

        Runs in the app's own process, so the TCC answers are for the
        Trnscrb.app identity — the one that actually records.
        """
        try:
            import trnscrb

            storage.write_app_state(
                version=trnscrb.__version__,
                system_audio_permission=rec_module.Recorder.system_audio_available(),
                **extra,
            )
        except Exception:
            _log.debug("Could not publish app state", exc_info=True)

    def _preload_model(self):
        try:
            backend = str(get_setting("transcription_backend") or "auto")
            # Loads happen on the transcriber's dedicated inference thread —
            # MLX models must be loaded and evaluated on the same thread.
            transcriber.preload(backend)
        except Exception as e:
            _log.debug("Model preload skipped: %s", e)

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
            _notify(
                "Trnscrb",
                "Auto-transcribe on",
                "Will start when mic is active for 5+ seconds",
            )

    def toggle_auto_integrate(self, sender):
        if get_setting("auto_integrate"):
            put_setting("auto_integrate", False)
            sender.title = "Auto-integrate notes: Off"
            _notify("Trnscrb", "Auto-integrate off", "")
        else:
            put_setting("auto_integrate", True)
            sender.title = "Auto-integrate notes: On ✓"
            if _find_claude_cli():
                msg = "Transcripts will be added to your notes via Claude Code"
            else:
                msg = "Claude CLI not found — install it for integration to work"
            _notify("Trnscrb", "Auto-integrate on", msg)

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
            item = rumps.MenuItem(
                enricher.provider_label(option), callback=self.select_enrich_provider
            )
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

    def open_latest(self, _):
        # During recording, open the live transcript; otherwise open the newest file
        target = self._live_path
        if not target or not target.exists():
            files = sorted(storage.NOTES_DIR.glob("*.txt"), reverse=True)
            target = files[0] if files else None
        if target and target.exists():
            subprocess.run(["open", str(target)])
        else:
            subprocess.run(["open", str(storage.ensure_notes_dir())])

    def open_folder(self, _):
        subprocess.run(["open", str(storage.ensure_notes_dir())])

    def quit_app(self, _):
        self._shutdown("Quit")

    def _shutdown(self, reason: str) -> None:
        """Stop cleanly, never losing an in-progress recording.

        Shared by the Quit menu item and the SIGTERM/SIGINT handlers, so a
        restart, upgrade, or logout saves the meeting instead of killing it.
        """
        if self._watcher:
            self._watcher.stop()

        # If a recording is in progress, stop it and save the WAV so it isn't lost.
        if self._recorder and self._recorder.is_recording:
            _log.info("%s while recording; stopping recorder and saving audio", reason)
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
                    _notify(
                        "Trnscrb",
                        "Recording saved",
                        f"Interrupted by {reason.lower()} — audio kept as {saved.name}",
                    )
                except Exception:
                    _log.error("Failed to rescue recording from %s", audio_path, exc_info=True)

        storage.clear_live_session()

        # If a background transcription thread is running, give it a few seconds.
        if self._process_thread and self._process_thread.is_alive():
            _log.info("Waiting up to 5 s for transcription thread to finish")
            self._process_thread.join(timeout=5)
            if self._process_thread.is_alive():
                _log.warning("Transcription thread still running; quitting anyway")

        rumps.quit_application()

    def _install_signal_handlers(self) -> None:
        """Route SIGTERM/SIGINT into the clean shutdown path.

        launchd sends SIGTERM on `launchctl kickstart -k`, upgrades, and
        logout. Plain signal handlers are unreliable under the AppKit run
        loop, so use libdispatch sources on the main queue.
        """
        try:
            import signal

            import libdispatch

            self._signal_sources = []
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)  # required for dispatch sources
                source = libdispatch.dispatch_source_create(
                    libdispatch.DISPATCH_SOURCE_TYPE_SIGNAL,
                    sig,
                    0,
                    libdispatch.dispatch_get_main_queue(),
                )
                name = signal.Signals(sig).name
                libdispatch.dispatch_source_set_event_handler(
                    source, lambda n=name: self._shutdown(n)
                )
                libdispatch.dispatch_resume(source)
                self._signal_sources.append(source)  # keep refs alive
            _log.debug("Signal handlers installed (SIGTERM, SIGINT)")
        except Exception:
            _log.warning("Could not install signal handlers", exc_info=True)

    # ── shared start / stop ───────────────────────────────────────────────────

    def _do_start(self, meeting_name: str = ""):
        # Blocking calls outside the lock to avoid deadlock
        if not meeting_name:
            try:
                evt = get_current_or_upcoming_event()
                meeting_name = evt["title"] if evt else ""
            except Exception:
                meeting_name = ""
        # Kick the model load now so it's ready before the first live pass.
        self._cancel_model_unload()
        threading.Thread(target=self._preload_model, daemon=True).start()

        with self._rec_lock:
            if self._recorder and self._recorder.is_recording:
                return
            self._recorder = rec_module.Recorder()
            self._started_at = datetime.now()
            self._recorder.start()
            self._set_state("recording")

        self._meeting_name = meeting_name or f"meeting-{self._started_at.strftime('%H%M')}"
        self._live_path = storage.get_transcript_path(self._meeting_name, self._started_at)
        # Write a placeholder so the file exists immediately. Say plainly when
        # live updates are paused, so an empty file isn't mistaken for a
        # failed recording — the audio is still being captured either way.
        if _on_battery() and not get_setting("live_on_battery"):
            note = (
                "[Recording in progress — live updates paused on battery; full transcript on stop]"
            )
        else:
            note = "[Recording in progress — live updates every 60s]"
        storage.save_transcript(
            self._live_path,
            storage.format_transcript([], self._started_at, self._meeting_name) + f"\n\n{note}\n",
        )
        self._open_latest_item.title = f"Open Latest ({self._meeting_name})"
        storage.set_live_session(self._live_path, self._meeting_name, self._started_at)

        # Start live transcription thread
        self._live_thread = threading.Thread(target=self._live_transcribe, daemon=True)
        self._live_thread.start()

        source = "system audio + mic" if self._recorder.system_audio_active else "built-in mic"
        # Ground truth from the actual capture attempt (permission may have
        # just been granted or revoked) — keep the published state current.
        self._publish_app_state(system_audio_active=self._recorder.system_audio_active)
        _log.info(
            "Recording started: meeting=%s device=%s",
            meeting_name or "(unnamed)",
            source,
        )
        label = f" — {meeting_name}" if meeting_name else ""
        _notify("Trnscrb", f"Transcription started{label}", f"via {source}")

    _LIVE_INTERVAL = 60  # seconds between live transcription updates

    def _write_paused_placeholder(self, frames: int) -> None:
        """Show captured duration while live transcription is paused.

        Proves the recording is progressing even though no text is appearing,
        and confirms the audio on disk is safe.
        """
        if not self._live_path or not self._started_at:
            return
        minutes = frames / rec_module.SAMPLE_RATE / 60
        try:
            storage.save_transcript(
                self._live_path,
                storage.format_transcript([], self._started_at, self._meeting_name)
                + f"\n\n[Recording in progress — {minutes:.0f} min captured and saved; "
                "live updates paused on battery, full transcript on stop]\n",
            )
        except Exception:
            _log.debug("Could not update paused placeholder", exc_info=True)

    def _live_transcribe(self):
        """Incrementally transcribe new audio during recording.

        Each pass transcribes only the audio captured since the previous pass
        (constant work per tick, instead of re-transcribing the whole meeting).
        Paused while on battery unless the `live_on_battery` setting is set.
        """
        import time

        transcribed_frames = 0
        segments_acc: list[dict] = []

        time.sleep(self._LIVE_INTERVAL)  # wait before first snapshot
        while self._recorder and self._recorder.is_recording:
            try:
                # Safety net that runs every tick regardless of power state:
                # keeps the WAV on disk valid and playable, so an abrupt end
                # (kill, crash, power loss) costs at most one interval.
                recorder = self._recorder
                frames = recorder.flush_to_disk() if recorder else 0

                if _on_battery() and not get_setting("live_on_battery"):
                    _log.debug("Live transcription paused (on battery)")
                    self._write_paused_placeholder(frames)
                else:
                    recorder = self._recorder
                    result = recorder.snapshot_since(transcribed_frames) if recorder else None
                    if result:
                        snap, end_frame = result
                        try:
                            offset = transcribed_frames / rec_module.SAMPLE_RATE
                            new_segments = transcriber.transcribe(snap)
                            for seg in new_segments:
                                seg["start"] += offset
                                seg["end"] += offset
                            attribution.label_segments(
                                new_segments, recorder.attribution_timeline()
                            )
                            segments_acc.extend(new_segments)
                            transcribed_frames = end_frame
                            text = storage.format_transcript(
                                segments_acc, self._started_at, self._meeting_name
                            )
                            text += "\n\n[Live — recording in progress…]\n"
                            if self._live_path:
                                storage.save_transcript(self._live_path, text)
                            _log.debug(
                                "Live transcription updated (+%d segments, %d total)",
                                len(new_segments),
                                len(segments_acc),
                            )
                        finally:
                            snap.unlink(missing_ok=True)
            except Exception:
                _log.debug("Live transcription update failed", exc_info=True)
            time.sleep(self._LIVE_INTERVAL)

    def _do_stop(self):
        with self._rec_lock:
            if not self._recorder or not self._recorder.is_recording:
                return
            _log.info("Recording stopped, starting transcription")
            started_at = self._started_at or datetime.now()
            recorder = self._recorder
            self._recorder = None
            self._set_state("transcribing")

        self._open_latest_item.title = "Open Latest"
        storage.clear_live_session()

        # Keep the watcher in step: it only fires on_start on a fresh
        # warming → recording transition, so without this a manual Stop leaves
        # it stuck in `recording` and auto-record never triggers again.
        if self._watcher:
            self._watcher.notify_recording_stopped()

        self._process_thread = threading.Thread(
            target=self._process,
            args=(recorder, started_at, self._meeting_name, self._live_path),
            daemon=True,
        )
        self._process_thread.start()

    # ── auto-record callbacks ─────────────────────────────────────────────────

    def _auto_start(self, meeting_name: str):
        # Guard on the recorder itself: the icon state can lag or be stale,
        # and a missed start is worse than a redundant check.
        if self._recorder and self._recorder.is_recording:
            return
        if getattr(self, "_current_state", "idle") == "transcribing":
            return
        self._do_start(meeting_name=meeting_name)

    def _auto_stop(self):
        if self._recorder and self._recorder.is_recording:
            self._do_stop()

    # ── background transcription ──────────────────────────────────────────────

    def _process(
        self,
        recorder: rec_module.Recorder,
        started_at: datetime,
        meeting_name: str = "",
        live_path: Path | None = None,
    ):
        audio_path = None
        transcript_saved = False
        try:
            audio_path = recorder.stop()
            if not audio_path:
                _notify("Trnscrb", "Error", "No audio captured.")
                return

            evt = None
            if not meeting_name:
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

            hf_token = read_hf_token()
            if hf_token and segments:
                try:
                    diar = diarizer.diarize(audio_path, hf_token)
                    segments = diarizer.merge(segments, diar)
                except Exception as e:
                    _log.warning("Diarization skipped: %s", e)
                    _notify("Trnscrb", "Speaker labels skipped", str(e)[:180])

            if segments:
                attribution.label_segments(segments, recorder.attribution_timeline())

            text = storage.format_transcript(segments, started_at, meeting_name)
            path = live_path or storage.get_transcript_path(meeting_name, started_at)
            storage.save_transcript(path, text)
            transcript_saved = True
            _log.info("Transcription complete: %s -> %s", meeting_name, path.name)
            _notify("Trnscrb", f"Saved: {meeting_name}", f"~/meeting-notes/{path.name}")

            # Auto-enrich if enabled
            if get_setting("auto_enrich"):
                try:
                    _log.info("Auto-enriching: %s", meeting_name)
                    calendar_event = evt if evt else None
                    result = enrich_transcript(text, calendar_event=calendar_event)
                    enriched = result["enriched_transcript"]
                    updated = enriched + "\n\n" + "=" * 60 + "\n\n" + result["enrichment"]
                    storage.save_transcript(path, updated)
                    _log.info(
                        "Auto-enrich complete: %s (provider=%s)",
                        meeting_name,
                        result["provider"],
                    )
                    _notify(
                        "Trnscrb",
                        f"Enriched: {meeting_name}",
                        "Summary + action items added",
                    )
                except Exception as e:
                    _log.warning("Auto-enrich failed for %s: %s", meeting_name, e)
                    _notify("Trnscrb", "Enrichment skipped", str(e)[:180])

            # Auto-integrate into notes via Claude Code (after enrich, so the
            # CLI sees the final transcript content)
            if get_setting("auto_integrate"):
                _integrate_notes(path)
        except Exception as e:
            _log.error("Unexpected error in _process: %s", e, exc_info=True)
            _notify("Trnscrb", "Error", str(e)[:180])
        finally:
            if audio_path:
                if transcript_saved:
                    audio_path.unlink(missing_ok=True)
                else:
                    # Never discard the meeting because transcription failed.
                    name = meeting_name or f"meeting-{started_at.strftime('%H%M')}"
                    saved = storage.preserve_audio(audio_path, name, started_at)
                    if saved:
                        _notify(
                            "Trnscrb",
                            "Audio saved for retry",
                            f"Transcription failed — audio kept at {saved.name}",
                        )
            self._restore_idle()

    def _restore_idle(self):
        """Called from background thread when transcription finishes."""
        state = "watching" if (self._watcher and self._watcher.is_watching) else "idle"
        self._set_state(state)
        self._schedule_model_unload()

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
        rec_icon = icon_path(recording=True)
        idle_icon = icon_path(recording=False)
        if state in ("recording", "transcribing"):
            self.icon, self.title = (rec_icon, None) if rec_icon else (None, _EMOJI_RECORDING)
        else:
            self.icon, self.title = (idle_icon, None) if idle_icon else (None, _EMOJI_IDLE)
        self._duration_shown = False

    def _update_duration_title(self, _timer):
        """Show elapsed recording time next to the menu bar icon."""
        if getattr(self, "_current_state", "idle") == "recording" and self._started_at:
            secs = int((datetime.now() - self._started_at).total_seconds())
            if secs >= 3600:
                elapsed = f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
            else:
                elapsed = f"{secs // 60}:{secs % 60:02d}"
            prefix = "" if self.icon else _EMOJI_RECORDING
            self.title = f"{prefix} {elapsed}".strip()
            self._duration_shown = True
        elif self._duration_shown:
            self._set_icon_state(getattr(self, "_current_state", "idle"))


def main():
    import AppKit

    from trnscrb.single_instance import SingleInstance

    # Hold for the whole app lifetime; a second copy (manual start while the
    # launchd one runs, or vice versa) exits cleanly instead of double-recording.
    lock = SingleInstance()
    if not lock.acquire():
        pid = lock.holder_pid()
        msg = f"trnscrb is already running (pid {pid})." if pid else "trnscrb is already running."
        _log.warning("%s Exiting.", msg)
        print(msg)
        return

    app = TrnscrbApp()
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(
        AppKit.NSApplicationActivationPolicyAccessory
    )
    app.run()
