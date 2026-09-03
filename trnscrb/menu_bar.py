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

from trnscrb import (
    action_items,
    analytics,
    attribution,
    diarizer,
    enricher,
    health,
    obsidian,
    storage,
    titles,
    transcriber,
)
from trnscrb import recorder as rec_module
from trnscrb.calendar_integration import get_current_or_upcoming_event
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


def _notes_root() -> Path:
    """Where note integration is allowed to work — the vault, else the notes folder.

    Never the home directory. A subprocess inherits this app's TCC identity, so
    anything it reads is attributed to Trnscrb: an agent searching from $HOME
    walks into ~/Pictures and macOS asks the user whether *Trnscrb* may read
    their photos. Starting inside the notes root keeps that from being a
    question anyone has to answer.
    """
    try:
        from trnscrb import obsidian

        vault = obsidian.vault_path()
        if vault and vault.is_dir():
            return vault
    except Exception:
        _log.debug("Could not resolve the Obsidian vault", exc_info=True)
    return storage.ensure_notes_dir()


def _integrate_notes(transcript_path: Path) -> None:
    """Fire-and-forget: ask Claude Code to fold the transcript into the user's notes.

    Prompt and tool allowlist come from the `integrate_prompt` and
    `integrate_allowed_tools` settings. The subprocess runs as this app as far
    as macOS privacy is concerned, so it is confined to the notes root and
    given the narrowest useful toolset.
    """
    claude = _find_claude_cli()
    if not claude:
        _log.warning("Auto-integrate skipped: claude CLI not found on PATH or common locations")
        _notify("Trnscrb", "Note integration skipped", "Claude CLI not found")
        return
    template = str(get_setting("integrate_prompt") or "")
    allowed = str(get_setting("integrate_allowed_tools") or "")
    notes_root = _notes_root()
    try:
        # Both placeholders are always supplied, so a prompt customised before
        # {notes_dir} existed keeps working.
        prompt = template.format(transcript_path=transcript_path, notes_dir=notes_root)
    except (KeyError, IndexError) as e:
        _log.error("Invalid integrate_prompt template (%s); skipping note integration", e)
        return
    cmd = [claude, "-p", prompt]
    if allowed:
        cmd += ["--allowedTools", allowed]
    _log.info(
        "Note integration via Claude Code started for %s (cwd %s)",
        transcript_path.name,
        notes_root,
    )
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(notes_root),
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
        self._summary_item = rumps.MenuItem(
            "Meeting summary: Off", callback=self.toggle_auto_summary
        )
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
        self._bookmark_item = rumps.MenuItem("Bookmark This Moment", callback=self.add_bookmark)
        self._voices_item = rumps.MenuItem("Label Voices…", callback=self.label_voices)
        # Standing state, not a notification: a component that has been broken
        # all week should be readable from the menu at any moment, rather than
        # having announced itself once, days ago, in a banner that vanished.
        self._health_item = rumps.MenuItem("Diagnostics…", callback=self.show_health)

        self.menu = [
            self._start_item,
            self._stop_item,
            None,
            self._auto_item,
            self._summary_item,
            self._integrate_item,
            self._bookmark_item,
            self._voices_item,
            self._open_latest_item,
            self._health_item,
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
        self._calendar_event: dict | None = None  # captured at recording start

        self._set_state("idle")
        self._install_signal_handlers()

        if get_setting("auto_record"):
            self._start_watcher()
            self._auto_item.title = "Auto-transcribe: On ✓"
        if get_setting("auto_enrich"):
            self._summary_item.title = "Meeting summary: On ✓"
        if get_setting("auto_integrate"):
            self._integrate_item.title = "Auto-integrate notes: On ✓"
        self._refresh_enrich_settings_menu()
        self._update_health_item()

        # Models load lazily when a recording starts (see _do_start) and are
        # released again after a long idle period to free ~1 GB of memory.
        self._unload_timer: threading.Timer | None = None

        # Elapsed recording time next to the menu bar icon (🔴 12:34).
        self._duration_shown = False
        self._duration_timer = rumps.Timer(self._update_duration_title, 15)
        self._duration_timer.start()

        self._start_backlog_retry()

        # An upgrade deletes the tree this process runs from; watch for it so
        # the app restarts itself instead of quietly losing lazily-imported
        # features (see trnscrb/rollout.py).
        self._stale_notified = False
        self._rollout_timer = rumps.Timer(self._check_rollout, self._ROLLOUT_CHECK_SECS)
        self._rollout_timer.start()

        # Nothing else ever records that startup succeeded, so a recovered
        # restart loop would keep being reported long after it was fixed.
        self._healthy_timer = threading.Timer(health.HEALTHY_UPTIME_SECS, self._mark_startup_ok)
        self._healthy_timer.daemon = True
        self._healthy_timer.start()

    def _start_backlog_retry(self):
        """Transcribe any preserved audio left un-transcribed by an earlier run.

        A backend that was broken during a meeting (missing dependency,
        undownloaded model) otherwise leaves the WAV sitting there forever.
        Runs off the main thread so the menu bar appears immediately.
        """
        from trnscrb import backlog

        def _run():
            try:
                written = backlog.process_pending()
            except Exception:
                _log.warning("Backlog retry pass failed", exc_info=True)
                return
            if written:
                _notify(
                    "Trnscrb",
                    f"Transcribed {len(written)} earlier recording(s)",
                    written[-1].name,
                )

        threading.Thread(target=_run, daemon=True).start()

    _MODEL_IDLE_UNLOAD_SECS = 30 * 60
    _ROLLOUT_CHECK_SECS = 60

    def _check_rollout(self, _timer=None) -> None:
        """Restart when an upgrade has replaced the tree we run from.

        Deferred while anything is in flight: a restart mid-meeting would end
        the recording, and the whole point is that an upgrade costs nothing.
        The stale process keeps working for everything already imported, so
        waiting is safe — it is only the *next* import that would fail.
        """
        from trnscrb import rollout

        if not rollout.is_stale():
            return
        if not self._stale_notified:
            self._stale_notified = True
            _log.warning(
                "Running from %s, which no longer exists — trnscrb was upgraded "
                "underneath this process; restarting when idle",
                rollout.install_root(),
            )
        busy = (self._recorder and self._recorder.is_recording) or (
            self._process_thread and self._process_thread.is_alive()
        )
        if busy:
            _log.debug("Upgrade restart deferred: recording or transcription in flight")
            return

        _notify("Trnscrb", "Updating", "Restarting to finish an update…")
        self._rollout_timer.stop()
        if rollout.restart():
            self._shutdown("Upgrade")
            rumps.quit_application()
        else:
            _notify("Trnscrb", "Update needs a restart", "Quit and start Trnscrb again.")

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
            from trnscrb import rollout

            storage.write_app_state(
                version=trnscrb.__version__,
                system_audio_permission=rec_module.Recorder.system_audio_available(),
                # Lets `trnscrb status` see that a running app was upgraded out
                # from under itself, which is invisible from the files on disk.
                install_root=str(rollout.install_root()),
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

        # Diarization is only needed at stop, but it is imported lazily: an
        # upgrade during the meeting would delete the tree torch comes from
        # and the recording would finish without speaker labels. Loading now
        # closes that window (see diarizer.preload).
        try:
            if diarizer.preload(read_hf_token() or ""):
                _log.info("Diarization pipeline preloaded for this recording")
        except Exception as e:
            _log.debug("Diarization preload skipped: %s", e)

    # ── watcher ───────────────────────────────────────────────────────────────

    def _start_watcher(self):
        self._watcher = MicWatcher(
            on_start=self._auto_start,
            on_stop=self._auto_stop,
            speech_ratio=lambda secs: attribution.live_speech_ratio(self._recorder, secs),
        )
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

    def add_bookmark(self, _):
        """Mark the current moment so it can be found in the transcript."""
        offset = storage.add_bookmark()
        if offset is None:
            _notify("Trnscrb", "No recording in progress", "Start a recording to bookmark it")
            return
        stamp = f"{int(offset) // 60:02d}:{int(offset) % 60:02d}"
        _notify("Trnscrb", f"Bookmarked at {stamp}", self._meeting_name or "")

    def toggle_auto_summary(self, sender):
        if get_setting("auto_enrich"):
            put_setting("auto_enrich", False)
            sender.title = "Meeting summary: Off"
            _notify("Trnscrb", "Meeting summary off", "")
        else:
            put_setting("auto_enrich", True)
            sender.title = "Meeting summary: On ✓"
            if _find_claude_cli():
                msg = "Summary + action items added to each transcript"
            else:
                msg = "Needs the Claude CLI or a configured LLM provider"
            _notify("Trnscrb", "Meeting summary on", msg)

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

    def _mark_startup_ok(self):
        """This launch has been up long enough to call it a good one."""
        try:
            health.note_healthy_uptime()
            self._update_health_item()
        except Exception:
            _log.debug("Could not record healthy uptime", exc_info=True)

    def _update_health_item(self):
        """Put any standing failure in the menu title itself.

        A menu entry that always reads "Diagnostics…" is one nobody opens.
        The whole point is that a component which quietly stopped working is
        visible without being looked for.
        """
        try:
            broken = health.unhealthy()
            if not broken:
                self._health_item.title = "Diagnostics: all clear"
                return
            name, entry = broken[0]
            label = health.LABELS.get(name, name)
            failures = int(entry.get("failures", 1))
            suffix = f" ({failures}×)" if failures > 1 else ""
            more = f" +{len(broken) - 1} more" if len(broken) > 1 else ""
            self._health_item.title = f"⚠ {label} failing{suffix}{more}"
        except Exception:
            _log.debug("Could not refresh the diagnostics item", exc_info=True)

    def show_health(self, _):
        """What each component did the last time it ran, and for how long."""
        broken = health.unhealthy()
        lines = [
            f"{health.LABELS.get(name, name)}: {health.describe(name)}"
            for name in sorted(health.LABELS)
            if health.get(name)
        ]
        if not lines:
            lines = ["Nothing has run yet."]
        if broken:
            lines.append("")
            lines.append("Run `trnscrb doctor` in a terminal to test the stack end to end.")
        rumps.alert("Trnscrb diagnostics", "\n".join(lines))

    def label_voices(self, _):
        """Play each unnamed voice and ask who it is.

        Identification is a listening task, so the clip plays while the prompt
        is up rather than before it — the answer arrives while the voice is
        still in the user's ear.
        """
        from trnscrb import voiceprints

        try:
            unnamed = [r for r in voiceprints.summary() if not r["name"]]
        except Exception:
            _log.warning("Could not read voices", exc_info=True)
            return
        if not unnamed:
            rumps.alert("Voices", "Every voice trnscrb has learned already has a name.")
            return

        named = 0
        playing: subprocess.Popen | None = None
        for row in unnamed:
            clip = voiceprints.sample_path(row["id"])
            if clip.is_file():
                # Stop the last one first. Answering Name or Skip quickly
                # would otherwise start this clip over the previous one, and
                # two voices at once is worse than either alone.
                if playing is not None and playing.poll() is None:
                    try:
                        playing.terminate()
                    except OSError:
                        pass
                try:
                    playing = subprocess.Popen(
                        ["afplay", str(clip)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    _log.debug("Could not play %s", clip, exc_info=True)

            meetings = ", ".join(dict.fromkeys(row["meetings"])) or "unknown meeting"
            detail = (
                f"{row['observations']} meeting(s), {row['speech_secs'] / 60:.0f} min\n{meetings}"
            )
            if not clip.is_file():
                detail += "\n\n(No audio clip — recorded before clips were kept.)"
            result = rumps.Window(
                message=detail,
                title=f"Who is {row['id']}?",
                default_text="",
                ok="Name",
                cancel="Skip",
                dimensions=(320, 24),
            ).run()
            if not result.clicked:
                continue
            name = result.text.strip()
            if name and voiceprints.name_voice(row["id"], name):
                named += 1

        # Don't leave the last clip playing into whatever the user does next.
        if playing is not None and playing.poll() is None:
            try:
                playing.terminate()
            except OSError:
                pass

        if named:
            _notify("Trnscrb", f"Named {named} voice(s)", "They apply to past and future meetings.")

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

            # SIGUSR1 toggles recording. `trnscrb toggle` sends it, so a hotkey
            # bound via Shortcuts/Raycast can start/stop with no permission.
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            toggle_source = libdispatch.dispatch_source_create(
                libdispatch.DISPATCH_SOURCE_TYPE_SIGNAL,
                signal.SIGUSR1,
                0,
                libdispatch.dispatch_get_main_queue(),
            )
            libdispatch.dispatch_source_set_event_handler(toggle_source, self._on_toggle_signal)
            libdispatch.dispatch_resume(toggle_source)
            self._signal_sources.append(toggle_source)
            _log.debug("Signal handlers installed (SIGTERM, SIGINT, SIGUSR1)")
        except Exception:
            _log.warning("Could not install signal handlers", exc_info=True)

    def _on_toggle_signal(self) -> None:
        """Start or stop recording — invoked on the main queue by SIGUSR1."""
        try:
            if self._recorder and self._recorder.is_recording:
                _log.info("SIGUSR1: stopping recording")
                self.stop_recording(None)
            else:
                _log.info("SIGUSR1: starting recording")
                self.start_recording(None)
        except Exception:
            _log.warning("Toggle signal handler failed", exc_info=True)

    # ── shared start / stop ───────────────────────────────────────────────────

    def _do_start(self, meeting_name: str = ""):
        # Blocking calls outside the lock to avoid deadlock. Always look the
        # event up: attendee names are used later to label the other speaker,
        # and by the end of the meeting the calendar no longer returns it.
        try:
            self._calendar_event = get_current_or_upcoming_event()
        except Exception:
            self._calendar_event = None
        if not meeting_name:
            meeting_name = (self._calendar_event or {}).get("title") or ""
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
                                segments_acc,
                                self._started_at,
                                self._meeting_name,
                                bookmarks=storage.read_bookmarks(),
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
        # Read before clearing: clear_live_session drops the bookmarks too.
        bookmarks = storage.read_bookmarks()
        storage.clear_live_session()

        # Keep the watcher in step: it only fires on_start on a fresh
        # warming → recording transition, so without this a manual Stop leaves
        # it stuck in `recording` and auto-record never triggers again.
        if self._watcher:
            self._watcher.notify_recording_stopped()

        self._process_thread = threading.Thread(
            target=self._process,
            args=(recorder, started_at, self._meeting_name, self._live_path, bookmarks),
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
        bookmarks: list[dict] | None = None,
    ):
        audio_path = None
        transcript_saved = False
        # Speaker labels and voiceprints can only ever be derived from the
        # audio, and the audio is deleted the moment the transcript saves.
        # When diarization fails, that deletion turns a recoverable error into
        # a permanent one, so the recording is kept instead.
        speakers_lost = False
        try:
            # Read before stopping — stop() clears the capture state.
            system_audio_used = recorder.system_audio_active
            audio_path = recorder.stop()
            if not audio_path:
                _notify("Trnscrb", "Error", "No audio captured.")
                return

            # Captured when the meeting started; by now the event may be over
            # and no longer returned by the calendar.
            evt = self._calendar_event
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
                    diar, embeddings = diarizer.diarize_with_embeddings(audio_path, hf_token)
                    segments = diarizer.merge(segments, diar)
                    health.record_ok(
                        health.DIARIZATION,
                        f"{len({t['speaker'] for t in diar})} speaker(s)",
                        meeting_name,
                    )
                    # system_audio_used was read before stop() cleared it.
                    self._learn_voices(
                        diar,
                        embeddings,
                        recorder,
                        system_audio_used,
                        meeting_name,
                        audio_path,
                        event=evt,
                    )
                except Exception as e:
                    # The transcript still saves, so nothing about this meeting
                    # looks wrong — which is exactly why it gets written down.
                    entry = health.record_failure(health.DIARIZATION, e, meeting_name)
                    speakers_lost = True
                    if health.should_notify(entry):
                        _notify(
                            "Trnscrb",
                            f"Speaker labels failing ({entry.get('failures', 1)}×)",
                            f"{str(e)[:120]} — run `trnscrb doctor`",
                        )

            if segments:
                attribution.label_segments(segments, recorder.attribution_timeline())
                # A 1:1 can be named from the calendar; larger meetings stay
                # generic rather than risk attaching the wrong name.
                attribution.name_from_calendar(segments, evt)

            capture = analytics.capture_health(
                segments,
                recorded_secs=(datetime.now() - started_at).total_seconds(),
                system_audio=system_audio_used,
            )
            if capture.get("mostly_silent"):
                _log.warning(
                    "%s looks mostly silent (%.0f%% speech) — likely an idle tab",
                    meeting_name,
                    capture["speech_ratio"] * 100,
                )

            text = storage.format_transcript(
                segments, started_at, meeting_name, bookmarks=bookmarks, health=capture
            )
            path = live_path or storage.get_transcript_path(meeting_name, started_at)
            storage.save_transcript(path, text)
            transcript_saved = True
            _log.info("Transcription complete: %s -> %s", meeting_name, path.name)
            _notify("Trnscrb", f"Saved: {meeting_name}", f"~/meeting-notes/{path.name}")

            # Auto-summary: prepend a summary + action items to the top of the
            # transcript. Best-effort — a missing LLM just means no summary.
            summary_block = None
            enrichment = None
            # Snapshot the user's open action items so the model can say which
            # this meeting resolved (indices map back to this list).
            track_items = bool(get_setting("track_action_items"))
            open_snapshot = action_items.open_items() if track_items else []
            if get_setting("auto_enrich") and segments:
                _log.info("Auto-summarizing: %s", meeting_name)
                result = enricher.summarize_for_auto(
                    text,
                    calendar_event=evt or None,
                    open_items=[i["text"] for i in open_snapshot] or None,
                )
                if result:
                    enrichment = result["enrichment"]
                    summary_block = enricher.summary_block(enrichment) or None
                    if summary_block:
                        _log.info(
                            "Auto-summary added: %s (provider=%s)", meeting_name, result["provider"]
                        )

            # Give generically-named (non-calendar) meetings a content title,
            # from the summary model if it ran, else a keyword heuristic.
            if titles.is_generic(meeting_name):
                new_title = titles.from_enrichment(enrichment) or titles.local(segments)
                if new_title:
                    meeting_name = new_title
                    _log.info("Derived meeting title: %s", new_title)

            # Rewrite once if the summary or the new title changed anything,
            # renaming the file when the title yields a new path.
            final_path = storage.get_transcript_path(meeting_name, started_at)
            if summary_block or final_path != path:
                text = storage.format_transcript(
                    segments,
                    started_at,
                    meeting_name,
                    bookmarks=bookmarks,
                    health=health,
                    ai_summary=summary_block,
                )
                storage.save_transcript(final_path, text)
                if final_path != path:
                    path.unlink(missing_ok=True)  # drop the provisional file
                    path = final_path
                if summary_block:
                    _notify(
                        "Trnscrb",
                        f"Summary added: {meeting_name}",
                        "Summary + action items at the top",
                    )

            # Mirror the transcript into Obsidian and track the user's own action
            # items (best-effort — never fail the meeting over this).
            if track_items:
                try:
                    spoken = segments[-1]["end"] if segments else 0
                    note = obsidian.mirror_transcript(
                        meeting_name,
                        started_at,
                        text,
                        duration=f"{int(spoken) // 60:02d}:{int(spoken) % 60:02d}",
                    )
                    if enrichment:
                        stats = action_items.record_meeting(
                            enrichment,
                            open_snapshot,
                            meeting_id=path.stem,
                            meeting_title=meeting_name,
                            note_name=note,
                            when=started_at.strftime("%Y-%m-%d"),
                        )
                        if stats["added"] or stats["resolved"]:
                            _notify(
                                "Trnscrb",
                                f"Action items: +{stats['added']}, {stats['resolved']} done",
                                "Updated in Obsidian",
                            )
                except Exception:
                    _log.warning("Action-item / Obsidian update failed", exc_info=True)

            # Auto-integrate into notes via Claude Code (after enrich, so the
            # CLI sees the final transcript content)
            if get_setting("auto_integrate"):
                _integrate_notes(path)
        except Exception as e:
            _log.error("Unexpected error in _process: %s", e, exc_info=True)
            _notify("Trnscrb", "Error", str(e)[:180])
        finally:
            if audio_path:
                if transcript_saved and not speakers_lost:
                    audio_path.unlink(missing_ok=True)
                else:
                    # Never discard the meeting because a stage failed: the
                    # transcript can be redone from the audio, and nothing can
                    # redo the audio.
                    reason = "Speaker labels failed" if transcript_saved else "Transcription failed"
                    name = meeting_name or f"meeting-{started_at.strftime('%H%M')}"
                    saved = storage.preserve_audio(audio_path, name, started_at, reason)
                    if saved:
                        _notify(
                            "Trnscrb",
                            "Audio kept for retry",
                            f"{reason} — redo with `trnscrb transcribe {saved.name}`",
                        )
            self._update_health_item()
            self._restore_idle()

    def _learn_voices(
        self,
        diar: list[dict],
        embeddings: dict,
        recorder,
        system_audio: bool,
        meeting: str,
        audio_path=None,
        event: dict | None = None,
    ) -> None:
        """Carry this meeting's speakers into the persistent voice identities.

        The user's own voice comes from the mic/system split, which only holds
        because the system stream was genuinely captured: a conferencing app
        never plays your own mic back, so anything absent from that stream is
        you. Without system audio the premise collapses — on laptop speakers
        the other participant bleeds into the mic and the whole meeting looks
        mic-only — so self-enrolment is skipped rather than risk training "Me"
        on a colleague.

        Everyone else is clustered only when `cluster_voices` is on: those are
        fingerprints of people who did not consent to being enrolled.

        Never fails the transcription: an identity is a nice-to-have, the
        transcript is not.
        """
        if not embeddings:
            return
        learn_self = bool(get_setting("learn_my_voice")) and system_audio
        cluster_others = bool(get_setting("cluster_voices"))
        if not learn_self and not cluster_others:
            return
        try:
            from trnscrb import voiceprints

            self_label = None
            if learn_self:
                self_label, _ = attribution.self_speaker(diar, recorder.attribution_timeline())
                if self_label is None or self_label not in embeddings:
                    self_label = None
                    _log.debug("No unambiguous self speaker; skipping self enrolment")

            learned = voiceprints.enrol(
                embeddings,
                diar,
                model=diarizer.pipeline_id(),
                space=diarizer.embedding_space(),
                meeting=meeting,
                self_label=self_label,
                cluster_others=cluster_others,
                audio_path=audio_path,
            )
            # A 1:1's counterpart, or the one invitee not yet known in a
            # group, gets their name from the invite.
            attribution.name_voice_from_calendar(learned, event, speakers=len(embeddings))
            ok, detail = voiceprints.enrolment_health(learned, diar)
            (health.record_ok if ok else health.record_failure)(
                health.VOICE_ENROLMENT, detail, meeting
            )
        except Exception as e:
            _log.debug("Voice identity update failed", exc_info=True)
            health.record_failure(health.VOICE_ENROLMENT, e, meeting)

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


def _startup_is_sane() -> bool:
    """False when the app is being restarted in a loop rather than started.

    Whatever kills the app during startup — a bad icon file, a broken
    dependency after an upgrade, a native crash in a model load — launchd's
    answer is the same: start it again in ten seconds, forever. The process
    that has to break that cycle is this one, because it is the only party
    that can tell "started" from "started for the fifth time in two minutes".

    Stopping is the useful behaviour even though the app then isn't running:
    a loop is not a degraded app, it is a machine burning CPU and a user who
    finds out days later. What it leaves behind — the health record, the log
    line, the notification — is how they find out today.
    """
    starts = health.note_start()
    if starts < health.CRASH_LOOP_STARTS:
        return True

    detail = (
        f"{starts} starts in under {health.CRASH_LOOP_WINDOW_SECS // 60} minutes — "
        "something is killing the app during startup"
    )
    health.record_failure(health.APP_START, detail)
    # Let the next launch run: the guard has done its job, and the user
    # restarting by hand should get a real attempt, not this message again.
    health.clear_starts()
    _log.error(
        "Restart loop detected (%s). Not starting again — see `trnscrb doctor`. "
        "Check ~/Library/Logs/DiagnosticReports for a crash report, and "
        "`launchctl print gui/$(id -u)/io.trnscrb.app` for the exit status "
        "(128+signal).",
        detail,
    )
    print(f"trnscrb: restart loop detected ({detail}). Not starting again.")
    _notify("Trnscrb", "Stopped: restart loop", f"{detail}. Run `trnscrb doctor`.")
    return False


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

    if not _startup_is_sane():
        return  # exit 0 — the one status KeepAlive-on-failure will not restart

    app = TrnscrbApp()
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(
        AppKit.NSApplicationActivationPolicyAccessory
    )
    app.run()
