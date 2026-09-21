"""Menu-bar app dictation: no-save plumbing, HUD wiring, auto-stop actuation."""

import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from trnscrb import dictation as d
from trnscrb import menu_bar


def _app():
    """A bare app instance (no __init__) with just the dictation state."""
    app = menu_bar.TrnscrbApp.__new__(menu_bar.TrnscrbApp)
    app._current_state = "idle"
    app._recorder = None
    app._dict_preset = None
    app._dict_started_at = None
    app._dict_save_note = True
    app._dict_live_text = ""
    app._dict_live_stop = threading.Event()
    app._dict_stop_requested = False
    app._dict_live_paused = False
    app._hud_window = None
    app._hud_label = None
    app._hud_shown_text = ""
    app._restore_idle = lambda: None
    app._set_state = lambda state: app.__setattr__("_current_state", state)
    return app


def _recorder(wav: Path):
    rec = mock.Mock()
    rec.stop.return_value = wav
    return rec


class ProcessDictationTest(unittest.TestCase):
    """The background tail: save_note plumbing and the user-facing outcome."""

    def _wav(self):
        fd, name = tempfile.mkstemp(suffix=".wav", prefix="trnscrb-dict-app-")
        path = Path(name)
        path.write_bytes(b"RIFF")
        return path

    def test_no_save_finishes_without_note(self):
        app = _app()
        wav = self._wav()
        with (
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": None,
                    "plain": "just paste this",
                    "on_clipboard": True,
                    "pasted": True,
                    "paste_detail": "pasted into active text field",
                },
            ) as finish,
            mock.patch.object(menu_bar, "_notify") as notify,
        ):
            app._process_dictation(_recorder(wav), datetime.now(), "message", save_note=False)
        self.assertEqual(finish.call_args.kwargs["save_note"], False)
        args = notify.call_args.args
        self.assertIn("no note saved", args[1])
        self.assertIn("pasted into active text field", args[2])

    def test_save_reports_saved_note(self):
        app = _app()
        wav = self._wav()
        with (
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": "/tmp/notes/2026-09-21_16-02_message-1602.txt",
                    "plain": "save this one",
                    "on_clipboard": True,
                    "pasted": False,
                },
            ) as finish,
            mock.patch.object(menu_bar, "_notify") as notify,
        ):
            app._process_dictation(_recorder(wav), datetime.now(), "message", save_note=True)
        self.assertEqual(finish.call_args.kwargs["save_note"], True)
        args = notify.call_args.args
        self.assertIn("saved", args[1])
        self.assertIn("message-1602.txt", args[2])

    def test_no_speech_reports_nothing_said(self):
        app = _app()
        wav = self._wav()
        with (
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": None,
                    "plain": "",
                    "on_clipboard": False,
                },
            ),
            mock.patch.object(menu_bar, "_notify") as notify,
        ):
            app._process_dictation(_recorder(wav), datetime.now(), "message", save_note=False)
        args = notify.call_args.args
        self.assertIn("Nothing was said.", args[2])


class StartDictationTest(unittest.TestCase):
    """_start_dictation: no-save state, HUD, and the two background loops."""

    def _call(self, preset: str, save_note: bool):
        app = _app()
        with (
            mock.patch.object(menu_bar.rec_module, "Recorder") as recorder_cls,
            mock.patch.object(menu_bar, "_notify"),
            mock.patch.object(d, "live_transcribe"),
            mock.patch.object(d, "silence_watchdog"),
        ):
            app._show_dictation_hud = mock.Mock()
            app._start_dictation(preset, save_note)
        return app, recorder_cls

    def test_no_save_start_stores_flag_and_starts_extras(self):
        app, recorder_cls = self._call("message", False)
        self.assertIs(app._dict_save_note, False)
        self.assertEqual(app._dict_preset, "message")
        self.assertEqual(app._current_state, "dictating")
        app._show_dictation_hud.assert_called_once()
        recorder_cls.assert_called_once_with(system_audio=False)
        recorder_cls.return_value.start.assert_called_once()

    def test_default_start_stores_save_flag(self):
        app, recorder_cls = self._call("message", True)
        self.assertIs(app._dict_save_note, True)
        self.assertEqual(app._dict_preset, "message")

    def test_live_text_accumulates(self):
        app = _app()
        app._on_live_text("I hope testing")
        app._on_live_text("if the microphone is working.")
        self.assertEqual(app._dict_live_text, "I hope testing if the microphone is working.")


class HudTest(unittest.TestCase):
    """The HUD timer actuates UI and auto-stop on the main thread."""

    def test_auto_stop_flag_triggers_stop(self):
        app = _app()
        app._current_state = "dictating"
        app._dict_stop_requested = True
        app.stop_dictation = mock.Mock()
        app._update_hud(None)
        app.stop_dictation.assert_called_once_with(None)

    def test_live_text_reaches_the_label(self):
        app = _app()
        app._current_state = "dictating"
        app._hud_window = mock.Mock()
        app._hud_label = mock.Mock()
        app._dict_live_text = "I hope testing"
        app._update_hud(None)
        app._hud_label.setStringValue_.assert_called_once_with("I hope testing")

    def test_hud_hidden_when_not_dictating(self):
        app = _app()
        app._current_state = "transcribing"
        app._hud_window = mock.Mock()
        app._update_hud(None)
        app._hud_window.orderOut_.assert_called_once()

    def test_hud_creation_failure_is_guarded(self):
        """No AppKit (headless) must never break a dictation."""
        app = _app()
        with mock.patch.dict(sys.modules, {"AppKit": None}):
            app._show_dictation_hud()
        self.assertIsNone(app._hud_window)

    def test_hide_is_a_noop_without_window(self):
        app = _app()
        app._hide_dictation_hud()  # must not raise
        self.assertEqual(app._hud_shown_text, "")


class DictationSignalTest(unittest.TestCase):
    """The SIGUSR2 request channel carries the no-save flag to the app."""

    def _request_file(self, payload: dict) -> Path:
        fd, name = tempfile.mkstemp(prefix="trnscrb-dict-req-")
        path = Path(name)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_no_save_request_starts_no_save_dictation(self):
        app = _app()
        request_file = self._request_file({"preset": "message", "save_note": False})
        with (
            mock.patch.object(d, "_START_REQUEST_FILE", request_file),
            mock.patch.object(app, "_start_dictation") as start,
            mock.patch.object(app, "_current_state", "idle"),
        ):
            app._on_dictation_signal()
        start.assert_called_once_with("message", False)

    def test_legacy_request_defaults_to_save(self):
        app = _app()
        request_file = self._request_file({"preset": "brain-dump"})
        with (
            mock.patch.object(d, "_START_REQUEST_FILE", request_file),
            mock.patch.object(app, "_start_dictation") as start,
        ):
            app._on_dictation_signal()
        start.assert_called_once_with("brain-dump", True)

    def test_signal_toggles_off_running_dictation(self):
        app = _app()
        app._dict_preset = "message"
        app.stop_dictation = mock.Mock()
        request_file = self._request_file({"preset": "message", "save_note": False})
        with mock.patch.object(d, "_START_REQUEST_FILE", request_file):
            app._on_dictation_signal()
        app.stop_dictation.assert_called_once_with(None)
