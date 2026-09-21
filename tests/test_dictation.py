"""Tests for Phase 1 dictation: raw formatting, the engine, CLI commands,
MCP tools, and the LLM draft pass."""

import json
import os
import signal
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from trnscrb import cli, enricher, mcp_server, storage
from trnscrb import dictation as d
from trnscrb import glossary as g


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "speaker": "Me", "text": text}


class _FakeRecorder:
    """Stands in for Recorder: captures whether start/stop were called."""

    def __init__(self, audio_path=None):
        self.started = False
        self.audio_path = audio_path

    @property
    def is_recording(self):
        return self.started

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        return self.audio_path


class RawFormattingTest(unittest.TestCase):
    """format_transcript keeps dictation verbatim, meetings readable."""

    def test_message_keeps_fillers_verbatim(self):
        segs = [
            _seg(0.0, 2.5, "um so I need to, like, check the billing"),
            _seg(2.5, 4.0, "Can we I mean can we ship today"),
        ]
        out = storage.format_transcript(
            segs, datetime(2026, 9, 20, 9, 41), "message-0941", kind="message"
        )
        self.assertIn("Message", out.splitlines()[0])
        self.assertIn("um so I need to, like, check the billing", out)
        self.assertIn("Can we I mean can we ship today", out)
        self.assertNotIn("Meeting:", out)

    def test_brain_dump_header_and_verbatim(self):
        segs = [
            _seg(0.0, 1.0, "Header idea one"),
            _seg(1.0, 2.0, "I I think we should refactor"),
        ]
        out = storage.format_transcript(
            segs, datetime(2026, 9, 20, 10, 5), "brain-dump-1005", kind="brain-dump"
        )
        self.assertIn("Brain dump", out.splitlines()[0])
        # The raw pass does not collapse the stutter.
        self.assertIn("I I think we should refactor", out)

    def test_meeting_default_uses_readable_text(self):
        segs = [_seg(0.0, 2.0, "um let me check the billing")]
        meeting = storage.format_transcript(segs, datetime(2026, 9, 20, 9, 0), "standup")
        message = storage.format_transcript(
            segs, datetime(2026, 9, 20, 9, 0), "message-0900", kind="message"
        )
        self.assertIn("Meeting: standup", meeting)
        self.assertIn("let me check the billing", meeting)  # filler stripped
        self.assertNotIn("um let me", meeting)
        self.assertIn("um let me check the billing", message)  # kept verbatim

    def test_dictation_body_is_one_flowing_passage(self):
        segs = [
            _seg(0.0, 1.0, "I"),
            _seg(1.0, 4.0, "hope testing if the microphone is working."),
        ]
        out = storage.format_transcript(
            segs, datetime(2026, 9, 21, 16, 2), "message-1602", kind="message"
        )
        body = out.split("=" * 60 + "\n\n", 1)[1]
        self.assertEqual(body, "I hope testing if the microphone is working.")


class DictationEngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trnscrb-dictation-test-"))
        self.pid_file = self.tmp / "dictation.pid"
        self.result_file = self.tmp / "dictation_result.json"
        patchers = [
            mock.patch.object(d, "_PID_FILE", self.pid_file),
            mock.patch.object(d, "_RESULT_FILE", self.result_file),
            # Keep finish() tests from writing into a real Obsidian vault.
            mock.patch.object(d, "_mirror_note", return_value=None),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_preset_label_and_is_preset(self):
        self.assertEqual(d.preset_label("message"), "Message")
        self.assertEqual(d.preset_label("brain-dump"), "Brain dump")
        self.assertTrue(d.is_preset("message"))
        self.assertFalse(d.is_preset("meeting"))

    def test_plain_text_is_verbatim(self):
        segs = [_seg(0, 1, "um first"), _seg(1, 2, "then, like, second"), _seg(2, 3, "")]
        # One flowing passage: the transcriber's pause-splits must not break
        # the dictation into mid-sentence line breaks.
        self.assertEqual(d.plain_text(segs), "um first then, like, second")

    def test_note_body_extracts_after_separator(self):
        note = (
            "Message\nDate:    2026-09-20 09:41\nDuration: 00:07\n\n"
            + ("=" * 60)
            + "\n\nSpoken body here"
        )
        self.assertEqual(d.note_body(note), "Spoken body here")

    @mock.patch("subprocess.run")
    def test_copy_to_clipboard(self, run):
        run.return_value = mock.Mock(returncode=0)
        self.assertTrue(d.copy_to_clipboard("hello world"))
        run.assert_called_once_with(["pbcopy"], input=b"hello world")
        # Empty text never touches pbcopy.
        run.reset_mock()
        self.assertFalse(d.copy_to_clipboard(""))
        run.assert_not_called()

    @mock.patch("subprocess.run", side_effect=OSError("no pbcopy"))
    def test_copy_to_clipboard_swallows_errors(self, run):
        self.assertFalse(d.copy_to_clipboard("text"))

    def test_save_note_writes_verbatim_message(self):
        segs = [_seg(0.0, 1.5, "grab milk um and eggs")]
        path, text = d.save_note("message", datetime(2026, 9, 20, 9, 41), segs)
        self.assertIsNotNone(path)
        self.assertTrue(path.name.endswith("message-0941.txt"))
        content = path.read_text(encoding="utf-8")
        self.assertIn("um and eggs", content)
        self.assertTrue(content.startswith("Message"))
        # The shell prints a stray "%" when a file lacks a final newline.
        self.assertTrue(content.endswith("\n"))

    def test_start_request_roundtrip_honors_no_save(self):
        with mock.patch.object(d, "_START_REQUEST_FILE", self.tmp / "dictation_request.json"):
            d.write_start_request("message", save_note=False)
            self.assertEqual(d.read_start_request(), {"preset": "message", "save_note": False})
            d.write_start_request("brain-dump")
            self.assertEqual(d.read_start_request(), {"preset": "brain-dump", "save_note": True})

    def test_start_request_legacy_payload_defaults_to_save(self):
        request_file = self.tmp / "dictation_request.json"
        request_file.write_text(json.dumps({"preset": "brain-dump"}), encoding="utf-8")
        with mock.patch.object(d, "_START_REQUEST_FILE", request_file):
            self.assertEqual(d.read_start_request(), {"preset": "brain-dump", "save_note": True})

    def test_save_note_empty_segments_saves_nothing(self):
        path, text = d.save_note("message", datetime(2026, 9, 20, 9, 41), [])
        self.assertIsNone(path)
        self.assertEqual(text, "")

    def test_finish_message_transcribes_saves_and_copies(self):
        wav = self.tmp / "audio.wav"
        wav.write_bytes(b"RIFF")
        segs = [_seg(0.0, 3.0, "remember to water the plants")]
        with (
            mock.patch.object(d.transcriber, "transcribe", return_value=segs),
            mock.patch.object(d, "copy_to_clipboard", return_value=True) as clip,
        ):
            result = d.finish("message", datetime(2026, 9, 20, 9, 41), wav)
        self.assertEqual(result["plain"], "remember to water the plants")
        self.assertTrue(result["on_clipboard"])
        self.assertTrue(result["path"].endswith("message-0941.txt"))
        clip.assert_called_once_with("remember to water the plants")
        self.assertFalse(wav.exists(), "audio cleaned up after finishing")

    def test_finish_message_without_audio_copies_nothing(self):
        wav = self.tmp / "silence.wav"
        wav.write_bytes(b"RIFF")
        with (
            mock.patch.object(d.transcriber, "transcribe", return_value=[]),
            mock.patch.object(d, "copy_to_clipboard") as clip,
        ):
            result = d.finish("message", datetime(2026, 9, 20, 9, 41), wav)
        self.assertIsNone(result["path"])
        self.assertEqual(result["plain"], "")
        clip.assert_not_called()
        self.assertFalse(wav.exists())

    def test_finish_brain_dump_does_not_copy(self):
        wav = self.tmp / "audio.wav"
        wav.write_bytes(b"RIFF")
        segs = [_seg(0.0, 2.0, "draft the proposal outline")]
        with (
            mock.patch.object(d.transcriber, "transcribe", return_value=segs),
            mock.patch.object(d, "copy_to_clipboard") as clip,
        ):
            result = d.finish("brain-dump", datetime(2026, 9, 20, 10, 5), wav)
        self.assertTrue(result["path"].endswith("brain-dump-1005.txt"))
        clip.assert_not_called()

    def test_finish_no_save_copies_but_writes_no_note(self):
        wav = self.tmp / "audio.wav"
        wav.write_bytes(b"RIFF")
        segs = [_seg(0.0, 3.0, "email the report by five")]
        with (
            mock.patch.object(d, "save_note", return_value=(None, "")) as save,
            mock.patch.object(d, "copy_to_clipboard", return_value=True) as clip,
            mock.patch.object(d.transcriber, "transcribe", return_value=segs),
        ):
            result = d.finish("message", datetime(2026, 9, 20, 9, 41), wav, save_note=False)
        self.assertIsNone(result["path"])
        self.assertEqual(result["plain"], "email the report by five")
        self.assertTrue(result["on_clipboard"])
        save.assert_not_called()
        clip.assert_called_once_with("email the report by five")
        self.assertFalse(wav.exists(), "audio cleaned up after finishing")

    def test_finish_mirrors_dictation_note_when_saved(self):
        wav = self.tmp / "audio.wav"
        wav.write_bytes(b"RIFF")
        segs = [_seg(0.0, 2.0, "draft the outline now")]
        with (
            mock.patch.object(d.transcriber, "transcribe", return_value=segs),
            mock.patch.object(d, "copy_to_clipboard", return_value=True),
            mock.patch.object(d, "_mirror_note") as mirror,
        ):
            result = d.finish("brain-dump", datetime(2026, 9, 20, 10, 5), wav)
        self.assertTrue(result["path"].endswith("brain-dump-1005.txt"))
        mirror.assert_called_once()
        preset, started, text = mirror.call_args.args
        self.assertEqual(preset, "brain-dump")
        self.assertEqual(started, datetime(2026, 9, 20, 10, 5))
        self.assertIn("draft the outline now", text)

    def test_finish_no_save_skips_obsidian_mirror(self):
        wav = self.tmp / "audio.wav"
        wav.write_bytes(b"RIFF")
        segs = [_seg(0.0, 2.0, "just paste this")]
        with (
            mock.patch.object(d.transcriber, "transcribe", return_value=segs),
            mock.patch.object(d, "copy_to_clipboard", return_value=True),
            mock.patch.object(d, "_mirror_note") as mirror,
        ):
            result = d.finish("message", datetime(2026, 9, 20, 10, 5), wav, save_note=False)
        self.assertIsNone(result["path"])
        self.assertTrue(result["on_clipboard"])
        mirror.assert_not_called()

    def test_refusing_reason_when_meeting_recording(self):
        with (
            mock.patch.object(d.storage, "get_live_session_info", return_value={"pid": 1}),
            mock.patch.object(d.storage, "read_app_state", return_value={"state": "recording"}),
        ):
            self.assertIn("meeting", d.refusing_reason().lower())

    def test_refusing_reason_when_meeting_transcribing(self):
        # Dictation should NOT be refused when a meeting is transcribing
        # (the recorder has already released the mic).
        with (
            mock.patch.object(d.storage, "get_live_session_info", return_value={"pid": 1}),
            mock.patch.object(d.storage, "read_app_state", return_value={"state": "transcribing"}),
        ):
            self.assertIsNone(d.refusing_reason())

    def test_refusing_reason_none_when_free(self):
        with mock.patch.object(d.storage, "get_live_session_info", return_value=None):
            self.assertIsNone(d.refusing_reason())

    def test_pid_helpers(self):
        self.assertIsNone(d.running_pid())
        d.write_pid(os.getpid())  # a pid that provably exists
        self.assertEqual(d.running_pid(), os.getpid())
        d.remove_pid()
        self.assertIsNone(d.running_pid())

    def test_write_and_read_result(self):
        self.assertIsNone(d.read_result())
        d.write_result({"preset": "message", "path": "/tmp/none.txt"})
        self.assertEqual(d.read_result()["path"], "/tmp/none.txt")

    def test_clear_result_removes_stale(self):
        d.write_result({"preset": "message", "path": "/tmp/none.txt"})
        d.clear_result()
        self.assertIsNone(d.read_result())

    def test_wait_for_stop_reads_result_once_child_exits(self):
        d.write_pid(4242)
        d.write_result({"preset": "message", "path": "/tmp/none.txt", "plain": "hi"})
        with mock.patch.object(d, "running_pid", side_effect=[4242, None, None]):
            result = d.wait_for_stop(timeout=2.0)
        self.assertEqual(result["plain"], "hi")

    def test_wait_for_stop_returns_none_without_result(self):
        # Child vanished but never wrote a result (e.g. it was killed).
        d.write_pid(4242)
        with mock.patch.object(d, "running_pid", side_effect=[4242, None, None]):
            result = d.wait_for_stop(timeout=0.5)
        self.assertIsNone(result)

    def test_resolve_note_matches_short_suffix(self):
        dt = datetime(2026, 9, 20, 9, 41)
        path1, _ = d.save_note("message", dt, [_seg(0, 1, "first")])
        path2, _ = d.save_note("message", datetime(2026, 9, 20, 10, 2), [_seg(0, 1, "second")])
        self.assertEqual(d.resolve_note("message-0941"), path1)
        self.assertEqual(d.resolve_note(path2.stem), path2)
        self.assertIsNone(d.resolve_note("message-9999"))


class LiveTranscribeTest(unittest.TestCase):
    """live_transcribe: the display-only loop that emits words as they land."""

    class _Recorder:
        """Feeds one snapshot, then silence."""

        def __init__(self, frames: int):
            self._frames = frames
            self._done = False
            self.snapshots: list[Path] = []

        def snapshot_since(self, start_frame: int):
            if self._done or start_frame >= self._frames:
                return None
            fd, name = tempfile.mkstemp(suffix=".snap.wav", prefix="trnscrb-live-")
            os.close(fd)
            path = Path(name)
            path.write_bytes(b"RIFF")
            self.snapshots.append(path)
            self._done = True
            return path, self._frames

    def test_emits_new_chunk_then_stops(self):
        rec = self._Recorder(16_000)
        emitted: list[str] = []
        stop = threading.Event()

        def on_text(text):
            emitted.append(text)
            stop.set()

        with (
            mock.patch.object(d.transcriber, "preload"),
            mock.patch.object(
                d.transcriber, "transcribe", return_value=[_seg(0, 1, "I hope testing")]
            ),
        ):
            segments = d.live_transcribe(rec, stop, interval=0.01, on_text=on_text)

        self.assertEqual(emitted, ["I hope testing"])
        self.assertEqual(segments, [_seg(0, 1, "I hope testing")])
        for snap in rec.snapshots:
            self.assertFalse(snap.exists(), "snapshot WAV must be cleaned up")

    def test_already_stopped_returns_immediately(self):
        rec = self._Recorder(16_000)
        stop = threading.Event()
        stop.set()
        with (
            mock.patch.object(d.transcriber, "preload"),
            mock.patch.object(d.transcriber, "transcribe") as t,
        ):
            segments = d.live_transcribe(rec, stop, interval=0.01)
        self.assertEqual(segments, [])
        t.assert_not_called()

    def test_failed_pass_does_not_kill_the_loop(self):
        rec = self._Recorder(16_000)
        stop = threading.Event()
        threading.Timer(0.15, stop.set).start()
        with (
            mock.patch.object(d.transcriber, "preload"),
            mock.patch.object(d.transcriber, "transcribe", side_effect=RuntimeError("model busy")),
        ):
            segments = d.live_transcribe(rec, stop, interval=0.05)
        self.assertEqual(segments, [])
        for snap in rec.snapshots:
            self.assertFalse(snap.exists())


class SilenceWatchdogTest(unittest.TestCase):
    """silence_watchdog: the dictation ends when the speaker stops talking."""

    class _Recorder:
        """Plays back a mic-energy script as if audio were arriving live.

        Script entries are ``(seconds, energy)``; the timeline grows in real
        time at 16 blocks per second (1024 frames @ 16 kHz).
        """

        BLOCKS_PER_SEC = 16

        def __init__(self, script):
            self._script = script
            self._t0 = time.monotonic()

        def attribution_timeline(self):
            import numpy as np

            elapsed = time.monotonic() - self._t0
            blocks: list[float] = []
            t = 0.0
            for secs, energy in self._script:
                for _ in range(int(secs * self.BLOCKS_PER_SEC)):
                    if t <= elapsed:
                        blocks.append(energy)
                    t += 1.0 / self.BLOCKS_PER_SEC
            n = len(blocks)
            offsets = np.arange(n, dtype=np.int64) * 1024
            energies = np.array(blocks, dtype=np.float32)
            return offsets, energies, np.zeros(n, dtype=np.float32)

    def test_stops_after_silence_following_speech(self):
        rec = self._Recorder([(1.5, 1e-2), (5.0, 1e-9)])
        stopped: list = []
        d.silence_watchdog(
            rec,
            threading.Event(),
            on_stop=lambda: stopped.append(True),
            interval=0.05,
            min_secs=1.0,
            silence_secs=0.3,
            max_secs=60.0,
        )
        self.assertEqual(len(stopped), 1)

    def test_keeps_running_while_speech_continues(self):
        rec = self._Recorder([(4.0, 1e-2)])
        stopped: list = []
        stop = threading.Event()
        threading.Timer(1.0, stop.set).start()
        d.silence_watchdog(
            rec,
            stop,
            on_stop=lambda: stopped.append(True),
            interval=0.05,
            min_secs=0.5,
            silence_secs=0.3,
            max_secs=60.0,
        )
        self.assertEqual(stopped, [])

    def test_leading_silence_cannot_stop_before_min_duration(self):
        rec = self._Recorder([(5.0, 1e-9)])
        stopped: list = []
        stop = threading.Event()
        threading.Timer(1.0, stop.set).start()
        d.silence_watchdog(
            rec,
            stop,
            on_stop=lambda: stopped.append(True),
            interval=0.05,
            min_secs=1.5,
            silence_secs=0.3,
            max_secs=60.0,
        )
        self.assertEqual(stopped, [])

    def test_max_duration_forces_stop(self):
        rec = self._Recorder([(30.0, 1e-2)])
        stopped: list = []
        d.silence_watchdog(
            rec,
            threading.Event(),
            on_stop=lambda: stopped.append(True),
            interval=0.05,
            min_secs=0.0,
            silence_secs=60.0,
            max_secs=0.5,
        )
        self.assertEqual(len(stopped), 1)


class DictationCliTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = Path(tempfile.mkdtemp(prefix="trnscrb-dict-cli-"))
        names = {"_PID_FILE": "dictation.pid", "_RESULT_FILE": "dictation_result.json"}
        for attr, name in names.items():
            p = mock.patch.object(d, attr, self.tmp / name)
            p.start()
            self.addCleanup(p.stop)

    def _tmp_wav(self):
        fd, name = tempfile.mkstemp(suffix=".wav", prefix="trnscrb-dict-")
        path = Path(name)
        path.write_bytes(b"RIFF")
        return path

    def test_record_refuses_when_meeting_in_progress(self):
        with (
            mock.patch.object(
                d, "refusing_reason", return_value="A meeting transcription is in progress."
            ),
            mock.patch.object(d, "new_recorder") as rec,
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "record"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("meeting transcription is in progress", result.output)
        rec.assert_not_called()

    def test_record_message_flow(self):
        wav = self._tmp_wav()
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "new_recorder", return_value=_FakeRecorder(wav)),
            mock.patch("click.pause"),
            mock.patch.object(d, "live_transcribe"),
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": "/tmp/notes/2026-09-20_09-41-00_message-0941.txt",
                    "plain": "cancel the subscription",
                    "on_clipboard": True,
                    "duration_secs": 1.0,
                },
            ),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "record"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Copied to the clipboard", result.output)
        self.assertIn("cancel the subscription", result.output)

    def test_record_no_save_flow(self):
        wav = self._tmp_wav()
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "new_recorder", return_value=_FakeRecorder(wav)),
            mock.patch("click.pause"),
            mock.patch.object(d, "live_transcribe"),
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": None,
                    "plain": "paste this only",
                    "on_clipboard": True,
                    "duration_secs": 1.0,
                },
            ) as finish,
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "record", "--no-save"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No note saved", result.output)
        self.assertIn("Copied to the clipboard", result.output)
        self.assertIn("paste this only", result.output)
        self.assertEqual(finish.call_args.kwargs["save_note"], False)

    def test_record_saves_note_by_default(self):
        wav = self._tmp_wav()
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "new_recorder", return_value=_FakeRecorder(wav)),
            mock.patch("click.pause"),
            mock.patch.object(d, "live_transcribe"),
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": "/tmp/notes/message-0941.txt",
                    "plain": "say it and save it",
                    "on_clipboard": True,
                    "duration_secs": 1.0,
                },
            ) as finish,
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "record"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(finish.call_args.kwargs.get("save_note"), True)

    def test_record_starts_live_transcription(self):
        wav = self._tmp_wav()
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "new_recorder", return_value=_FakeRecorder(wav)),
            mock.patch("click.pause"),
            mock.patch.object(d, "live_transcribe") as live,
            mock.patch.object(
                d,
                "finish",
                return_value={
                    "preset": "message",
                    "path": None,
                    "plain": "words on screen",
                    "on_clipboard": True,
                    "duration_secs": 1.0,
                },
            ),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "record"])
        self.assertEqual(result.exit_code, 0)
        live.assert_called_once()
        recorder_arg, stop_event = live.call_args.args
        self.assertIsInstance(recorder_arg, _FakeRecorder)
        self.assertIsInstance(stop_event, threading.Event)
        self.assertTrue(stop_event.is_set(), "the live loop must be stopped before finishing")
        self.assertTrue(callable(live.call_args.kwargs.get("on_text")))

    @mock.patch("os.kill")
    def test_dictate_no_save_signals_app_with_flag(self, kill):
        with (
            mock.patch.object(cli, "_running_app_pid", return_value=12345),
            mock.patch.object(d, "write_start_request") as write,
        ):
            result = self.runner.invoke(cli.cli, ["dictate", "message", "--no-save"])
        self.assertEqual(result.exit_code, 0)
        write.assert_called_once_with("message", save_note=False)
        kill.assert_called_once_with(12345, signal.SIGUSR2)

    @mock.patch("os.kill")
    def test_dictate_default_signals_app_with_save(self, kill):
        with (
            mock.patch.object(cli, "_running_app_pid", return_value=12345),
            mock.patch.object(d, "write_start_request") as write,
        ):
            result = self.runner.invoke(cli.cli, ["dictate", "message"])
        self.assertEqual(result.exit_code, 0)
        write.assert_called_once_with("message", save_note=True)

    @mock.patch("trnscrb.cli.subprocess.Popen")
    def test_start_spawns_detached_child(self, popen):
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "running_pid", return_value=None),
            mock.patch.object(d, "wait_for_pid", return_value=5678) as wait,
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "start", "--preset", "brain-dump"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("pid 5678", result.output)
        args = popen.call_args.args[0]
        self.assertEqual(args[:3], [cli.sys.executable, "-m", "trnscrb.dictation"])
        self.assertEqual(args[3], "brain-dump")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        wait.assert_called_once()

    @mock.patch("trnscrb.cli.subprocess.Popen")
    def test_start_no_save_passes_flag_to_child(self, popen):
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "running_pid", return_value=None),
            mock.patch.object(d, "wait_for_pid", return_value=5678),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "start", "--no-save"])
        self.assertEqual(result.exit_code, 0)
        args = popen.call_args.args[0]
        self.assertEqual(args[3], "message")
        self.assertEqual(args[4:], ["--no-save"])

    def test_start_refuses_when_already_running(self):
        with (
            mock.patch.object(d, "refusing_reason", return_value=None),
            mock.patch.object(d, "running_pid", return_value=999),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "start"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("already running", result.output)

    @mock.patch("os.kill")
    def test_stop_signals_and_reports(self, kill):
        with (
            mock.patch.object(d, "running_pid", return_value=5678),
            mock.patch.object(
                d,
                "wait_for_stop",
                return_value={
                    "preset": "message",
                    "path": "/tmp/notes/2026-09-20_09-41-00_message-0941.txt",
                    "plain": "remind me about lunch",
                    "on_clipboard": True,
                    "duration_secs": 1.0,
                },
            ),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "stop"])
        self.assertEqual(result.exit_code, 0)
        kill.assert_called_once_with(5678, signal.SIGUSR1)
        self.assertIn("Saved →", result.output)
        self.assertIn("remind me about lunch", result.output)

    def test_stop_reports_no_dictation(self):
        with mock.patch.object(d, "running_pid", return_value=None):
            result = self.runner.invoke(cli.cli, ["dictation", "stop"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No dictation is running", result.output)

    def test_draft_uses_note_body_and_prints_draft(self):
        dt = datetime(2026, 9, 20, 10, 5)
        d.save_note("brain-dump", dt, [_seg(0, 2, "um draft this idea")])
        with (
            mock.patch(
                "trnscrb.enricher.get_active_provider_config",
                return_value=("claude_code", {"model": "sonnet"}),
            ),
            mock.patch("trnscrb.enricher.provider_label", return_value="Claude Code"),
            mock.patch(
                "trnscrb.enricher.draft_dictation",
                return_value={
                    "draft": "Draft this idea cleanly.",
                    "provider": "claude_code",
                    "model": "sonnet",
                },
            ),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "draft", "brain-dump-1005"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Draft this idea cleanly.", result.output)

    def test_draft_unknown_id_fails(self):
        result = self.runner.invoke(cli.cli, ["dictation", "draft", "message-0000"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No dictation note matching", result.output)

    def test_status_shows_running_and_last_result(self):
        with (
            mock.patch.object(d, "running_pid", return_value=5678),
            mock.patch.object(
                d,
                "read_result",
                return_value={
                    "preset": "message",
                    "path": "/tmp/notes/2026-09-20_09-41-00_message-0941.txt",
                    "on_clipboard": True,
                },
            ),
        ):
            result = self.runner.invoke(cli.cli, ["dictation", "status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Dictation running (pid 5678)", result.output)
        self.assertIn("on clipboard", result.output)


class DraftDictationTest(unittest.TestCase):
    """enricher.draft_dictation uses draft.md or the built-in default."""

    _SETTINGS_WITH_CLAUDE_CODE = {
        "enrich": {
            "provider": "claude_code",
            "profiles": {
                "claude_code": {
                    "endpoint": "",
                    "api_key": "",
                    "model": "sonnet",
                    "models": ["sonnet"],
                }
            },
        }
    }

    class _FakeAdapter:
        def __init__(self, response="Draft text"):
            self.last_prompt = ""
            self.last_config = {}
            self.response = response

        def test_connection(self, config):
            return True, "ok"

        def list_models(self, config):
            return ["model-a"]

        def enrich(self, prompt, config):
            self.last_prompt = prompt
            self.last_config = config
            return self.response

    def test_uses_default_prompt_and_returns_draft(self):
        fake = self._FakeAdapter(response="Clean draft output")
        with (
            mock.patch("trnscrb.settings.load", return_value=self._SETTINGS_WITH_CLAUDE_CODE),
            mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}),
            mock.patch.object(enricher, "_PROMPTS_DIR", Path("/nonexistent/trnscrb-prompts")),
        ):
            result = enricher.draft_dictation("um could we push the launch")
        self.assertEqual(result["draft"], "Clean draft output")
        self.assertEqual(result["provider"], "claude_code")
        self.assertIn("um could we push the launch", fake.last_prompt)

    def test_prefers_custom_draft_md_template(self):
        fake = self._FakeAdapter(response="Draft text")
        custom = "Polish this raw dictation:\n{dictation}\n\nRESULT:"
        with (
            mock.patch("trnscrb.settings.load", return_value=self._SETTINGS_WITH_CLAUDE_CODE),
            mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}),
            mock.patch.object(enricher, "_load_prompt", return_value=custom),
        ):
            enricher.draft_dictation("raw words")
        self.assertTrue(fake.last_prompt.startswith("Polish this raw dictation:"))
        self.assertIn("raw words", fake.last_prompt)


class McpDictationTest(unittest.TestCase):
    def setUp(self):
        for attr in ("_recorder", "_recording_started_at", "_dictation_active"):
            setattr(mcp_server, attr, None)

    def test_start_refuses_unknown_preset(self):
        out = mcp_server.start_dictation(preset="weekly")
        self.assertIn("Unknown preset", out)

    def test_start_refuses_when_recording(self):
        with mock.patch.object(mcp_server, "_recorder", _FakeRecorder(audio_path=None)):
            # _FakeRecorder.is_recording is False until start(); force it.
            mcp_server._recorder.started = True
            out = mcp_server.start_dictation(preset="message")
        self.assertIn("already in progress", out)

    def test_start_refuses_when_meeting_recording_elsewhere(self):
        with (
            mock.patch.object(mcp_server.dictation, "meeting_in_progress", return_value=True),
            mock.patch.object(mcp_server, "_stale_notice", return_value=""),
        ):
            out = mcp_server.start_dictation(preset="message")
        self.assertIn("Stop the meeting first", out)

    def test_start_and_stop_roundtrip(self):
        wav = Path(tempfile.mkstemp(suffix=".wav", prefix="trnscrb-mcp-dict-")[1])
        wav.write_bytes(b"RIFF")
        fake = _FakeRecorder(audio_path=wav)
        with (
            mock.patch.object(mcp_server.rec_module, "Recorder", return_value=fake),
            mock.patch.object(mcp_server, "_stale_notice", return_value=""),
        ):
            out = mcp_server.start_dictation(preset="message")
        self.assertIn("Message dictation started", out)
        self.assertEqual(mcp_server._dictation_active, "message")

        with mock.patch.object(
            mcp_server.dictation,
            "finish",
            return_value={
                "preset": "message",
                "path": "/tmp/notes/2026-09-20_09-41-00_message-0941.txt",
                "plain": "ping the design file",
                "on_clipboard": True,
                "duration_secs": 1.0,
            },
        ):
            out = mcp_server.stop_dictation()
        self.assertIn("Saved:", out)
        self.assertIn("Copied to clipboard", out)
        self.assertIn("ping the design file", out)
        self.assertIsNone(mcp_server._dictation_active)

    def test_stop_without_dictation(self):
        out = mcp_server.stop_dictation()
        self.assertIn("No dictation is in progress", out)

    def test_stop_no_audio(self):
        with (
            mock.patch.object(mcp_server, "_dictation_active", "brain-dump"),
            mock.patch.object(mcp_server, "_recorder", _FakeRecorder(audio_path=None)),
        ):
            mcp_server._recorder.started = True
            out = mcp_server.stop_dictation()
        self.assertIn("no audio was captured", out)


class DictationBackgroundChildTest(unittest.TestCase):
    """run_background: records until SIGUSR1, then reports via the result file."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trnscrb-dict-child-"))
        patchers = [
            mock.patch.object(d, "_PID_FILE", self.tmp / "dictation.pid"),
            mock.patch.object(d, "_RESULT_FILE", self.tmp / "dictation_result.json"),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_records_then_finishes_on_signal(self):
        # run_background installs real signal handlers, which only work on the
        # main thread — so the test runs it here and a timer delivers the
        # SIGUSR1 that a `dictation stop` would send.
        wav = self.tmp / "audio.wav"
        wav.write_bytes(b"RIFF")
        fake = _FakeRecorder(audio_path=wav)
        segs = [_seg(0.0, 2.0, "buy more coffee")]
        timer = threading.Timer(0.5, os.kill, args=[os.getpid(), signal.SIGUSR1])
        timer.daemon = True
        timer.start()
        with (
            mock.patch.object(d, "new_recorder", return_value=fake),
            mock.patch.object(d.transcriber, "transcribe", return_value=segs),
        ):
            d.run_background("brain-dump")
        timer.cancel()

        self.assertFalse(self.tmp.joinpath("dictation.pid").exists())
        result = d.read_result()
        self.assertIsNotNone(result)
        self.assertEqual(result["preset"], "brain-dump")
        self.assertIn("buy more coffee", result["plain"])
        self.assertFalse(wav.exists(), "child cleans up its audio")

    def test_unknown_preset_exits_early(self):
        with mock.patch.object(d, "new_recorder") as rec:
            with self.assertRaises(SystemExit) as ctx:
                d.run_background("meeting")
        self.assertEqual(ctx.exception.code, 2)
        rec.assert_not_called()


# ── command mode matching ─────────────────────────────────────────────────────


class CommandMatcherTest(unittest.TestCase):
    """Pure function: dictation.match_command(text) -> dict | None."""

    def test_stop_meeting_exact(self):
        result = d.match_command("stop the meeting")
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "stop_meeting")

    def test_end_meeting(self):
        result = d.match_command("end meeting")
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "stop_meeting")

    def test_bookmark(self):
        result = d.match_command("bookmark this")
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "bookmark")

    def test_stop_dictation(self):
        result = d.match_command("stop dictation")
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "stop_dictation")

    def test_start_meeting(self):
        result = d.match_command("start the meeting")
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "start_meeting")

    def test_no_match_returns_none(self):
        result = d.match_command("hello world how are you")
        self.assertIsNone(result)

    def test_empty_returns_none(self):
        result = d.match_command("")
        self.assertIsNone(result)

    def test_stop_meeting_with_filler(self):
        result = d.match_command("stop the meeting please")
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "stop_meeting")


# ── voice-trained glossary ────────────────────────────────────────────────────


class VoiceTrainedGlossaryTest(unittest.TestCase):
    """glossary.train_term: learns a term from a spoken sample."""

    def test_train_term_adds_canonical(self):
        result = g.train_term("TestTerm")
        self.assertIn("TestTerm", [e["term"] for e in result])

    def test_train_term_adds_alias_when_different(self):
        result = g.train_term("MyTerm", heard="my term")
        entries = [e for e in result if e["term"] == "MyTerm"]
        self.assertEqual(len(entries), 1)
        self.assertIn("my term", entries[0]["aliases"])

    def test_train_term_skips_alias_when_same(self):
        result = g.train_term("Hivenet", heard="Hivenet")
        entries = [e for e in result if e["term"] == "Hivenet"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["aliases"], [])

    def test_train_term_merges_alias(self):
        g.add_terms([{"term": "MyTerm", "aliases": ["variant1"]}])
        result = g.train_term("MyTerm", heard="variant2")
        entries = [e for e in result if e["term"] == "MyTerm"]
        self.assertEqual(len(entries), 1)
        self.assertIn("variant1", entries[0]["aliases"])
        self.assertIn("variant2", entries[0]["aliases"])

    def test_train_term_empty_raises(self):
        with self.assertRaises(ValueError):
            g.train_term("")


# ── meeting-aware dictation ───────────────────────────────────────────────────


class MeetingAwareDictationTest(unittest.TestCase):
    """Meeting-aware dictation: per-meeting storage and injection."""

    def test_get_meeting_context_when_none(self):
        with mock.patch.object(d.storage, "get_live_session_info", return_value=None):
            ctx = d.get_meeting_context()
            self.assertIsNone(ctx)

    def test_get_meeting_context_returns_info(self):
        mock_info = {
            "path": "/tmp/test.txt",
            "meeting": "Team Standup",
            "started_at": "2026-01-01T00:00:00",
        }
        with mock.patch.object(d.storage, "get_live_session_info", return_value=mock_info):
            ctx = d.get_meeting_context()
            self.assertIsNotNone(ctx)
            self.assertEqual(ctx["meeting"], "Team Standup")

    def test_meeting_slug(self):
        self.assertEqual(d._meeting_slug("Team Standup"), "Team-Standup")
        self.assertEqual(d._meeting_slug("Hello World!"), "Hello-World-")

    def test_save_note_meeting_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notes_dir = Path(tmpdir) / "notes"
            with mock.patch.object(d.storage, "NOTES_DIR", notes_dir):
                segments = [_seg(0, 1, "hello")]
                path, _ = d.save_note(
                    "message", datetime.now(), segments, meeting_name="Team Meeting"
                )
                self.assertIsNotNone(path)
                self.assertTrue(str(path).startswith(str(notes_dir / "Team-Meeting")))

    def test_inject_into_meeting_transcript(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"[Alice]\n  00:00  hello\n")
            f.flush()
            target = Path(f.name)
        result = d.inject_into_meeting_transcript("dictated text", target)
        self.assertTrue(result)
        content = target.read_text(encoding="utf-8")
        self.assertIn("[Dictation", content)
        self.assertIn("dictated text", content)


# ── paste to active app ───────────────────────────────────────────────────────


class PasteTextTest(unittest.TestCase):
    """dictation.paste_text_to_active_app: copies + pastes via AppleScript."""

    def test_empty_text_returns_failure(self):
        ok, detail = d.paste_text_to_active_app("")
        self.assertFalse(ok)

    def test_no_text_returns_failure(self):
        ok, detail = d.paste_text_to_active_app(None)
        self.assertFalse(ok)

    def test_paste_text_calls_osascript(self):
        # Stub the osascript call to simulate success.
        class FakeProc:
            returncode = 0

        with mock.patch("subprocess.run", return_value=FakeProc()):
            ok, detail = d.paste_text_to_active_app("hello")
        self.assertTrue(ok)
        self.assertIn("pasted", detail)


if __name__ == "__main__":
    unittest.main()
