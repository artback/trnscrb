"""Dictation — mic-only capture for short voice notes.

Phase 1 of the dictation feature. Two presets, both deliberately outside the
meeting pipeline:

  message    — capture with the mic only, transcribe with the glossary applied,
               copy the verbatim text to the clipboard (pbcopy), and save a
               `message-<HHMM>` note. No AI summary, no filler removal.
  brain-dump — captured the same way and saved as a `brain-dump-<HHMM>` note;
               optionally turned into a draft afterwards with
               `trnscrb dictation draft <id>` (see prompts/draft.md).

Raw output is the point: a dictation keeps the speaker's exact phrasing, so
nothing is stripped or "readability"-processed. Command mode and the
voice-trained glossary are Phase 2, not here.

Every surface ends in one function — `finish()` — which transcribes, saves,
and copies to the clipboard. The menu bar app and MCP server own their
Recorders in-process; the CLI's `dictation start` spawns a detached child that
records until `dictation stop` signals it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from trnscrb import storage, transcriber
from trnscrb.log import get_logger
from trnscrb.recorder import Recorder

_log = get_logger("trnscrb.dictation")

PRESETS = ("message", "brain-dump")
PRESET_LABELS = {"message": "Message", "brain-dump": "Brain dump"}

_CONTROL_DIR = Path.home() / ".config" / "trnscrb"
_PID_FILE = _CONTROL_DIR / "dictation.pid"
_RESULT_FILE = _CONTROL_DIR / "dictation_result.json"

# Matches storage._SEPARATOR: the line that splits a note's header from its
# spoken body. Kept in sync deliberately so note_body() can find it.
_BODY_SEP = "=" * 60


def preset_label(preset: str) -> str:
    """User-facing label for a preset ("Message", "Brain dump")."""
    return PRESET_LABELS.get(preset, preset.replace("-", " ").title())


def is_preset(value: str) -> bool:
    return value in PRESETS


def meeting_in_progress() -> bool:
    """True when a meeting is being recorded, so dictation must wait.

    The live-session file is written by the meeting recorder and validated
    against its PID, so a crashed session can never block dictation forever.
    """
    return storage.get_live_session_info() is not None


def refusing_reason() -> str | None:
    """Why dictation cannot start right now, or None when it can."""
    if meeting_in_progress():
        return "A meeting transcription is in progress. Stop the meeting first, then dictate."
    return None


def new_recorder() -> Recorder:
    """A mic-only recorder, started and ready to capture."""
    recorder = Recorder(system_audio=False)
    recorder.start()
    return recorder


def plain_text(segments: list[dict]) -> str:
    """The verbatim dictation: segment text joined with newlines, unedited."""
    return "\n".join(
        str(seg.get("text") or "").strip() for seg in segments if str(seg.get("text") or "").strip()
    )


def note_body(note: str) -> str:
    """The spoken body of a saved dictation note (everything after the header)."""
    _, _, body = note.partition(f"\n{_BODY_SEP}\n")
    return body.strip() if body else note.strip()


def copy_to_clipboard(text: str) -> bool:
    """Best-effort pbcopy. True when the text landed on the clipboard."""
    if not text:
        return False
    try:
        proc = subprocess.run(["pbcopy"], input=text.encode("utf-8"))
        return proc.returncode == 0
    except Exception:
        _log.debug("pbcopy failed", exc_info=True)
        return False


def save_note(preset: str, started_at: datetime, segments: list[dict]) -> tuple[Path | None, str]:
    """Format and save the dictation note. Returns (path, text).

    A dictation with no spoken content (silence) is not saved — a header-only
    note is worse than no note.
    """
    name = f"{preset}-{started_at.strftime('%H%M')}"
    if not any(str(seg.get("text") or "").strip() for seg in segments):
        return None, ""
    text = storage.format_transcript(segments, started_at, name, kind=preset)
    if not text.strip():
        return None, ""
    path = storage.get_transcript_path(name, started_at)
    storage.save_transcript(path, text)
    return path, text


def finish(preset: str, started_at: datetime, audio_path: Path) -> dict:
    """Transcribe, save the note, and copy the text for a message preset.

    The audio file is always cleaned up — the note and clipboard are the
    product, and a few seconds of dictation is cheap to redo. Returns:
      {preset, path, text, plain, on_clipboard, duration_secs}
    """
    _log.info("Dictation finishing (preset=%s, audio=%s)", preset, audio_path)
    try:
        segments = transcriber.transcribe(audio_path)
    except Exception as e:
        _log.error("Dictation transcription failed: %s", e, exc_info=True)
        _cleanup_audio(audio_path)
        raise
    _cleanup_audio(audio_path)

    plain = plain_text(segments)
    duration_secs = float(segments[-1]["end"]) if segments else 0.0

    path, text = None, ""
    if plain:
        path, text = save_note(preset, started_at, segments)
    on_clipboard = False
    if preset == "message" and plain:
        on_clipboard = copy_to_clipboard(plain)

    return {
        "preset": preset,
        "path": str(path) if path else None,
        "text": text,
        "plain": plain,
        "on_clipboard": on_clipboard,
        "duration_secs": duration_secs,
    }


def resolve_note(short_id: str) -> Path | None:
    """Resolve a dictation note id to its file.

    Accepts a full filename stem or a short suffix such as `message-0941`;
    a suffix resolves to the newest note whose stem ends with it.
    """
    target = str(short_id).strip()
    if not target:
        return None
    try:
        notes = storage.ensure_notes_dir()
        candidates = [path for path in notes.glob("*.txt") if path.stem.endswith(target)]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── background CLI child ─────────────────────────────────────────────────────


def _cleanup_audio(audio_path: Path) -> None:
    try:
        audio_path.unlink(missing_ok=True)
    except OSError:
        _log.debug("Could not remove dictation audio %s", audio_path, exc_info=True)


def running_pid() -> int | None:
    """PID of the background dictation process, or None."""
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # raises if the process is gone
        return pid
    except Exception:
        return None


def write_pid(pid: int) -> None:
    _CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(pid))


def remove_pid() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def write_result(payload: dict) -> None:
    try:
        _CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        _RESULT_FILE.write_text(json.dumps(payload))
    except Exception:
        _log.debug("Could not write dictation result", exc_info=True)


def read_result() -> dict | None:
    """The last background dictation's result, or None."""
    try:
        payload = json.loads(_RESULT_FILE.read_text())
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def clear_result() -> None:
    """Drop any previous result, so a stale one is never mistaken for fresh."""
    try:
        _RESULT_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def wait_for_pid(timeout: float = 5.0) -> int | None:
    """Wait (briefly) for the background child to publish its pid."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = running_pid()
        if pid is not None:
            return pid
        time.sleep(0.1)
    return None


def wait_for_stop(timeout: float = 180.0) -> dict | None:
    """After signalling the child, wait for its result. None on timeout.

    The child removes its pid file before transcribing, so this polls in two
    phases: for the pid to disappear, then for the result file to appear —
    which `starts` guarantees is fresh by clearing it first.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if running_pid() is None:
            break
        time.sleep(0.2)
    while time.monotonic() < deadline:
        result = read_result()
        if result is not None:
            return result
        time.sleep(0.2)
    return None  # still finishing — the caller can check `status` again


def run_background(preset: str) -> None:
    """Child entry point: record until SIGUSR1, then finish and report.

    Launched detached by `trnscrb dictation start` as
    `python -m trnscrb.dictation <preset>`. stdout is /dev/null, so the
    outcome lands in dictation_result.json for `trnscrb dictation stop` to
    read (and the app log carries any failure).
    """
    if not is_preset(preset):
        sys.exit(2)

    recorder = new_recorder()
    started_at = datetime.now()
    write_pid(os.getpid())

    stop_event = threading.Event()

    def _stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGUSR1, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        while not stop_event.wait(1.0):
            pass
    finally:
        remove_pid()

    _log.info("Dictation child finishing (preset=%s)", preset)
    audio_path = None
    try:
        audio_path = recorder.stop()
        if not audio_path:
            write_result({"preset": preset, "error": "No audio captured."})
            return
        result = finish(preset, started_at, audio_path)
        result["preset"] = preset
        write_result(result)
    except Exception as e:
        _log.error("Dictation child failed", exc_info=True)
        write_result({"preset": preset, "error": str(e)})
    finally:
        if audio_path:
            _cleanup_audio(audio_path)


def _main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="trnscrb.dictation")
    parser.add_argument("preset", choices=PRESETS)
    args = parser.parse_args(argv)
    run_background(args.preset)


if __name__ == "__main__":
    _main()
