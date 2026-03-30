"""MCP server exposing trnscrb tools to Claude Desktop.

Runs as a stdio server — Claude Desktop starts it automatically via the
entry in claude_desktop_config.json.

Tools available to Claude:
  start_recording        — begin audio capture
  stop_recording         — stop immediately, process in background
  recording_status       — check if recording / if processing is done
  get_last_transcript    — get the transcript from the most recent stop
  get_current_transcript — live partial transcript during recording
  list_transcripts       — all saved meeting files
  get_transcript         — full text of one transcript
  get_calendar_context   — current/upcoming calendar event
  enrich_transcript      — post-call summary + action items via configured LLM
"""

import json
import threading
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from trnscrb import diarizer, storage, transcriber
from trnscrb import recorder as rec_module
from trnscrb.calendar_integration import get_current_or_upcoming_event
from trnscrb.log import get_logger

_log = get_logger("trnscrb.mcp_server")

mcp = FastMCP("trnscrb")

# ── Shared state ──────────────────────────────────────────────────────────────
_recorder: rec_module.Recorder | None = None
_recording_started_at: datetime | None = None
_state_lock = threading.Lock()

# Background processing state
_processing = False  # True while transcription/diarization is running
_last_result: str | None = None  # last stop_recording outcome (path + preview)
_last_error: str | None = None  # last processing error if any


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def start_recording() -> str:
    """Start capturing audio for a meeting transcript."""
    global _recorder, _recording_started_at
    with _state_lock:
        if _recorder and _recorder.is_recording:
            return "Already recording."
        device = rec_module.Recorder.find_blackhole_device()
        _recorder = rec_module.Recorder(device=device)
        _recorder.start()
        _recording_started_at = datetime.now()
    source = "BlackHole (system + mic)" if device is not None else "built-in mic"
    _log.info("start_recording: device=%s", source)
    return f"Recording started at {_recording_started_at.strftime('%H:%M')} using {source}."


@mcp.tool()
def stop_recording(meeting_name: str = "") -> str:
    """
    Stop the recording immediately and kick off transcription in the background.
    Returns right away — use recording_status or get_last_transcript to get the result.

    Args:
        meeting_name: Optional name. Defaults to calendar event title or timestamp.
    """
    global _recorder, _recording_started_at, _processing, _last_result, _last_error
    with _state_lock:
        if not _recorder or not _recorder.is_recording:
            return "Not currently recording."
        started_at = _recording_started_at or datetime.now()
        audio_path = _recorder.stop()  # stops the stream, returns WAV path
        _recorder = None

    if not audio_path:
        return "Recording stopped but no audio was captured."

    # Resolve name before background thread (calendar call is fast)
    if not meeting_name:
        evt = get_current_or_upcoming_event()
        meeting_name = evt["title"] if evt else f"meeting-{started_at.strftime('%H%M')}"

    _log.info("stop_recording: meeting=%s", meeting_name)

    with _state_lock:
        _processing = True
        _last_result = None
        _last_error = None

    thread = threading.Thread(
        target=_process_audio,
        args=(audio_path, started_at, meeting_name),
        daemon=True,
    )
    thread.start()

    duration_s = int((datetime.now() - started_at).total_seconds())
    return (
        f'Recording stopped. {duration_s}s of audio captured for "{meeting_name}".\n'
        f"Transcription is running in the background.\n"
        f"Ask me for `recording_status` to check progress, or `get_last_transcript` once done."
    )


@mcp.tool()
def recording_status() -> str:
    """Check whether a recording is active or background transcription is in progress."""
    with _state_lock:
        is_recording = bool(_recorder and _recorder.is_recording)
        started_at = _recording_started_at

    if is_recording and started_at:
        elapsed = int((datetime.now() - started_at).total_seconds())
        m, s = divmod(elapsed, 60)
        return f"Recording in progress — {m}m {s}s elapsed."

    with _state_lock:
        processing = _processing
        last_error = _last_error
        last_result = _last_result

    if processing:
        return "Transcription in progress — processing audio, please wait."

    if last_error:
        return f"Last transcription failed: {last_error}"

    if last_result:
        return "Transcription complete. Use get_last_transcript to read it."

    return "Idle — no active recording or pending transcription."


@mcp.tool()
def get_last_transcript() -> str:
    """Return the transcript from the most recently completed recording."""
    with _state_lock:
        processing = _processing
        last_error = _last_error
        last_result = _last_result
    if processing:
        return "Still transcribing — check back in a moment."
    if last_error:
        return f"Transcription failed: {last_error}"
    if last_result:
        return last_result
    return "No transcript available yet. Start and stop a recording first."


@mcp.tool()
def get_current_transcript() -> str:
    """Return whatever has been transcribed so far during an active recording (not yet available — live transcription coming soon)."""
    with _state_lock:
        is_recording = bool(_recorder and _recorder.is_recording)
    if not is_recording:
        return "Not currently recording."
    return "Recording in progress. Live transcript is not yet available — stop the recording to get the full transcript."


@mcp.tool()
def list_transcripts() -> str:
    """List all saved meeting transcripts (most recent first)."""
    transcripts = storage.list_transcripts()
    if not transcripts:
        return "No transcripts found in ~/meeting-notes/"
    lines = [f"{t['id']}  ({t['modified'][:16]})" for t in transcripts[:30]]
    return "\n".join(lines)


@mcp.tool()
def get_transcript(transcript_id: str) -> str:
    """
    Return the full text of a saved transcript.

    Args:
        transcript_id: The filename stem (e.g. 2024-01-15_14-30_standup).
    """
    text = storage.read_transcript(transcript_id)
    if text is None:
        return f"Transcript '{transcript_id}' not found."
    return text


@mcp.tool()
def get_calendar_context() -> str:
    """Return the current or next upcoming calendar event for meeting context."""
    evt = get_current_or_upcoming_event()
    if not evt:
        return "No current or upcoming calendar events found."
    return json.dumps(evt, indent=2)


@mcp.tool()
def get_weekly_transcripts(week: str = "") -> str:
    """
    Get all transcripts for a given week, concatenated and ready for summarization.
    Returns the meeting name and full text for each transcript in the week.

    Args:
        week: ISO week (e.g. '2026-W13'). Defaults to the current week.
    """
    from datetime import date, timedelta

    if week:
        try:
            year, w = week.split("-W")
            monday = date.fromisocalendar(int(year), int(w), 1)
        except (ValueError, TypeError):  # fmt: skip
            return f"Invalid week format: '{week}'. Use YYYY-WNN (e.g. 2026-W13)."
    else:
        today = date.today()
        monday = today - timedelta(days=today.weekday())

    friday = monday + timedelta(days=4)
    files = sorted(storage.NOTES_DIR.glob("*.txt"))
    parts = []
    for f in files:
        if f.name.startswith("weekly-") or f.name.startswith("annual-"):
            continue
        try:
            file_date = date.fromisoformat(f.name[:10])
        except ValueError:
            continue
        if monday <= file_date <= friday:
            text = f.read_text(encoding="utf-8").strip()
            parts.append(f"--- {f.name} ---\n{text}")

    if not parts:
        week_label = monday.strftime("%G-W%V")
        return f"No transcripts found for {week_label} ({monday} to {friday})."

    header = (
        f"Week: {monday.strftime('%G-W%V')} ({monday} to {friday})\nTranscripts: {len(parts)}\n\n"
    )
    return header + "\n\n".join(parts)


@mcp.tool()
def get_weekly_summaries(year: str = "") -> str:
    """
    Get all saved weekly summaries for a given year.
    Use this as input when generating an annual summary.

    Args:
        year: Year (e.g. '2026'). Defaults to current year.
    """
    from datetime import date

    target_year = year or str(date.today().year)
    files = sorted(storage.NOTES_DIR.glob(f"weekly-{target_year}-W*.txt"))
    if not files:
        return f"No weekly summaries found for {target_year}."

    parts = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        parts.append(f"{'=' * 40}\n{f.stem}\n{'=' * 40}\n{text}")

    return f"Year: {target_year}\nWeekly summaries: {len(files)}\n\n" + "\n\n".join(parts)


@mcp.tool()
def search_transcripts(query: str) -> str:
    """
    Search across all saved transcripts for a keyword or phrase.
    Returns matching lines with context from each transcript.

    Args:
        query: The search term (case-insensitive).
    """
    import re

    files = sorted(storage.NOTES_DIR.glob("*.txt"), reverse=True)
    if not files:
        return "No transcripts found in ~/meeting-notes/"

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        if not hits:
            continue
        matches = []
        for hit in hits:
            start = max(0, hit - 1)
            end = min(len(lines), hit + 2)
            matches.append("\n".join(lines[start:end]))
        results.append(f"## {f.name}\n" + "\n---\n".join(matches))

    if not results:
        return f"No matches for '{query}'."
    return "\n\n".join(results[:10])


@mcp.tool()
def enrich_transcript(transcript_id: str) -> str:
    """
    Run an LLM pass on a saved transcript to produce a summary,
    action items, and inferred speaker names.

    Args:
        transcript_id: The filename stem of the transcript to enrich.
    """
    text = storage.read_transcript(transcript_id)
    if text is None:
        return f"Transcript '{transcript_id}' not found."

    from trnscrb.enricher import (
        enrich_transcript as _enrich,
    )
    from trnscrb.enricher import (
        get_active_provider_config,
        provider_label,
    )

    provider, profile = get_active_provider_config()
    evt = get_current_or_upcoming_event()
    try:
        result = _enrich(text, calendar_event=evt)
    except Exception as e:
        model_name = str(profile.get("model") or "<not selected>")
        return f"Enrichment failed ({provider_label(provider)} / {model_name}): {e}"

    path = storage.NOTES_DIR / f"{transcript_id}.txt"
    if path.exists():
        updated = result["enriched_transcript"] + "\n\n" + "=" * 60 + "\n\n" + result["enrichment"]
        storage.save_transcript(path, updated)

    return result["enrichment"]


# ── Background processing ─────────────────────────────────────────────────────


def _process_audio(audio_path: Path, started_at: datetime, meeting_name: str) -> None:
    global _processing, _last_result, _last_error
    try:
        try:
            size = audio_path.stat().st_size
        except FileNotFoundError:
            raise RuntimeError(f"Audio file missing: {audio_path}")
        if size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_path}")

        _log.info("Transcription starting: %s", meeting_name)
        segments = transcriber.transcribe(audio_path)

        from trnscrb.settings import read_hf_token

        hf_token = read_hf_token()
        if hf_token and segments:
            try:
                diar = diarizer.diarize(audio_path, hf_token)
                segments = diarizer.merge(segments, diar)
            except Exception:
                _log.warning("Diarization skipped", exc_info=True)
                pass

        audio_path.unlink(missing_ok=True)

        transcript_text = storage.format_transcript(segments, started_at, meeting_name)
        path = storage.get_transcript_path(meeting_name, started_at)
        storage.save_transcript(path, transcript_text)

        _log.info("Transcription complete: %s -> %s", meeting_name, path.name)
        preview = transcript_text[:800] + ("…" if len(transcript_text) > 800 else "")
        with _state_lock:
            _last_result = f"Saved: {path.name}\n\n{preview}"
    except Exception as e:
        _log.error("Transcription failed for %s: %s", meeting_name, e)
        audio_path.unlink(missing_ok=True)
        with _state_lock:
            _last_error = str(e)
    finally:
        with _state_lock:
            _processing = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
