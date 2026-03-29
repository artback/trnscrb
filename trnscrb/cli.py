"""CLI entry point.

trnscrb install   — smart dependency checker / installer
trnscrb start     — launch the menu bar app
trnscrb server    — start MCP server (Claude Desktop calls this)
trnscrb list      — list saved transcripts
trnscrb show <id> — print a transcript
trnscrb enrich <id> — run LLM enrichment pass on a transcript
trnscrb watch     — headless auto-record watcher
trnscrb devices   — list audio input devices
"""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from trnscrb.log import get_logger

_log = get_logger("trnscrb.cli")

# Path to Claude Desktop's MCP config file
_CLAUDE_CONFIG = (
    Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
)
_DEFAULT_PARAKEET_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"


# ── CLI group ─────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """Trnscrb — lightweight offline meeting transcription for Claude Desktop."""


# ── install ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--force", is_flag=True, help="Re-install packages even if already present.")
def install(force: bool):
    """Smart installer — checks each dependency and skips what's already installed."""
    click.echo()
    click.echo(click.style("Trnscrb Setup", bold=True))
    click.echo("=" * 42)

    # ── 1. Python version ────────────────────────────────────────────────────
    py_ok = sys.version_info >= (3, 11)
    _row("Python 3.11+", py_ok, sys.version.split()[0])
    if not py_ok:
        click.echo(click.style("  Python 3.11+ is required. Install from python.org.", fg="red"))
        sys.exit(1)
    click.echo()

    # ── 2. Python packages ───────────────────────────────────────────────────
    click.echo("  Python packages:")
    packages = {
        "rumps": "rumps>=0.4.0",
        "sounddevice": "sounddevice>=0.4.6",
        "parakeet_mlx": "parakeet-mlx>=0.5.1",
        "faster_whisper": "faster-whisper>=1.0.0",
        "pyannote.audio": "pyannote.audio>=3.1",
        "mcp": "mcp>=1.0.0",
        "anthropic": "anthropic>=0.25",
        "openai": "openai>=2.24.0",
        "scipy": "scipy>=1.11",
        "numpy": "numpy>=1.24",
    }
    to_install = []
    for import_name, pkg_spec in packages.items():
        installed = _pkg_installed(import_name) and not force
        _row(f"  {pkg_spec.split('>=')[0]}", installed, "  ", indent=4)
        if not installed:
            to_install.append(pkg_spec)

    if to_install:
        click.echo()
        if click.confirm(f"  Install {len(to_install)} missing package(s)?", default=True):
            _run([sys.executable, "-m", "pip", "install", "--quiet", *to_install])
    click.echo()

    # ── 3. Audio setup ───────────────────────────────────────────────────────
    bh_ok = _blackhole_installed()
    _row("BlackHole 2ch", bh_ok, "captures other participants' audio")
    if not bh_ok:
        if click.confirm("  Install BlackHole via Homebrew?", default=True):
            _run(["brew", "install", "blackhole-2ch"])
            click.echo("  After install: open System Settings → Sound → Output and select")
            click.echo("  'Multi-Output Device' that includes BlackHole to capture system audio.")
    click.echo()

    # ── 4. Transcription model ───────────────────────────────────────────────
    from trnscrb.settings import load as load_settings
    from trnscrb.settings import save as save_settings

    settings = load_settings()
    backend = _normalize_backend(settings.get("transcription_backend"))
    if backend == "whisper":
        model_size = str(settings.get("model_size") or "small")
        model_ok = _whisper_model_cached(model_size)
        _row(f"Whisper '{model_size}' model", model_ok)
        if not model_ok and click.confirm("  Download now? (~500 MB)", default=True):
            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415

                WhisperModel(model_size, device="cpu")
                click.echo(click.style("  Model ready.", fg="green"))
            except Exception as e:
                click.echo(click.style(f"  Download failed: {e}", fg="yellow"))
    else:
        model_id = str(settings.get("parakeet_model_id") or _DEFAULT_PARAKEET_MODEL_ID)
        model_ok = _parakeet_model_cached(model_id)
        _row("Parakeet model", model_ok)
        if not model_ok and click.confirm("  Download now?", default=True):
            try:
                from parakeet_mlx import from_pretrained  # noqa: PLC0415

                from_pretrained(model_id)
                click.echo(click.style("  Model ready.", fg="green"))
            except Exception as e:
                click.echo(click.style(f"  Download failed: {e}", fg="yellow"))
    click.echo()

    # ── 5. Speaker diarization (optional) ────────────────────────────────────
    hf_ok = bool(_get_hf_token())
    _row("Speaker labels", hf_ok, "optional — requires free HuggingFace token")
    if not hf_ok:
        click.echo("  Get a token at https://hf.co/settings/tokens")
        click.echo("  Accept terms at https://hf.co/pyannote/speaker-diarization-3.1")
        token = click.prompt(
            "  Paste token (or Enter to skip)",
            default="",
            show_default=False,
        )
        if token.strip():
            _save_hf_token(token.strip())
            click.echo(click.style("  Token saved.", fg="green"))
    click.echo()

    # ── 6. Permissions ───────────────────────────────────────────────────────
    click.echo("  Permissions (macOS will prompt if needed):")
    _request_mic_permission()
    _request_calendar_permission()
    click.echo()

    # ── 7. Notes directory ───────────────────────────────────────────────────
    from trnscrb.storage import ensure_notes_dir

    folder = ensure_notes_dir()
    click.echo(f"  Transcripts saved to: {folder}")
    click.echo()

    # ── 8. Optional integrations ─────────────────────────────────────────────
    # Claude Desktop MCP — only offer if Claude Desktop is installed
    if _CLAUDE_CONFIG.parent.exists():
        mcp_ok = _mcp_configured()
        _row("Claude Desktop integration", mcp_ok)
        if not mcp_ok:
            if click.confirm("  Register trnscrb with Claude Desktop?", default=True):
                _write_mcp_config()
                click.echo(click.style("  Done. Restart Claude Desktop to apply.", fg="green"))
        click.echo()

    # Launch at login
    login_ok = _login_item_exists()
    _row("Launch at login", login_ok)
    if not login_ok:
        if click.confirm("  Start trnscrb automatically on login?", default=False):
            import shutil

            binary = shutil.which("trnscrb") or sys.executable
            if _setup_login_item(binary):
                click.echo(click.style("  Enabled.", fg="green"))
            else:
                click.echo(click.style("  Could not set up login item.", fg="yellow"))

    # ── Defaults ─────────────────────────────────────────────────────────────
    settings = load_settings()
    changed = False
    if settings.get("auto_record") is not True:
        settings["auto_record"] = True
        changed = True
    configured_backend = str(settings.get("transcription_backend") or "").strip().lower()
    if configured_backend not in {"parakeet", "whisper"}:
        settings["transcription_backend"] = "parakeet"
        changed = True
    if not settings.get("parakeet_model_id"):
        settings["parakeet_model_id"] = _DEFAULT_PARAKEET_MODEL_ID
        changed = True
    if changed:
        save_settings(settings)

    click.echo()
    click.echo("=" * 42)
    click.echo(click.style("Setup complete!", fg="green", bold=True))
    click.echo()
    click.echo("  trnscrb start    launch menu bar app")
    click.echo("  trnscrb watch    headless auto-record")
    click.echo("  trnscrb list     list saved transcripts")
    click.echo()


# ── start ──────────────────────────────────────────────────────────────────────


@cli.command()
def start():
    """Launch the menu bar app."""
    from trnscrb.menu_bar import main

    main()


# ── server ────────────────────────────────────────────────────────────────────


@cli.command()
def server():
    """Start the MCP server (used internally by Claude Desktop)."""
    from trnscrb.mcp_server import main

    main()


# ── watch ─────────────────────────────────────────────────────────────────────


@cli.command()
def watch():
    """Watch for mic activity and auto-record meetings (headless, no menu bar)."""
    import signal

    from trnscrb import diarizer, storage, transcriber
    from trnscrb import recorder as rec_module
    from trnscrb.recorder import cleanup_stale_temp_files
    from trnscrb.watcher import GRACE_SECS, WARMUP_SECS, MicWatcher

    cleanup_stale_temp_files()
    from trnscrb.calendar_integration import get_current_or_upcoming_event

    _recorder_ref: list = [None]
    _started_ref: list = [None]

    def on_start(meeting_name: str):
        _log.info("watch: recording started, meeting=%s", meeting_name)
        click.echo(f"  🔴 Meeting detected: {meeting_name} — recording started")
        device = rec_module.Recorder.find_blackhole_device()
        r = rec_module.Recorder(device=device)
        r.start()
        _recorder_ref[0] = r
        _started_ref[0] = datetime.now()

    def on_stop():
        r = _recorder_ref[0]
        started_at = _started_ref[0] or datetime.now()
        _recorder_ref[0] = None
        if not r:
            return
        _log.info("watch: recording stopped, transcribing")
        click.echo("  ⏹  Meeting ended — transcribing…")
        audio_path = r.stop()
        if not audio_path:
            click.echo("  ⚠️  No audio captured.")
            return

        evt = get_current_or_upcoming_event()
        meeting_name = evt["title"] if evt else f"meeting-{started_at.strftime('%H%M')}"

        from trnscrb.settings import load as _load_settings

        _backend = _normalize_backend(_load_settings().get("transcription_backend"))
        try:
            segments = transcriber.transcribe(audio_path)
        except Exception as e:
            audio_path.unlink(missing_ok=True)
            _log.error(
                "Transcription failed: backend=%s file=%s err=%s",
                _backend,
                audio_path.name,
                e,
            )
            click.echo(f"  ✗ Transcription failed ({_backend}, {audio_path.name}): {e}")
            return

        import os

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            tf = Path.home() / ".cache" / "huggingface" / "token"
            hf_token = tf.read_text().strip() if tf.exists() else None
        if hf_token and segments:
            try:
                diar = diarizer.diarize(audio_path, hf_token)
                segments = diarizer.merge(segments, diar)
            except Exception as e:
                _log.warning("Diarization skipped: %s", e)
                click.echo(f"  ⚠  Speaker diarization skipped: {e}")

        audio_path.unlink(missing_ok=True)
        text = storage.format_transcript(segments, started_at, meeting_name)
        path = storage.get_transcript_path(meeting_name, started_at)
        storage.save_transcript(path, text)
        click.echo(f"  ✓ Saved: {path.name}")

        from trnscrb.settings import get as _get_setting

        if _get_setting("auto_enrich"):
            click.echo("  🔄 Enriching transcript…")
            try:
                from trnscrb.enricher import enrich_transcript

                result = enrich_transcript(text, calendar_event=evt)
                enriched = result["enriched_transcript"]
                updated = enriched + "\n\n" + "=" * 60 + "\n\n" + result["enrichment"]
                storage.save_transcript(path, updated)
                click.echo(f"  ✓ Enriched: summary + action items added ({result['provider']})")
            except Exception as e:
                _log.warning("Auto-enrich failed: %s", e)
                click.echo(f"  ⚠  Enrichment skipped: {e}")

    watcher = MicWatcher(on_start=on_start, on_stop=on_stop)
    watcher.start()

    click.echo(f"Watching for mic activity (warmup={WARMUP_SECS}s, grace={GRACE_SECS}s).")
    click.echo("Press Ctrl-C to stop.\n")

    def _shutdown(sig, frame):
        click.echo("\nStopping watcher…")
        watcher.stop()
        if _recorder_ref[0]:
            on_stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    import time

    while watcher.is_watching:
        time.sleep(1)


# ── list ──────────────────────────────────────────────────────────────────────


@cli.command(name="list")
def list_cmd():
    """List all saved meeting transcripts."""
    from trnscrb import storage

    transcripts = storage.list_transcripts()
    if not transcripts:
        click.echo("No transcripts found in ~/meeting-notes/")
        return
    for t in transcripts:
        size_kb = t["size"] // 1024 or 1
        click.echo(f"  {t['id']}  ({t['modified'][:16]})  {size_kb} KB")


# ── search ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("-n", "--context", default=1, help="Number of context lines around each match.")
def search(query: str, context: int):
    """Search across all transcripts for a keyword or phrase."""
    import re

    from trnscrb import storage

    files = sorted(storage.NOTES_DIR.glob("*.txt"), reverse=True)
    if not files:
        click.echo("No transcripts found in ~/meeting-notes/")
        return

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    total_matches = 0

    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        if not hits:
            continue

        total_matches += len(hits)
        click.echo(click.style(f"\n{f.name}", fg="cyan", bold=True))

        shown = set()
        for hit in hits:
            start = max(0, hit - context)
            end = min(len(lines), hit + context + 1)
            for i in range(start, end):
                if i in shown:
                    continue
                shown.add(i)
                line = lines[i]
                if i == hit:
                    # Highlight matches
                    highlighted = pattern.sub(
                        lambda m: click.style(m.group(), fg="yellow", bold=True),
                        line,
                    )
                    click.echo(f"  {i + 1:4d}  {highlighted}")
                else:
                    click.echo(click.style(f"  {i + 1:4d}  {line}", dim=True))

    if total_matches:
        click.echo(f"\n{total_matches} match(es) across {len(files)} transcript(s).")
    else:
        click.echo(f"No matches for '{query}'.")


# ── show ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("transcript_id")
def show(transcript_id: str):
    """Print a transcript to stdout."""
    from trnscrb import storage

    text = storage.read_transcript(transcript_id)
    if text is None:
        click.echo(f"Transcript '{transcript_id}' not found.", err=True)
        sys.exit(1)
    click.echo(text)


# ── enrich ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("transcript_id")
def enrich(transcript_id: str):
    """Run an LLM pass on a transcript: summary, action items, speaker names."""
    from trnscrb import storage
    from trnscrb.calendar_integration import get_current_or_upcoming_event
    from trnscrb.enricher import (
        enrich_transcript,
        get_active_provider_config,
        provider_label,
    )

    text = storage.read_transcript(transcript_id)
    if text is None:
        click.echo(f"Transcript '{transcript_id}' not found.", err=True)
        sys.exit(1)

    provider, config = get_active_provider_config()
    model_name = str(config.get("model") or "<not selected>")
    click.echo(f"Running enrichment with {provider_label(provider)} ({model_name})…")
    evt = get_current_or_upcoming_event()
    try:
        result = enrich_transcript(text, calendar_event=evt)
    except Exception as e:
        click.echo(f"Enrichment failed: {e}", err=True)
        sys.exit(1)

    # Overwrite file with resolved speaker names + enrichment appended
    path = storage.NOTES_DIR / f"{transcript_id}.txt"
    updated = result["enriched_transcript"] + "\n\n" + "=" * 60 + "\n\n" + result["enrichment"]
    storage.save_transcript(path, updated)

    click.echo(result["enrichment"])
    click.echo(f"\nTranscript updated at {path}")


# ── weekly / annual ──────────────────────────────────────────────────────────


@cli.command()
@click.option("--week", default=None, help="ISO week (e.g. 2026-W13). Defaults to last week.")
@click.option(
    "--prompt",
    "prompt_file",
    default=None,
    type=click.Path(exists=True),
    help="Custom prompt template file. Use {week_start}, {week_end}, {transcripts}.",
)
@click.option("--save/--no-save", default=True, help="Save summary to ~/meeting-notes/")
def weekly(week: str | None, prompt_file: str | None, save: bool):
    """Generate a weekly summary from all transcripts in a given week.

    Prompt templates are loaded from (in order):
      1. --prompt flag (explicit file)
      2. ~/.config/trnscrb/prompts/weekly.md (custom default)
      3. Built-in default
    """
    from datetime import date, timedelta

    from trnscrb import storage
    from trnscrb.enricher import (
        generate_weekly_summary,
        get_active_provider_config,
        provider_label,
    )

    if week:
        try:
            year, w = week.split("-W")
            monday = date.fromisocalendar(int(year), int(w), 1)
        except (ValueError, TypeError):
            click.echo(
                f"Invalid week format: '{week}'. Use YYYY-WNN (e.g. 2026-W13).",
                err=True,
            )
            sys.exit(1)
    else:
        today = date.today()
        monday = today - timedelta(days=today.weekday() + 7)  # last Monday

    friday = monday + timedelta(days=4)
    week_start = monday.strftime("%Y-%m-%d")
    week_end = friday.strftime("%Y-%m-%d")
    iso_week = monday.strftime("%G-W%V")

    click.echo(f"Collecting transcripts for {iso_week} ({week_start} to {week_end})…")

    files = sorted(storage.NOTES_DIR.glob("*.txt"))
    transcripts = []
    for f in files:
        try:
            file_date = date.fromisoformat(f.name[:10])
        except ValueError:
            continue
        if monday <= file_date <= friday:
            text = f.read_text(encoding="utf-8")
            if text.strip():
                transcripts.append({"name": f.name, "text": text})

    if not transcripts:
        click.echo(f"No transcripts found for {iso_week}.")
        return

    # Load custom prompt if provided via --prompt flag
    prompt_override = None
    if prompt_file:
        prompt_override = Path(prompt_file).read_text(encoding="utf-8")
        click.echo(f"Using prompt template: {prompt_file}")

    provider, profile = get_active_provider_config()
    model_name = str(profile.get("model") or "default")
    prov = provider_label(provider)
    click.echo(f"Found {len(transcripts)} transcript(s). Summarizing with {prov} ({model_name})…")

    try:
        summary = generate_weekly_summary(
            transcripts,
            week_start,
            week_end,
            prompt_override=prompt_override,
        )
    except Exception as e:
        click.echo(f"Summary generation failed: {e}", err=True)
        sys.exit(1)

    click.echo()
    click.echo(summary)

    if save:
        path = storage.NOTES_DIR / f"weekly-{iso_week}.txt"
        storage.save_transcript(path, summary)
        click.echo(f"\nSaved to {path}")


@cli.command()
@click.option(
    "--year",
    default=None,
    help="Year to summarize (e.g. 2026). Defaults to current year.",
)
@click.option(
    "--prompt",
    "prompt_file",
    default=None,
    type=click.Path(exists=True),
    help="Custom prompt template file. Use {summaries} and {year} as placeholders.",
)
@click.option("--save/--no-save", default=True, help="Save summary to ~/meeting-notes/")
def annual(year: str | None, prompt_file: str | None, save: bool):
    """Generate an annual summary from all weekly summaries.

    Prompt templates are loaded from (in order):
      1. --prompt flag (explicit file)
      2. ~/.config/trnscrb/prompts/annual.md (custom default)
      3. Built-in default
    """
    from datetime import date

    from trnscrb import storage
    from trnscrb.enricher import (
        generate_annual_summary,
        get_active_provider_config,
        provider_label,
    )

    target_year = year or str(date.today().year)
    click.echo(f"Collecting weekly summaries for {target_year}…")

    files = sorted(storage.NOTES_DIR.glob(f"weekly-{target_year}-W*.txt"))
    if not files:
        click.echo(f"No weekly summaries found for {target_year}. Run `trnscrb weekly` first.")
        return

    combined = ""
    for f in files:
        text = f.read_text(encoding="utf-8")
        combined += f"\n{'=' * 40}\n{f.stem}\n{'=' * 40}\n{text}\n"

    prompt_override = None
    if prompt_file:
        prompt_override = Path(prompt_file).read_text(encoding="utf-8")
        click.echo(f"Using prompt template: {prompt_file}")

    provider, profile = get_active_provider_config()
    model_name = str(profile.get("model") or "default")
    prov = provider_label(provider)
    click.echo(f"Found {len(files)} weekly summary(ies). Summarizing with {prov} ({model_name})…")

    try:
        summary = generate_annual_summary(combined, target_year, prompt_override=prompt_override)
    except Exception as e:
        click.echo(f"Annual summary failed: {e}", err=True)
        sys.exit(1)

    click.echo()
    click.echo(summary)

    if save:
        path = storage.NOTES_DIR / f"annual-{target_year}.txt"
        storage.save_transcript(path, summary)
        click.echo(f"\nSaved to {path}")


# ── devices ───────────────────────────────────────────────────────────────────


@cli.command()
def icons():
    """Generate menu bar icons (mic PNG). Run once after install."""
    from trnscrb.icon import generate_icons_cli

    generate_icons_cli()


@cli.command(name="mic-status")
def mic_status():
    """Check live mic activity and which meeting app is detected."""
    import time

    from trnscrb.watcher import GRACE_SECS, WARMUP_SECS, detect_meeting, is_mic_in_use

    active = is_mic_in_use()
    status = click.style("IN USE 🔴", fg="red") if active else click.style("idle  ⚪", fg="white")
    click.echo(f"\n  Microphone: {status}")

    if active:
        click.echo(f"  Detected app: {detect_meeting()}")
    click.echo(
        f"\n  Watcher thresholds: warmup={WARMUP_SECS}s  grace={GRACE_SECS}s  min_save={30}s"
    )
    click.echo()
    click.echo("  Watching for 10 seconds (press Ctrl-C to stop early)…")
    for i in range(10):
        time.sleep(1)
        active = is_mic_in_use()
        mark = "🔴" if active else "⚪"
        click.echo(f"  {i + 1:2d}s  {mark}", nl=True)
    click.echo()


@cli.command()
def devices():
    """List available audio input devices."""
    from trnscrb.recorder import Recorder

    devs = Recorder.list_input_devices()
    if not devs:
        click.echo("No input devices found.")
        return
    for d in devs:
        tag = "  (BlackHole)" if "BlackHole" in d["name"] else ""
        click.echo(f"  [{d['index']}] {d['name']}  {d['channels']}ch{tag}")


# ── helpers ───────────────────────────────────────────────────────────────────


def _row(label: str, ok: bool, detail: str = "", indent: int = 2):
    mark = click.style("✓", fg="green") if ok else click.style("✗", fg="red")
    status = click.style("ok", fg="green") if ok else click.style("missing", fg="yellow")
    pad = " " * indent
    click.echo(f"{pad}{mark} {label:<30} {status}  {detail}")


def _pkg_installed(import_name: str) -> bool:
    return importlib.util.find_spec(import_name.split(".")[0]) is not None


def _blackhole_installed() -> bool:
    try:
        import sounddevice as sd

        return any(
            "BlackHole" in d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0
        )
    except Exception:
        return False


def _get_hf_token() -> str | None:
    import os

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip() or None
    return None


def _save_hf_token(token: str):
    d = Path.home() / ".cache" / "huggingface"
    d.mkdir(parents=True, exist_ok=True)
    (d / "token").write_text(token)


def _whisper_model_cached(size: str) -> bool:
    # faster-whisper stores models under ~/.cache/huggingface/hub/models--Systran--faster-whisper-*
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if any(hf_hub.glob(f"models--Systran--faster-whisper-{size}")):
        return True
    # also check ct2 local cache
    ct2_cache = Path.home() / ".cache" / "faster_whisper"
    return ct2_cache.exists() and any(ct2_cache.glob(f"*{size}*"))


def _parakeet_model_cached(model_id: str) -> bool:
    # huggingface cache paths use models--org--repo naming
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    cache_prefix = "models--" + model_id.replace("/", "--")
    return any(hf_hub.glob(f"{cache_prefix}*"))


def _normalize_backend(value) -> str:
    backend = str(value or "parakeet").strip().lower()
    if backend in {"parakeet", "whisper"}:
        return backend
    return "parakeet"


def _mcp_configured() -> bool:
    if not _CLAUDE_CONFIG.exists():
        return False
    try:
        config = json.loads(_CLAUDE_CONFIG.read_text())
        return "trnscrb" in config.get("mcpServers", {})
    except Exception:
        return False


def _write_mcp_config():
    config: dict = {}
    if _CLAUDE_CONFIG.exists():
        try:
            config = json.loads(_CLAUDE_CONFIG.read_text())
        except Exception:
            pass
    # Prefer the installed binary on PATH; fall back to python -m
    import shutil

    binary = shutil.which("trnscrb") or sys.executable
    if binary.endswith("trnscrb"):
        cmd_entry = {"command": binary, "args": ["server"]}
    else:
        cmd_entry = {"command": binary, "args": ["-m", "trnscrb.mcp_server"]}

    config.setdefault("mcpServers", {})
    config["mcpServers"]["trnscrb"] = cmd_entry
    _CLAUDE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CLAUDE_CONFIG.write_text(json.dumps(config, indent=2))


def _run(cmd: list[str]):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"  Command failed: {e}", fg="yellow"))
    except FileNotFoundError:
        click.echo(click.style(f"  Not found: {cmd[0]}", fg="yellow"))


# ── permission helpers ────────────────────────────────────────────────────────


def _request_mic_permission() -> None:
    """Briefly open the audio input stream to trigger the macOS mic permission dialog."""
    try:
        import time

        import sounddevice as sd

        stream = sd.InputStream(channels=1, samplerate=16000, dtype="float32")
        stream.start()
        try:
            time.sleep(0.3)
        finally:
            stream.stop()
            stream.close()
        click.echo(click.style("    ✓ Microphone access granted", fg="green"))
    except Exception as e:
        click.echo(click.style(f"    ⚠  Microphone: {e}", fg="yellow"))


def _request_calendar_permission() -> None:
    """Call Calendar via AppleScript to trigger the macOS calendar permission dialog."""
    try:
        from trnscrb.calendar_integration import get_current_or_upcoming_event

        get_current_or_upcoming_event()
        click.echo(click.style("    ✓ Calendar access granted (or skipped)", fg="green"))
    except Exception as e:
        click.echo(click.style(f"    ⚠  Calendar: {e}", fg="yellow"))


# ── login item helpers ────────────────────────────────────────────────────────

_LAUNCH_AGENT_LABEL = "io.trnscrb.app"
_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCH_AGENT_LABEL}.plist"


def _login_item_exists() -> bool:
    return _PLIST_PATH.exists()


def _setup_login_item(binary_path: str) -> bool:
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_LAUNCH_AGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary_path}</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DYLD_LIBRARY_PATH</key>
        <string>/opt/homebrew/lib:/usr/local/lib</string>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/trnscrb.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/trnscrb.err</string>
</dict>
</plist>
"""
    try:
        _PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PLIST_PATH.write_text(plist)
        # Load it for the current session (unload first in case it was previously loaded)
        subprocess.run(
            ["launchctl", "unload", str(_PLIST_PATH)],
            capture_output=True,
        )
        subprocess.run(
            ["launchctl", "load", str(_PLIST_PATH)],
            capture_output=True,
            check=True,
        )
        return True
    except Exception:
        return False
