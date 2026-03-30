# trnscrb

> Offline meeting transcription for macOS — no cloud, no subscription.

trnscrb lives in your menu bar, auto-detects meetings (Google Meet, Zoom, Teams, Slack, FaceTime), records and transcribes them locally, cleans up filler words, and makes every transcript searchable from Claude Desktop.

---

## Install

```bash
brew tap artback/trnscrb
brew install trnscrb
trnscrb install
```

Or with `uv`:

```bash
uv tool install trnscrb && trnscrb install
```

`trnscrb install` handles BlackHole audio driver, model downloads, Claude Desktop MCP config, and launch-at-login.

---

## Quick start

```bash
trnscrb start       # launch the menu bar app
```

With **Auto-transcribe** on (the default), trnscrb detects when a meeting starts and begins recording automatically. When the meeting ends, it transcribes and saves to `~/meeting-notes/`.

---

## Transcription backends

| Backend | Language | Speed | Model size | Best for |
|---------|----------|-------|------------|----------|
| **auto** (default) | All | Fast | ~1.1 GB | Best of both worlds |
| Parakeet | English only | Fastest | ~600 MB | English-only teams |
| Whisper | 99 languages | Fast | ~500 MB | Multilingual teams |
| Voxtral | Multilingual | Slower | ~6 GB | Experimental |

**Auto mode** detects the spoken language and routes English to Parakeet (best accuracy) and everything else to Whisper.

```json
{
  "transcription_backend": "auto"
}
```

---

## Meeting detection

trnscrb detects active meetings through multiple signals:

- **Browser tabs** — Google Meet, Teams, Zoom in Chrome/Safari
- **Native apps** — Zoom (CptHost/caphost), FaceTime, Tuple via CoreAudio mic check
- **Teams desktop** — window count detection (2 windows = active call)
- **Calendar** — macOS Calendar integration for meeting names

Detection runs in parallel for minimal latency. Muting doesn't stop the recording — only leaving the meeting does.

---

## Enrichment

After transcription, enrich with an LLM to get summaries, action items, and speaker name inference.

**Providers** (configured in menu bar Settings):

| Provider | Setup |
|----------|-------|
| **Claude Code** | No config needed — uses local `claude` CLI |
| Ollama | `http://127.0.0.1:11434` |
| llama.cpp | `http://127.0.0.1:8080` |
| LM Studio | `http://127.0.0.1:1234` |
| Anthropic | API key required |
| OpenAI | API key required |

---

## Weekly & annual summaries

Generate summaries from your meeting transcripts — useful for performance reviews.

```bash
trnscrb weekly                      # summarize last week's meetings
trnscrb weekly --week 2026-W13      # specific week
trnscrb weekly --prompt template.md # custom prompt
trnscrb annual                      # compile weekly summaries into annual review
```

Custom prompt templates can also be placed in `~/.config/trnscrb/prompts/weekly.md`.

---

## Search

```bash
trnscrb search "auth migration"     # search all transcripts
trnscrb search "Miguel" -n 3        # with context lines
```

Also available as an MCP tool (`search_transcripts`) in Claude Desktop.

---

## CLI

```bash
trnscrb start            # launch menu bar app
trnscrb install          # guided setup
trnscrb watch            # headless auto-transcribe (no menu bar)
trnscrb list             # list saved transcripts
trnscrb show <id>        # print a transcript
trnscrb search <query>   # full-text search across all transcripts
trnscrb enrich <id>      # add summary + action items
trnscrb weekly           # weekly summary from transcripts
trnscrb annual           # annual summary from weekly summaries
trnscrb mic-status       # live mic activity monitor
trnscrb devices          # list audio input devices
```

---

## Claude Desktop / MCP tools

After `trnscrb install`, Claude Desktop has these tools:

| Tool | Description |
|------|-------------|
| `start_recording` | Start capturing audio |
| `stop_recording` | Stop and transcribe in background |
| `recording_status` | Check recording/transcription status |
| `get_last_transcript` | Most recent transcript |
| `list_transcripts` | All saved meetings |
| `get_transcript` | Read a specific transcript |
| `search_transcripts` | Full-text search across transcripts |
| `get_weekly_transcripts` | All transcripts for a given week |
| `get_weekly_summaries` | All weekly summaries for a year |
| `get_calendar_context` | Current/upcoming calendar event |
| `enrich_transcript` | Summary + action items via LLM |

---

## System audio setup

To capture both your mic and other participants' audio:

1. Install BlackHole via `trnscrb install` (or `brew install blackhole-2ch`)
2. Open **Audio MIDI Setup** → create a **Multi-Output Device** with BlackHole + speakers
3. Set the Multi-Output Device as your system output

trnscrb auto-detects BlackHole or an Aggregate Device. Without either, only your mic is recorded.

---

## Transcript format

```
Meeting: Weekly Standup
Date:    2026-03-23 10:00
Duration: 23:14

============================================================

[Alice]
  00:12  Good morning, let's get started.

[Bob]
  00:18  Morning! I finished the auth PR yesterday.
```

Filler words (um, uh, like, basically, etc.) are automatically removed in 5 languages.

---

## Requirements

- macOS 13+
- Python 3.12+
- Apple Silicon recommended for fastest local transcription

---

## Privacy

Everything runs locally. Enrichment sends transcript text to whichever LLM provider you configure (local by default with Claude Code or Ollama).

---

## License

MIT
