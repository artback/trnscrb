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

`trnscrb install` handles the system-audio (Screen Recording) permission, model downloads, Claude Desktop MCP config, and launch-at-login. It also creates a `~/Applications/Trnscrb.app` wrapper so macOS permission prompts are attributed to **Trnscrb** rather than your terminal.

---

## Quick start

```bash
trnscrb start       # launch the menu bar app
```

With **Auto-transcribe** on (the default), trnscrb detects when a meeting starts and begins recording automatically. When the meeting ends, it transcribes and saves to `~/meeting-notes/`.

A meeting ends when the app or tab closes, or after 15 minutes with nobody speaking — a Meet tab left open after the call looks exactly like a live one, and without the silence rule it would keep the recording running until the browser quits, folding the rest of the day's calls into the same file. Change the wait with `trnscrb config set quiet_stop_minutes 10`, or set it to `0` to rely on the tab alone.

**During a call**, click **Open Latest** in the menu bar to read the live transcript — it updates every 60 seconds. Or stream it to your terminal:

```bash
trnscrb live        # tail the live transcript as it updates
```

---

## Transcription backends

| Backend | Language | Speed | Model size | Best for |
|---------|----------|-------|------------|----------|
| **auto** (default) | All | Fast | ~1.8 GB | Parakeet for English, Qwen3 otherwise |
| Parakeet | English + 24 EU languages | Fastest | ~600 MB | English-only teams |
| Qwen3 | 52 languages | Fast | ~1.2 GB | Multilingual teams |
| Whisper | 99 languages | Fast | ~500 MB | Legacy / fallback |
| Voxtral | Multilingual | Slower | ~6 GB | Experimental |

Speaker labels use pyannote **community-1** (falls back to 3.1) — accept its terms once at [hf.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). The repo is gated, so an HF token on its own is not enough; `trnscrb status` reports whether labelling will actually work, not just whether a token is set.

**Auto mode** detects the spoken language and routes English to Parakeet (best accuracy) and everything else to Whisper.

If the chosen backend is unavailable — an uninstalled dependency, a model that was never downloaded — trnscrb falls back to Whisper rather than failing the transcription. Any recording that still ends up without a transcript keeps its audio, is retried the next time the app starts, and is never deleted by retention. `trnscrb status` shows the count, and `trnscrb retry` works through them on demand.

```json
{
  "transcription_backend": "auto"
}
```

---

## Live transcription

During recording, trnscrb transcribes what it has so far every 60 seconds and writes it to the transcript file. You can read along during the call:

- **Menu bar** — click "Open Latest" to open the transcript in your default editor
- **Terminal** — `trnscrb live` streams new content as it appears
- **Claude Desktop** — use `get_transcript` on the latest ID

When the call ends, the final full transcription (with diarization and enrichment) replaces the live version.

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
trnscrb search "who owns billing" --semantic  # search by meaning, not keywords
```

Also available as MCP tools (`search_transcripts`, `semantic_search`) in Claude Desktop.

---

## Custom vocabulary

Teach trnscrb the names, product terms, and acronyms your meetings actually use, so they land correctly in the transcript itself rather than needing a find-and-replace afterwards:

```bash
trnscrb glossary add Hivenet --alias "high vnet" --alias "hive net"
trnscrb glossary list
trnscrb glossary remove Hivenet
```

Aliases are rewritten to the canonical term (with canonical casing) as each segment is transcribed, so the saved transcript already carries your terminology — this isn't a post-hoc edit or part of enrichment. With `glossary_fuzzy` on (the default), single tokens that are close-spelling matches to a term get nudged onto it too. On the Whisper backend, glossary terms are also handed to the model as decode hotwords; Parakeet has no such hook, so correction is doing all the work there.

Also available as MCP tools in Claude Desktop: `list_glossary`, `add_glossary_terms`, `add_glossary_correction`, `remove_glossary_term`, `suggest_glossary_terms`.

---

## Speaker & voice identities

Diarization labels speakers `SPEAKER_00`, `SPEAKER_01`, etc. trnscrb can learn what those voices actually sound like and carry names across meetings:

```bash
trnscrb voices                        # list learned identities
trnscrb voices --label                # play each unnamed voice and name it interactively
trnscrb voices --name SPEAKER_02 Sara # name one directly
trnscrb voices --verbose              # show which meetings each voice appeared in
trnscrb voices --forget SPEAKER_02    # delete an identity
```

Your own voice is learned by default (`learn_my_voice`) — the mic stream identifies you unambiguously, and it's your voice to keep. Everyone else's voice is **not** clustered by default (`cluster_voices` is off): these are biometric fingerprints of people who haven't consented to being enrolled. Turn it on with `trnscrb config set cluster_voices true` once that's a call you're comfortable making for your team. Naming a voice applies retroactively to every past meeting it appeared in, not just the current one.

If you have an exported Google Meet transcript (Docs → File → Download → Plain text), `trnscrb import-meet <file>` matches its speaker names onto trnscrb's own transcript and voiceprints, and suggests glossary terms from words Meet's captions heard differently:

```bash
trnscrb import-meet meet-transcript.txt --apply-glossary
```

---

## Action items & Obsidian

trnscrb tracks *your own* commitments — not tasks assigned to someone else in a standup — across meetings:

```bash
trnscrb tasks            # open action items
trnscrb tasks --all      # include completed ones
```

With `track_action_items` on (the default) and an Obsidian vault available, each meeting is also mirrored into the vault as a note, and open items are kept in sync both ways with an `Action Items.md` note using Tasks-plugin checkboxes and `[[backlinks]]` — ticking a box in Obsidian marks the item done in trnscrb too. The vault is auto-detected from Obsidian's own config, or set explicitly:

```bash
trnscrb config set obsidian_vault ~/Documents/Notes
trnscrb config set obsidian_subdir Meetings   # default
```

`trnscrb vault-sync` refreshes existing notes with the latest attendee/topic properties (worth re-running after glossary changes, since topics are derived from it); `--all` also mirrors transcripts that don't have a note yet, so it can't quietly flood a personal vault by default. Separately, `auto_integrate` (off by default) hands each transcript to the local `claude` CLI to fold key decisions and action items straight into your existing notes — configure its prompt with `trnscrb config set integrate_prompt "..."`.

Also available as MCP tools in Claude Desktop: `list_action_items`, `add_action_item`, `resolve_action_item`, `link_action_item`.

---

## Configuration

```bash
trnscrb config list              # every setting and its current value
trnscrb config get quiet_stop_minutes
trnscrb config set quiet_stop_minutes 10
```

Settings worth knowing about beyond the ones covered above:

| Setting | Default | What it does |
|---------|---------|---------------|
| `auto_record` | `true` | Start watching for mic activity on launch |
| `auto_enrich` | `true` | Summary + action items after each recording (no-op if no LLM is reachable) |
| `retention_audio_days` | `30` | Delete *preserved* audio (from a failed transcription) after N days; `0` keeps forever |
| `retention_transcript_days` | `0` | Delete transcripts after N days; `0` keeps forever |
| `voice_match_threshold` / `voice_match_margin` | `0.75` / `0.10` | How confidently two recordings must match to be treated as the same person — tuned to favor two identities for one person over merging two people into one |
| `mlx_cache_limit_mb` | `512` | Cap on MLX's GPU buffer cache; `0` disables the cap |
| `user_name` | *(macOS username)* | Your display name in meetings, used to tell which action items are yours |

---

## CLI

```bash
trnscrb start            # launch menu bar app
trnscrb install          # guided setup
trnscrb watch            # headless auto-transcribe (no menu bar)
trnscrb live             # stream live transcript to terminal
trnscrb toggle           # start/stop recording in the running app (bind to a hotkey)
trnscrb bookmark [label] # mark this moment in the running recording

trnscrb list             # list saved transcripts
trnscrb show <id>        # print a transcript
trnscrb search <query>   # full-text search across all transcripts (--semantic for meaning-based)
trnscrb transcribe <wav> # transcribe a saved recording
trnscrb retry            # transcribe every recording still missing a transcript
trnscrb import-meet <f>  # name speakers from an exported Google Meet transcript

trnscrb enrich <id>      # add summary + action items
trnscrb weekly           # weekly summary from transcripts
trnscrb annual           # annual summary from weekly summaries
trnscrb tasks            # list tracked action items
trnscrb vault-sync       # refresh Obsidian notes from current transcripts

trnscrb glossary list|add|remove   # manage custom vocabulary
trnscrb voices                     # list/name/forget learned voice identities
trnscrb config list|get|set        # inspect/change settings

trnscrb mic-status       # live mic activity monitor
trnscrb devices          # list audio input devices
trnscrb status           # health check across recording, diarization, MCP, etc.
trnscrb doctor           # run the speaker-labelling stack end to end
trnscrb icons            # regenerate menu bar icons (run once after install)
```

### When speaker labels stop working

A meeting that transcribes but fails to diarize looks like a success: the transcript saves, and the only trace is one line in a log. `trnscrb status` reports what actually happened the last time diarization ran — "failing since 2026-08-14 (9 meetings)" rather than a ✓ because the model file is on disk — and the menu bar carries the same standing state.

`trnscrb doctor` is the one that tells you *where* it broke: it decodes audio, loads the pipeline and diarizes a clip, checking each stage in turn. `--quick` skips the model load. Once it passes, the recorded failure clears.

When diarization fails, the recording is kept instead of deleted, so the speaker labels and voiceprints for that meeting are not lost — redo it with `trnscrb transcribe <wav>` once `doctor` is green.

`trnscrb transcribe` and `trnscrb retry` enrol voices as well (when `cluster_voices` is on), so a meeting recovered after the fact still feeds the voice store. They cannot name your own voice: identifying it needs the microphone and the system audio kept apart, and a saved recording is already mixed. Your voice still lands on the right identity when it matches one already learned.

### If the app keeps restarting

launchd restarts a failing job every 10 seconds and never gives up, so anything that kills trnscrb during startup turns into a silent loop. The app counts its own starts: five inside two minutes and it stops instead, exits cleanly so launchd leaves it alone, and records why. `trnscrb status` then shows an **App startup** row, and the log names the exit status — `128 + signal`, so 138 is SIGBUS and 137 is a kill. Start it again with `trnscrb start` once the cause is fixed; the guard resets itself.

---

## Claude Desktop / MCP tools

After `trnscrb install`, Claude Desktop has these tools:

| Tool | Description |
|------|-------------|
| `start_recording` | Start capturing audio |
| `stop_recording` | Stop and transcribe in background |
| `recording_status` | Check recording/transcription status |
| `get_last_transcript` | Most recent transcript |
| `get_current_transcript` | Live transcript of a recording in progress |
| `list_transcripts` | All saved meetings |
| `get_transcript` | Read a specific transcript |
| `search_transcripts` | Full-text search across transcripts |
| `semantic_search` | Meaning-based search across transcripts |
| `get_weekly_transcripts` | All transcripts for a given week |
| `get_weekly_summaries` | All weekly summaries for a year |
| `get_calendar_context` | Current/upcoming calendar event |
| `enrich_transcript` | Summary + action items via LLM |
| `list_glossary` | Show all glossary terms |
| `add_glossary_terms` | Add custom vocabulary terms |
| `add_glossary_correction` | Add a mis-heard → correct spelling pair |
| `remove_glossary_term` | Remove a glossary term |
| `suggest_glossary_terms` | Suggest terms from recent transcripts |
| `list_action_items` | Tracked action items (open or all) |
| `add_action_item` | Add an action item |
| `resolve_action_item` | Mark an action item done |
| `link_action_item` | Attach a Jira/GitHub reference to an item |

---

## System audio setup

trnscrb captures both your mic and other participants' audio natively via **ScreenCaptureKit** — no virtual audio driver or Multi-Output Device needed.

**Upgrading to 0.51.0 asks for Screen Recording once more.** System audio now carries its capture timestamp so it can be laid down beside the microphone at the moment it happened rather than the moment it arrived — without that, anything coming out of your speakers is recorded twice, once as mic bleed and once through the capture path a few hundred milliseconds later. The fix changes the capture binary, and macOS ties the grant to that binary's signature.

The only requirement is the **Screen Recording** permission (macOS 15+ shows it as "Screen & System Audio Recording"). `trnscrb install` requests it, or grant it manually under **System Settings → Privacy & Security**. Without it, only your mic is recorded.

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

Everything runs locally. Enrichment sends transcript text to whichever LLM provider you configure (local by default, via llama.cpp) — swap in Claude Code, Ollama, LM Studio, or a hosted Anthropic/OpenAI key from menu bar Settings.

---

## License

MIT
