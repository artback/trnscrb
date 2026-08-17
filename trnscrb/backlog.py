"""Retry transcription for audio that was preserved after a failure.

When transcription fails the WAV is kept instead of discarded (see
`storage.preserve_audio`), and an interrupted session leaves its audio behind
too. Nothing used to revisit those files: they sat un-transcribed until
retention deleted them. This module is the missing second half — it finds
preserved audio with no transcript and transcribes it.

Runs at app startup, so a backend that was broken during a meeting (a missing
dependency, a model that had not been downloaded yet) costs a delay rather
than the meeting.
"""

from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.backlog")


def finalize_wav_header(path: Path) -> None:
    """Repair the placeholder header a killed recorder leaves behind.

    No-op on a WAV that was closed properly, so this is safe to call on any
    file before reading it.
    """
    from trnscrb.recorder import SAMPLE_RATE, _wav_header

    with open(path, "r+b") as f:
        if f.read(4) == b"RIFF":
            return
        f.seek(0)
        f.write(_wav_header(SAMPLE_RATE, 1, path.stat().st_size - 44))
    _log.info("Finalized interrupted WAV header: %s", path.name)


def _learn_voices(turns: list[dict], embeddings: dict, meeting: str, audio_file: Path) -> None:
    """Cluster this recording's speakers into the persistent identities.

    Only the clustering half of what the live path does. Identifying the user
    needs the microphone and the system audio kept apart, and a file on disk
    is already mixed — so "Me" is never asserted here. The user's own voice
    still lands on the right identity when it *matches* one already known,
    which is what recovering a meeting after the fact can honestly claim.

    Gated on `cluster_voices` like the live path: these are fingerprints of
    people who did not consent to being enrolled. Never fails the
    transcription.
    """
    from trnscrb import diarizer, health, voiceprints
    from trnscrb.settings import get as get_setting

    if not embeddings or not get_setting("cluster_voices"):
        return
    try:
        learned = voiceprints.enrol(
            embeddings,
            turns,
            model=diarizer.pipeline_id(),
            space=diarizer.embedding_space(),
            meeting=meeting,
            cluster_others=True,
            audio_path=audio_file,
        )
        ok, detail = voiceprints.enrolment_health(learned, turns)
        (health.record_ok if ok else health.record_failure)(health.VOICE_ENROLMENT, detail, meeting)
        if learned:
            _log.info("Enrolled %d voice(s) from %s", len(learned), audio_file.name)
    except Exception:
        _log.warning("Voice enrolment failed for %s", audio_file.name, exc_info=True)


def transcribe_file(audio_file: Path, meeting_name: str = "") -> tuple[Path, int]:
    """Transcribe one WAV and save the transcript. Returns (path, segment count).

    Shared by `trnscrb transcribe` and the startup retry pass so both produce
    identical output — and so the transcript lands where `has_transcript`
    looks for it.
    """
    from trnscrb import diarizer, health, storage, transcriber
    from trnscrb.settings import read_hf_token

    finalize_wav_header(audio_file)
    parsed_name, started_at = storage.meeting_from_filename(audio_file)
    meeting_name = meeting_name or parsed_name

    segments = transcriber.transcribe(audio_file)

    hf_token = read_hf_token()
    if hf_token and segments:
        try:
            diar, embeddings = diarizer.diarize_with_embeddings(audio_file, hf_token)
            segments = diarizer.merge(segments, diar)
            health.record_ok(
                health.DIARIZATION, f"{len({t['speaker'] for t in diar})} speaker(s)", meeting_name
            )
            _learn_voices(diar, embeddings, meeting_name, audio_file)
        except Exception as e:
            health.record_failure(health.DIARIZATION, e, meeting_name)

    text = storage.format_transcript(segments, started_at, meeting_name)
    out = storage.get_transcript_path(meeting_name, started_at)
    storage.save_transcript(out, text)
    return out, len(segments)


def process_pending(notes_dir: Path | None = None) -> list[Path]:
    """Transcribe every preserved recording that still has no transcript.

    Stops early if a recording starts, so a live meeting is never left waiting
    behind hours of catch-up work on the shared inference thread; whatever is
    left is picked up on the next run. Returns the transcripts written.
    """
    from trnscrb import storage

    pending = storage.pending_audio(notes_dir)
    if not pending:
        return []

    _log.info("Retrying %d preserved recording(s) with no transcript", len(pending))
    written: list[Path] = []
    for audio_file in pending:
        if storage.get_live_session() is not None:
            _log.info(
                "Recording started — deferring %d remaining file(s)", len(pending) - len(written)
            )
            break
        try:
            out, count = transcribe_file(audio_file)
            written.append(out)
            _log.info(
                "Backlog transcribed %s -> %s (%d segments)", audio_file.name, out.name, count
            )
        except Exception as e:
            # Leave the audio in place; the next run tries again.
            _log.error("Backlog transcription failed for %s: %s", audio_file.name, e)
    return written
