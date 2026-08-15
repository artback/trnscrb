"""Whether each part of the pipeline actually worked, remembered across runs.

A meeting that transcribes but fails to diarize looks like a success. The
transcript saves, the notification is about the transcript, and the only
trace of the failure is one WARNING line in a log nobody reads. That is how
speaker labels stayed broken for six days and nine meetings while
`trnscrb status` called them healthy — it was checking that a model file
existed on disk, which it did, rather than that anything had ever run.

Checking for the parts is not checking that the machine turns over. This
module records what happened the last time each component actually ran, and
how long it has been failing, so `trnscrb status`, `trnscrb doctor` and the
menu bar can answer "is this working?" from evidence.

Nothing here may raise: a component reporting its own health must never be
the thing that breaks the meeting.
"""

import json
from datetime import datetime
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.health")

STORE = Path.home() / ".config" / "trnscrb" / "health.json"
_VERSION = 1

# Components worth remembering the health of. Each is something that can fail
# on its own while everything around it still looks fine.
DIARIZATION = "diarization"
VOICE_ENROLMENT = "voice_enrolment"
TRANSCRIPTION = "transcription"

LABELS = {
    DIARIZATION: "Speaker labels",
    VOICE_ENROLMENT: "Voice identities",
    TRANSCRIPTION: "Transcription",
}

# Failure text is for a human reading a status line, not a stack trace.
_MAX_DETAIL = 300


def _empty() -> dict:
    return {"version": _VERSION, "components": {}}


def load() -> dict:
    """The store, or an empty one when absent, unreadable, or from the future."""
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    if data.get("version") != _VERSION or not isinstance(data.get("components"), dict):
        return _empty()
    return data


def _save(data: dict) -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        _log.debug("Could not write the health store", exc_info=True)


def _record(component: str, ok: bool, detail: str, meeting: str) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    data = load()
    entry = dict(data["components"].get(component) or {})

    entry["ok"] = ok
    entry["detail"] = str(detail)[:_MAX_DETAIL]
    entry["at"] = now
    entry["meeting"] = meeting
    entry["runs"] = int(entry.get("runs", 0)) + 1
    if ok:
        entry["failures"] = 0
        entry["failing_since"] = ""
        entry["last_ok_at"] = now
    else:
        # The streak is the number that matters: one failure is a bad day,
        # nine in a row is a broken install nobody was told about.
        entry["failures"] = int(entry.get("failures", 0)) + 1
        entry.setdefault("last_ok_at", "")
        if not entry.get("failing_since"):
            entry["failing_since"] = now

    data["components"][component] = entry
    _save(data)
    return entry


def record_ok(component: str, detail: str = "", meeting: str = "") -> dict:
    """Note that this component ran and worked."""
    try:
        return _record(component, True, detail, meeting)
    except Exception:
        _log.debug("Could not record health for %s", component, exc_info=True)
        return {}


def record_failure(component: str, error, meeting: str = "") -> dict:
    """Note that this component ran and failed. Returns the entry, streak included."""
    try:
        entry = _record(component, False, str(error), meeting)
    except Exception:
        _log.debug("Could not record health for %s", component, exc_info=True)
        return {}
    _log.warning(
        "%s failed (%d in a row since %s): %s",
        LABELS.get(component, component),
        entry.get("failures", 1),
        entry.get("failing_since", "?"),
        entry.get("detail", ""),
    )
    return entry


def get(component: str) -> dict | None:
    """What happened the last time this component ran, or None if it never has."""
    return (load().get("components") or {}).get(component)


def unhealthy() -> list[tuple[str, dict]]:
    """(component, entry) for everything whose last run failed."""
    return [
        (name, entry)
        for name, entry in sorted((load().get("components") or {}).items())
        if not entry.get("ok", True)
    ]


def describe(component: str) -> str:
    """One line about this component's last run, for a status row.

    Says how long a failure has been going on rather than just that it
    happened: "broken since Tuesday, 9 meetings" is a different problem from
    "failed once this afternoon", and the log line looks identical for both.
    """
    entry = get(component)
    if not entry:
        return "never run"
    when = str(entry.get("at", ""))[:16].replace("T", " ")
    if entry.get("ok"):
        detail = entry.get("detail") or "ok"
        return f"{detail} (last ran {when})"

    failures = int(entry.get("failures", 1))
    since = str(entry.get("failing_since", ""))[:10]
    run_word = "meeting" if failures == 1 else "meetings"
    return f"failing since {since} ({failures} {run_word}): {entry.get('detail', '')}"


def should_notify(entry: dict) -> bool:
    """Whether this failure is worth interrupting the user over.

    The first one, then every fifth. A notification per meeting for a
    component that has been broken all week is noise the user learns to
    dismiss, and the menu bar carries the standing state anyway.
    """
    failures = int(entry.get("failures", 0))
    return failures == 1 or (failures > 0 and failures % 5 == 0)


def clear(component: str = "") -> None:
    """Forget one component's history, or all of it."""
    data = load()
    if component:
        data["components"].pop(component, None)
    else:
        data = _empty()
    _save(data)
