"""Keep the test suite off the user's real data.

Two escapes so far, both the same shape: a module-level path pointing into
the user's home, and a test that writes through it. One run rewrote
~/.config/trnscrb/health.json; another called forget("voice-1") and deleted a
real voice clip — the only copy, since the meeting audio it was cut from is
normally gone by then.

Both were fixed by remembering to patch one more constant in one more test
class, which is not a fix: it works until the next test, or the next
constant. This redirects every path that leads to the user's own data, for
every test, whether or not the test knows it needs it. Tests that patch these
themselves still win — an autouse fixture is applied before the test body.

Deliberately not redirected: the log file (writing there is harmless and
useful) and the HuggingFace cache (read-only, and a redirect would trigger
gigabyte downloads).
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_data(tmp_path, monkeypatch):
    from trnscrb import health, icon, settings, storage, voiceprints

    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(voiceprints, "STORE", config / "voiceprints.json")
    monkeypatch.setattr(voiceprints, "SAMPLES_DIR", config / "voice-samples")
    monkeypatch.setattr(health, "STORE", config / "health.json")
    monkeypatch.setattr(settings, "_SETTINGS_FILE", config / "settings.json")
    monkeypatch.setattr(storage, "_APP_STATE_FILE", config / "app_state.json")
    monkeypatch.setattr(storage, "NOTES_DIR", tmp_path / "meeting-notes")
    monkeypatch.setattr(icon, "ICON_DIR", tmp_path / "icons")
    monkeypatch.setattr(icon, "ICON_IDLE", tmp_path / "icons" / "mic.png")
    monkeypatch.setattr(icon, "ICON_RECORDING", tmp_path / "icons" / "mic_active.png")
