"""Fake mode for testing the full call-names pipeline: watch → attribute → label.
on any platform without a real meeting.

Triggered via ``TRNSCRB_FAKE_CALL_NAMES`` environment variable:
- Set to ``"1"`` or ``"true"`` → uses the built-in canned fixture.
- Set to a file path → loads samples from a JSON file.

The built-in fixture mirrors yogurt's fake.ts:
    Zoom call with 4 people (Ann, Bo, Cy, Sam="me")
    Ann speaks for the first segment, Bo for the second.
"""

from __future__ import annotations

import json
import os
import time

from trnscrb.callnames import attribute as _attr_mod
from trnscrb.log import get_logger

_log = get_logger("trnscrb.callnames.fake")

# ── built-in fixture ──────────────────────────────────────────────────────────

def _fixture_samples() -> list[_attr_mod.Sample]:
    """Canned fixture: Zoom call with Sam Rivera (me), Ann Lee, Bo Park, Cy Dao.

    Wall-clock timestamps spaced ~400ms apart, spanning 30 seconds.
    Ann Lee is marked speaking during the first ~10s; Bo Park during the next ~14s.
    """
    base = time.time() - 30.0  # 30s of meeting history
    samples: list[_attr_mod.Sample] = []
    all_participants = ["Sam Rivera", "Ann Lee", "Bo Park", "Cy Dao"]

    for i in range(75):  # 75 samples × 400ms ≈ 30s
        ts = base + i * 0.4
        if i < 25:  # first 10s
            speaking = ["Ann Lee"]
        elif i < 60:  # next 14s
            speaking = ["Bo Park"]
        else:  # last 6s
            speaking = []

        samples.append(_attr_mod.Sample(
            ts=ts,
            speaking=speaking,
            participants=all_participants[:],
            me="Sam Rivera",
            ok=True,
        ))

    return samples


# ── public API ──────────────────────────────────────────────────────────────────

def _fake_mode() -> bool:
    """Is fake mode enabled?"""
    val = os.environ.get("TRNSCRB_FAKE_CALL_NAMES", "").strip().lower()
    return val in ("1", "true", "yes")


def load_fake_samples() -> list[_attr_mod.Sample] | None:
    """Load fake samples from the env var.

    Returns None when fake mode is not enabled.
    """
    val = os.environ.get("TRNSCRB_FAKE_CALL_NAMES", "").strip()
    if not val:
        return None

    if val.lower() in ("1", "true", "yes"):
        return _fixture_samples()

    # Treat as a file path.
    if os.path.isfile(val):
        try:
            raw = json.loads(open(val).read())
            return _parse_fixture(raw)
        except Exception:
            _log.warning("Could not load fake sample file %s", val, exc_info=True)
            return None

    return None


def _parse_fixture(raw: list[dict]) -> list[_attr_mod.Sample]:
    """Parse a JSON array of sample dicts into ``Sample`` objects."""
    samples: list[_attr_mod.Sample] = []
    for entry in raw:
        samples.append(_attr_mod.Sample(
            ts=float(entry.get("ts", 0)),
            speaking=list(entry.get("speaking", [])),
            participants=list(entry.get("participants", [])),
            me=entry.get("me"),
            ok=bool(entry.get("ok", True)),
        ))
    return samples