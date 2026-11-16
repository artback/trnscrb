"""Background watcher: samples the call app's AX tree during a meeting recording.

Started on meeting start, stopped on meeting end.  Collects samples in
memory; on stop, runs temporal attribution (via ``attribute.py``) to assign
names to transcript segments.

Usage (menu_bar.py integration):
    watch = CallNamesWatch(bundle_ids=["us.zoom.xos"], app_name="Zoom")
    watch.start()           # in _do_start
    ...
    result = watch.stop_and_attr(segments, my_names)  # in _process_audio

Graceful stop: the background thread is joined with a timeout.
Failure in this module never breaks recording or transcription.
"""

from __future__ import annotations

import threading
import time

from trnscrb.callnames import ax as _ax_mod
from trnscrb.callnames import attribute as _attr_mod
from trnscrb.callnames import parse as _parse_mod
from trnscrb.log import get_logger

_log = get_logger("trnscrb.callnames.watch")

# ── App → bundle ID map ──────────────────────────────────────────────────────

_APP_BUNDLES: dict[str, list[str]] = {
    "Zoom": ["us.zoom.xos", "zoom"],
    "Microsoft Teams": ["com.microsoft.teams2", "com.microsoft.teams"],
    "Google Meet": [
        "com.google.Chrome", "company.thebrowser.Browser",
        "com.brave.Browser", "com.microsoft.edgemac",
        "com.apple.Safari", "org.mozilla.firefox",
    ],
    "Webex": ["com.cisco.webexmeetingsapp"],
}


class CallNamesWatch:
    """Samples the call app's AX tree during a meeting recording.

    Thread-safe.  Start it when recording begins; call ``stop_and_attr()``
    after the transcript segments are available.
    """

    WATCH_EVERY_MS = 400
    APP_LAG_MS = 750  # apps mark who talks ~0.75–1.2s after they start

    def __init__(
        self,
        bundle_ids: list[str],
        app_name: str | None = None,
    ):
        self._bundle_ids = bundle_ids
        self._app_name = app_name
        self._samples: list[_attr_mod.Sample] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._meeting_offset_ms: float = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin sampling on a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._samples = []
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="callnames-watch"
        )
        self._thread.start()
        _log.debug("Call names watch started (app=%s)", self._app_name)

    def stop(self) -> list[_attr_mod.Sample]:
        """Stop sampling and return the collected samples."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                _log.warning("Call names watch thread did not stop cleanly")
        with self._lock:
            return list(self._samples)

    def set_offset(self, ms: float) -> None:
        """Set the meeting time zero offset (from audio capture timing)."""
        self._meeting_offset_ms = ms

    # ── attribution ───────────────────────────────────────────────────────────

    def stop_and_attr(
        self,
        segments: list[dict],
        my_names: list[str],
    ) -> dict | None:
        """Stop sampling, run attribution, and return the result.

        Returns a dict ``{name_map, participants, samples_count}`` or
        ``None`` when there are no samples.

        ``segments`` = list of transcript segment dicts with ``start``,
        ``end``, ``speaker`` keys (as produced by diarizer.merge).
        ``my_names`` = the user's own name(s) for identifying "me".
        """
        raw_samples = self.stop()
        if not raw_samples:
            return None

        try:
            return self.attr_from_samples(raw_samples, segments, my_names)
        except Exception:
            _log.warning("Attribution failed", exc_info=True)
            return None

    def attr_from_samples(
        self,
        raw_samples: list[_attr_mod.Sample],
        segments: list[dict],
        my_names: list[str],
    ) -> dict | None:
        """Run attribution on already-collected samples.

        Same shape as ``stop_and_attr`` but does not stop the watcher.
        """
        if not raw_samples:
            return None

        segs = [
            _attr_mod.Segment(
                start_ms=float(seg.get("start", 0)) * 1000.0,
                end_ms=float(seg.get("end", 0)) * 1000.0,
                speaker=seg.get("speaker", "Unknown"),
            )
            for seg in segments
        ]

        result = _attr_mod.attributed_meeting(
            raw_samples,
            segs,
            my_names=my_names,
            meeting_offset_ms=self._meeting_offset_ms,
        )

        return {
            "name_map": result.result,
            "participants": result.participants,
            "samples_count": len(raw_samples),
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _sample_loop(self) -> None:
        """Background loop: sample every WATCH_EVERY_MS until stopped."""
        bundle_id = self._bundle_ids[0] if self._bundle_ids else None
        next_sample_time = time.monotonic()

        while self._running:
            now = time.monotonic()
            if now < next_sample_time:
                # Use condition wait instead of sleep for responsive stop.
                diff = next_sample_time - now
                if hasattr(self, "_stop_event"):
                    self._stop_event.wait(min(diff, 1.0))
                else:
                    time.sleep(diff)
                continue

            next_sample_time = now + self.WATCH_EVERY_MS / 1000.0

            try:
                snap = _ax_mod.sample(bundle_id=bundle_id, app_name=self._app_name)
                if snap.get("ok") and snap.get("tree"):
                    reading = _parse_mod.parse(
                        self._app_name or "Zoom", snap["tree"]
                    )
                    sample = _attr_mod.Sample(
                        ts=snap["ts"],
                        speaking=reading.speaking,
                        participants=reading.participants,
                        me=reading.me,
                        ok=True,
                    )
                    with self._lock:
                        self._samples.append(sample)
                else:
                    sample = _attr_mod.Sample(
                        ts=time.time(),
                        speaking=[],
                        participants=[],
                        me=None,
                        ok=False,
                    )
                    with self._lock:
                        self._samples.append(sample)
            except Exception:
                _log.debug("AX sample failed", exc_info=True)
                sample = _attr_mod.Sample(
                    ts=time.time(),
                    speaking=[],
                    participants=[],
                    me=None,
                    ok=False,
                )
                with self._lock:
                    self._samples.append(sample)

    def _stop_and_attr(
        self,
        segments: list[dict],
        my_names: list[str],
    ) -> dict | None:
        """Stops the watcher, runs attribution, and returns the result."""
        raw_samples = self.stop()
        if not raw_samples:
            return None
        try:
            return self.attr_from_samples(raw_samples, segments, my_names)
        except Exception:
            _log.warning("Attribution failed", exc_info=True)
            return None


# ── convenience ────────────────────────────────────────────────────────────────

def watch_for(meeting_app: str | None) -> CallNamesWatch:
    """Create a watcher for the given meeting app name.

    Returns ``None`` when the app is not supported.
    """
    bundle_ids = _APP_BUNDLES.get(meeting_app)
    if not bundle_ids:
        _log.debug("No bundle IDs for %s", meeting_app)
        return None
    return CallNamesWatch(bundle_ids=bundle_ids, app_name=meeting_app)