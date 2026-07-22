"""Automatic meeting detector and recording trigger.

Uses CoreAudio's kAudioDevicePropertyDeviceIsRunningSomewhere to detect
microphone activity — the same signal that lights up the orange menu-bar dot.
A muted call still records: when the mic is off but a browser/meeting app is
producing audio output (the other participants talking) and a meeting app/tab
is confirmed present, that counts as call activity too.

Two stop conditions (whichever comes first):
  1. Mic goes idle AND meeting app/tab is no longer detected for GRACE_SECS.
     When mic is off but the meeting app is still running (user is muted),
     recording continues to avoid splitting a single call into multiple files.
  2. No meeting app/tab detected for APP_GONE_POLLS consecutive app-checks,
     even if mic is still technically active (handles Chrome keeping mic warm
     after leaving Google Meet).  App is checked every APP_POLL_EVERY mic
     polls so slow osascript doesn't block the mic-poll loop.

State machine:
  idle  ──(mic on 5s)──► warming ──(5s elapsed)──► recording
                             │                          │
                         (mic off)          (mic off AND app gone)
                             │                 OR (meeting gone)
                             ▼                          │
                           idle                      cooling
                                                        │
                                              (5s elapsed + app gone)
                                                        │
                                                  stop + save
"""

import ctypes
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Callable

from trnscrb.log import get_logger

_log = get_logger("trnscrb.watcher")

# ── Timing thresholds ─────────────────────────────────────────────────────────
WARMUP_SECS = 5  # mic must be active this long before we start
GRACE_SECS = 5  # mic must be idle this long before we stop
MIN_SAVE_SECS = 30  # recordings shorter than this are discarded
POLL_SECS = 1.0  # how often we check mic (fast CoreAudio call)
APP_POLL_EVERY = 4  # run the slow meeting-app check every N mic polls (~4s)
APP_GONE_POLLS = 3  # N consecutive app-gone checks → start cooling (~12s)
IDLE_FALLBACK_SECS = 30.0  # safety re-poll while idle in event-driven mode

# ── Meeting app detection ─────────────────────────────────────────────────────
# Used by detect_meeting() at recording START — can be broad because the mic
# activity signal already confirms something real is happening.
_NATIVE_APPS = [
    ("zoom.us", "Zoom"),
    ("Slack Helper (Renderer)", "Slack Huddle"),
    ("Microsoft Teams Helper", "Microsoft Teams"),
    ("Webex", "Webex"),
    ("Around Helper", "Around"),
    ("Tuple", "Tuple"),
    ("Loom", "Loom"),
    ("FaceTime", "FaceTime"),
    ("Discord Helper", "Discord"),
]

# Used by is_meeting_app_running() during STOP detection — must be NARROW.
# "Slack Helper", "Teams Helper", "Discord Helper" etc. are ALWAYS present
# when those apps are open, even when NOT in a meeting → false positives.
# Only list processes that exist exclusively during an active session.
# NOTE: FaceTime is NOT included — it persists as a background process on
# modern macOS even when no call is active.  It is still detected via the
# per-process CoreAudio mic check (step 1) when actually in a call.
_ACTIVE_SESSION_PROCS = [
    "CptHost",  # Zoom: meeting capture host — only present during an active Zoom call
    "caphost",  # Zoom: secondary capture process (newer versions)
    "Tuple",  # Tuple — only runs during an active screen-share session
]

# CoreAudio process-level constants (macOS 14+, verified against
# AudioHardware.h) — the API behind the orange privacy indicator. Lets us see
# which PID is using mic input or producing audio output.
_kProcessObjectList = 0x70727323  # 'prs#' kAudioHardwarePropertyProcessObjectList
_kProcessPID = 0x70706964  # 'ppid' kAudioProcessPropertyPID
_kProcessIsRunningIn = 0x70697269  # 'piri' kAudioProcessPropertyIsRunningInput
_kProcessIsRunningOut = 0x7069726F  # 'piro' kAudioProcessPropertyIsRunningOutput

# ── CoreAudio constants ───────────────────────────────────────────────────────
_kSysObject = 1
_kDefaultInputDevice = 0x64496E20  # 'dIn '
_kDefaultOutputDevice = 0x644F7574  # 'dOut'
_kScopeGlobal = 0x676C6F62  # 'glob'
_kElementMain = 0
_kIsRunningSomewhere = 0x676F6E65  # 'gone' (kAudioDevicePropertyDeviceIsRunningSomewhere)

# While muted in a call, the mic is silent but the meeting app still plays the
# other participants — that output is the "call is live" signal. The slow
# osascript confirmation is throttled to this interval.
OUTPUT_APP_CHECK_SECS = 10.0

# Processes whose audio output counts as meeting audio (matched against
# `ps -ax -o comm=` paths of output-producing PIDs). WebKit covers Safari's
# GPU process, which is what actually plays tab audio.
_MEETING_OUTPUT_PROC_FRAGMENTS = [
    "Google Chrome",
    "Safari",
    "WebKit",
    "firefox",
    "Firefox",
    "zoom.us",
    "Teams",
    "Slack",
    "Discord",
    "Webex",
    "FaceTime",
]


class MicWatcher:
    """
    Polls CoreAudio every POLL_SECS seconds and fires:
      on_start(meeting_name: str)  — when a meeting is confirmed to have started
      on_stop()                    — when the meeting has ended
    """

    def __init__(
        self,
        on_start: Callable[[str], None],
        on_stop: Callable[[], None],
    ):
        self.on_start = on_start
        self.on_stop = on_stop

        self._thread: threading.Thread | None = None
        self._running = False
        self._state = "idle"  # idle | warming | recording | cooling
        self._since: datetime | None = None
        self._rec_started: datetime | None = None
        self._no_app_polls = 0  # consecutive polls without a meeting app
        self._wake = threading.Event()
        self._listener: _MicActivityListener | None = None
        self._event_driven = False
        # Set when the app stops a recording we did not stop ourselves.
        self._reset_requested = threading.Event()
        # True between an app-side stop and the end of that call, so a
        # deliberate Stop is not immediately undone by auto-record.
        self._suppressed = False
        self._last_output_app_check = 0.0  # throttles the muted-call osascript
        self._output_meeting_cached = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._state = "idle"
        self._since = None
        self._no_app_polls = 0
        self._suppressed = False
        # Event-driven idle: CoreAudio wakes us on mic/device changes so the
        # idle loop doesn't have to poll every second. Falls back to polling.
        self._listener = _MicActivityListener(self._wake)
        self._event_driven = self._listener.start()
        _log.info("watcher started (event_driven=%s)", self._event_driven)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()  # unblock an idle wait promptly
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._event_driven = False

    @property
    def is_watching(self) -> bool:
        return self._running

    @property
    def state(self) -> str:
        return self._state

    def notify_recording_stopped(self) -> None:
        """Tell the watcher a recording ended by some route other than us.

        on_start only fires on the warming → recording transition, so a
        watcher left in `recording` after a manual Stop (or any app-side stop)
        can never trigger auto-record again — auto-record silently dies until
        the app restarts. Handled in the loop rather than here so state is
        only mutated on the watcher thread.
        """
        self._reset_requested.set()
        self._wake.set()

    def _call_activity(self) -> bool:
        """Mic in use, OR a meeting app playing audio while the mic is muted.

        The mic path is cheap and checked first. The muted-call path needs an
        osascript meeting-app confirmation, throttled to
        OUTPUT_APP_CHECK_SECS; between confirmations the cached verdict holds
        for as long as meeting audio keeps playing.
        """
        if is_mic_in_use():
            return True
        if not meeting_audio_output_active():
            return False
        now = time.monotonic()
        if now - self._last_output_app_check >= OUTPUT_APP_CHECK_SECS:
            self._last_output_app_check = now
            self._output_meeting_cached = is_meeting_app_running()
            if self._output_meeting_cached:
                _log.debug("muted-call signal: meeting app playing audio, mic off")
        return self._output_meeting_cached

    # ── event loop ────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            self._loop_inner()
        except Exception:
            _log.exception("watcher thread crashed — restarting in 5s")
            time.sleep(5)
            if self._running:
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

    def _loop_inner(self) -> None:
        # Separate the fast mic check (every POLL_SECS) from the slow app check
        # (osascript can take 3-4 s — running it every poll would block the loop).
        _app_counter = 0  # counts mic polls; app is checked every APP_POLL_EVERY
        _app_running = False  # cached result of the last meeting-app check

        while self._running:
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                if self._state != "idle":
                    _log.info("state %s → idle (recording stopped by the app)", self._state)
                self._state = "idle"
                self._since = None
                self._rec_started = None
                self._no_app_polls = 0
                _app_counter = 0
                _app_running = False
                # Stopping mid-call is a deliberate act ("stop recording this
                # meeting"), so don't re-arm until the call actually ends —
                # otherwise auto-record would restart seconds later.
                self._suppressed = True

            active = self._call_activity()
            now = datetime.now()
            elapsed = (now - self._since).total_seconds() if self._since else 0

            if self._suppressed:
                if active:
                    time.sleep(POLL_SECS)
                    continue
                _log.debug("call ended — auto-record re-armed")
                self._suppressed = False

            if self._state == "idle":
                if active:
                    _log.debug("state %s → %s", "idle", "warming")
                    self._state = "warming"
                    self._since = now
                    self._no_app_polls = 0

            elif self._state == "warming":
                if not active:
                    # False positive — Siri, dictation, brief mic check
                    _log.debug("state %s → %s", "warming", "idle")
                    self._state = "idle"
                    self._since = None
                elif elapsed >= WARMUP_SECS:
                    # Mic has been active long enough — but only start recording
                    # if a meeting app/tab is actually detected.  Without this
                    # gate, ambient mic activity (Siri, dictation, voice memos)
                    # would trigger hours-long ghost recordings.
                    if not is_meeting_app_running():
                        _log.debug(
                            "mic active %.1fs but no meeting app detected — staying idle",
                            elapsed,
                        )
                        self._state = "idle"
                        self._since = None
                        continue

                    _log.debug("state %s → %s", "warming", "recording")
                    self._rec_started = now
                    self._state = "recording"
                    self._since = now
                    self._no_app_polls = 0
                    _app_counter = APP_POLL_EVERY  # check app on first recording poll
                    _app_running = True
                    meeting_name = detect_meeting()
                    _log.info("on_start firing — meeting_name=%s", meeting_name)
                    self.on_start(meeting_name)

            elif self._state == "recording":
                # Periodically check if the meeting app is still running.
                # This is the slow check (osascript) so we only do it every
                # APP_POLL_EVERY polls.
                _app_counter += 1
                if _app_counter >= APP_POLL_EVERY:
                    _app_counter = 0
                    _app_running = is_meeting_app_running()
                    if _app_running:
                        self._no_app_polls = 0
                    else:
                        self._no_app_polls += 1
                    _log.debug(
                        "app check: running=%s, _no_app_polls=%d",
                        _app_running,
                        self._no_app_polls,
                    )

                if not active:
                    if _app_running:
                        # Mic off but meeting app still open — user is muted.
                        # Stay in recording to avoid splitting the call.
                        _log.debug("mic idle, meeting app active — treating as muted")
                        self._no_app_polls = 0
                    elif self._no_app_polls >= APP_GONE_POLLS:
                        # Mic off AND meeting app gone for multiple consecutive
                        # checks — start grace period.  Requiring multiple checks
                        # prevents a single slow/failed osascript from stopping
                        # the recording while the user is just muted.
                        _log.debug("state %s → %s", "recording", "cooling")
                        self._state = "cooling"
                        self._since = now
                        self._no_app_polls = 0
                else:
                    # Mic still active — if meeting app has been gone for multiple
                    # consecutive checks, Chrome/Safari is keeping mic "warm"
                    # after leaving the meeting.
                    if self._no_app_polls >= APP_GONE_POLLS:
                        _log.debug("state %s → %s", "recording", "cooling")
                        self._state = "cooling"
                        self._since = now
                        self._no_app_polls = 0

            elif self._state == "cooling":
                if active:
                    # Mic came back — resume recording
                    _log.debug("state %s → %s", "cooling", "recording")
                    self._state = "recording"
                    self._since = now
                    self._no_app_polls = 0
                    _app_counter = APP_POLL_EVERY
                else:
                    # Mic still off — check if meeting app is still running
                    _app_counter += 1
                    if _app_counter >= APP_POLL_EVERY:
                        _app_counter = 0
                        _app_running = is_meeting_app_running()
                        if _app_running:
                            # Meeting still active — user is muted, resume recording
                            _log.debug("state %s → %s", "cooling", "recording")
                            self._state = "recording"
                            self._since = now
                            self._no_app_polls = 0

                # Only stop if still in cooling (app check may have moved us back)
                # and the grace period has elapsed.
                if self._state == "cooling" and elapsed >= GRACE_SECS:
                    duration = (now - self._rec_started).total_seconds() if self._rec_started else 0
                    _log.debug("state %s → %s", "cooling", "idle")
                    self._state = "idle"
                    self._since = None
                    self._rec_started = None
                    self._no_app_polls = 0
                    if duration >= MIN_SAVE_SECS:
                        _log.info("on_stop firing — recording duration=%.1fs", duration)
                        self.on_stop()

            if self._state == "idle" and self._event_driven and not active:
                # Nothing to time — sleep until CoreAudio signals mic activity
                # (or the safety fallback elapses).
                self._wake.wait(IDLE_FALLBACK_SECS)
                self._wake.clear()
                if self._listener:
                    self._listener.refresh()
            else:
                time.sleep(POLL_SECS)


# ── CoreAudio mic detection ────────────────────────────────────────────────────


class _PropAddr(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


_ca_handle = None


def _coreaudio():
    """Cached CoreAudio library handle."""
    global _ca_handle
    if _ca_handle is None:
        _ca_handle = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    return _ca_handle


def _default_device(selector: int) -> int:
    """AudioObjectID of the default input/output device, or 0."""
    try:
        ca = _coreaudio()
        addr = _PropAddr(selector, _kScopeGlobal, _kElementMain)
        dev = ctypes.c_uint32(0)
        sz = ctypes.c_uint32(ctypes.sizeof(dev))
        ca.AudioObjectGetPropertyData(
            _kSysObject,
            ctypes.byref(addr),
            0,
            None,
            ctypes.byref(sz),
            ctypes.byref(dev),
        )
        return dev.value
    except Exception:
        return 0


def _default_input_device() -> int:
    return _default_device(_kDefaultInputDevice)


def is_mic_in_use() -> bool:
    """True if ANY process is currently using the default audio input device."""
    try:
        dev = _default_input_device()
        if dev == 0:
            return False
        ca = _coreaudio()
        addr = _PropAddr(_kIsRunningSomewhere, _kScopeGlobal, _kElementMain)
        running = ctypes.c_uint32(0)
        sz = ctypes.c_uint32(ctypes.sizeof(running))
        status = ca.AudioObjectGetPropertyData(
            dev,
            ctypes.byref(addr),
            0,
            None,
            ctypes.byref(sz),
            ctypes.byref(running),
        )
        return status == 0 and bool(running.value)
    except Exception:
        return False


# OSStatus (*)(AudioObjectID, UInt32 nAddresses, const AudioObjectPropertyAddress*, void*)
_ListenerProc = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


class _MicActivityListener:
    """Wakes the watcher via CoreAudio property listeners instead of polling.

    Registers for changes to the default input AND output device selections
    and to those devices' is-running-somewhere state. Input activity is the
    orange-dot mic signal; output activity is the muted-call signal (the
    meeting app playing other participants). Callbacks arrive on a CoreAudio
    thread and just set the wake event — all real work stays in the watcher
    thread. If registration fails, the watcher falls back to 1 Hz polling.
    """

    _DEVICE_SELECTORS = (_kDefaultInputDevice, _kDefaultOutputDevice)

    def __init__(self, wake: threading.Event):
        self._wake = wake
        self._proc = _ListenerProc(self._on_change)  # keep ref: C callback target
        self._devices: dict[int, int] = {}  # selector -> attached device id
        self._started = False

    def _on_change(self, obj_id, n_addresses, addresses, client_data) -> int:
        self._wake.set()
        return 0

    def start(self) -> bool:
        try:
            ca = _coreaudio()
            for selector in self._DEVICE_SELECTORS:
                addr = _PropAddr(selector, _kScopeGlobal, _kElementMain)
                if ca.AudioObjectAddPropertyListener(
                    _kSysObject, ctypes.byref(addr), self._proc, None
                ):
                    return False
            self._started = True
            self.refresh()
            return True
        except Exception:
            return False

    def refresh(self) -> None:
        """Re-attach running-state listeners if a default device changed."""
        if not self._started:
            return
        try:
            ca = _coreaudio()
            addr = _PropAddr(_kIsRunningSomewhere, _kScopeGlobal, _kElementMain)
            for selector in self._DEVICE_SELECTORS:
                dev = _default_device(selector)
                attached = self._devices.get(selector, 0)
                if dev == attached:
                    continue
                if attached:
                    ca.AudioObjectRemovePropertyListener(
                        attached, ctypes.byref(addr), self._proc, None
                    )
                if dev:
                    ca.AudioObjectAddPropertyListener(dev, ctypes.byref(addr), self._proc, None)
                self._devices[selector] = dev
        except Exception:
            _log.debug("Audio listener refresh failed", exc_info=True)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            ca = _coreaudio()
            run_addr = _PropAddr(_kIsRunningSomewhere, _kScopeGlobal, _kElementMain)
            for selector in self._DEVICE_SELECTORS:
                addr = _PropAddr(selector, _kScopeGlobal, _kElementMain)
                ca.AudioObjectRemovePropertyListener(
                    _kSysObject, ctypes.byref(addr), self._proc, None
                )
                attached = self._devices.pop(selector, 0)
                if attached:
                    ca.AudioObjectRemovePropertyListener(
                        attached, ctypes.byref(run_addr), self._proc, None
                    )
        except Exception:
            pass


# ── Meeting presence checks ───────────────────────────────────────────────────


def _meeting_app_pids() -> set[int]:
    """Return PIDs of known native meeting apps that are currently running."""
    pids: set[int] = set()
    try:
        ps = subprocess.run(
            ["ps", "-ax", "-o", "pid=,comm="],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for line in ps.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                pid_str, comm = parts
                for frag in _ACTIVE_SESSION_PROCS:
                    if frag in comm:
                        try:
                            pids.add(int(pid_str))
                        except ValueError:
                            pass
    except Exception:
        pass
    return pids


def _pids_using_mic_input() -> set[int]:
    """PIDs of all processes currently capturing audio input."""
    return _process_pids_where(_kProcessIsRunningIn)


def _pids_producing_output() -> set[int]:
    """PIDs of all processes currently producing audio output."""
    return _process_pids_where(_kProcessIsRunningOut)


def meeting_audio_output_active() -> bool:
    """True if a browser or meeting app is currently playing audio.

    The muted-call signal: your mic is off, but the other participants'
    voices still play through the meeting app.
    """
    pids = _pids_producing_output()
    if not pids:
        return False
    try:
        ps = subprocess.run(
            ["ps", "-ax", "-o", "pid=,comm="],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for line in ps.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            pid_str, comm = parts
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid in pids and any(frag in comm for frag in _MEETING_OUTPUT_PROC_FRAGMENTS):
                return True
    except Exception:
        pass
    return False


def _process_pids_where(selector: int) -> set[int]:
    """
    Return PIDs of processes whose given CoreAudio process property is true.

    Uses CoreAudio's kAudioHardwarePropertyProcessObjectList API (macOS 14+),
    the same mechanism that drives the orange privacy-indicator dot.
    Returns an empty set on older macOS or on any error.
    """
    pids: set[int] = set()
    try:
        ca = _coreaudio()
        # 1. How many process objects are there?
        addr = _PropAddr(_kProcessObjectList, _kScopeGlobal, _kElementMain)
        sz = ctypes.c_uint32(0)
        if (
            ca.AudioObjectGetPropertyDataSize(
                _kSysObject, ctypes.byref(addr), 0, None, ctypes.byref(sz)
            )
            != 0
            or sz.value == 0
        ):
            return pids

        n = sz.value // ctypes.sizeof(ctypes.c_uint32)
        objs = (ctypes.c_uint32 * n)()
        if (
            ca.AudioObjectGetPropertyData(
                _kSysObject,
                ctypes.byref(addr),
                0,
                None,
                ctypes.byref(sz),
                ctypes.byref(objs),
            )
            != 0
        ):
            return pids

        own_pid = os.getpid()
        for obj_id in objs:
            # Is this process using audio input?
            addr_in = _PropAddr(selector, _kScopeGlobal, _kElementMain)
            running = ctypes.c_uint32(0)
            sz_r = ctypes.c_uint32(ctypes.sizeof(running))
            if (
                ca.AudioObjectGetPropertyData(
                    obj_id,
                    ctypes.byref(addr_in),
                    0,
                    None,
                    ctypes.byref(sz_r),
                    ctypes.byref(running),
                )
                != 0
                or not running.value
            ):
                continue
            # Get the PID
            addr_pid = _PropAddr(_kProcessPID, _kScopeGlobal, _kElementMain)
            pid = ctypes.c_int32(0)
            sz_p = ctypes.c_uint32(ctypes.sizeof(pid))
            if (
                ca.AudioObjectGetPropertyData(
                    obj_id,
                    ctypes.byref(addr_pid),
                    0,
                    None,
                    ctypes.byref(sz_p),
                    ctypes.byref(pid),
                )
                == 0
                and pid.value != own_pid
            ):
                pids.add(pid.value)
    except Exception:
        pass
    return pids


def is_meeting_app_running() -> bool:
    """
    Accurate check: is an active meeting session in progress right now?

    Strategy (in order):
    1. CoreAudio per-process mic check — if any known meeting-app PID is
       capturing audio input, the meeting is still live.
    2. Active-session process check (CptHost for Zoom, etc.) via ps.
    3. Browser tab URL + title check (handles Google Meet / Teams in browser).
    """
    # 1. Per-process mic check (macOS 14+)
    mic_pids = _pids_using_mic_input()
    if mic_pids:
        meeting_pids = _meeting_app_pids()
        # If any process using the mic belongs to a meeting app → still in meeting
        if mic_pids & meeting_pids:
            _log.debug(
                "is_meeting_app_running: per-process mic check succeeded (pids=%s)",
                mic_pids & meeting_pids,
            )
            return True
        # If mic_pids is non-empty but no meeting app PID matches,
        # fall through — browser-based meeting processes may not be in
        # _ACTIVE_SESSION_PROCS, so we still check the browser tabs.

    # 2. Active-session native process check (narrow list — no helper
    #    false-positives). Keep the process listing: steps 3 and 4 use it to
    #    avoid spawning osascript for apps that aren't even running.
    ps_out = ""
    try:
        ps = subprocess.run(
            ["ps", "-ax", "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=3,
        )
        ps_out = ps.stdout
        for frag in _ACTIVE_SESSION_PROCS:
            if frag in ps_out:
                _log.debug("is_meeting_app_running: ps check succeeded (process=%s)", frag)
                return True
    except Exception:
        pass

    # 3. Teams desktop app — window count > 1 means active call
    if (not ps_out or "MSTeams" in ps_out) and _teams_call_active():
        _log.debug("is_meeting_app_running: Teams window count check succeeded")
        return True

    # 4. Browser tab URL + title check — only for browsers actually running
    result = _browser_has_meeting_tab(browsers=_running_browsers(ps_out))
    if result:
        _log.debug("is_meeting_app_running: browser tab check succeeded")
    return result


def detect_meeting() -> str:
    """Best-effort: identify which meeting app is active when recording starts."""
    # 1. Browser tabs first — most reliable (URL + title, no false positives)
    browser_name = _browser_has_meeting_tab(return_name=True)
    if browser_name:
        _log.debug("detect_meeting: browser tab match (name=%s)", browser_name)
        return browser_name

    # 2. Native apps — cross-reference with mic usage to avoid matching
    #    background processes (e.g. FaceTime sitting idle).
    mic_pids = _pids_using_mic_input()
    meeting_pids = _meeting_app_pids()
    active_meeting_pids = mic_pids & meeting_pids
    if active_meeting_pids:
        try:
            ps = subprocess.run(
                ["ps", "-ax", "-o", "pid=,comm="],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in ps.stdout.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    pid_str, comm = parts
                    try:
                        pid = int(pid_str)
                    except ValueError:
                        continue
                    if pid in active_meeting_pids:
                        for fragment, name in _NATIVE_APPS:
                            if fragment in comm:
                                _log.debug(
                                    "detect_meeting: native app match "
                                    "(process=%s, name=%s, pid=%d)",
                                    fragment,
                                    name,
                                    pid,
                                )
                                return name
        except Exception:
            pass

    # 3. Fallback: check for session-only processes (CptHost for Zoom, etc.).
    #    Skip apps like FaceTime / Slack Helper that persist when idle.
    try:
        ps = subprocess.run(["ps", "-ax", "-o", "comm="], capture_output=True, text=True, timeout=3)
        _SESSION_FALLBACK = [
            ("CptHost", "Zoom"),
            ("Tuple", "Tuple"),
        ]
        for frag, name in _SESSION_FALLBACK:
            if frag in ps.stdout:
                _log.debug(
                    "detect_meeting: session-only process match (process=%s, name=%s)",
                    frag,
                    name,
                )
                return name
    except Exception:
        pass

    try:
        from trnscrb.calendar_integration import get_current_or_upcoming_event

        evt = get_current_or_upcoming_event()
        if evt and evt.get("title"):
            _log.debug("detect_meeting: calendar match (title=%s)", evt["title"])
            return evt["title"]
    except Exception:
        pass

    fallback = f"meeting-{datetime.now().strftime('%H%M')}"
    _log.debug("detect_meeting: no method matched, using fallback=%s", fallback)
    return fallback


_MEET_URLS = [
    "meet.google.com",
    "teams.microsoft.com/meet",
    "teams.microsoft.com/v2",
    "app.huddle.team",
    "zoom.us/j/",
]

_CHROME_TAB_SCRIPT = """
tell application "System Events"
    if not (exists process "Google Chrome") then return ""
end tell
tell application "Google Chrome"
    repeat with w in windows
        repeat with t in tabs of w
            set u to URL of t
            if u contains "meet.google.com/" and u does not contain "/landing" then
                set ttl to title of t
                -- Filter out post-meeting pages ("Meeting ended", "You left the meeting")
                if ttl does not contain "ended" and ttl does not contain "left" then return "Google Meet"
            end if
            if u contains "teams.microsoft.com/meet" or u contains "teams.microsoft.com/v2/meet" then return "Microsoft Teams"
            if u contains "app.huddle.team" then return "Huddle"
            if u contains "zoom.us/j/" then return "Zoom"
        end repeat
    end repeat
end tell
return ""
"""

_SAFARI_TAB_SCRIPT = """
tell application "System Events"
    if not (exists process "Safari") then return ""
end tell
tell application "Safari"
    repeat with w in windows
        repeat with t in tabs of w
            try
                set u to URL of t
                if u contains "meet.google.com/" and u does not contain "/landing" then
                    set ttl to name of t
                    -- Filter out post-meeting pages ("Meeting ended", "You left the meeting")
                    if ttl does not contain "ended" and ttl does not contain "left" then return "Google Meet"
                end if
                if u contains "teams.microsoft.com/meet" or u contains "teams.microsoft.com/v2/meet" then return "Microsoft Teams"
                if u contains "app.huddle.team" then return "Huddle"
                if u contains "zoom.us/j/" then return "Zoom"
            end try
        end repeat
    end repeat
end tell
return ""
"""


# Firefox has no AppleScript tab API, so match window titles instead.
# Meeting tabs put a recognisable prefix in the window title while active.
_FIREFOX_WINDOW_SCRIPT = """
tell application "System Events"
    if not (exists process "firefox") then return ""
end tell
tell application "Firefox"
    repeat with w in windows
        set t to name of w
        if t is "Meet" or t starts with "Meet -" or t starts with "Meet —" then
            if t does not contain "ended" then return "Google Meet"
        end if
        if t contains "Microsoft Teams" then return "Microsoft Teams"
        if t contains "Zoom Meeting" then return "Zoom"
    end repeat
end tell
return ""
"""


def _teams_call_active() -> bool:
    """True if Microsoft Teams has an active call (detected by window count > 1)."""
    try:
        r = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to tell process "MSTeams"'
                " to get count of windows",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        return int(r.stdout.strip()) > 1
    except Exception:
        return False


def _run_osascript(label: str, script: str) -> str | None:
    """Run a single osascript and return the stripped output, or None."""
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=4,
        )
        name = r.stdout.strip()
        if name:
            _log.debug("_browser_has_meeting_tab: found tab in %s (name=%s)", label, name)
            return name
    except subprocess.TimeoutExpired:
        _log.debug("_browser_has_meeting_tab: osascript timeout for %s", label)
    except Exception:
        pass
    return None


_BROWSER_DEFS = [
    # (label, ps comm fragment, AppleScript)
    ("Chrome", "Google Chrome.app/", _CHROME_TAB_SCRIPT),
    ("Safari", "Safari.app/Contents/MacOS/Safari", _SAFARI_TAB_SCRIPT),
    ("Firefox", "Firefox.app/", _FIREFOX_WINDOW_SCRIPT),
]


def _running_browsers(ps_out: str) -> list[tuple[str, str]]:
    """(label, script) pairs for browsers with live processes.

    No point spawning osascript for browsers that aren't running — a meeting
    tab can only exist in a running browser. Empty/unknown ps output → all.
    """
    if not ps_out:
        return [(label, script) for label, _frag, script in _BROWSER_DEFS]
    return [(label, script) for label, frag, script in _BROWSER_DEFS if frag in ps_out]


def _browser_has_meeting_tab(return_name: bool = False, browsers=None):
    """
    Check browsers for open meeting tabs in parallel.
    return_name=False → returns bool (fast presence check)
    return_name=True  → returns str name or None
    browsers=None → check all supported browsers (Chrome, Safari, Firefox)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if browsers is None:
        browsers = [(label, script) for label, _frag, script in _BROWSER_DEFS]
    if not browsers:
        return None if return_name else False
    with ThreadPoolExecutor(max_workers=len(browsers)) as pool:
        futures = {pool.submit(_run_osascript, label, script): label for label, script in browsers}
        for future in as_completed(futures, timeout=5):
            try:
                name = future.result()
                if name:
                    return name if return_name else True
            except Exception:
                pass
    return None if return_name else False
