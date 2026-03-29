"""macOS Calendar integration via AppleScript.

No extra dependencies — uses the osascript CLI that ships with macOS.
Returns the current or next upcoming meeting within a 30-minute window.
"""

import subprocess
import time
from typing import Optional

from trnscrb.log import get_logger

_log = get_logger("trnscrb.calendar_integration")

_SCRIPT = """
tell application "Calendar"
    set now to current date
    set windowEnd to now + (30 * minutes)
    set found to {}
    repeat with cal in calendars
        try
            set evts to (events of cal whose start date >= (now - 5 * minutes) and start date <= windowEnd)
            set found to found & evts
        end try
    end repeat
    if (count of found) is 0 then return ""
    set evt to item 1 of found
    set evtTitle to summary of evt
    set evtStart to start date of evt as string
    set evtEnd to end date of evt as string
    set attendeeList to ""
    try
        repeat with a in attendees of evt
            set attendeeList to attendeeList & (name of a) & ","
        end repeat
    end try
    return evtTitle & "||" & evtStart & "||" & evtEnd & "||" & attendeeList
end tell
"""

_CACHE_TTL = 30  # seconds
_cache: Optional[dict] = None
_cache_time: float = 0


def get_current_or_upcoming_event() -> Optional[dict]:
    """Return the nearest calendar event, or None if none found / Calendar denied."""
    global _cache, _cache_time

    now = time.time()
    if _cache is not None and (now - _cache_time) < _CACHE_TTL:
        return _cache

    try:
        result = subprocess.run(
            ["osascript", "-e", _SCRIPT],
            capture_output=True,
            text=True,
            timeout=3,
        )
        output = result.stdout.strip()
        if not output or "||" not in output:
            _cache = None
            _cache_time = now
            return None

        parts = output.split("||")
        attendees = [a for a in parts[3].split(",") if a] if len(parts) > 3 else []
        evt = {
            "title": parts[0],
            "start": parts[1],
            "end": parts[2] if len(parts) > 2 else "",
            "attendees": attendees,
        }
        _cache = evt
        _cache_time = now
        return evt
    except subprocess.TimeoutExpired:
        _log.debug("Calendar lookup timed out, returning cached result")
        return _cache
    except Exception:
        return _cache
