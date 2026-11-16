"""Accessibility-based call participant naming.

Reads the accessibility tree of the meeting app (Zoom, Teams, Meet, Webex)
during a recording, extracts who is in the call and who is speaking, then
assigns names to transcript segments by temporal overlap — same signal as
yogurt's call names, different target.

The package is safe to import headless (CI, Linux). PyObjC is only imported
inside ax.py; everything else is pure Python.

Usage:
    watch = CallNamesWatch(bundle_ids=["us.zoom.xos"], app_name="Zoom")
    watch.start()        # call on meeting start
    samples = watch.stop()  # call on meeting end
    result = watch.meeting_result(segments, my_names)  # attribution
"""

from trnscrb.callnames.watch import CallNamesWatch

__all__ = ["CallNamesWatch"]