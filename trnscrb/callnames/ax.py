"""macOS AX tree adapter — the only file touching PyObjC/ApplicationServices.

Given a bundle ID or app name, reads the accessibility tree and returns a
plain-Python snapshot dict.  On any failure (no AX permission, app not
running, tree unreadable, headless) → returns ``ok=False``, never raises.

Headless: module imports anywhere; ``sample()`` returns ``ok=False`` when
Quartz / ApplicationServices is unavailable.
"""

from __future__ import annotations

try:
    import ApplicationServices as _AS
    from AppKit import NSRunningApplication
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False

import ctypes
import time

from trnscrb.log import get_logger

_log = get_logger("trnscrb.callnames.ax")

# ── constants ─────────────────────────────────────────────────────────────────

_MAX_DEPTH = 20
_MAX_ELEMENTS = 500
_CHILDREN_LIMIT = 100  # max children to fetch per element

# ── AX attribute names ─────────────────────────────────────────────────────────

_kAXRoleAttribute = b"AXRole"
_kAXDescriptionAttribute = b"AXDescription"
_kAXTitleAttribute = b"AXTitle"
_kAXValueAttribute = b"AXValue"
_kAXChildrenAttribute = b"AXChildren"


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_pid(bundle_id: str) -> int | None:
    """PID of a running app by bundle ID, or ``None``."""
    if not AX_AVAILABLE:
        return None
    try:
        app = NSRunningApplication.runningApplicationWithBundleIdentifier_(bundle_id)
        if app:
            return app.processIdentifier()
    except Exception:
        _log.debug("Could not get PID for %s", bundle_id, exc_info=True)
    return None


def _frontmost_bundle() -> str | None:
    """Bundle ID of the frontmost app, or ``None`` (never raises)."""
    if not AX_AVAILABLE:
        return None
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is not None:
            return app.bundleIdentifier()
    except Exception:
        _log.debug("Could not get frontmost app", exc_info=True)
    return None


def pick_bundle(bundle_ids: list[str]) -> str | None:
    """Choose which candidate bundle to sample.

    Browsers are the common case (several may be installed): prefer the
    frontmost app when it is one of the candidates, else the first running
    candidate, else the first candidate (``sample()`` reports "not
    running" if none is).  Headless: the input order is kept.
    """
    if not bundle_ids:
        return None
    if not AX_AVAILABLE:
        return bundle_ids[0]
    front = _frontmost_bundle()
    running = [b for b in bundle_ids if _get_pid(b) is not None]
    if front in running:
        return front
    if running:
        return running[0]
    return bundle_ids[0]


def _bundle_for_app(app_name: str) -> str | None:
    """Map a display name to a bundle ID."""
    return {
        "Zoom": "us.zoom.xos",
        "Microsoft Teams": "com.microsoft.teams2",
        "Google Meet": "com.google.Chrome",
        "Webex": "com.cisco.webexmeetingsapp",
        "Slack Huddle": "com.tinyspeck.slackmacgap",
    }.get(app_name)


def _read_attr(element, attr_bytes: bytes) -> str | None:
    """Read one AX attribute as a string.  Returns None on any error."""
    if not AX_AVAILABLE:
        return None
    try:
        value_ref = ctypes.c_void_p()
        err = _AS.AXUIElementCopyAttributeValue(element, attr_bytes, ctypes.byref(value_ref))
        if err != 0 or not value_ref.value:
            return None
        # CFString → Python str via ctypes auto-conversion.
        return str(value_ref.value) if value_ref.value else None
    except Exception:
        _log.debug("AX attribute read failed", exc_info=True)
        return None


# ── tree walking ──────────────────────────────────────────────────────────────

def _walk(element, depth: int = 0, seen: set | None = None) -> list[dict]:
    """Recursively walk an AX element tree.

    Returns a list of node dicts {role, description, title, value, children}.
    Depth-capped at ``_MAX_DEPTH`` and element-capped at ``_MAX_ELEMENTS``.
    """
    if seen is None:
        seen = set()
    if depth > _MAX_DEPTH or len(seen) > _MAX_ELEMENTS:
        return []
    if id(element) in seen:
        return []
    seen.add(id(element))

    node: dict = {
        "role": _read_attr(element, _kAXRoleAttribute) or "",
        "description": _read_attr(element, _kAXDescriptionAttribute) or "",
        "title": _read_attr(element, _kAXTitleAttribute) or "",
        "value": _read_attr(element, _kAXValueAttribute) or "",
    }

    # Fetch children.
    try:
        child_array_ref = ctypes.c_void_p()
        err = _AS.AXUIElementCopyAttributeValue(
            element, _kAXChildrenAttribute, ctypes.byref(child_array_ref)
        )
        if err == 0 and child_array_ref.value:
            from Foundation import CFArrayGetCount, CFArrayGetValueAtIndex
            count = CFArrayGetCount(child_array_ref.value)
            children = []
            for i in range(min(int(count), _CHILDREN_LIMIT)):
                child_ref = ctypes.c_void_p()
                CFArrayGetValueAtIndex(child_array_ref.value, i, ctypes.byref(child_ref))
                if child_ref.value:
                    children.extend(_walk(child_ref.value, depth + 1, seen))
            if children:
                node["children"] = children
    except Exception:
        _log.debug("AX children fetch failed", exc_info=True)

    return [node]


# ── public API ─────────────────────────────────────────────────────────────────

def sample(
    bundle_id: str | None = None,
    app_name: str | None = None,
) -> dict:
    """Read the AX tree of a single call app and return a snapshot.

    Returns ``{ts, app, bundle_id, tree, ok}`` where ``tree`` is the AX
    node list.  On any failure (no permission, app not running, headless)
    returns ``ok=False`` and never raises.
    """
    result: dict = {
        "ts": time.time(),
        "app": app_name or bundle_id or "unknown",
        "bundle_id": bundle_id or "",
        "tree": [],
        "ok": False,
    }

    bid = bundle_id or (_bundle_for_app(app_name) if app_name else None)
    if not bid:
        return result

    pid = _get_pid(bid)
    if pid is None:
        return result

    app_ref = ctypes.c_void_p()
    err = _AS.AXUIElementCreateApplication(pid, ctypes.byref(app_ref))
    if err != 0:
        return result

    try:
        elements = _walk(app_ref.value, 0, set())
        result["tree"] = elements
        result["ok"] = True
    except Exception:
        _log.debug("AX sample failed for %s (%s)", app_name or bid, exc_info=True)

    return result


def has_permission() -> bool:
    """Best-effort check: is the accessibility permission granted?

    Tries to read the system-wide focused element.  Never raises; returns
    ``False`` on any error.
    """
    if not AX_AVAILABLE:
        return False
    try:
        sys_ref = ctypes.c_void_p()
        err = _AS.AXUIElementCreateSystemWide(ctypes.byref(sys_ref))
        if err != 0 or not sys_ref.value:
            return False
        focused_ref = ctypes.c_void_p()
        err = _AS.AXUIElementCopyAttributeValue(
            sys_ref.value, b"AXFocusedUIElement", ctypes.byref(focused_ref)
        )
        return bool(err == 0 and focused_ref.value)
    except Exception:
        return False