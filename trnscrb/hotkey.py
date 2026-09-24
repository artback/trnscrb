"""Push-to-talk hotkey: key specs, virtual key codes, and display.

The ``dictation_ptt_key`` setting holds a spec such as ``"ctrl+alt+f8"`` or
``"f8"``. The push-to-talk monitor (``trnscrb.ptt``) matches live key events
against the parsed (key code, modifier mask) pair; this module owns that
translation in both directions and is pure — no AppKit, no PyObjC — so it
stays importable on headless CI.

Key codes are Apple virtual key codes (``kVK_*``), a stable ABI since the
early Mac OS X. The table deliberately lists only keys whose codes are
trusted; the menu bar's "Record PTT key…" captures whatever the user
actually presses, so a spec naming an unsupported key is rejected rather
than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass

# CGEventFlags modifier masks, as exposed by CoreGraphics (and PyObjC's
# Quartz). The monitor compares these bits against a key event's flags.
FLAG_SHIFT = 1 << 17  # kCGEventFlagMaskShift
FLAG_CONTROL = 1 << 18  # kCGEventFlagMaskControl
FLAG_OPTION = 1 << 19  # kCGEventFlagMaskAlternate
FLAG_COMMAND = 1 << 20  # kCGEventFlagMaskCommand

# Modifiers that count as "held" when recording a PTT combo. Shift is
# deliberately excluded: it is noise in a push-to-talk combo (finger slips
# while reaching for the key), so the recorder drops it.
FLAG_PTT_MODS = FLAG_CONTROL | FLAG_OPTION | FLAG_COMMAND

_MODIFIER_NAMES: dict[str, int] = {
    "shift": FLAG_SHIFT,
    "shft": FLAG_SHIFT,
    "\u21e7": FLAG_SHIFT,  # ⇧
    "ctrl": FLAG_CONTROL,
    "control": FLAG_CONTROL,
    "\u2303": FLAG_CONTROL,  # ⌃
    "alt": FLAG_OPTION,
    "option": FLAG_OPTION,
    "opt": FLAG_OPTION,
    "\u2325": FLAG_OPTION,  # ⌥
    "cmd": FLAG_COMMAND,
    "command": FLAG_COMMAND,
    "meta": FLAG_COMMAND,
    "\u2318": FLAG_COMMAND,  # ⌘
}

# Apple virtual key codes for the keys a PTT spec may name. Only codes this
# project trusts: function keys, the two main letter rows, digits, and a
# handful of special keys. See the module docstring for why the list is
# conservative — "Record PTT key…" covers everything else.
KEY_CODES: dict[str, int] = {
    # Function keys
    "f1": 96,
    "f2": 97,
    "f3": 98,
    "f4": 100,
    "f5": 103,
    "f6": 105,
    "f7": 106,
    "f8": 99,
    "f9": 111,
    "f10": 109,
    "f11": 108,
    "f12": 107,
    "f13": 104,
    "f14": 115,
    "f15": 114,
    # ASDF row
    "a": 0,
    "s": 1,
    "d": 2,
    "f": 3,
    "h": 4,
    "g": 5,
    "j": 6,
    "k": 7,
    "l": 8,
    "z": 10,
    # QWERTY row
    "q": 12,
    "w": 13,
    "e": 14,
    "r": 15,
    "t": 17,
    "y": 18,
    "u": 19,
    "i": 20,
    "o": 21,
    "p": 22,
    # Number row
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "5": 23,
    "6": 22,
    "7": 27,
    "8": 26,
    "9": 25,
    "0": 29,
    # Special keys
    "space": 49,
    "tab": 48,
    "return": 36,
    "enter": 36,
    "escape": 53,
    "delete": 51,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
}

# Reverse lookup (first name wins — there are no collisions in KEY_CODES).
_CODE_TO_KEY: dict[int, str] = {}
for _name, _code in KEY_CODES.items():
    _CODE_TO_KEY.setdefault(_code, _name)

# Canonical modifier order for specs and display: ⌃⌥⇧⌘.
_CANONICAL_MODS: list[tuple[int, str, str]] = [
    (FLAG_CONTROL, "ctrl", "\u2303"),
    (FLAG_OPTION, "alt", "\u2325"),
    (FLAG_SHIFT, "shift", "\u21e7"),
    (FLAG_COMMAND, "cmd", "\u2318"),
]


@dataclass(frozen=True)
class PTTKey:
    """A parsed push-to-talk combo: a key plus the modifiers it requires."""

    key_code: int
    flags: int  # required modifier mask (CGEventFlags bits); 0 = bare key
    spec: str  # canonical spec, e.g. "ctrl+alt+f8"


def parse(spec: str) -> PTTKey | None:
    """Parse a PTT spec like ``"ctrl+alt+f8"`` into a :class:`PTTKey`.

    Grammar: zero or more modifier names, then exactly one key name, joined
    by ``+`` (case-insensitive, spaces ignored). A bare key name ("f8") is
    a valid spec; a bare modifier is not. The key may also be ``keycode:N``
    for a raw Apple virtual key code — the record mode stores combos this
    way when the pressed key is not in the named table. Returns None for
    anything else, including an empty spec (the "off" setting).
    """
    if not isinstance(spec, str):
        return None
    tokens = [t.strip().lower() for t in spec.split("+") if t.strip()]
    if not tokens or len(tokens) > 5:
        return None
    key_token = tokens[-1]
    if key_token in KEY_CODES:
        key_code, key_name = KEY_CODES[key_token], key_token
    elif key_token.startswith("keycode:"):
        raw = key_token.split(":", 1)[1]
        if not raw.isdigit():
            return None
        key_code, key_name = int(raw), key_token
    else:
        return None
    flags = 0
    for token in tokens[:-1]:
        if token not in _MODIFIER_NAMES:
            return None
        flags |= _MODIFIER_NAMES[token]
    return PTTKey(key_code=key_code, flags=flags, spec=canonical_spec(key_name, flags))


def canonical_spec(key_name: str, flags: int) -> str:
    """Canonical spec string for a key name and modifier mask ("ctrl+alt+f8")."""
    parts = [name for bit, name, _sym in _CANONICAL_MODS if flags & bit]
    parts.append(key_name.lower())
    return "+".join(parts)


def key_name(key_code: int) -> str | None:
    """The table name for a virtual key code, or None when unknown."""
    return _CODE_TO_KEY.get(key_code)


def display_spec(key_code: int, flags: int) -> str:
    """Menu-bar display for a combo: ``⌃⌥F8``; ``key 45`` when the key is
    not in the table (record mode can capture keys the parser cannot name)."""
    symbols = "".join(sym for bit, _name, sym in _CANONICAL_MODS if flags & bit)
    name = _CODE_TO_KEY.get(key_code)
    if name is None:
        return f"{symbols}key {key_code}"
    label = name.title() if name[:1].isalpha() else name.upper()
    # "space" reads better as a word than "Space" next to a modifier.
    if name == "space":
        label = "Space"
    if name.startswith("f") and name[1:].isdigit():
        label = "F" + name[1:]
    return f"{symbols}{label}"


def ptt_mods(flags: int) -> int:
    """Only the modifiers that count toward a PTT combo (shift is noise)."""
    return flags & FLAG_PTT_MODS
