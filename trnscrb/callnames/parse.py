"""Per-app accessibility-tree parsers.

Input = a tree as a list of node dicts (role, description, title, value,
children).  Output = ``CallReading{speaking, participants, me}``.

Every rule fails towards reading nothing, so a changed app tree means "Them"
(not a wrong name).  Each parser is independently unit-testable with
synthetic tree dicts.
"""

from __future__ import annotations

import re
from typing import Callable

from trnscrb.log import get_logger

_log = get_logger("trnscrb.callnames.parse")

# ── node types ────────────────────────────────────────────────────────────────

# Role sets that the various apps use.  macOS / AT-SPI map differently.
TAB_GROUP = {"AXTabGroup", "page tab"}
STATIC_TEXT = {"AXStaticText", "static", "label", "text", "paragraph", "heading", "caption"}
ROW = {"AXRow", "table row", "list item", "row"}
OUTLINE = {"AXOutline", "tree table", "table", "list", "list box", "outline", "tree"}


class CallReading:
    """Result of parsing one call app's AX tree snapshot."""
    __slots__ = ("speaking", "participants", "me")

    def __init__(
        self,
        speaking: list[str] | None = None,
        participants: list[str] | None = None,
        me: str | None = None,
    ):
        self.speaking: list[str] = speaking or []
        self.participants: list[str] = participants or []
        self.me: str | None = me


# ── tree walker ────────────────────────────────────────────────────────────────

def walk(
    node: dict,
    fn: Callable[[dict, list[dict]], None],
    up: list[dict] | None = None,
) -> None:
    """Recursively walk the tree, calling ``fn(node, ancestors)``.

    ``up`` is the path from root to parent (exclusive).
    """
    if up is None:
        up = []
    fn(node, up)
    for child in node.get("children") or []:
        walk(child, fn, [*up, node])


def texts(node: dict) -> list[str]:
    """Text content of a node: description, title, value — truthy only."""
    return [node[k] for k in ("description", "title", "value") if node.get(k)]


def uniq(names: list[str]) -> list[str]:
    """Case-insensitive unique, preserving first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        k = n.casefold()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


# ── Zoom (desktop app) ────────────────────────────────────────────────────────

# Multilingual strings from Zoom's Localizable.strings.
_ZOOM_AUDIO = {
    "computer audio unmuted", "computer audio muted",
    "bei computeraudio stummschaltung deaktiviert",
    "bei computeraudio stummschaltung aktiviert",
    "audio de la computadora no silenciado",
    "audio de la computadora silenciado",
    "son de l'ordinateur rétabli", "ordinateur mis en sourdine",
    "audio computer attivato", "audio computer disattivato",
    "áudio do computador ativado", "áudio do computador desativado",
    "computeraudio ingeschakeld", "computeraudio gedempt",
    "datorljudet aktiverades", "datorljudet inaktiverades",
    "ミュート解除されたコンピュータ オーディオ",
    "ミュートされたコンピュータ オーディオ",
    "电脑音频开启", "电脑音频静音",
}
_ZOOM_ACTIVE = {
    "active speaker", "aktiver sprecher", "orador activo",
    "intervenant actif", "oratore attivo", "orador ativo",
    "actieve spreker", "aktiv talare", "アクティブ スピーカー",
    "当前演讲者",
}
_ZOOM_TALKING = ["talking:", "sprechen:", "hablando:", "la parole est à :",
    "conversazione in corso:", "falando:", "aan het spreken:", "talar:",
    "発言中:", "正在发言："]
_ZOOM_LISTS = {
    "participants list", "teilnehmerliste", "lista de participantes",
    "liste des participants", "elenco dei partecipanti",
    "lista de participantes", "deelnemerslijst", "deltagarlista",
    "参加者リスト", "参会者列表",
}
_ZOOM_SELF = {"me", "ich", "yo", "moi", "io", "eu", "ik", "jag", "自分", "我"}
_ZOOM_TAGS_RE = (
    r"\s*[(（]\s*(?:me|host|co-host|guest|interpreter|ich|yo|moi|io|eu|ik|jag|自分|我)"
    r"(?:\s*[,，、]\s*(?:me|host|co-host|guest|interpreter|ich|yo|moi|io|eu|ik|jag|自分|我))*"
    r"\s*[)）]\s*$"
)


def _clean_zoom(raw: str) -> str | None:
    """Strip Zoom suffixes from a tile label, return the name."""
    import re
    name = raw.strip()
    while re.search(_ZOOM_TAGS_RE, name):
        name = re.sub(_ZOOM_TAGS_RE, "", name)
    name = name.strip()
    if not name or name.lower().startswith("participants ("):
        return None
    return name


def _zoom_tile(description: str) -> tuple[str, bool] | None:
    """Parse a Zoom tile description → (name, talking).

    "Ana Ruiz, Computer audio unmuted, Video off, Active speaker" → ("Ana Ruiz", True).
    """
    if not description:
        return None
    import re
    parts = re.split(r",\s*|、", description)
    talking = False
    known = False
    while len(parts) > 1:
        tail = parts[-1].strip().casefold()
        if tail in _ZOOM_ACTIVE:
            talking = True
            known = True
            parts.pop()
        elif tail in _ZOOM_AUDIO:
            known = True
            parts.pop()
        elif re.match(
            r"(video (on|off)|no audio|telephone.*|screen sharing.*|pinned|spotlighted)$",
            tail, re.IGNORECASE,
        ):
            parts.pop()
        else:
            break
    if not known:
        return None
    name = _clean_zoom(", ".join(parts))
    return (name, talking) if name else None


def _zoom_linux_tile(name: str) -> tuple[str, bool] | None:
    """Linux Zoom Workplace: 'video item <name> Active Speaker'."""
    import re
    m = re.match(r"^video item\s+(.+)$", name.strip(), re.IGNORECASE)
    if not m:
        return None
    raw = m[1].strip()
    talking = bool(re.search(r"\s+active speaker$", raw, re.IGNORECASE))
    person = _clean_zoom(raw.replace("active speaker", "", 1).strip())
    if person and len(person) <= 60:
        return (person, talking)
    return None


def zoom(tree: list[dict]) -> CallReading:
    """Parse a Zoom desktop app AX tree snapshot.

    Returns a ``CallReading`` with the detected participants and the person
    marked as actively speaking.  Returns empty lists on any error or
    unrecognized tree shape.
    """
    speaking: list[str] = []
    participants: list[str] = []
    me: str | None = None
    linux_tiles: list[tuple[str, bool]] = []

    def visit(node: dict, up: list[dict]) -> None:
        nonlocal me
        # Video tile: AXTabGroup with a tile description.
        if node.get("role") in TAB_GROUP and node.get("description"):
            result = _zoom_tile(node["description"])
            if result:
                name, talking = result
                if name not in participants:
                    participants.append(name)
                if talking:
                    speaking.append(name)
            return

        # Linux tile: push/toggle button "video item <name>".
        if node.get("role") in ("push button", "toggle button"):
            for raw in (node.get("title") or "") + " " + (node.get("description") or ""):
                result = _zoom_linux_tile(raw.strip())
                if result:
                    name, talking = result
                    if name not in participants:
                        participants.append(name)
                    linux_tiles.append(result)

        # "Talking: Name" floating text.
        if node.get("role") in STATIC_TEXT:
            raw = node.get("value") or node.get("title") or ""
            raw_lower = raw.lower()
            for key in _ZOOM_TALKING:
                if raw_lower.startswith(key):
                    name = _clean_zoom(raw[len(key):].lstrip(":：").strip())
                    if name:
                        speaking.append(name)
                    break

        # Participants list row.
        if node.get("role") in ROW:
            in_part_list = any(
                parent.get("role") in OUTLINE
                and (parent.get("description") or parent.get("title") or "")
                .casefold().strip() in _ZOOM_LISTS
                for parent in up
            )
            if not in_part_list:
                return
            for raw in texts(node):
                name = _clean_zoom(raw)
                if not name:
                    continue
                import re
                tags = re.findall(
                    r"(?:me|host|co-host|guest|interpreter|ich|yo|moi|io|eu|ik|jag|自分|我)",
                    raw, re.IGNORECASE,
                )
                participants.append(name)
                if any(t in _ZOOM_SELF for t in tags):
                    me = name
                break

    for w in tree:
        walk(w, visit)

    # Linux active-speaker only with 3+ tiles (same threshold as Mac).
    if len(set(n for n, _ in linux_tiles)) >= 3:
        for name, talking in linux_tiles:
            if talking and name not in speaking:
                speaking.append(name)

    return CallReading(
        speaking=uniq(speaking),
        participants=uniq(participants),
        me=me,
    )


# ── Google Meet (browser) ─────────────────────────────────────────────────────

# Yogurt detected a CSS class on Meet's talking tile (v5h6Xc as of Oct 2026).
# Our AX walker doesn't see DOM classes — the Rust helper does.  So we detect
# participants but not who is talking from DOM classes alone.  The attribute
# module falls back to temporal overlap.
_MEET_TALKING_CLASSES = ["v5h6Xc"]

_NOT_A_NAME = (
    r"^\d{1,2}[:.]\d{2}(\s?[AP]M)?$"
    r"|^[a-z]{3}-[a-z]{4}-[a-z]{3}$"
    r"|^[a-z0-9]+-[a-z0-9]+-[a-z0-9]+$"
    r"|says in chat|^(AM|PM)$|^\d+$"
)


def _meet_name(text: str) -> str | None:
    """A name from a Meet tile, or ``None``."""
    if not text:
        return None
    import re
    name = re.sub(
        r"\s*\((?:presenting|presentation|you)[^)]*\)\s*$", "", text.strip()
    )
    if not (2 <= len(name) <= 60):
        return None
    if any(ch.isdigit() for ch in name):
        return None
    if re.search(_NOT_A_NAME, name, re.IGNORECASE):
        return None
    if not re.search(r"[A-Za-z]", name):
        return None
    return name


def meet(tree: list[dict]) -> CallReading:
    """Parse a Google Meet (browser) AX tree snapshot.

    Returns participants found in video tiles.  Who is talking is not
    detectable without the DOM-class signal (not available in our walker),
    so ``speaking`` stays empty — the attribution module uses temporal
    overlap instead.
    """
    groups: dict[int, str] = {}  # root id → detected name
    speaking: list[str] = []
    participants: list[str] = []
    me: str | None = None
    seen: set[str] = set()

    TILE_UP = 7  # one tile is ~7 levels above the name text

    def visit(node: dict, up: list[dict]) -> None:
        nonlocal me
        if node.get("role") not in STATIC_TEXT or len(up) < TILE_UP:
            return
        text = node.get("value") or node.get("title") or ""
        name = _meet_name(text)
        if not name:
            return
        # Skip the user's own tile but still record me
        if name.casefold() == "you":
            me = name
            return
        # Skip duplicates
        if name.casefold() in seen:
            return
        seen.add(name.casefold())

        root = up[-TILE_UP] if len(up) >= TILE_UP else node
        groups[id(root)] = name

    for w in tree:
        walk(w, visit)

    for name in groups.values():
        if name.casefold() == "you":
            me = name
        participants.append(name)

    return CallReading(
        speaking=speaking,  # empty — detected via temporal overlap
        participants=uniq(participants),
        me=me,
    )


# ── Microsoft Teams ────────────────────────────────────────────────────────────

_TEAMS_SPEAKING_RE = [
    r"^(.{2,60}?)(?:,\s*|\s+is\s+)(?:speaking|talking)\b",
    r"\b(?:active speaker|now speaking|speaking now)[:,]?\s+(.{2,60})$",
]
_TEAMS_STATUS = (
    r"^(?:muted|unmuted|mic(?:rophone)? (?:on|off)"
    r"|video (?:is )?(?:on|off)|camera (?:is )?(?:on|off))$"
)
_TEAMS_ROLE = (
    r"^(?:organizer|presenter|guest|external|co-?organizer"
    r"|attendee|host|co-?host|interpreter)$"
)
_TEAMS_ME_RE = r"^(.{2,60}?)\s*\((?:you|me)\)"
_TEAMS_LIST_NAME = (
    r"^(?:meeting )?(?:participants|attendees|people|in (?:this|the) meeting)\b"
)
_TEAMS_ROW = {"tree item", "table row", "list item", "row", "AXRow"}
_TEAMS_LIST = {"tree", "tree table", "table", "list", "list box", "outline", "AXOutline", "AXTable"}


def _clean_teams(s: str) -> str:
    import re
    s = re.sub(r"\s*\((?:organizer|presenter|guest|external|you|me)\)", "", s)
    s = re.sub(r"[,.:;]+?$", "", s)
    return s.strip()


def _teams_persona(label: str) -> tuple[str, bool] | None:
    """"Dana Smith, Organizer, Unmuted" → (Dana Smith, False)."""
    if not label:
        return None
    import re
    fields = re.split(r",\s*", label)
    state_idx = None
    for i, f in enumerate(fields):
        if re.match(_TEAMS_STATUS, f.strip(), re.IGNORECASE):
            state_idx = i
            break
    if state_idx is None or state_idx < 1:
        return None
    before = fields[:state_idx]
    while len(before) > 1 and re.match(
        _TEAMS_ROLE, before[-1].strip(), re.IGNORECASE
    ):
        before.pop()
    name = _clean_teams(", ".join(before)).strip()
    if not (2 <= len(name) <= 60):
        return None
    speaking = any(
        re.match(r"^(?:speaking|talking|active speaker|now speaking)$", f.strip(), re.IGNORECASE)
        for f in fields
    )
    return (name, speaking)


def _in_teams_roster(node: dict, up: list[dict]) -> bool:
    """Is this node inside the People/Attendees panel?"""
    is_row = node.get("role") in _TEAMS_ROW or any(
        p.get("role") in _TEAMS_ROW for p in up
    )
    if not is_row:
        return False
    return any(
        p.get("role") in _TEAMS_LIST
        and re.search(_TEAMS_LIST_NAME, (p.get("description") or p.get("title") or "").strip(), re.IGNORECASE)
        for p in up
    )


def teams(tree: list[dict]) -> CallReading:
    """Parse a Microsoft Teams AX tree snapshot.

    Returns participants from the People panel and anyone marked speaking.
    """
    speaking: list[str] = []
    participants: list[str] = []
    me: str | None = None

    def visit(node: dict, up: list[dict]) -> None:
        nonlocal me
        roster = _in_teams_roster(node, up)
        for raw in texts(node):
            # Speaking markers outside roster (e.g. "Dana, speaking")
            for pattern in _TEAMS_SPEAKING_RE:
                m = re.search(pattern, raw.strip())
                if m:
                    name = m.group(1).strip()
                    if name.lower() not in ("you", "me") and len(name) <= 60:
                        speaking.append(name)
                        if name not in participants:
                            participants.append(name)
                    break

            # "You (me)" or "Name (you)" outside roster
            m = re.match(_TEAMS_ME_RE, raw.strip())
            if m:
                name = m.group(1).strip()
                if 2 <= len(name) <= 60:
                    me = name
                    if name not in participants:
                        participants.append(name)

            if not roster:
                continue
            persona = _teams_persona(raw)
            if persona:
                name, is_speaking = persona
                if name not in participants:
                    participants.append(name)
                if is_speaking and name not in speaking:
                    speaking.append(name)

            m = re.match(_TEAMS_ME_RE, raw)
            if m:
                name = m.group(1).strip()
                name = name.strip()
                if len(name) >= 2 and len(name) <= 60:
                    me = name
                    if name not in participants:
                        participants.append(name)

    for w in tree:
        walk(w, visit)

    return CallReading(
        speaking=uniq(speaking),
        participants=uniq(participants),
        me=me,
    )


# ── Webex ──────────────────────────────────────────────────────────────────────

_WEBEX_TILE_RE = r"^(.+?)[,，].*speaking.*Video (?:on|off)"
_WEBEX_ROSTER_RE = r"^participants(?:\s|$)"
_WEBEX_PERSON_MARK = (
    r"^(?:host|presenter|me|muted|unmuted|speaking|using phone|unverified)$"
)


def _webex_row_name(description: str) -> str | None:
    """Parse a Webex roster row description → name."""
    if not description:
        return None
    import re
    fields = [f.strip() for f in re.split(r"[,،]", description) if f.strip()]
    status_idx = None
    for i, f in enumerate(fields):
        if i > 0 and re.match(_WEBEX_PERSON_MARK, f, re.IGNORECASE):
            status_idx = i
            break
    if status_idx is None:
        return None
    name = ", ".join(fields[:status_idx]).strip()
    if not name or re.match(_WEBEX_PERSON_MARK, name, re.IGNORECASE):
        return None
    return name if 2 <= len(name) <= 60 else None


def webex(tree: list[dict]) -> CallReading:
    """Parse a Webex desktop app AX tree snapshot."""
    speaking: list[str] = []
    participants: list[str] = []
    me: str | None = None

    def visit(node: dict, up: list[dict]) -> None:
        nonlocal me
        # Floating video tile: "Name, speaking, Video on/off"
        if node.get("role") == "AXGroup" and node.get("title"):
            m = re.match(_WEBEX_TILE_RE, node["title"].strip())
            if m:
                name = m.group(1).strip()
                if 2 <= len(name) <= 60:
                    speaking.append(name)

        # Roster row: AXRow inside an AXOutline named "participants".
        in_roster = any(
            p.get("role") == "AXOutline"
            and re.search(_WEBEX_ROSTER_RE, (p.get("description") or p.get("title") or "").strip(), re.IGNORECASE)
            for p in up
        )
        if not in_roster:
            return

        name = None
        is_row = False
        if node.get("role") == "AXRow" and node.get("description"):
            name = _webex_row_name(node["description"])
            if name:
                is_row = True

        # Walk children for more info.
        labels: list[str] = []
        for child in node.get("children") or []:
            for t in texts(child):
                labels.append(t)

        if not is_row and not name:
            for child in node.get("children") or []:
                if (
                    child.get("role") == "AXCell"
                    and child.get("identifier") == "GenericParticipantListCellView"
                ):
                    is_row = True
                if child.get("role") in STATIC_TEXT:
                    c = child.get("value") or child.get("title") or ""
                    if c and 2 <= len(c) <= 60:
                        name = c

        if not is_row or not name:
            return
        labels_text = ", ".join(labels)
        # Check speaking from row description or child labels
        row_desc = node.get("description", "")
        check_text = f"{name}, {labels_text}, {row_desc}"
        if name not in participants:
            participants.append(name)
        if re.search(r"(?:^|[,，]\s*)speaking(?:\s*[,，]|$)", check_text, re.IGNORECASE):
            if name not in speaking:
                speaking.append(name)
        if re.search(r"(?:^|[,，]\s*)me(?:\s*[,，]|$)", check_text, re.IGNORECASE):
            me = name

    for w in tree:
        walk(w, visit)

    return CallReading(
        speaking=uniq(speaking),
        participants=uniq(participants),
        me=me,
    )


# ── Slack Huddle ───────────────────────────────────────────────────────────────
#
# A huddle lives in its own floating window (Electron AX tree): a "Huddle"
# or "Video huddle" header, one video tile per participant with the name as
# a static text under it, and the user's own tile labelled "You".  There is
# no reliable speaking signal in the tree, so ``speaking`` stays empty and
# attribution falls back to the two-person rule for 1:1 huddles (group
# huddles land in the meeting metadata as a participant list only).

_HUDDLE_ANCHOR_RE = re.compile(r"\bhuddle\b", re.IGNORECASE)
_SLACK_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,57}[A-Za-z]$")
_SLACK_UI_WORDS = {
    "huddle", "huddles", "video huddle", "muted", "unmuted", "mute",
    "unmute", "video", "mic", "microphone", "camera", "call", "leave",
    "end", "end call", "end huddle", "share", "share screen", "screen",
    "participants", "invite", "invite people", "add people", "people",
    "settings", "more", "options", "join", "start", "stop", "sound",
    "follow", "pin", "copy", "you", "you're", "you are", "now speaking",
    "speaking", "talking", "you're talking",
}


def _slack_name(text: str) -> str | None:
    """A name-shaped static text, or ``None`` (UI label / chat / junk)."""
    t = text.strip()
    if not (2 <= len(t) <= 60):
        return None
    if not _SLACK_NAME_RE.match(t):
        return None
    if len(t.split()) > 4:
        return None
    if t.casefold() in _SLACK_UI_WORDS:
        return None
    return t


def slack(tree: list[dict]) -> CallReading:
    """Parse a Slack Huddle AX tree snapshot.

    Names come from the huddle window's tile labels; chat lives in a
    different window and is excluded by the anchor rule (a root only counts
    as a huddle when it contains a "huddle" text).  ``speaking`` is always
    empty — see the module note above.
    """
    huddle_roots: set[int] = set()
    for w in tree:
        found = False

        def _anchor(node: dict, up: list[dict]) -> None:
            if node.get("role") in STATIC_TEXT:
                t = (node.get("value") or node.get("title") or "").strip()
                if t and _HUDDLE_ANCHOR_RE.search(t):
                    nonlocal found
                    found = True

        walk(w, _anchor)
        if found:
            huddle_roots.add(id(w))

    participants: list[str] = []
    me: str | None = None
    seen: set[str] = set()

    for w in tree:
        if id(w) not in huddle_roots:
            continue

        def _names(node: dict, up: list[dict]) -> None:
            nonlocal me
            if node.get("role") not in STATIC_TEXT:
                return
            t = (node.get("value") or node.get("title") or "").strip()
            if not t:
                return
            if _HUDDLE_ANCHOR_RE.search(t):
                return  # the header itself
            if t.casefold() in ("you", "you (you)", "you (me)"):
                me = t
                return
            name = _slack_name(t)
            if not name or name.casefold() in seen:
                return
            seen.add(name.casefold())
            participants.append(name)

        walk(w, _names)

    return CallReading(
        speaking=[],
        participants=uniq(participants),
        me=me,
    )


# ── dispatcher ──────────────────────────────────────────────────────────────────

_APP_PARSERS: dict[str, Callable[[list[dict]], CallReading]] = {
    "Zoom": zoom,
    "Microsoft Teams": teams,
    "Google Meet": meet,
    "Webex": webex,
    "Slack Huddle": slack,
}


def parse(app_name: str, tree: list[dict]) -> CallReading:
    """Dispatch to the correct parser for the given app.

    Returns an empty ``CallReading`` when the app is unknown or the tree
    is empty — never raises.
    """
    parser = _APP_PARSERS.get(app_name)
    if parser is None:
        return CallReading()
    try:
        return parser(tree)
    except Exception:
        _log.debug("Parser failed for %s", app_name, exc_info=True)
        return CallReading()