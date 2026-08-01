"""Cross-meeting action-item tracking, surfaced in Obsidian.

Only the *user's own* commitments are tracked — a task assigned to someone else
in a standup never lands on the list. Items are extracted from each meeting's
summary, de-duplicated across meetings, and written to an Obsidian note as
Tasks-plugin checkboxes with Dataview fields and [[backlinks]].

The JSON store is the source of truth; the note is kept in sync both ways —
ticking a box in Obsidian marks the item done here, and the summary model can
propose which open items a new meeting resolved.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from trnscrb import obsidian, settings
from trnscrb.log import get_logger

_log = get_logger("trnscrb.action_items")

_STORE = Path.home() / ".config" / "trnscrb" / "action_items.json"
_NOTE = "Action Items.md"

_JIRA = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_GITHUB = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+")
# A checkbox line in the Obsidian note, anchored by its block id (^abc123).
_TASK_LINE = re.compile(r"^- \[([ xX])\]\s+.*\s\^([0-9a-f]{6,})\s*$")


# ── store ───────────────────────────────────────────────────────────────────


def load() -> list[dict]:
    try:
        raw = json.loads(_STORE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw.get("items", []) if isinstance(raw, dict) else []


def save(items: list[dict]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")


def open_items() -> list[dict]:
    return [i for i in load() if i.get("status") == "open"]


# ── ownership (only the user's own tasks) ────────────────────────────────────


def _owner_kind(owner: str) -> str:
    """Classify an action-item owner as 'self', 'unknown', or 'other'."""
    name = str(owner or "").strip().casefold()
    if name in ("", "unknown", "n/a", "tbd", "none"):
        return "unknown"
    if name in ("me", "i", "myself", "self"):
        return "self"
    configured = str(settings.get("user_name") or "").strip().casefold()
    if configured and (configured == name or configured in name.split()):
        return "self"
    if _looks_like_self(owner):
        return "self"
    return "other"


def _looks_like_self(owner: str) -> bool:
    import getpass
    import os

    candidates = {
        os.environ.get("USER", ""),
        getpass.getuser(),
        os.environ.get("TRNSCRB_USER_NAME", ""),
    }
    name = owner.casefold()
    for candidate in candidates:
        candidate = candidate.strip().casefold()
        if candidate and (candidate == name or candidate in name.split()):
            return True
    return False


def _is_mine(owner: str) -> bool:
    # Keep the user's own items and genuinely-unassigned ones (the summary is
    # already scoped to the user); drop anything owned by a named other person.
    return _owner_kind(owner) in ("self", "unknown")


# ── parsing enrichment output ────────────────────────────────────────────────


def parse_action_items(enrichment: str) -> list[dict]:
    """Extract the user's action items from an enrichment's ACTION ITEMS section."""
    items = []
    for line in _section(enrichment, "ACTION ITEMS:"):
        stripped = line.strip()
        if not stripped.startswith(("-", "*")):
            continue
        body = stripped[1:].strip()
        if not body:
            continue
        owner = ""
        match = re.search(r"\(owner:\s*(.*?)\)\s*$", body, re.IGNORECASE)
        if match:
            owner = match.group(1).strip()
            body = body[: match.start()].strip()
        if not body or not _is_mine(owner):
            continue
        items.append(
            {
                "text": body,
                "owner": owner,
                "jira": sorted(set(_JIRA.findall(body))),
                "github": sorted(set(_GITHUB.findall(body))),
            }
        )
    return items


def parse_resolved_indices(enrichment: str) -> list[int]:
    """1-based indices the model marked resolved in the RESOLVED section."""
    nums: list[int] = []
    for line in _section(enrichment, "RESOLVED:"):
        for token in re.findall(r"\d+", line):
            nums.append(int(token))
    return nums


def _section(text: str, header: str):
    """Yield the lines under ``header`` up to the next ALLCAPS section header."""
    lines = (text or "").splitlines()
    inside = False
    for line in lines:
        stripped = line.strip()
        if not inside:
            if stripped.upper().startswith(header.upper()):
                inside = True
            continue
        if re.match(r"^[A-Z][A-Z ]+:$", stripped):  # next section
            break
        yield line


# ── recording a meeting ──────────────────────────────────────────────────────


def record_meeting(
    enrichment: str,
    open_snapshot: list[dict],
    meeting_id: str,
    meeting_title: str,
    note_name: str | None,
    when: str,
) -> dict:
    """Fold a meeting's action items into the store and refresh the Obsidian note.

    ``open_snapshot`` is the open-items list handed to the model (same order), so
    the RESOLVED indices map back to real items. Returns a small summary dict.
    """
    sync_from_obsidian()  # pick up any boxes ticked in Obsidian first
    items = load()
    by_id = {i["id"]: i for i in items}

    # 1. Close items the model says this meeting resolved.
    resolved = 0
    for idx in parse_resolved_indices(enrichment):
        if 1 <= idx <= len(open_snapshot):
            target = by_id.get(open_snapshot[idx - 1]["id"])
            if target and target.get("status") == "open":
                target["status"] = "done"
                target["done_date"] = when
                target["done_reason"] = f"resolved in {meeting_title}"
                resolved += 1

    # 2. Add / refresh the user's own action items from this meeting.
    added = 0
    for parsed in parse_action_items(enrichment):
        item_id = _item_id(parsed["text"])
        existing = by_id.get(item_id)
        if existing:
            existing["last_seen"] = when
            existing["jira"] = sorted(set(existing.get("jira", []) + parsed["jira"]))
            existing["github"] = sorted(set(existing.get("github", []) + parsed["github"]))
        else:
            record = {
                "id": item_id,
                "text": parsed["text"],
                "owner": parsed["owner"] or "Me",
                "status": "open",
                "created": when,
                "last_seen": when,
                "meeting_id": meeting_id,
                "meeting_title": meeting_title,
                "note": note_name or "",
                "jira": parsed["jira"],
                "github": parsed["github"],
                "done_date": "",
                "done_reason": "",
            }
            items.append(record)
            by_id[item_id] = record
            added += 1

    save(items)
    render()
    _log.info("Action items: +%d new, %d resolved (%s)", added, resolved, meeting_title)
    return {"added": added, "resolved": resolved}


# ── manual operations (CLI / MCP) ────────────────────────────────────────────


def resolve(item_id: str, reason: str = "") -> bool:
    items = load()
    for item in items:
        if item["id"] == item_id and item.get("status") == "open":
            item["status"] = "done"
            item["done_date"] = _today()
            item["done_reason"] = reason or "closed manually"
            save(items)
            render()
            return True
    return False


def add(text: str, owner: str = "Me") -> dict | None:
    text = str(text).strip()
    if not text:
        return None
    items = load()
    item_id = _item_id(text)
    if any(i["id"] == item_id for i in items):
        return None  # already tracked
    record = {
        "id": item_id,
        "text": text,
        "owner": owner or "Me",
        "status": "open",
        "created": _today(),
        "last_seen": _today(),
        "meeting_id": "",
        "meeting_title": "manual",
        "note": "",
        "jira": sorted(set(_JIRA.findall(text))),
        "github": sorted(set(_GITHUB.findall(text))),
        "done_date": "",
        "done_reason": "",
    }
    items.append(record)
    save(items)
    render()
    return record


def link(item_id: str, jira: str = "", github: str = "") -> bool:
    items = load()
    for item in items:
        if item["id"] == item_id:
            if jira:
                item["jira"] = sorted(set(item.get("jira", []) + [jira.strip()]))
            if github:
                item["github"] = sorted(set(item.get("github", []) + [github.strip()]))
            save(items)
            render()
            return True
    return False


# ── Obsidian round-trip ──────────────────────────────────────────────────────


def sync_from_obsidian() -> int:
    """Mark items done whose checkbox was ticked in the Obsidian note."""
    text = obsidian.read_note(_NOTE)
    if not text:
        return 0
    checked = set()
    for line in text.splitlines():
        match = _TASK_LINE.match(line.strip())
        if match and match.group(1).lower() == "x":
            checked.add(match.group(2))
    items = load()
    changed = 0
    for item in items:
        if item["id"] in checked and item.get("status") == "open":
            item["status"] = "done"
            item["done_date"] = _today()
            item["done_reason"] = "checked in Obsidian"
            changed += 1
    if changed:
        save(items)
    return changed


def render() -> None:
    """Write the Obsidian action-items note from the store (no-op without a vault)."""
    if obsidian.meetings_dir() is None:
        return
    items = load()
    open_i = [i for i in items if i.get("status") == "open"]
    done_i = [i for i in items if i.get("status") == "done"]
    lines = [
        "# Action Items",
        "",
        "*Maintained by trnscrb — tick a box to close an item. "
        "Ask Claude to sync these with Jira/GitHub.*",
        "",
        f"## Open ({len(open_i)})",
        "",
    ]
    lines += [_render_item(i, done=False) for i in open_i] or ["*Nothing open.*"]
    lines += ["", f"## Done ({len(done_i)})", ""]
    lines += [_render_item(i, done=True) for i in done_i[-50:]] or ["*Nothing done yet.*"]
    obsidian.write_note(_NOTE, "\n".join(lines) + "\n")


def _render_item(item: dict, done: bool) -> str:
    box = "[x]" if done else "[ ]"
    fields = [f"[owner:: {item.get('owner', 'Me')}]"]
    if item.get("note"):
        fields.append(f"[meeting:: [[{item['note']}]]]")
    if done and item.get("done_date"):
        fields.append(f"[done:: {item['done_date']}]")
    elif item.get("created"):
        fields.append(f"[created:: {item['created']}]")
    if item.get("jira"):
        fields.append(f"[jira:: {', '.join(item['jira'])}]")
    if item.get("github"):
        fields.append(f"[github:: {', '.join(item['github'])}]")
    return f"- {box} {item['text']}  " + "  ".join(fields) + f" ^{item['id']}"


# ── helpers ───────────────────────────────────────────────────────────────


def _item_id(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")
