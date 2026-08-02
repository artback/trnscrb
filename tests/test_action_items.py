"""Tests for cross-meeting action-item tracking (only the user's own items)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb import action_items

_ENRICHMENT = """TITLE:
Cloud sync

SUMMARY:
We discussed the rollout.

ACTION ITEMS:
- Add auth to shared links PROJ-123 (Owner: Me)
- Draft the pricing note (Owner: Andre)
- Review the numbers (Owner: Unknown)

SPEAKER MAPPING:
- SPEAKER_00 → Me
"""


class OwnershipFilterTest(unittest.TestCase):
    def setUp(self):
        # Deterministic ownership: no configured name, nothing "looks like self".
        patch.object(action_items.settings, "get", return_value="").start()
        patch.object(action_items, "_looks_like_self", return_value=False).start()
        self.addCleanup(patch.stopall)

    def test_only_my_items_are_parsed(self):
        items = action_items.parse_action_items(_ENRICHMENT)
        texts = [i["text"] for i in items]
        # "Me" and "Unknown" kept; "Andre" (a named other) dropped.
        self.assertIn("Add auth to shared links PROJ-123", texts)
        self.assertIn("Review the numbers", texts)
        self.assertTrue(all("pricing note" not in t for t in texts))

    def test_extracts_jira_and_github(self):
        items = action_items.parse_action_items(
            "ACTION ITEMS:\n- Fix login PROJ-9 https://github.com/o/r/issues/5 (Owner: Me)"
        )
        self.assertEqual(items[0]["jira"], ["PROJ-9"])
        self.assertEqual(items[0]["github"], ["https://github.com/o/r/issues/5"])

    def test_configured_user_name_counts_as_self(self):
        with patch.object(action_items.settings, "get", return_value="Andre"):
            items = action_items.parse_action_items(_ENRICHMENT)
        texts = [i["text"] for i in items]
        self.assertTrue(any("pricing note" in t for t in texts))  # now "Andre" == me


class RecordAndResolveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patch.object(action_items, "_STORE", Path(self._tmp.name) / "ai.json").start()
        patch.object(action_items.settings, "get", return_value="").start()
        patch.object(action_items, "_looks_like_self", return_value=False).start()
        # No Obsidian vault → render/sync are no-ops.
        patch.object(action_items.obsidian, "meetings_dir", return_value=None).start()
        patch.object(action_items.obsidian, "read_note", return_value=None).start()
        self.addCleanup(patch.stopall)

    def test_records_only_my_items(self):
        stats = action_items.record_meeting(
            _ENRICHMENT, [], "m1", "Cloud sync", "2026-07-30 Cloud sync", "2026-07-30"
        )
        self.assertEqual(stats["added"], 2)  # Me + Unknown, not Andre
        self.assertEqual(len(action_items.open_items()), 2)

    def test_dedup_across_meetings(self):
        action_items.record_meeting(_ENRICHMENT, [], "m1", "A", None, "2026-07-30")
        stats = action_items.record_meeting(_ENRICHMENT, [], "m2", "B", None, "2026-07-31")
        self.assertEqual(stats["added"], 0)  # same items, not duplicated
        self.assertEqual(len(action_items.load()), 2)

    def test_resolved_indices_close_items(self):
        action_items.record_meeting(_ENRICHMENT, [], "m1", "A", None, "2026-07-30")
        snapshot = action_items.open_items()
        # A later meeting says item #1 is done.
        later = "ACTION ITEMS:\n- Something new (Owner: Me)\n\nRESOLVED:\n1"
        stats = action_items.record_meeting(later, snapshot, "m2", "B", None, "2026-08-01")
        self.assertEqual(stats["resolved"], 1)
        done = [i for i in action_items.load() if i["status"] == "done"]
        self.assertEqual(done[0]["id"], snapshot[0]["id"])

    def test_manual_resolve_and_add_and_link(self):
        rec = action_items.add("Ship the thing", owner="Me")
        self.assertTrue(action_items.link(rec["id"], jira="PROJ-1"))
        self.assertIn("PROJ-1", action_items.load()[0]["jira"])
        self.assertTrue(action_items.resolve(rec["id"], "shipped"))
        self.assertEqual(action_items.load()[0]["status"], "done")


class ObsidianRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.vault = Path(self._tmp.name)
        patch.object(action_items, "_STORE", self.vault / "ai.json").start()
        patch.object(action_items.settings, "get", return_value="").start()
        patch.object(action_items, "_looks_like_self", return_value=False).start()
        patch.object(action_items.obsidian, "meetings_dir", return_value=self.vault).start()
        self.addCleanup(patch.stopall)

    def test_render_writes_tasks_note_with_ids(self):
        action_items.add("Add auth PROJ-5", owner="Me")
        note = (self.vault / "Action Items.md").read_text()
        self.assertIn("- [ ] Add auth PROJ-5", note)
        self.assertIn("[jira:: PROJ-5]", note)
        self.assertRegex(note, r"\^[0-9a-f]{6,}")  # block-id anchor for round-trip

    def test_ticking_box_in_obsidian_closes_item(self):
        rec = action_items.add("Do the thing", owner="Me")
        note_path = self.vault / "Action Items.md"
        note_path.write_text(note_path.read_text().replace("- [ ]", "- [x]"))
        changed = action_items.sync_from_obsidian()
        self.assertEqual(changed, 1)
        self.assertEqual(action_items.load()[0]["status"], "done")
        self.assertEqual(action_items.load()[0]["id"], rec["id"])


if __name__ == "__main__":
    unittest.main()
