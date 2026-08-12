"""Tests for the Obsidian properties that make a graph out of transcripts.

Before this, every meeting note was a leaf: the action-items index linked to
them and they linked to nothing, so the graph was a star with one hub.
"""

import unittest
from datetime import datetime
from unittest import mock

from trnscrb import obsidian

_BODY = """\
Meeting: Polycloud Network Architecture Redesign
Date:    2026-08-10 15:01

============================================================

[Mahmoud Barhamgi]
  01:07  The Policloud network uses Tailscale across every site.

[SPEAKER_02]
  03:38  So either one machine, or three for high availability.

[Me]
  04:26  The main issue we have seen is NAT traversal.

[Mahmoud Barhamgi]
  07:16  We can talk about that.
"""


class SpeakersTest(unittest.TestCase):
    def test_real_names_are_found_once_each(self):
        self.assertEqual(obsidian.speakers_in(_BODY), ["Mahmoud Barhamgi"])

    def test_placeholder_labels_are_not_people(self):
        """SPEAKER_02 and Me would be junk nodes joined to every meeting."""
        found = obsidian.speakers_in(_BODY)
        for junk in ("SPEAKER_02", "Me", "Them", "Unknown"):
            self.assertNotIn(junk, found)

    def test_no_speakers_is_empty(self):
        self.assertEqual(obsidian.speakers_in("no headings here"), [])


class TopicsTest(unittest.TestCase):
    def _topics(self, terms):
        with mock.patch("trnscrb.glossary.terms", return_value=terms):
            return obsidian.topics_in(_BODY)

    def test_mentioned_glossary_terms_become_topics(self):
        self.assertEqual(
            sorted(self._topics(["Policloud", "Tailscale", "Proxmox"])), ["Policloud", "Tailscale"]
        )

    def test_unmentioned_terms_are_skipped(self):
        self.assertEqual(self._topics(["Kubernetes"]), [])

    def test_substrings_do_not_count(self):
        """'NAT' must not match inside 'Nation'."""
        with mock.patch("trnscrb.glossary.terms", return_value=["ail"]):
            self.assertEqual(obsidian.topics_in("Tailscale is fine"), [])


class SeriesTest(unittest.TestCase):
    def test_recurring_meeting_collapses_to_its_series(self):
        self.assertEqual(obsidian.series_name("Daily Standup 2026-08-10"), "Daily Standup")
        self.assertEqual(obsidian.series_name("Sprint Review #12"), "Sprint Review")

    def test_plain_name_is_its_own_series(self):
        self.assertEqual(obsidian.series_name("Polycloud Redesign"), "Polycloud Redesign")


class BuildNoteTest(unittest.TestCase):
    def _note(self, name="Polycloud Network Architecture Redesign", terms=("Policloud",)):
        with mock.patch("trnscrb.glossary.terms", return_value=list(terms)):
            return obsidian.build_note(name, datetime(2026, 8, 10, 15, 1), _BODY, "20:51")

    def test_frontmatter_is_first(self):
        note = self._note()
        self.assertTrue(note.startswith("---\n"))
        self.assertIn("\n---\n", note)

    def test_properties_are_present(self):
        note = self._note()
        self.assertIn("date: 2026-08-10", note)
        self.assertIn('time: "15:01"', note)
        self.assertIn('duration: "20:51"', note)
        self.assertIn("- meeting", note)

    def test_people_and_topics_are_links(self):
        note = self._note()
        self.assertIn('"[[Mahmoud Barhamgi]]"', note)
        self.assertIn('"[[Policloud]]"', note)

    def test_transcript_body_is_preserved_verbatim(self):
        self.assertTrue(self._note().endswith(_BODY))

    def test_series_link_only_for_recurring_names(self):
        self.assertNotIn("series:", self._note())
        self.assertIn('series: "[[Daily Standup]]"', self._note(name="Daily Standup 2026-08-10"))

    def test_note_without_people_or_topics_still_valid(self):
        with mock.patch("trnscrb.glossary.terms", return_value=[]):
            note = obsidian.build_note("X", datetime(2026, 8, 10, 15, 1), "plain text")
        self.assertNotIn("attendees:", note)
        self.assertNotIn("topics:", note)
        self.assertTrue(note.endswith("plain text"))


if __name__ == "__main__":
    unittest.main()
