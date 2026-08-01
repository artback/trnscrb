"""Tests for content-derived meeting titles."""

import unittest

from trnscrb import enricher, titles

_ENRICHMENT = """TITLE:
Cloud artifacts POC

SUMMARY:
We agreed to ship the cloud-artifacts POC behind auth.

ACTION ITEMS:
- Add authentication (Owner: Jonathan)

SPEAKER MAPPING:
- SPEAKER_00 → Jonathan
"""


class IsGenericTest(unittest.TestCase):
    def test_generic_names(self):
        for name in ["", "meeting-1549", "Google Meet", "google meet", "Zoom", "meeting"]:
            self.assertTrue(titles.is_generic(name), name)

    def test_real_names_are_not_generic(self):
        for name in ["Q3 Planning", "Standup with Andre", "1:1 Jonathan / David"]:
            self.assertFalse(titles.is_generic(name), name)


class FromEnrichmentTest(unittest.TestCase):
    def test_parses_title_on_next_line(self):
        self.assertEqual(titles.from_enrichment(_ENRICHMENT), "Cloud artifacts POC")

    def test_parses_inline_title(self):
        self.assertEqual(
            titles.from_enrichment("TITLE: Budget review\nSUMMARY:\nx"), "Budget review"
        )

    def test_strips_quotes_and_trailing_period(self):
        self.assertEqual(titles.from_enrichment('TITLE: "Roadmap sync."'), "Roadmap sync")

    def test_none_when_absent(self):
        self.assertIsNone(titles.from_enrichment("SUMMARY:\nno title here"))
        self.assertIsNone(titles.from_enrichment(None))


class LocalFallbackTest(unittest.TestCase):
    def test_picks_repeated_keywords(self):
        segs = [
            {"text": "The kubernetes migration is blocking the kubernetes rollout"},
            {"text": "We need the migration plan for kubernetes"},
        ]
        title = titles.local(segs)
        self.assertIsNotNone(title)
        self.assertIn("Kubernetes", title)

    def test_none_when_too_little_signal(self):
        self.assertIsNone(titles.local([{"text": "hi there"}]))


class SummaryBlockSkipsTitleTest(unittest.TestCase):
    def test_summary_block_excludes_title_and_mapping(self):
        block = enricher.summary_block(_ENRICHMENT)
        self.assertNotIn("TITLE:", block)
        self.assertNotIn("Cloud artifacts POC", block)
        self.assertNotIn("SPEAKER MAPPING", block)
        self.assertTrue(block.startswith("SUMMARY:"))
        self.assertIn("ACTION ITEMS:", block)

    def test_fallback_when_no_summary_header(self):
        # Older/looser model output without a SUMMARY header still yields text.
        text = "Some free-form notes\n\nSPEAKER MAPPING:\n- SPEAKER_00 → X"
        self.assertEqual(enricher.summary_block(text), "Some free-form notes")


if __name__ == "__main__":
    unittest.main()
