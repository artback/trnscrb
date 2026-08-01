"""Tests for the custom-vocabulary glossary applied during transcription."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb import glossary


class GlossaryStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "glossary.json"
        patcher = patch.object(glossary, "_GLOSSARY_FILE", path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_empty_when_no_file(self):
        self.assertEqual(glossary.load(), [])

    def test_add_plain_terms(self):
        glossary.add_terms(["Hivenet", "Kubernetes"])
        self.assertEqual(glossary.terms(), ["Hivenet", "Kubernetes"])

    def test_add_term_with_aliases_and_merge(self):
        glossary.add_terms([{"term": "Hivenet", "aliases": ["high vnet"]}])
        glossary.add_terms([{"term": "hivenet", "aliases": ["hive net"]}])  # same term, new alias
        entries = glossary.load()
        self.assertEqual(len(entries), 1)  # merged, not duplicated
        self.assertEqual(entries[0]["term"], "Hivenet")
        self.assertEqual(entries[0]["aliases"], ["high vnet", "hive net"])

    def test_alias_equal_to_term_is_dropped(self):
        glossary.add_terms([{"term": "Hivenet", "aliases": ["Hivenet", "high vnet"]}])
        self.assertEqual(glossary.load()[0]["aliases"], ["high vnet"])

    def test_remove_term(self):
        glossary.add_terms(["Hivenet", "Kubernetes"])
        self.assertTrue(glossary.remove_term("hivenet"))  # case-insensitive
        self.assertEqual(glossary.terms(), ["Kubernetes"])
        self.assertFalse(glossary.remove_term("nope"))

    def test_whisper_hotwords(self):
        self.assertIsNone(glossary.whisper_hotwords())
        glossary.add_terms(["Hivenet", "pyannote"])
        self.assertEqual(glossary.whisper_hotwords(), "Hivenet, pyannote")

    def test_corrupt_file_is_treated_as_empty(self):
        glossary._GLOSSARY_FILE.write_text("{ not json", encoding="utf-8")
        self.assertEqual(glossary.load(), [])


class GlossaryCorrectTest(unittest.TestCase):
    def _entries(self):
        return [
            {"term": "Hivenet", "aliases": ["high vnet", "hive net"]},
            {"term": "Kubernetes", "aliases": ["kubernets"]},
        ]

    def test_alias_phrase_is_rewritten(self):
        out = glossary.correct("We deployed to high vnet today.", self._entries())
        self.assertEqual(out, "We deployed to Hivenet today.")

    def test_alias_is_case_insensitive_and_keeps_punctuation(self):
        out = glossary.correct("Is High Vnet up?", self._entries())
        self.assertEqual(out, "Is Hivenet up?")

    def test_canonical_casing_is_normalised(self):
        out = glossary.correct("running kubernetes in prod", self._entries())
        self.assertIn("Kubernetes", out)
        self.assertNotIn("kubernetes", out)

    def test_does_not_touch_unrelated_words(self):
        out = glossary.correct("the meeting went well", self._entries())
        self.assertEqual(out, "the meeting went well")

    def test_empty_glossary_is_noop(self):
        self.assertEqual(glossary.correct("anything", []), "anything")

    @patch.object(glossary.settings, "get", return_value=True)
    def test_fuzzy_corrects_near_miss(self, _get):
        out = glossary.correct("we run kubernetese here", self._entries())
        self.assertIn("Kubernetes", out)

    @patch.object(glossary.settings, "get", return_value=False)
    def test_fuzzy_disabled_leaves_near_miss(self, _get):
        out = glossary.correct("we run kubernetese here", self._entries())
        self.assertIn("kubernetese", out)

    @patch.object(glossary.settings, "get", return_value=True)
    def test_fuzzy_does_not_wreck_common_words(self, _get):
        # "meetings" is close to nothing distinctive; must be left alone.
        out = glossary.correct("back-to-back meetings today", self._entries())
        self.assertIn("meetings", out)


if __name__ == "__main__":
    unittest.main()
