"""Tests for learning real speaker names from an exported Meet transcript.

Meet is the only source that knows who a voice belongs to. Matching is by
text rather than time, because Meet's clock starts when someone presses
transcribe and ours starts on mic activity.
"""

import unittest
from pathlib import Path

from trnscrb import meet, storage

# ruff: noqa: E501
_MEET_EXPORT = """\
Polycloud Network Architecture Redesign - 2026/08/10 15:01 CEST
Attendees: Jonathan Artback, Mahmoud Barhamgi, Justin Lee

Transcript
00:00:18
Justin Lee: I am feeling a little outnumbered here today honestly and I would like to take a look at how we firewall those subnets across the sites
00:01:07
Mahmoud Barhamgi: I will give you some introduction but I wanted to talk about the proposal and the general idea for each site to have a dedicated control plane
00:03:38
Jonathan Artback: So either one machine because it is not so many nodes but we get no high availability
"""

_OURS = f"""\
Meeting: Polycloud Network Architecture Redesign
Date:    2026-08-10 15:01

{storage._SEPARATOR}

[SPEAKER_01]
  00:18  Man, I'm feeling a little outnumbered here today honestly and I'd have to take a look at how we firewall those subnets across the sites.

[SPEAKER_02]
  01:07  I'll I will give you some introduction but I wanted to talk about the proposal and the general idea for each site to have a dedicated control plane.

[Me]
  03:38  So either one machine because it is not so many nodes but we get no high availability.
"""


class ParseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "meet.txt"
        self.path.write_text(_MEET_EXPORT, encoding="utf-8")

    def test_turns_are_extracted(self):
        turns = meet.parse(self.path)
        self.assertEqual(
            [t["speaker"] for t in turns], ["Justin Lee", "Mahmoud Barhamgi", "Jonathan Artback"]
        )

    def test_header_lines_are_not_speakers(self):
        speakers = {t["speaker"] for t in meet.parse(self.path)}
        self.assertNotIn("Attendees", speakers)

    def test_timestamps_are_skipped(self):
        self.assertTrue(all(":" not in t["speaker"] for t in meet.parse(self.path)))

    def test_missing_file_is_empty(self):
        self.assertEqual(meet.parse(Path(self._tmp.name) / "nope.txt"), [])


class ParseOurTranscriptTest(unittest.TestCase):
    def test_blocks_and_text(self):
        blocks = meet.parse_transcript(_OURS)
        self.assertEqual([b["speaker"] for b in blocks], ["SPEAKER_01", "SPEAKER_02", "Me"])
        self.assertTrue(blocks[0]["text"].startswith("Man, I'm feeling"))
        self.assertNotIn("00:18", blocks[0]["text"])

    def test_header_is_excluded(self):
        self.assertNotIn("Meeting:", " ".join(b["text"] for b in meet.parse_transcript(_OURS)))


class MapSpeakersTest(unittest.TestCase):
    def setUp(self):
        tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "m.txt"
        path.write_text(_MEET_EXPORT, encoding="utf-8")
        self.theirs = meet.parse(path)
        self.ours = meet.parse_transcript(_OURS)

    def test_labels_map_to_real_names(self):
        mapping = meet.map_speakers(self.ours, self.theirs)
        self.assertEqual(mapping.get("SPEAKER_01"), "Justin Lee")
        self.assertEqual(mapping.get("SPEAKER_02"), "Mahmoud Barhamgi")

    def test_empty_inputs_map_nothing(self):
        self.assertEqual(meet.map_speakers([], self.theirs), {})
        self.assertEqual(meet.map_speakers(self.ours, []), {})

    def test_unrelated_transcript_names_nobody(self):
        other = [{"speaker": "X", "text": "completely unrelated words about gardening and soil"}]
        self.assertEqual(meet.map_speakers(other, self.theirs), {})

    def test_apply_names_rewrites_headings(self):
        out = meet.apply_names(_OURS, {"SPEAKER_01": "Justin Lee"})
        self.assertIn("[Justin Lee]", out)
        self.assertNotIn("[SPEAKER_01]", out)

    def test_apply_names_without_mapping_is_a_noop(self):
        self.assertEqual(meet.apply_names(_OURS, {}), _OURS)


class GlossaryCandidatesTest(unittest.TestCase):
    def test_repeated_substitution_is_suggested(self):
        ours = [{"speaker": "A", "text": "the poly cloud team and the poly cloud network"}]
        theirs = [{"speaker": "B", "text": "the policloud team and the policloud network"}]
        self.assertIn(("poly cloud", "policloud"), meet.glossary_candidates(ours, theirs))

    def test_one_off_difference_is_not_suggested(self):
        ours = [{"speaker": "A", "text": "alpha bravo charlie delta"}]
        theirs = [{"speaker": "B", "text": "alpha bravo chorlie delta"}]
        self.assertEqual(meet.glossary_candidates(ours, theirs), [])

    def test_identical_text_suggests_nothing(self):
        same = [{"speaker": "A", "text": "identical words here"}]
        self.assertEqual(meet.glossary_candidates(same, same), [])


if __name__ == "__main__":
    unittest.main()
