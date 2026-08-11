"""Tests for speaker-accurate segmentation and chunk-seam de-duplication.

Both defects were visible in a real 20-minute meeting: a three-minute block
labelled "Me" that contained four other people, and text like "also talk
about also talk about" from the 15s overlap between decoding chunks.
"""

import unittest

from trnscrb import diarizer, transcriber


def _words(spec):
    """[(text, start, end), …] -> word dicts."""
    return [{"text": t, "start": s, "end": e} for t, s, e in spec]


def _turns(spec):
    return [{"start": s, "end": e, "speaker": spk} for s, e, spk in spec]


class SplitBySpeakerTest(unittest.TestCase):
    def test_segment_is_cut_where_the_speaker_changes(self):
        seg = {
            "start": 0.0,
            "end": 8.0,
            "text": "one two three four",
            "speaker": None,
            "words": _words(
                [("one", 0.0, 1.5), ("two", 1.5, 3.0), ("three", 5.0, 6.5), ("four", 6.5, 8.0)]
            ),
        }
        turns = _turns([(0.0, 4.0, "SPEAKER_00"), (4.0, 8.0, "SPEAKER_01")])
        out = diarizer.merge([seg], turns)
        self.assertEqual(
            [(s["speaker"], s["text"]) for s in out],
            [("SPEAKER_00", "one two"), ("SPEAKER_01", "three four")],
        )

    def test_single_speaker_segment_is_left_whole(self):
        seg = {
            "start": 0.0,
            "end": 4.0,
            "text": "one two",
            "speaker": None,
            "words": _words([("one", 0.0, 2.0), ("two", 2.0, 4.0)]),
        }
        out = diarizer.merge([seg], _turns([(0.0, 4.0, "SPEAKER_00")]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], "SPEAKER_00")

    def test_momentary_run_is_folded_into_its_neighbour(self):
        """One word on another label is a boundary being slightly out."""
        seg = {
            "start": 0.0,
            "end": 6.0,
            "text": "a b c",
            "speaker": None,
            "words": _words([("a", 0.0, 2.0), ("b", 2.0, 2.4), ("c", 2.4, 6.0)]),
        }
        turns = _turns(
            [(0.0, 2.0, "SPEAKER_00"), (2.0, 2.4, "SPEAKER_01"), (2.4, 6.0, "SPEAKER_00")]
        )
        out = diarizer.merge([seg], turns)
        self.assertEqual(len(out), 1, "a 0.4s run should not split the sentence")

    def test_segments_without_words_keep_whole_segment_labelling(self):
        seg = {"start": 0.0, "end": 4.0, "text": "hi", "speaker": None}
        out = diarizer.merge([seg], _turns([(0.0, 4.0, "SPEAKER_02")]))
        self.assertEqual(out[0]["speaker"], "SPEAKER_02")

    def test_unmatched_segment_is_unknown(self):
        seg = {"start": 90.0, "end": 95.0, "text": "hi", "speaker": None}
        self.assertEqual(diarizer.merge([seg], _turns([(0.0, 4.0, "S")]))[0]["speaker"], "Unknown")


class DropRepeatsTest(unittest.TestCase):
    def _text(self, sentence):
        parts = sentence.split()
        words = [
            {"text": w, "raw": f" {w}", "start": float(i), "end": float(i + 1)}
            for i, w in enumerate(parts)
        ]
        return transcriber.words_to_text(transcriber._drop_repeats(words))

    def test_punctuation_from_the_repeat_is_kept(self):
        """ "Primary primary area area." must keep its full stop."""
        self.assertEqual(self._text("Primary primary area area."), "Primary area.")

    def test_subword_tokens_rebuild_into_words(self):
        class Tok:
            def __init__(self, text, start, end):
                self.text, self.start, self.end = text, start, end

        sentence = type("S", (), {})()
        sentence.tokens = [
            Tok(t, i * 0.1, i * 0.1 + 0.1)
            for i, t in enumerate(
                [" Pri", "mar", "y", " pri", "mar", "y", " are", "a", " are", "a", "."]
            )
        ]
        words = transcriber._parakeet_words(sentence)
        self.assertEqual([w["text"] for w in words], ["Primary", "primary", "area", "area."])
        self.assertEqual(
            transcriber.words_to_text(transcriber._drop_repeats(words)), "Primary area."
        )

    def test_repeated_phrase_is_collapsed(self):
        self.assertEqual(
            self._text("also talk about also talk about his proposal"),
            "also talk about his proposal",
        )

    def test_repeated_content_word_is_collapsed(self):
        self.assertEqual(
            self._text("fulfill all teams requirements requirements"),
            "fulfill all teams requirements",
        )

    def test_natural_repeats_are_kept(self):
        for phrase in ("yeah yeah of course", "okay okay lets start", "no no here is fine"):
            self.assertEqual(self._text(phrase), phrase)

    def test_short_words_are_kept(self):
        self.assertEqual(self._text("it is is fine"), "it is is fine")

    def test_unrepeated_text_is_untouched(self):
        self.assertEqual(
            self._text("the network requirements are clear"), "the network requirements are clear"
        )

    def test_longer_repeated_run(self):
        self.assertEqual(
            self._text("we can start by we can start by creating"), "we can start by creating"
        )

    def test_empty_input(self):
        self.assertEqual(transcriber._drop_repeats([]), [])


if __name__ == "__main__":
    unittest.main()
