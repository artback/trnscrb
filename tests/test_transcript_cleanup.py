"""Tests for transcript readability cleanup — filler removal + de-stuttering."""

import unittest

from trnscrb import storage


class CollapseRepeatsTest(unittest.TestCase):
    def test_collapses_stutter_function_words(self):
        # The recogniser's classic artifacts: a single duplicate is enough.
        self.assertEqual(storage.collapse_repeats("you you know"), "you know")
        self.assertEqual(storage.collapse_repeats("it it was fine"), "it was fine")
        self.assertEqual(storage.collapse_repeats("I I think so"), "I think so")
        self.assertEqual(storage.collapse_repeats("the the the report"), "the report")

    def test_collapses_stuttered_contractions(self):
        self.assertEqual(storage.collapse_repeats("it's it's only fair"), "it's only fair")
        self.assertEqual(storage.collapse_repeats("that's that's the plan"), "that's the plan")

    def test_keeps_deliberate_content_word_doubling(self):
        # Two in a row of a content word is likely emphasis — leave it.
        self.assertEqual(storage.collapse_repeats("very very good"), "very very good")
        self.assertEqual(storage.collapse_repeats("no no thanks"), "no no thanks")

    def test_collapses_triple_content_word(self):
        # Three in a row is a stutter regardless of the word.
        self.assertEqual(storage.collapse_repeats("no no no thanks"), "no thanks")

    def test_collapses_repeated_bigram(self):
        self.assertEqual(storage.collapse_repeats("I think I think we should"), "I think we should")
        self.assertEqual(
            storage.collapse_repeats("you know you know what I mean"),
            "you know what I mean",
        )

    def test_preserves_case_and_punctuation_of_survivor(self):
        self.assertEqual(storage.collapse_repeats("So so, anyway"), "So, anyway")

    def test_is_case_insensitive_when_matching(self):
        self.assertEqual(storage.collapse_repeats("The the thing"), "The thing")

    def test_leaves_clean_text_untouched(self):
        text = "This is a perfectly normal sentence."
        self.assertEqual(storage.collapse_repeats(text), text)

    def test_handles_short_input(self):
        self.assertEqual(storage.collapse_repeats(""), "")
        self.assertEqual(storage.collapse_repeats("hi"), "hi")

    def test_does_not_merge_distinct_words(self):
        # A repeat must be immediate — "you and you" keeps both.
        self.assertEqual(storage.collapse_repeats("you and you"), "you and you")


class ReadableTextTest(unittest.TestCase):
    def test_strips_fillers_then_destutters(self):
        # "uh" is a filler; the doubled "the" is a stutter — both go.
        self.assertEqual(storage.readable_text("uh the the plan"), "the plan")

    def test_realistic_noisy_line(self):
        raw = "I s I tested the the one that I I just did"
        # filler removal leaves the "s" (not a filler); de-stutter fixes repeats.
        out = storage.readable_text(raw)
        self.assertNotIn("the the", out)
        self.assertNotIn("I I", out)


class FillerDialBackTest(unittest.TestCase):
    def test_strips_hesitation_sounds(self):
        self.assertEqual(storage.clean_filler_words("um so uh yeah"), "so yeah")
        self.assertEqual(storage.clean_filler_words("erm okay"), "okay")

    def test_keeps_meaning_bearing_words(self):
        # These were previously stripped and corrupted meaning — must survive now.
        self.assertEqual(storage.clean_filler_words("I like it"), "I like it")
        self.assertEqual(storage.clean_filler_words("turn right here"), "turn right here")
        self.assertEqual(storage.clean_filler_words("it actually works"), "it actually works")
        self.assertEqual(storage.clean_filler_words("kind of blue"), "kind of blue")

    def test_you_know_i_mean_no_longer_eaten(self):
        # The reported bug: this used to collapse to "what".
        out = storage.readable_text("you know you know what I mean")
        self.assertEqual(out, "you know what I mean")


if __name__ == "__main__":
    unittest.main()
