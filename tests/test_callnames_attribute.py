"""Tests for callnames.attribute — temporal attribution.

Covers:
- >50% overlap rule (a name must dominate a segment)
- HOLD_MS: a stuck reading never keeps naming someone
- Two-person special case (non-me gets non-me cluster)
- No sample coverage → no name
- Short segments
- Multiple speakers (only the dominant one wins)
"""

from __future__ import annotations

import unittest

from trnscrb.callnames import attribute as _attr


class TestNameFor(unittest.TestCase):
    """name_for: temporal attribution of a single segment."""

    def _sample(self, ts: float, speaking: list[str], participants: list[str], me=None, ok=True) -> _attr.Sample:
        return _attr.Sample(ts=ts, speaking=speaking, participants=participants, me=me, ok=ok)

    def test_no_samples(self):
        self.assertIsNone(_attr.name_for(0, 5000, []))

    def test_no_coverage(self):
        samples = [self._sample(100.0, [], [], ok=False)]
        self.assertIsNone(_attr.name_for(0, 5000, samples))

    def test_five_second_segment_ann_dominates(self):
        """Ann speaks for 80% of the segment → name is Ann."""
        samples = [
            self._sample(0.0, [], ["Ann", "Bo"], me=None),
            self._sample(0.4, ["Ann"], ["Ann", "Bo"], me=None),
            self._sample(0.8, ["Ann"], ["Ann", "Bo"], me=None),
            self._sample(1.2, ["Ann"], ["Ann", "Bo"], me=None),
            self._sample(1.6, ["Ann"], ["Ann", "Bo"], me=None),
            self._sample(2.0, [], ["Ann", "Bo"], me=None),
        ]
        name = _attr.name_for(100, 2200, samples)
        self.assertEqual(name, "Ann")

    def test_no_one_dominates(self):
        """Two people each speak ~50% → neither crosses the 50% threshold."""
        samples = [
            self._sample(0.0, [], ["Ann", "Bo"], me=None),
            self._sample(0.4, ["Ann"], ["Ann", "Bo"], me=None),
            self._sample(0.8, ["Ann"], ["Ann", "Bo"], me=None),
            self._sample(1.2, ["Bo"], ["Ann", "Bo"], me=None),
            self._sample(1.6, ["Bo"], ["Ann", "Bo"], me=None),
            self._sample(2.0, [], ["Ann", "Bo"], me=None),
        ]
        name = _attr.name_for(400, 5400, samples)
        self.assertIsNone(name)

    def test_200ms_hold_capped(self):
        """A reading stands at most HOLD_MS.  A 2.5s gap → the reading at t=0
        is alive until t=2.0, not until the next reading at t=2.5."""
        s0 = self._sample(0.0, ["Ann"], ["Ann", "Bo"], me=None)
        s1 = self._sample(2.5, [], ["Ann", "Bo"], me=None)
        # Segment [0.2, 2.2] spans t=0→2.0 (HOLD_MS) and t=2.0→2.5
        # Ann gets 1.8s out of 2.0s → 90% → wins.
        name = _attr.name_for(200, 2200, [s0, s1])
        self.assertEqual(name, "Ann")

    def test_short_segment_returns_immediate(self):
        """A segment shorter than HOLD_MS: return the name at its start."""
        s0 = self._sample(0.0, ["Ann"], ["Ann", "Bo"], me=None)
        s1 = self._sample(0.5, [], ["Ann", "Bo"], me=None)
        # Segment 100ms at 0.2s → only s0 covers it.
        name = _attr.name_for(200, 300, [s0, s1])
        self.assertEqual(name, "Ann")

    def test_gap_in_samples_no_coverage(self):
        """Both samples are before the segment → no name."""
        s0 = self._sample(0.0, ["Ann"], ["Ann", "Bo"], me=None)
        s1 = self._sample(0.4, ["Ann"], ["Ann", "Bo"], me=None)
        # Segment starts at 5s — beyond all samples.
        name = _attr.name_for(5000, 5500, [s0, s1])
        self.assertIsNone(name)

    def test_2000ms_hold_edge(self):
        """At exactly HOLD_MS boundary, the reading expires and no name is
        assigned (next reading has no one speaking)."""
        s0 = self._sample(0.0, ["Ann"], ["Ann", "Bo"], me=None)
        s1 = self._sample(2.0, [], ["Ann", "Bo"], me=None)  # exactly 2000ms later
        # s0's window: [0, 2000]ms. s1's window: [2000, 4000]ms.
        # Segment [1000, 3000]ms → 1000ms overlap with Ann, 1000ms with nobody.
        # 1000/2000 = 50% → NOT >50% → no name.
        name = _attr.name_for(1000, 3000, [s0, s1])
        self.assertIsNone(name)


class TestAttributedMeeting(unittest.TestCase):
    """attributed_meeting: full meeting attribution."""

    def _samples(self, speaking_pattern: list[list[str]]) -> list[_attr.Sample]:
        """Build 75 samples at 400ms intervals."""
        return [
            _attr.Sample(
                ts=0.4 * i,
                speaking=speaking_pattern[min(i, len(speaking_pattern) - 1)]
                if speaking_pattern
                else [],
                participants=["Ann", "Bo", "Cy"],
                me=None,
                ok=True,
            )
            for i in range(75)
        ]

    def test_full_meeting_ann_dominates_first_half(self):
        speaking_pattern = [
            ["Ann"] if i < 25 else (["Bo"] if i < 50 else [])
            for i in range(75)
        ]
        samples = self._samples(speaking_pattern)
        segments = [
            _attr.Segment(start_ms=100, end_ms=2200, speaker="SPEAKER_0"),  # Ann time
            _attr.Segment(start_ms=10100, end_ms=22000, speaker="SPEAKER_1"),  # Bo time
        ]
        result = _attr.attributed_meeting(samples, segments)
        self.assertEqual(result.result.get("SPEAKER_0"), "Ann")
        self.assertEqual(result.result.get("SPEAKER_1"), "Bo")

    def test_no_meeting_coverage(self):
        samples = self._samples([])
        segments = [_attr.Segment(start_ms=10000, end_ms=15000, speaker="SPEAKER_0")]
        result = _attr.attributed_meeting(samples, segments)
        self.assertEqual(result.result, {})
        self.assertEqual(result.participants, ["Ann", "Bo", "Cy"])

    def test_empty_samples(self):
        result = _attr.attributed_meeting([], [])
        self.assertEqual(result.result, {})
        self.assertEqual(result.participants, [])


class TestTwoPersonSpecialCase(unittest.TestCase):
    """Two-person call: assign non-me name regardless of speaking markers."""

    def _s(self, ts, speaking, me):
        return _attr.Sample(
            ts=ts, speaking=speaking,
            participants=["You", "Ann"],
            me=me, ok=True,
        )

    def test_two_person_non_me_name(self):
        """Even with no speaking markers, the non-me person gets the name."""
        samples = [self._s(0.0, [], "You")]
        segments = [_attr.Segment(start_ms=100, end_ms=5000, speaker="SPEAKER_1")]
        result = _attr.attributed_meeting(samples, segments, my_names=["You"])
        self.assertEqual(result.result["SPEAKER_1"], "Ann")

    def test_two_person_me_cluster(self):
        """The non-me name goes to the non-me cluster, not "Me" cluster."""
        samples = [self._s(0.0, [], "You")]
        segments = [
            _attr.Segment(start_ms=100, end_ms=5000, speaker="Me"),
            _attr.Segment(start_ms=5000, end_ms=10000, speaker="SPEAKER_1"),
        ]
        result = _attr.attributed_meeting(samples, segments, my_names=["You"])
        self.assertNotIn("Me", result.result)
        self.assertEqual(result.result["SPEAKER_1"], "Ann")

    def test_three_people_no_special_case(self):
        """Three+ people: no two-person special case, use overlap."""
        samples = [
            _attr.Sample(
                ts=0.0, speaking=["Ann"],
                participants=["You", "Ann", "Bo"], me="You", ok=True,
            )
        ]
        segments = [_attr.Segment(start_ms=100, end_ms=5000, speaker="SPEAKER_1")]
        result = _attr.attributed_meeting(
            samples, segments, my_names=["You"],
        )
        # Ann is speaking, but with 3 participants the two-person logic
        # doesn't apply; name_for needs >50% overlap. One sample → Ann gets
        # 1900ms out of 4900ms ≈ 39%.
        self.assertIsNone(result.result.get("SPEAKER_1"))


class TestOthersIn(unittest.TestCase):
    def test_basic(self):
        samples = [
            _attr.Sample(ts=0.0, speaking=["Ann"], participants=["You", "Ann"], me="You"),
            _attr.Sample(ts=1.0, speaking=["Bo"], participants=["You", "Ann", "Bo"], me="You"),
        ]
        result = _attr.others_in(samples, my_names=["You"])
        self.assertEqual(result, ["Ann", "Bo"])

    def test_excludes_me(self):
        samples = [
            _attr.Sample(ts=0.0, speaking=["You", "Ann"], participants=["You", "Ann"], me="You"),
        ]
        result = _attr.others_in(samples, my_names=["You"])
        self.assertEqual(result, ["Ann"])

    def test_case_insensitive(self):
        samples = [
            _attr.Sample(ts=0.0, speaking=[], participants=["ann", "ANN", "Ann"], me="You"),
        ]
        result = _attr.others_in(samples, my_names=["You"])
        self.assertEqual(result, ["ann"])


if __name__ == "__main__":
    unittest.main()