"""Tests for Me/Them speaker attribution from the dual-stream energy timeline."""

import unittest

import numpy as np

from trnscrb import attribution
from trnscrb.recorder import SAMPLE_RATE, Recorder


def _timeline(spec):
    """Build a timeline from [(seconds, mic_energy, sys_energy), …] at 10 blocks/s."""
    offsets, mic, sys_ = [], [], []
    for second, mic_e, sys_e in spec:
        for i in range(10):
            offsets.append(int((second + i / 10) * SAMPLE_RATE))
            mic.append(mic_e)
            sys_.append(sys_e)
    return (
        np.array(offsets, dtype=np.int64),
        np.array(mic, dtype=np.float32),
        np.array(sys_, dtype=np.float32),
    )


class LabelSegmentsTest(unittest.TestCase):
    def test_mic_dominant_is_me(self):
        timeline = _timeline([(0, 0.01, 0.0001), (1, 0.01, 0.0001)])
        segments = [{"start": 0.0, "end": 2.0, "text": "hi", "speaker": None}]
        attribution.label_segments(segments, timeline)
        self.assertEqual(segments[0]["speaker"], "Me")

    def test_system_dominant_is_them(self):
        timeline = _timeline([(0, 0.0001, 0.01)])
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        attribution.label_segments(segments, timeline)
        self.assertEqual(segments[0]["speaker"], "Them")

    def test_mic_dominance_overrides_diarizer_label(self):
        """The diarizer can't know which voice is the user — Me wins."""
        timeline = _timeline([(0, 0.01, 0.0001)])
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}]
        attribution.label_segments(segments, timeline)
        self.assertEqual(segments[0]["speaker"], "Me")

    def test_system_dominance_keeps_diarizer_sublabel(self):
        """Them-segments keep pyannote's finer-grained speaker names."""
        timeline = _timeline([(0, 0.0001, 0.01)])
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_01"}]
        attribution.label_segments(segments, timeline)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_01")

    def test_silence_leaves_segment_untouched(self):
        timeline = _timeline([(0, 0.0, 0.0)])
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        attribution.label_segments(segments, timeline)
        self.assertIsNone(segments[0]["speaker"])

    def test_ambiguous_crosstalk_falls_back_to_louder_stream(self):
        timeline = _timeline([(0, 0.010, 0.008)])  # close, below dominance factor
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        attribution.label_segments(segments, timeline)
        self.assertEqual(segments[0]["speaker"], "Me")

    def test_empty_timeline_is_noop(self):
        empty = (np.array([], dtype=np.int64), np.array([]), np.array([]))
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        attribution.label_segments(segments, empty)
        self.assertIsNone(segments[0]["speaker"])

    def test_mixed_conversation(self):
        timeline = _timeline([(0, 0.01, 0.0001), (1, 0.0001, 0.01), (2, 0.01, 0.0001)])
        segments = [
            {"start": 0.0, "end": 1.0, "text": "me", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": "them", "speaker": None},
            {"start": 2.0, "end": 3.0, "text": "me again", "speaker": None},
        ]
        attribution.label_segments(segments, timeline)
        self.assertEqual([s["speaker"] for s in segments], ["Me", "Them", "Me"])


class RecorderTimelineTest(unittest.TestCase):
    def _make_recorder(self):
        import tempfile

        rec = Recorder(device=None)
        rec._tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        rec._tmpfile.write(b"\x00" * 44)
        rec._frame_count = 0
        rec._recording = True
        self.addCleanup(lambda: rec._tmpfile.close())
        return rec

    def test_callback_records_energy_blocks(self):
        rec = self._make_recorder()
        rec._system_audio_active = True
        rec._on_system_chunk(np.full(100, 0.5, dtype=np.float32))
        rec._callback(np.full((100, 1), 0.1, dtype=np.float32), 100, None, None)
        rec._callback(np.zeros((100, 1), dtype=np.float32), 100, None, None)

        offsets, mic, sys_ = rec.attribution_timeline()
        self.assertEqual(list(offsets), [0, 100])
        self.assertAlmostEqual(float(mic[0]), 0.01, places=4)
        self.assertAlmostEqual(float(sys_[0]), 0.25, places=4)
        self.assertAlmostEqual(float(mic[1]), 0.0, places=6)
        self.assertAlmostEqual(float(sys_[1]), 0.0, places=6)  # buffer drained

    def test_timeline_without_system_audio(self):
        rec = self._make_recorder()
        rec._callback(np.full((50, 1), 0.2, dtype=np.float32), 50, None, None)
        _offsets, mic, sys_ = rec.attribution_timeline()
        self.assertAlmostEqual(float(mic[0]), 0.04, places=4)
        self.assertEqual(float(sys_[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
