"""Tests for lining system audio up with the microphone.

Without alignment the same voice lands in the recording twice: once through
the mic as speaker bleed at the moment it was said, and once through
ScreenCaptureKit a few hundred milliseconds later. Measured on real
recordings at +0.21 to +0.29 waveform correlation, at lags that were steady
within a call and different between calls — a capture offset, not a room.
"""

import re
import struct
import unittest
from unittest import mock

import numpy as np

from trnscrb import sck_helper
from trnscrb.recorder import SAMPLE_RATE, Recorder


def _frame(samples: np.ndarray, age: float) -> bytes:
    return (
        sck_helper._MAGIC
        + struct.pack("<fI", age, len(samples))
        + samples.astype(np.float32).tobytes()
    )


class WireFormatTest(unittest.TestCase):
    """The helper's chunks carry how old the audio already is."""

    def setUp(self):
        self.chunks = []
        self.capture = sck_helper.HelperCapture(
            lambda samples, age: self.chunks.append((samples, age))
        )

    def _feed(self, data: bytes):
        pending = data
        while True:
            taken, pending = self.capture._take_frame(pending)
            if not taken:
                return pending

    def test_a_frame_yields_its_samples_and_age(self):
        self._feed(_frame(np.array([0.1, 0.2, 0.3], dtype=np.float32), 0.35))
        samples, age = self.chunks[0]
        np.testing.assert_allclose(samples, [0.1, 0.2, 0.3], atol=1e-6)
        self.assertAlmostEqual(age, 0.35, places=5)

    def test_two_frames_in_one_read(self):
        one = np.array([0.1], dtype=np.float32)
        two = np.array([0.2, 0.3], dtype=np.float32)
        self._feed(_frame(one, 0.1) + _frame(two, 0.2))
        self.assertEqual([len(s) for s, _ in self.chunks], [1, 2])

    def test_a_split_frame_waits_for_the_rest(self):
        data = _frame(np.arange(8, dtype=np.float32), 0.2)
        rest = self._feed(data[:10])
        self.assertEqual(self.chunks, [])
        self._feed(rest + data[10:])
        self.assertEqual(len(self.chunks[0][0]), 8)

    def test_garbage_resyncs_to_the_next_frame(self):
        """One torn frame must not end the capture."""
        good = _frame(np.array([0.5], dtype=np.float32), 0.1)
        self._feed(b"\x00\x01\x02\x03\x04" + good)
        self.assertEqual(len(self.chunks), 1)
        np.testing.assert_allclose(self.chunks[0][0], [0.5], atol=1e-6)


class AlignmentTest(unittest.TestCase):
    """The mic is held back to meet system audio of the same moment."""

    def setUp(self):
        self.rec = Recorder(system_audio=True)

    def test_no_delay_before_any_timestamp_is_known(self):
        block = np.ones(1024, dtype=np.float32)
        np.testing.assert_array_equal(self.rec._delayed_mic(block), block)

    def test_the_first_chunk_sets_the_delay(self):
        self.rec._on_system_chunk(np.zeros(160, dtype=np.float32), age=0.30)
        self.assertEqual(self.rec._align_frames, int(0.30 * SAMPLE_RATE))

    def test_a_later_chunk_does_not_move_it(self):
        """One measurement, not a moving target that would warp the audio."""
        self.rec._on_system_chunk(np.zeros(160, dtype=np.float32), age=0.30)
        self.rec._on_system_chunk(np.zeros(160, dtype=np.float32), age=0.05)
        self.assertEqual(self.rec._align_frames, int(0.30 * SAMPLE_RATE))

    def test_an_absurd_delay_is_capped(self):
        self.rec._on_system_chunk(np.zeros(160, dtype=np.float32), age=30.0)
        self.assertEqual(self.rec._align_frames, SAMPLE_RATE // 2)

    def test_an_ageless_chunk_leaves_alignment_off(self):
        """An older helper streams bare PCM; unaligned still beats not recording."""
        self.rec._on_system_chunk(np.zeros(160, dtype=np.float32), age=0.0)
        self.assertIsNone(self.rec._align_frames)

    def test_the_delay_line_fills_before_emitting(self):
        self.rec._align_frames = 2048
        self.assertIsNone(self.rec._delayed_mic(np.ones(1024, dtype=np.float32)))
        self.assertIsNone(self.rec._delayed_mic(np.ones(1024, dtype=np.float32)))
        out = self.rec._delayed_mic(np.ones(1024, dtype=np.float32))
        self.assertEqual(len(out), 1024)

    def test_audio_comes_out_delayed_not_dropped(self):
        """Every sample in must come out, just later."""
        self.rec._align_frames = 1000
        blocks = [np.full(1024, i, dtype=np.float32) for i in range(1, 6)]
        out = [self.rec._delayed_mic(b) for b in blocks]
        emitted = np.concatenate([o for o in out if o is not None])
        held = self.rec._mic_hold
        self.assertEqual(len(emitted) + len(held), 5 * 1024)
        # and in order: the delay line is FIFO
        np.testing.assert_array_equal(np.concatenate([emitted, held]), np.concatenate(blocks))

    def test_stopping_flushes_what_the_delay_line_holds(self):
        self.rec._align_frames = 4096
        self.rec._tmpfile = mock.Mock()
        self.rec._delayed_mic(np.ones(1024, dtype=np.float32))
        self.rec._flush_mic_hold()
        self.rec._tmpfile.write.assert_called_once()
        self.assertEqual(self.rec._frame_count, 1024)
        self.assertEqual(len(self.rec._mic_hold), 0)

    def test_flushing_an_empty_delay_line_writes_nothing(self):
        self.rec._tmpfile = mock.Mock()
        self.rec._flush_mic_hold()
        self.rec._tmpfile.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class HelperSourceWireFormatTest(unittest.TestCase):
    """The Swift writer and the Python reader must agree byte for byte.

    Every other test here builds frames from ``sck_helper._MAGIC``, so both
    sides share Python's constant and the helper's own encoding is never
    checked. That is how the magic shipped little-endian — "SNRT" on the
    wire — with the reader falling back to bare PCM and mixing each header
    into the audio as a full-scale click.
    """

    def setUp(self):
        from trnscrb.app_bundle import _APP_SOURCE

        self.source = _APP_SOURCE.read_text()

    def _field(self, expression: str) -> str:
        """The endianness the helper writes ``expression`` with."""
        match = re.search(
            re.escape(expression) + r"\)?\.(littleEndian|bigEndian)\) \{ header\.append",
            self.source,
        )
        self.assertIsNotNone(match, f"helper no longer writes {expression}")
        return match.group(1)

    def test_the_magic_lands_as_the_reader_spells_it(self):
        match = re.search(
            r"withUnsafeBytes\(of: UInt32\(0x([0-9A-Fa-f_]+)\)\.(littleEndian|bigEndian)\)"
            r" \{ header\.append",
            self.source,
        )
        self.assertIsNotNone(match, "helper no longer writes a magic word")
        value = int(match.group(1).replace("_", ""), 16)
        order = "little" if match.group(2) == "littleEndian" else "big"
        self.assertEqual(value.to_bytes(4, order), sck_helper._MAGIC)

    def test_age_and_count_are_written_little_endian(self):
        """``_take_frame`` unpacks them with "<fI"."""
        self.assertEqual(self._field("withUnsafeBytes(of: age.bitPattern"), "littleEndian")
        self.assertEqual(
            self._field("withUnsafeBytes(of: UInt32(length / 4"), "littleEndian"
        )

    def test_the_header_is_the_length_the_reader_expects(self):
        self.assertEqual(sck_helper._HEADER_BYTES, 4 + struct.calcsize("<fI"))
