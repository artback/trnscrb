"""Tests for learning a fingerprint of the user's own voice.

The guards matter more than the happy path: enrolling the wrong speaker
trains "Me" on a colleague's voice, and nothing downstream would notice.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from trnscrb import attribution, diarizer, voiceprints
from trnscrb.recorder import SAMPLE_RATE

_MODEL = "pyannote/speaker-diarization-community-1"


def _timeline(blocks):
    """(offsets, mic, sys) from [(mic_energy, sys_energy), …], 1s per block."""
    offsets = np.arange(len(blocks), dtype=np.int64) * SAMPLE_RATE
    mic = np.array([b[0] for b in blocks], dtype=np.float32)
    sys_ = np.array([b[1] for b in blocks], dtype=np.float32)
    return offsets, mic, sys_


def _turns(spans):
    return [{"start": s, "end": e, "speaker": spk} for s, e, spk in spans]


class SelfSpeakerTest(unittest.TestCase):
    """Identifying which diarized speaker is the user."""

    def test_mic_only_speaker_is_the_user(self):
        # 10s of user (mic hot, system silent), 10s of someone else.
        blocks = [(1.0, 1e-9)] * 10 + [(1e-9, 1.0)] * 10
        turns = _turns([(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01")])
        label, secs = attribution.self_speaker(turns, _timeline(blocks))
        self.assertEqual(label, "SPEAKER_00")
        self.assertAlmostEqual(secs, 10.0)

    def test_no_self_when_two_speakers_look_mic_only(self):
        """Ambiguous is a refusal, not a coin flip."""
        blocks = [(1.0, 1e-9)] * 20
        turns = _turns([(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01")])
        label, _ = attribution.self_speaker(turns, _timeline(blocks))
        self.assertIsNone(label)

    def test_no_self_when_the_speaker_is_mostly_on_the_system_stream(self):
        """A cluster that mixes the user with a remote voice must not enrol."""
        blocks = [(1.0, 1e-9)] * 2 + [(1e-9, 1.0)] * 8
        turns = _turns([(0, 2, "SPEAKER_00"), (2, 10, "SPEAKER_00")])
        label, _ = attribution.self_speaker(turns, _timeline(blocks))
        self.assertIsNone(label)

    def test_no_self_when_the_mic_never_spoke(self):
        blocks = [(1e-9, 1.0)] * 10
        turns = _turns([(0, 10, "SPEAKER_00")])
        label, _ = attribution.self_speaker(turns, _timeline(blocks))
        self.assertIsNone(label)

    def test_empty_timeline_is_safe(self):
        empty = (np.array([], dtype=np.int64), np.array([]), np.array([]))
        self.assertEqual(attribution.self_speaker(_turns([(0, 5, "A")]), empty), (None, 0.0))


class EnrollTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(voiceprints, "STORE", Path(self._tmp.name) / "vp.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_first_enrolment_is_stored_normalised(self):
        self.assertTrue(voiceprints.enroll("Me", np.array([3.0, 4.0]), _MODEL, 120))
        entry = voiceprints.load()["prints"]["Me"]
        self.assertAlmostEqual(np.linalg.norm(entry["vector"]), 1.0, places=6)
        self.assertEqual(entry["enrollments"], 1)

    def test_short_meetings_are_ignored(self):
        self.assertFalse(voiceprints.enroll("Me", np.array([1.0, 0.0]), _MODEL, 15))
        self.assertEqual(voiceprints.summary(), [])

    def test_repeated_enrolments_average(self):
        voiceprints.enroll("Me", np.array([1.0, 0.0]), _MODEL, 120)
        voiceprints.enroll("Me", np.array([0.0, 1.0]), _MODEL, 120)
        entry = voiceprints.load()["prints"]["Me"]
        self.assertEqual(entry["enrollments"], 2)
        self.assertAlmostEqual(entry["vector"][0], entry["vector"][1], places=6)
        self.assertAlmostEqual(entry["speech_secs"], 240.0)

    def test_one_long_meeting_does_not_dominate(self):
        """Averaging is per meeting, not per second."""
        voiceprints.enroll("Me", np.array([1.0, 0.0]), _MODEL, 120)
        voiceprints.enroll("Me", np.array([0.0, 1.0]), _MODEL, 6000)
        entry = voiceprints.load()["prints"]["Me"]
        self.assertAlmostEqual(entry["vector"][0], entry["vector"][1], places=6)

    def test_changing_pipeline_discards_incompatible_prints(self):
        """Embeddings from different models are not comparable."""
        voiceprints.enroll("Me", np.array([1.0, 0.0]), _MODEL, 120)
        voiceprints.enroll("Me", np.array([0.0, 1.0]), "some/other-pipeline", 120)
        data = voiceprints.load()
        self.assertEqual(data["model"], "some/other-pipeline")
        self.assertEqual(data["prints"]["Me"]["enrollments"], 1)

    def test_unusable_embedding_is_rejected(self):
        self.assertFalse(voiceprints.enroll("Me", np.array([np.nan, 1.0]), _MODEL, 120))
        self.assertEqual(voiceprints.summary(), [])

    def test_forget(self):
        voiceprints.enroll("Me", np.array([1.0, 0.0]), _MODEL, 120)
        self.assertTrue(voiceprints.forget("Me"))
        self.assertFalse(voiceprints.forget("Me"))
        self.assertEqual(voiceprints.summary(), [])

    def test_missing_store_reads_as_empty(self):
        self.assertEqual(voiceprints.summary(), [])

    def test_corrupt_store_reads_as_empty(self):
        voiceprints.STORE.parent.mkdir(parents=True, exist_ok=True)
        voiceprints.STORE.write_text("{not json", encoding="utf-8")
        self.assertEqual(voiceprints.summary(), [])


class EnrollOwnVoiceGuardTest(unittest.TestCase):
    """The mic-only inference depends on system audio having been captured."""

    def _app(self):
        from trnscrb.menu_bar import TrnscrbApp

        app = TrnscrbApp.__new__(TrnscrbApp)  # no rumps app needed for this method
        return app

    def _call(self, system_audio, self_label="SPEAKER_00"):
        recorder = mock.Mock()
        recorder.attribution_timeline.return_value = _timeline([(1.0, 1e-9)] * 120)
        turns = _turns([(0, 120, "SPEAKER_00")])
        embeddings = {"SPEAKER_00": np.array([1.0, 0.0])}
        with (
            mock.patch("trnscrb.settings.get", side_effect={"learn_my_voice": True}.get),
            mock.patch.object(attribution, "self_speaker", return_value=(self_label, 120.0)),
            mock.patch.object(diarizer, "pipeline_id", return_value=_MODEL),
            mock.patch.object(voiceprints, "enroll") as enroll,
        ):
            self._app()._enroll_own_voice(turns, embeddings, recorder, system_audio)
        return enroll

    def test_enrols_when_system_audio_was_captured(self):
        self._call(system_audio=True).assert_called_once()

    def test_refuses_without_system_audio(self):
        """Without it, speaker bleed makes a colleague look mic-only."""
        self._call(system_audio=False).assert_not_called()

    def test_refuses_when_no_speaker_is_clearly_the_user(self):
        self._call(system_audio=True, self_label=None).assert_not_called()


class EmbeddingsBySpeakerTest(unittest.TestCase):
    """Pairing centroids with labels — a mismatch would mislabel a voice."""

    def _result(self, labels, vectors):
        annotation = mock.Mock()
        annotation.labels.return_value = labels
        return mock.Mock(speaker_diarization=annotation, speaker_embeddings=vectors)

    def test_pairs_in_label_order(self):
        out = diarizer._embeddings_by_speaker(
            self._result(["SPEAKER_00", "SPEAKER_01"], np.array([[1.0, 0.0], [0.0, 1.0]]))
        )
        self.assertEqual(sorted(out), ["SPEAKER_00", "SPEAKER_01"])
        np.testing.assert_allclose(out["SPEAKER_01"], [0.0, 1.0])

    def test_count_mismatch_yields_nothing(self):
        out = diarizer._embeddings_by_speaker(
            self._result(["SPEAKER_00", "SPEAKER_01"], np.array([[1.0, 0.0]]))
        )
        self.assertEqual(out, {})

    def test_absent_embeddings_yield_nothing(self):
        self.assertEqual(diarizer._embeddings_by_speaker(object()), {})


if __name__ == "__main__":
    unittest.main()
