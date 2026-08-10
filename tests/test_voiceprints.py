"""Tests for learning voice identities across meetings.

The guards matter more than the happy path. Enrolling the wrong speaker
trains "Me" on a colleague, and fusing two identities puts one person's name
on another's words — neither is self-correcting, and nothing downstream
would notice.
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


def _vec(*values):
    return np.array(values, dtype=np.float64)


class _StoreTest(unittest.TestCase):
    """Isolates the on-disk store and pins the matching thresholds."""

    threshold = 0.55
    margin = 0.10

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(voiceprints, "STORE", Path(self._tmp.name) / "vp.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        thresholds = mock.patch.object(
            voiceprints, "_thresholds", return_value=(self.threshold, self.margin)
        )
        thresholds.start()
        self.addCleanup(thresholds.stop)


class SelfSpeakerTest(unittest.TestCase):
    """Identifying which diarized speaker is the user."""

    def test_mic_only_speaker_is_the_user(self):
        blocks = [(1.0, 1e-9)] * 10 + [(1e-9, 1.0)] * 10
        turns = _turns([(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01")])
        label, secs = attribution.self_speaker(turns, _timeline(blocks))
        self.assertEqual(label, "SPEAKER_00")
        self.assertAlmostEqual(secs, 10.0)

    def test_no_self_when_two_speakers_look_mic_only(self):
        """Ambiguous is a refusal, not a coin flip."""
        blocks = [(1.0, 1e-9)] * 20
        turns = _turns([(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01")])
        self.assertIsNone(attribution.self_speaker(turns, _timeline(blocks))[0])

    def test_no_self_when_the_speaker_is_mostly_on_the_system_stream(self):
        blocks = [(1.0, 1e-9)] * 2 + [(1e-9, 1.0)] * 8
        turns = _turns([(0, 2, "SPEAKER_00"), (2, 10, "SPEAKER_00")])
        self.assertIsNone(attribution.self_speaker(turns, _timeline(blocks))[0])

    def test_no_self_when_the_mic_never_spoke(self):
        blocks = [(1e-9, 1.0)] * 10
        self.assertIsNone(
            attribution.self_speaker(_turns([(0, 10, "SPEAKER_00")]), _timeline(blocks))[0]
        )

    def test_empty_timeline_is_safe(self):
        empty = (np.array([], dtype=np.int64), np.array([]), np.array([]))
        self.assertEqual(attribution.self_speaker(_turns([(0, 5, "A")]), empty), (None, 0.0))


class MatchTest(_StoreTest):
    """Deciding whether a new recording is someone already known."""

    def test_empty_store_matches_nothing(self):
        self.assertEqual(voiceprints.match(_vec(1.0, 0.0)), (None, 0.0))

    def test_same_voice_matches(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "SPEAKER_00")
        voice_id, score = voiceprints.match(_vec(0.99, 0.14))
        self.assertEqual(voice_id, "voice-1")
        self.assertGreater(score, self.threshold)

    def test_different_voice_does_not_match(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "SPEAKER_00")
        self.assertIsNone(voiceprints.match(_vec(0.0, 1.0))[0])

    def test_ambiguous_match_is_refused(self):
        """Two stored voices equally close means we cannot tell them apart."""
        voiceprints.observe(_vec(1.0, 0.0, 0.0), _MODEL, 120, "m1", "SPEAKER_00")
        voiceprints.observe(_vec(0.0, 1.0, 0.0), _MODEL, 120, "m2", "SPEAKER_00")
        # Equidistant from both stored voices.
        self.assertIsNone(voiceprints.match(_vec(1.0, 1.0, 0.0))[0])


class ObserveTest(_StoreTest):
    def test_first_observation_creates_an_identity(self):
        self.assertEqual(voiceprints.observe(_vec(3.0, 4.0), _MODEL, 120, "m1", "S0"), "voice-1")
        row = voiceprints.summary()[0]
        self.assertEqual(row["observations"], 1)
        self.assertEqual(row["meetings"], ["m1"])

    def test_same_voice_across_meetings_is_one_identity(self):
        first = voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "march", "SPEAKER_00")
        # Same person, different meeting, arbitrary diarizer label.
        second = voiceprints.observe(_vec(0.99, 0.14), _MODEL, 120, "april", "SPEAKER_02")
        self.assertEqual(first, second)
        row = voiceprints.summary()[0]
        self.assertEqual(row["observations"], 2)
        self.assertEqual(row["meetings"], ["march", "april"])

    def test_different_voices_stay_separate(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "SPEAKER_00")
        voiceprints.observe(_vec(0.0, 1.0), _MODEL, 120, "m1", "SPEAKER_01")
        self.assertEqual([r["id"] for r in voiceprints.summary()], ["voice-1", "voice-2"])

    def test_short_observations_are_ignored(self):
        self.assertIsNone(voiceprints.observe(_vec(1.0, 0.0), _MODEL, 15, "m1", "S0"))
        self.assertEqual(voiceprints.summary(), [])

    def test_unusable_embedding_is_rejected(self):
        self.assertIsNone(voiceprints.observe(_vec(np.nan, 1.0), _MODEL, 120, "m1", "S0"))
        self.assertEqual(voiceprints.summary(), [])

    def test_one_long_meeting_does_not_dominate(self):
        """Averaging is per observation, not per second."""
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "S0")
        with mock.patch.object(voiceprints, "_thresholds", return_value=(-1.0, 0.0)):
            voiceprints.observe(_vec(0.0, 1.0), _MODEL, 6000, "m2", "S0")
        vector = voiceprints.load()["voices"]["voice-1"]["vector"]
        self.assertAlmostEqual(vector[0], vector[1], places=6)

    def test_changing_pipeline_discards_incompatible_voices(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "S0")
        voiceprints.observe(_vec(0.0, 1.0), "some/other-pipeline", 120, "m2", "S0")
        data = voiceprints.load()
        self.assertEqual(data["model"], "some/other-pipeline")
        self.assertEqual(len(data["voices"]), 1)


class NamingTest(_StoreTest):
    def test_naming_applies_to_every_past_meeting(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "march", "SPEAKER_00")
        voiceprints.observe(_vec(0.99, 0.14), _MODEL, 120, "april", "SPEAKER_02")
        self.assertTrue(voiceprints.name_voice("voice-1", "Anna"))
        row = voiceprints.summary()[0]
        self.assertEqual(row["name"], "Anna")
        self.assertEqual(row["meetings"], ["march", "april"])

    def test_naming_an_unknown_voice_fails(self):
        self.assertFalse(voiceprints.name_voice("voice-99", "Anna"))

    def test_find_by_name(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "S0")
        voiceprints.name_voice("voice-1", voiceprints.SELF)
        self.assertEqual(voiceprints.find_by_name(voiceprints.SELF), "voice-1")
        self.assertIsNone(voiceprints.find_by_name("Nobody"))

    def test_forget(self):
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "m1", "S0")
        self.assertTrue(voiceprints.forget("voice-1"))
        self.assertFalse(voiceprints.forget("voice-1"))
        self.assertEqual(voiceprints.summary(), [])


class StoreTest(_StoreTest):
    def test_missing_store_reads_as_empty(self):
        self.assertEqual(voiceprints.summary(), [])

    def test_corrupt_store_reads_as_empty(self):
        voiceprints.STORE.parent.mkdir(parents=True, exist_ok=True)
        voiceprints.STORE.write_text("{not json", encoding="utf-8")
        self.assertEqual(voiceprints.summary(), [])

    def test_v1_store_is_migrated(self):
        """The shipped 0.35.0 store held named prints with no clusters."""
        voiceprints.STORE.parent.mkdir(parents=True, exist_ok=True)
        voiceprints.STORE.write_text(
            '{"version": 1, "model": "m", "prints": {"Me": '
            '{"vector": [1.0, 0.0], "enrollments": 3, "speech_secs": 600.0}}}',
            encoding="utf-8",
        )
        rows = voiceprints.summary()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Me")
        self.assertEqual(rows[0]["observations"], 3)

    def test_future_version_is_ignored(self):
        voiceprints.STORE.parent.mkdir(parents=True, exist_ok=True)
        voiceprints.STORE.write_text('{"version": 99, "voices": {"v": {}}}', encoding="utf-8")
        self.assertEqual(voiceprints.summary(), [])


class LearnVoicesTest(_StoreTest):
    """The recording-path wiring, including the consent gate."""

    def _app(self):
        from trnscrb.menu_bar import TrnscrbApp

        return TrnscrbApp.__new__(TrnscrbApp)

    def _call(self, *, system_audio=True, learn_self=True, cluster=False, self_label="SPEAKER_00"):
        recorder = mock.Mock()
        recorder.attribution_timeline.return_value = _timeline([(1.0, 1e-9)] * 200)
        turns = _turns([(0, 100, "SPEAKER_00"), (100, 200, "SPEAKER_01")])
        embeddings = {"SPEAKER_00": _vec(1.0, 0.0), "SPEAKER_01": _vec(0.0, 1.0)}
        flags = {"learn_my_voice": learn_self, "cluster_voices": cluster}
        with (
            # menu_bar binds `get` at import time, so the module attribute is
            # the only patch point that actually takes effect.
            mock.patch("trnscrb.menu_bar.get_setting", side_effect=flags.get),
            mock.patch.object(attribution, "self_speaker", return_value=(self_label, 100.0)),
            mock.patch.object(diarizer, "pipeline_id", return_value=_MODEL),
        ):
            self._app()._learn_voices(turns, embeddings, recorder, system_audio, "meeting-1")
        return voiceprints.summary()

    def test_self_only_by_default(self):
        rows = self._call()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], voiceprints.SELF)

    def test_clustering_enrols_everyone(self):
        rows = self._call(cluster=True)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["name"] for r in rows], [voiceprints.SELF, ""])

    def test_refuses_self_without_system_audio(self):
        """Without it, speaker bleed makes a colleague look mic-only."""
        self.assertEqual(self._call(system_audio=False), [])

    def test_others_are_still_clustered_without_system_audio(self):
        """Clustering never claims which voice is the user, so it stays safe."""
        rows = self._call(system_audio=False, cluster=True)
        self.assertEqual([r["name"] for r in rows], ["", ""])

    def test_refuses_when_no_speaker_is_clearly_the_user(self):
        self.assertEqual(self._call(self_label=None), [])

    def test_disabled_entirely_stores_nothing(self):
        self.assertEqual(self._call(learn_self=False, cluster=False), [])


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


class SpaceMigrationTest(_StoreTest):
    """A v1 store holds raw embeddings; the projected space has fewer dims.

    Comparing the two raised ValueError deep inside a best-effort path that
    swallows exceptions, so enrolment would have stopped silently forever.
    """

    def _write_v1(self, dim=256):
        import json

        voiceprints.STORE.parent.mkdir(parents=True, exist_ok=True)
        voiceprints.STORE.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model": _MODEL,
                    "prints": {
                        "Me": {
                            "vector": list(np.ones(dim)),
                            "enrollments": 3,
                            "speech_secs": 477.5,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_v1_is_marked_as_raw_embedding_space(self):
        self._write_v1()
        self.assertEqual(voiceprints.load()["space"], "embedding")

    def test_mismatched_dimensions_never_raise(self):
        self.assertEqual(voiceprints._cosine(np.ones(256), np.ones(128)), -1.0)

    def test_projected_observation_retires_the_v1_store(self):
        self._write_v1()
        voice_id = voiceprints.observe(np.ones(128), _MODEL, 120, "mtg", "S0", "plda")
        self.assertIsNotNone(voice_id)
        data = voiceprints.load()
        self.assertEqual(data["space"], "plda")
        self.assertEqual(len(data["voices"]), 1)
        self.assertEqual(len(data["voices"][voice_id]["vector"]), 128)

    def test_same_space_keeps_accumulating(self):
        voiceprints.observe(np.ones(128), _MODEL, 120, "m1", "S0", "plda")
        voiceprints.observe(np.ones(128), _MODEL, 120, "m2", "S0", "plda")
        self.assertEqual(voiceprints.summary()[0]["observations"], 2)
