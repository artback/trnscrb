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

from trnscrb import attribution, diarizer, health, voiceprints
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
    """Isolates the on-disk stores and pins the matching thresholds.

    Every file these tests can reach has to be redirected: enrolment now also
    records component health, and a test run must not rewrite the diagnostics
    the user reads.
    """

    threshold = 0.55
    margin = 0.10

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(voiceprints, "STORE", Path(self._tmp.name) / "vp.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        health_store = mock.patch.object(health, "STORE", Path(self._tmp.name) / "health.json")
        health_store.start()
        self.addCleanup(health_store.stop)
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

    # ── what the diagnostics record ───────────────────────────────────────

    def test_a_successful_enrolment_is_recorded(self):
        self._call()
        entry = health.get(health.VOICE_ENROLMENT)
        self.assertTrue(entry["ok"])
        self.assertIn("1 voice(s)", entry["detail"])

    def test_enrolling_nobody_despite_a_long_speaker_is_a_failure(self):
        """The reported symptom: meetings go by and `trnscrb voices` never grows."""
        self._call(self_label=None)
        entry = health.get(health.VOICE_ENROLMENT)
        self.assertFalse(entry["ok"])
        self.assertIn("spoke long enough", entry["detail"])

    def test_a_meeting_with_nobody_over_the_bar_is_not_a_failure(self):
        """The enrolment bar doing its job must not read as a broken component."""
        with mock.patch.object(voiceprints, "MIN_ENROLL_SECS", 10_000):
            self._call(self_label=None)
        entry = health.get(health.VOICE_ENROLMENT)
        self.assertTrue(entry["ok"])
        self.assertIn("needed to enrol", entry["detail"])


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


class BacklogEnrolmentTest(_StoreTest):
    """Transcribing a file after the fact must grow the voice store too.

    It cannot do the whole job: identifying the user needs the mic and system
    streams kept apart, and a file on disk is already mixed.
    """

    def _run(self, *, cluster=True, embeddings=None, turns=None):
        from trnscrb import backlog

        embeddings = embeddings or {"S0": _vec(1.0, 0.0), "S1": _vec(0.0, 1.0)}
        turns = turns if turns is not None else _turns([(0, 100, "S0"), (100, 200, "S1")])
        with (
            mock.patch("trnscrb.settings.get", side_effect={"cluster_voices": cluster}.get),
            mock.patch.object(diarizer, "pipeline_id", return_value=_MODEL),
            mock.patch.object(diarizer, "embedding_space", return_value="plda"),
        ):
            backlog._learn_voices(turns, embeddings, "recovered", Path("/tmp/none.wav"))
        return voiceprints.summary()

    def test_speakers_are_enrolled(self):
        rows = self._run()
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["meetings"] for r in rows], [["recovered"], ["recovered"]])

    def test_nobody_is_named_me(self):
        """A mixed file cannot tell the user's voice from anyone else's."""
        self.assertEqual([r["name"] for r in self._run()], ["", ""])

    def test_a_known_voice_still_lands_on_its_identity(self):
        """Recovering a meeting should extend Me, not fork it — via matching."""
        voiceprints.observe(_vec(1.0, 0.0), _MODEL, 120, "earlier", "S0", "plda")
        voiceprints.name_voice("voice-1", voiceprints.SELF)
        rows = self._run(embeddings={"S0": _vec(0.99, 0.14)}, turns=_turns([(0, 100, "S0")]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], voiceprints.SELF)
        self.assertEqual(rows[0]["observations"], 2)

    def test_clustering_off_enrols_nobody(self):
        self.assertEqual(self._run(cluster=False), [])

    def test_short_speakers_are_skipped(self):
        turns = _turns([(0, 10, "S0"), (10, 20, "S1")])
        self.assertEqual(self._run(turns=turns), [])

    def test_enrolment_never_fails_the_transcription(self):
        from trnscrb import backlog

        with (
            mock.patch("trnscrb.settings.get", return_value=True),
            mock.patch.object(voiceprints, "enrol", side_effect=RuntimeError("boom")),
        ):
            backlog._learn_voices([], {"S0": _vec(1.0)}, "m", Path("/tmp/none.wav"))


class EnrolmentHealthTest(unittest.TestCase):
    def test_enrolled_voices_are_ok(self):
        ok, detail = voiceprints.enrolment_health(["voice-1"], [])
        self.assertTrue(ok)
        self.assertIn("1 voice(s)", detail)

    def test_nobody_over_the_bar_is_ok(self):
        """The bar doing its job is not a broken component."""
        ok, detail = voiceprints.enrolment_health([], _turns([(0, 5, "S0")]))
        self.assertTrue(ok)
        self.assertIn("needed to enrol", detail)

    def test_a_long_speaker_that_was_not_enrolled_is_a_failure(self):
        ok, detail = voiceprints.enrolment_health([], _turns([(0, 300, "S0")]))
        self.assertFalse(ok)
        self.assertIn("spoke long enough", detail)


def _store(vectors: dict[str, tuple], names: dict[str, str] | None = None) -> dict:
    """Write a store of already-split identities, bypassing observe()'s matching."""
    data = voiceprints._empty(_MODEL)
    for voice_id, vector in vectors.items():
        data["voices"][voice_id] = {
            "vector": voiceprints._unit(_vec(*vector)).tolist(),
            "name": (names or {}).get(voice_id, ""),
            "observations": 1,
            "speech_secs": 100.0,
            "seen": [{"meeting": voice_id, "label": "S0", "secs": 100.0, "at": voice_id}],
            "updated_at": voice_id,
        }
    data["next_id"] = len(vectors) + 1
    voiceprints._save(data)
    return data


class DuplicateMatchTest(_StoreTest):
    """A voice stored twice must not make itself unmatchable."""

    def test_duplicate_runner_up_does_not_block_a_match(self):
        # Two copies of one person, and one genuinely different voice.
        _store({"voice-1": (1.0, 0.0, 0.0), "voice-2": (0.99, 0.14, 0.0), "voice-3": (0, 0, 1)})
        voice_id, _ = voiceprints.match(_vec(1.0, 0.05, 0.0))
        self.assertIn(voice_id, ("voice-1", "voice-2"))

    def test_two_distinct_close_voices_are_still_ambiguous(self):
        # 0.8 apart is well under the duplicate bar; equidistant stays refused.
        _store({"voice-1": (1.0, 0.0), "voice-2": (0.8, 0.6)})
        self.assertIsNone(voiceprints.match(_vec(0.95, 0.31))[0])

    def test_observe_folds_the_duplicate_in(self):
        _store({"voice-1": (1.0, 0.0, 0.0), "voice-2": (0.99, 0.14, 0.0)})
        voice_id = voiceprints.observe(_vec(1.0, 0.05, 0.0), _MODEL, 120, "m3", "S0")
        rows = voiceprints.summary()
        self.assertEqual([r["id"] for r in rows], [voice_id])
        self.assertEqual(rows[0]["observations"], 3)


class MergeTest(_StoreTest):
    def setUp(self):
        super().setUp()
        samples = mock.patch.object(voiceprints, "SAMPLES_DIR", Path(self._tmp.name) / "samples")
        samples.start()
        self.addCleanup(samples.stop)

    def test_named_identity_survives(self):
        data = _store({"voice-1": (1.0, 0.0), "voice-2": (0.99, 0.14)}, {"voice-2": "Anna"})
        self.assertEqual(voiceprints.dedupe(data), [("voice-2", "voice-1", mock.ANY)])
        self.assertEqual(list(data["voices"]), ["voice-2"])
        entry = data["voices"]["voice-2"]
        self.assertEqual(entry["name"], "Anna")
        self.assertEqual(entry["observations"], 2)
        self.assertEqual(entry["speech_secs"], 200.0)
        self.assertEqual([s["meeting"] for s in entry["seen"]], ["voice-1", "voice-2"])

    def test_two_different_names_are_never_fused(self):
        data = _store(
            {"voice-1": (1.0, 0.0), "voice-2": (0.99, 0.14)}, {"voice-1": "Anna", "voice-2": "Bo"}
        )
        self.assertEqual(voiceprints.dedupe(data), [])
        self.assertEqual(len(data["voices"]), 2)

    def test_chain_collapses_to_one(self):
        data = _store(
            {
                "voice-1": (1.0, 0.0, 0.0),
                "voice-2": (0.99, 0.14, 0.0),
                "voice-3": (0.98, 0.20, 0.0),
                "voice-4": (0.0, 0.0, 1.0),
            }
        )
        merged = voiceprints.dedupe(data)
        self.assertEqual(len(merged), 2)
        (survivor,) = [vid for vid in data["voices"] if vid != "voice-4"]
        self.assertEqual(data["voices"][survivor]["observations"], 3)

    def test_sample_follows_the_identity(self):
        data = _store({"voice-1": (1.0, 0.0), "voice-2": (0.99, 0.14)})
        voiceprints.SAMPLES_DIR.mkdir()
        voiceprints.sample_path("voice-2").write_bytes(b"RIFF")
        voiceprints.merge(data, "voice-1", "voice-2")
        self.assertTrue(voiceprints.sample_path("voice-1").is_file())
        self.assertFalse(voiceprints.sample_path("voice-2").exists())

    def test_merge_duplicates_dry_run_leaves_the_store_alone(self):
        _store({"voice-1": (1.0, 0.0), "voice-2": (0.99, 0.14)})
        self.assertEqual(len(voiceprints.merge_duplicates(dry_run=True)), 1)
        self.assertEqual(len(voiceprints.summary()), 2)
        self.assertEqual(len(voiceprints.merge_duplicates()), 1)
        self.assertEqual(len(voiceprints.summary()), 1)


class NameVoiceFromCalendarTest(_StoreTest):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(attribution, "_looks_like_self", lambda n: n == "Jonathan")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _name(self, learned, attendees, speakers):
        return attribution.name_voice_from_calendar(learned, {"attendees": attendees}, speakers)

    def test_one_to_one_names_the_other_voice(self):
        _store({"voice-1": (1.0, 0.0), "voice-2": (0.0, 1.0)}, {"voice-1": voiceprints.SELF})
        self.assertEqual(self._name(["voice-1", "voice-2"], ["Jonathan", "Anna"], 2), "Anna")
        self.assertEqual(voiceprints.find_by_name("Anna"), "voice-2")

    def test_group_names_the_last_unknown_by_elimination(self):
        _store(
            {"voice-1": (1, 0, 0), "voice-2": (0, 1, 0), "voice-3": (0, 0, 1)},
            {"voice-1": voiceprints.SELF, "voice-2": "Bo"},
        )
        learned = ["voice-1", "voice-2", "voice-3"]
        self.assertEqual(self._name(learned, ["Jonathan", "Bo", "Cleo"], 3), "Cleo")

    def test_two_unknown_voices_stay_unnamed(self):
        _store({"voice-1": (1, 0, 0), "voice-2": (0, 1, 0), "voice-3": (0, 0, 1)})
        learned = ["voice-1", "voice-2", "voice-3"]
        self.assertIsNone(self._name(learned, ["Jonathan", "Bo", "Cleo"], 3))
        self.assertIsNone(voiceprints.find_by_name("Bo"))

    def test_unenrolled_speaker_blocks_naming(self):
        """One voice short: the leftover could be the user's own, not the invitee's."""
        _store({"voice-1": (1.0, 0.0)})
        self.assertIsNone(self._name(["voice-1"], ["Jonathan", "Anna"], 2))

    def test_uninvited_speaker_blocks_naming(self):
        _store(
            {"voice-1": (1, 0, 0), "voice-2": (0, 1, 0), "voice-3": (0, 0, 1)}, {"voice-1": "Me"}
        )
        self.assertIsNone(self._name(["voice-1", "voice-2", "voice-3"], ["Jonathan", "Anna"], 3))
