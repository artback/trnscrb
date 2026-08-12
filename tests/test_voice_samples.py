"""Tests for keeping a listenable clip of each voice.

A meeting's audio is deleted as soon as its transcript is saved, so the clip
has to be cut during enrolment — afterwards there is nothing left to cut.
"""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from trnscrb import voiceprints

_RATE = 16000


def _wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_RATE)
        w.writeframes(b"\x01\x02" * int(_RATE * seconds))
    return path


def _turns(spec):
    return [{"start": s, "end": e, "speaker": spk} for s, e, spk in spec]


class SaveSampleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        patcher = mock.patch.object(voiceprints, "SAMPLES_DIR", self.dir / "samples")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.audio = _wav(self.dir / "meeting.wav", 30.0)

    def test_clip_is_cut_from_the_longest_turn(self):
        turns = _turns([(0.0, 1.0, "S0"), (10.0, 25.0, "S0")])
        out = voiceprints.save_sample("voice-1", self.audio, turns)
        self.assertIsNotNone(out)
        with wave.open(str(out), "rb") as w:
            self.assertEqual(w.getframerate(), _RATE)
            self.assertAlmostEqual(w.getnframes() / _RATE, voiceprints.SAMPLE_SECS, places=1)

    def test_short_speaker_yields_no_clip(self):
        short = _turns([(1.0, 1.3, "S0")])
        self.assertIsNone(voiceprints.save_sample("voice-1", self.audio, short))

    def test_no_turns_yields_no_clip(self):
        self.assertIsNone(voiceprints.save_sample("voice-1", self.audio, []))

    def test_existing_clip_is_not_overwritten(self):
        turns = _turns([(2.0, 20.0, "S0")])
        first = voiceprints.save_sample("voice-1", self.audio, turns)
        before = first.read_bytes()
        again = voiceprints.save_sample("voice-1", self.audio, turns)
        self.assertEqual(again, first)
        self.assertEqual(first.read_bytes(), before)

    def test_missing_audio_is_survivable(self):
        gone = self.dir / "nope.wav"
        self.assertIsNone(voiceprints.save_sample("voice-1", gone, _turns([(0.0, 20.0, "S0")])))

    def test_forgetting_a_voice_deletes_its_clip(self):
        store = mock.patch.object(voiceprints, "STORE", self.dir / "vp.json")
        store.start()
        self.addCleanup(store.stop)
        with mock.patch.object(voiceprints, "_thresholds", return_value=(0.75, 0.1)):
            voiceprints.observe([1.0, 0.0], "m", 120, "mtg", "S0", "plda")
        clip = voiceprints.save_sample("voice-1", self.audio, _turns([(2.0, 20.0, "S0")]))
        self.assertTrue(clip.is_file())
        voiceprints.forget("voice-1")
        self.assertFalse(clip.is_file())


class KeepVoiceSampleTest(unittest.TestCase):
    """The recording-path wrapper must never fail a transcription."""

    def test_no_audio_path_is_a_noop(self):
        from trnscrb import menu_bar

        with mock.patch.object(voiceprints, "save_sample") as save:
            menu_bar._keep_voice_sample("voice-1", None, [], "S0")
        save.assert_not_called()

    def test_only_this_speakers_turns_are_used(self):
        from trnscrb import menu_bar

        diar = _turns([(0.0, 5.0, "S0"), (5.0, 9.0, "S1")])
        with mock.patch.object(voiceprints, "save_sample") as save:
            menu_bar._keep_voice_sample("voice-1", Path("/tmp/a.wav"), diar, "S1")
        self.assertEqual([t["speaker"] for t in save.call_args.args[2]], ["S1"])

    def test_failure_is_swallowed(self):
        from trnscrb import menu_bar

        with mock.patch.object(voiceprints, "save_sample", side_effect=OSError("boom")):
            menu_bar._keep_voice_sample("voice-1", Path("/tmp/a.wav"), [], "S0")


if __name__ == "__main__":
    unittest.main()


class LabelVoicesMenuTest(unittest.TestCase):
    """The menu-bar labelling flow."""

    def _app(self):
        from trnscrb.menu_bar import TrnscrbApp

        return TrnscrbApp.__new__(TrnscrbApp)

    def _run(self, rows, answers, clip=True):
        from trnscrb import menu_bar

        results = [mock.Mock(clicked=bool(a is not None), text=a or "") for a in answers]
        with (
            mock.patch.object(voiceprints, "summary", return_value=rows),
            mock.patch.object(voiceprints, "sample_path") as path,
            mock.patch.object(voiceprints, "name_voice", return_value=True) as name,
            mock.patch.object(menu_bar.subprocess, "Popen") as popen,
            mock.patch.object(menu_bar.rumps, "Window") as window,
            mock.patch.object(menu_bar.rumps, "alert") as alert,
            mock.patch.object(menu_bar, "_notify"),
        ):
            path.return_value = mock.Mock(**{"is_file.return_value": clip})
            window.return_value.run.side_effect = results
            self._app().label_voices(None)
        return name, popen, alert

    def _row(self, vid, name=""):
        return {
            "id": vid,
            "name": name,
            "observations": 2,
            "speech_secs": 300.0,
            "updated_at": "",
            "dimension": 128,
            "meetings": ["standup"],
        }

    def test_named_voice_is_saved(self):
        name, _, _ = self._run([self._row("voice-2")], ["Justin Lee"])
        name.assert_called_once_with("voice-2", "Justin Lee")

    def test_skipping_names_nothing(self):
        name, _, _ = self._run([self._row("voice-2")], [None])
        name.assert_not_called()

    def test_blank_answer_names_nothing(self):
        name, _, _ = self._run([self._row("voice-2")], [""])
        name.assert_not_called()

    def test_clip_plays_while_the_prompt_is_up(self):
        _, popen, _ = self._run([self._row("voice-2")], ["X"])
        self.assertEqual(popen.call_args.args[0][0], "afplay")

    def test_missing_clip_still_prompts(self):
        name, popen, _ = self._run([self._row("voice-2")], ["X"], clip=False)
        popen.assert_not_called()
        name.assert_called_once()

    def test_already_named_voices_are_skipped(self):
        name, _, alert = self._run([self._row("voice-1", "Me")], [])
        name.assert_not_called()
        alert.assert_called_once()
