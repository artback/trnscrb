"""Tests for reporting whether speaker labelling actually works.

pyannote's repos are gated, so a valid HF token proves nothing on its own.
`trnscrb status` used to report "HF token ok" while every transcript came out
with no speaker names.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trnscrb import diarizer, health
from trnscrb.cli import _diarization_ready


class IsDownloadedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)

    def _cache(self, model_id, with_snapshot):
        d = self.home / ".cache" / "huggingface" / "hub"
        d /= f"models--{model_id.replace('/', '--')}"
        d /= "snapshots"
        d.mkdir(parents=True)
        if with_snapshot:
            (d / "abc123").mkdir()

    def _check(self, model_id):
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(Path, "home", return_value=self.home),
        ):
            return diarizer.is_downloaded(model_id)

    def test_absent_model_is_not_downloaded(self):
        self.assertFalse(self._check("pyannote/speaker-diarization-community-1"))

    def test_cached_model_is_downloaded(self):
        self._cache("pyannote/speaker-diarization-community-1", with_snapshot=True)
        self.assertTrue(self._check("pyannote/speaker-diarization-community-1"))

    def test_empty_snapshot_dir_is_not_downloaded(self):
        """A gated download that never completed leaves the directory behind."""
        self._cache("pyannote/speaker-diarization-community-1", with_snapshot=False)
        self.assertFalse(self._check("pyannote/speaker-diarization-community-1"))

    def test_hf_hub_cache_env_is_respected(self):
        cache = self.home / "custom" / "hub"
        d = cache / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "x"
        d.mkdir(parents=True)
        with mock.patch.dict("os.environ", {"HF_HUB_CACHE": str(cache)}, clear=True):
            self.assertTrue(diarizer.is_downloaded("pyannote/speaker-diarization-3.1"))


class DiarizationReadyTest(unittest.TestCase):
    def setUp(self):
        """Isolate the health store — status now reports what actually ran."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(setattr, health, "STORE", health.STORE)
        health.STORE = Path(tmp.name) / "health.json"

    def test_no_token_is_reported_as_optional(self):
        with mock.patch("trnscrb.settings.read_hf_token", return_value=None):
            ok, detail = _diarization_ready()
        self.assertFalse(ok)
        self.assertIn("no HF token", detail)

    def test_token_without_accepted_terms_is_not_ready(self):
        """The bug: a token alone used to report as ok."""
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", return_value=False),
        ):
            ok, detail = _diarization_ready()
        self.assertFalse(ok)
        self.assertIn("accept the model terms", detail)

    def test_downloaded_model_is_ready(self):
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", side_effect=lambda m: "community" in m),
        ):
            ok, detail = _diarization_ready()
        self.assertTrue(ok)
        self.assertIn("community", detail)

    def test_falls_back_to_the_second_candidate(self):
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "pipeline_candidates", return_value=["a/b", "c/d"]),
            mock.patch.object(diarizer, "is_downloaded", side_effect=lambda m: m == "c/d"),
        ):
            ok, detail = _diarization_ready()
        self.assertTrue(ok)
        self.assertIn("c/d", detail)

    def test_a_failing_last_run_beats_an_installed_model(self):
        """The six-day bug: everything on disk was correct and nothing worked."""
        health.record_failure(health.DIARIZATION, "torchcodec is not available")
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", return_value=True),
        ):
            ok, detail = _diarization_ready()
        self.assertFalse(ok)
        self.assertIn("torchcodec", detail)
        self.assertIn("doctor", detail)

    def test_a_successful_last_run_is_reported_with_the_model(self):
        health.record_ok(health.DIARIZATION, "3 speaker(s)")
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", side_effect=lambda m: "community" in m),
        ):
            ok, detail = _diarization_ready()
        self.assertTrue(ok)
        self.assertIn("community", detail)
        self.assertIn("3 speaker(s)", detail)

    def test_a_recovered_run_stops_being_reported_as_broken(self):
        health.record_failure(health.DIARIZATION, "torchcodec is not available")
        health.record_ok(health.DIARIZATION, "verified by `trnscrb doctor`")
        with (
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(diarizer, "is_downloaded", return_value=True),
        ):
            ok, _ = _diarization_ready()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
