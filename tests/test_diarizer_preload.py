"""Tests for loading the diarization stack when a recording starts.

torch and pyannote are imported lazily, at stop. That leaves a window where
`brew upgrade` deletes the tree they would come from, and a perfectly
recorded meeting finishes with "No module named 'torch'" — no speaker
labels, no voiceprints, no clips. Deferring the app's restart protects the
audio but not the import, so the import has to happen first.
"""

import unittest
from unittest import mock

from trnscrb import diarizer


class PreloadTest(unittest.TestCase):
    def test_loads_when_a_pipeline_is_cached(self):
        with (
            mock.patch.object(diarizer, "is_downloaded", return_value=True),
            mock.patch.object(diarizer, "_get_pipeline") as get,
        ):
            self.assertTrue(diarizer.preload("hf_x"))
        get.assert_called_once_with("hf_x")

    def test_no_token_does_not_load(self):
        with mock.patch.object(diarizer, "_get_pipeline") as get:
            self.assertFalse(diarizer.preload(""))
        get.assert_not_called()

    def test_uncached_pipeline_does_not_block_on_the_network(self):
        """Starting a recording must never wait on a model download."""
        with (
            mock.patch.object(diarizer, "is_downloaded", return_value=False),
            mock.patch.object(diarizer, "_get_pipeline") as get,
        ):
            self.assertFalse(diarizer.preload("hf_x"))
        get.assert_not_called()

    def test_failure_is_not_fatal(self):
        with (
            mock.patch.object(diarizer, "is_downloaded", return_value=True),
            mock.patch.object(diarizer, "_get_pipeline", side_effect=RuntimeError("gated")),
        ):
            self.assertFalse(diarizer.preload("hf_x"))

    def test_the_fallback_pipeline_also_counts_as_cached(self):
        candidates = diarizer.pipeline_candidates()
        with (
            mock.patch.object(diarizer, "is_downloaded", side_effect=lambda m: m == candidates[-1]),
            mock.patch.object(diarizer, "_get_pipeline") as get,
        ):
            self.assertTrue(diarizer.preload("hf_x"))
        get.assert_called_once()


class RecordingStartPreloadTest(unittest.TestCase):
    """The recording-start hook must load both stacks and survive either failing."""

    def _app(self):
        from trnscrb.menu_bar import TrnscrbApp

        return TrnscrbApp.__new__(TrnscrbApp)

    def _run(self, **patches):
        from trnscrb import menu_bar

        with (
            mock.patch.object(menu_bar, "get_setting", return_value="auto"),
            mock.patch.object(menu_bar, "read_hf_token", return_value="hf_x"),
            mock.patch.object(menu_bar.transcriber, "preload", **patches.get("t", {})) as t,
            mock.patch.object(menu_bar.diarizer, "preload", **patches.get("d", {})) as d,
        ):
            self._app()._preload_model()
        return t, d

    def test_both_stacks_are_loaded(self):
        t, d = self._run()
        t.assert_called_once()
        d.assert_called_once_with("hf_x")

    def test_diarizer_still_loads_when_transcriber_fails(self):
        """One backend being broken must not cost the other."""
        t, d = self._run(t={"side_effect": RuntimeError("no parakeet")})
        d.assert_called_once()

    def test_diarizer_failure_is_swallowed(self):
        self._run(d={"side_effect": RuntimeError("no torch")})


if __name__ == "__main__":
    unittest.main()
