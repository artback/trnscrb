"""Tests for diarization pipeline selection (community-1 with 3.1 fallback)."""

import unittest
from unittest.mock import patch

from trnscrb import diarizer


class _FakePipeline:
    def __init__(self, name):
        self.name = name

    def to(self, _device):
        return self


class PipelineSelectionTest(unittest.TestCase):
    def setUp(self):
        diarizer._pipeline = None
        self.addCleanup(setattr, diarizer, "_pipeline", None)

    def _get(self, load_side_effect, setting=None):
        with (
            patch.object(diarizer, "_load_pipeline", side_effect=load_side_effect) as load,
            patch("trnscrb.settings.get", return_value=setting),
        ):
            pipeline = diarizer._get_pipeline("hf_token")
        return pipeline, load

    def test_prefers_community_1(self):
        pipeline, load = self._get(lambda model_id, token: _FakePipeline(model_id))
        self.assertEqual(pipeline.name, "pyannote/speaker-diarization-community-1")
        self.assertEqual(load.call_count, 1)

    def test_falls_back_to_3_1_when_community_unavailable(self):
        def load(model_id, token):
            if "community" in model_id:
                raise RuntimeError("gated")
            return _FakePipeline(model_id)

        pipeline, mock_load = self._get(load)
        self.assertEqual(pipeline.name, "pyannote/speaker-diarization-3.1")
        self.assertEqual(mock_load.call_count, 2)

    def test_raises_when_no_pipeline_loads(self):
        with (
            patch.object(diarizer, "_load_pipeline", side_effect=RuntimeError("gated")),
            patch("trnscrb.settings.get", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "No diarization pipeline"):
                diarizer._get_pipeline("hf_token")

    def test_setting_overrides_preferred_pipeline(self):
        pipeline, load = self._get(
            lambda model_id, token: _FakePipeline(model_id),
            setting="my-org/custom-diarizer",
        )
        self.assertEqual(pipeline.name, "my-org/custom-diarizer")


if __name__ == "__main__":
    unittest.main()
