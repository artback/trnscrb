from pathlib import Path
import unittest
from unittest import mock

from click.testing import CliRunner

from trnscrb.cli import cli


class CliEnrichTests(unittest.TestCase):
    def test_enrich_uses_selected_provider_and_updates_transcript(self):
        runner = CliRunner()
        with mock.patch("trnscrb.storage.read_transcript", return_value="[SPEAKER_00] hi"), mock.patch(
            "trnscrb.storage.save_transcript"
        ) as save_mock, mock.patch(
            "trnscrb.storage.NOTES_DIR", Path("/tmp")
        ), mock.patch(
            "trnscrb.calendar_integration.get_current_or_upcoming_event", return_value=None
        ), mock.patch(
            "trnscrb.enricher.get_active_provider_config",
            return_value=("llama_cpp", {"model": "qwen2.5"}),
        ), mock.patch(
            "trnscrb.enricher.provider_label", return_value="llama.cpp"
        ), mock.patch(
            "trnscrb.enricher.enrich_transcript",
            return_value={
                "enrichment": "SUMMARY:\nHello",
                "speaker_map": {"SPEAKER_00": "Alex"},
                "enriched_transcript": "[Alex] hi",
            },
        ):
            result = runner.invoke(cli, ["enrich", "abc"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running enrichment with llama.cpp", result.output)
        self.assertTrue(save_mock.called)


if __name__ == "__main__":
    unittest.main()
