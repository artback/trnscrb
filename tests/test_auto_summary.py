"""Tests for the automatic post-call summary: summary_block, summarize_for_auto,
and the transcript header slot."""

import unittest
from datetime import datetime
from unittest.mock import patch

from trnscrb import enricher, storage

_ENRICHMENT = """SUMMARY:
We agreed to ship the cloud-artifacts POC behind auth.

ACTION ITEMS:
- Add authentication to shared links (Owner: Jonathan)
- Draft the pricing note (Owner: Andre)

SPEAKER MAPPING:
- SPEAKER_00 → Jonathan
- SPEAKER_01 → Andre
"""


class SummaryBlockTest(unittest.TestCase):
    def test_keeps_summary_and_actions_drops_speaker_mapping(self):
        block = enricher.summary_block(_ENRICHMENT)
        self.assertIn("SUMMARY:", block)
        self.assertIn("ACTION ITEMS:", block)
        self.assertIn("authentication to shared links", block)
        self.assertNotIn("SPEAKER MAPPING", block)
        self.assertNotIn("SPEAKER_00", block)

    def test_handles_missing_mapping_section(self):
        text = "SUMMARY:\nShort call.\n\nACTION ITEMS:\n- none"
        self.assertEqual(enricher.summary_block(text), text)


class SummarizeForAutoTest(unittest.TestCase):
    def test_falls_back_to_claude_cli_when_configured_provider_fails(self):
        calls = []

        def fake_enrich(text, event, provider=None, open_items=None):
            calls.append(provider)
            if provider == "llama_cpp":
                raise RuntimeError("connection refused")
            return {"enrichment": _ENRICHMENT, "provider": provider, "model": "sonnet"}

        with (
            patch.object(enricher, "get_active_provider_config", return_value=("llama_cpp", {})),
            patch("shutil.which", return_value="/opt/homebrew/bin/claude"),
            patch.object(enricher, "enrich_transcript", side_effect=fake_enrich),
        ):
            result = enricher.summarize_for_auto("transcript", calendar_event=None)

        self.assertIsNotNone(result)
        self.assertEqual(result["provider"], "claude_code")
        self.assertEqual(calls, ["llama_cpp", "claude_code"])  # tried configured, then CLI

    def test_returns_none_when_nothing_available(self):
        with (
            patch.object(enricher, "get_active_provider_config", return_value=("llama_cpp", {})),
            patch("shutil.which", return_value=None),  # no claude CLI
            patch.object(enricher, "enrich_transcript", side_effect=RuntimeError("no server")),
        ):
            result = enricher.summarize_for_auto("transcript", calendar_event=None)
        self.assertIsNone(result)

    def test_does_not_retry_cli_when_it_is_already_the_provider(self):
        with (
            patch.object(enricher, "get_active_provider_config", return_value=("claude_code", {})),
            patch("shutil.which", return_value="/opt/homebrew/bin/claude"),
            patch.object(
                enricher, "enrich_transcript", side_effect=RuntimeError("cli boom")
            ) as enrich,
        ):
            result = enricher.summarize_for_auto("transcript", calendar_event=None)
        self.assertIsNone(result)
        self.assertEqual(enrich.call_count, 1)  # claude_code tried once, not twice


class TranscriptSummarySlotTest(unittest.TestCase):
    def test_summary_appears_above_the_transcript_body(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Me"}]
        out = storage.format_transcript(
            segments,
            datetime(2026, 7, 30, 11, 0),
            "Google Meet",
            ai_summary="SUMMARY:\nShort call.",
        )
        self.assertIn("SUMMARY:", out)
        # Summary sits before the divider that precedes the transcript body.
        self.assertLess(out.index("SUMMARY:"), out.index("=" * 60))

    def test_no_summary_when_absent(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Me"}]
        out = storage.format_transcript(segments, datetime(2026, 7, 30, 11, 0), "Meet")
        self.assertNotIn("SUMMARY:", out)


if __name__ == "__main__":
    unittest.main()
