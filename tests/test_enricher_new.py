"""Tests for enricher module: ClaudeCodeAdapter, generate_weekly_summary,
generate_annual_summary, _load_prompt, and MCP tools (get_weekly_transcripts,
search_transcripts)."""

import subprocess
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from trnscrb import enricher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SETTINGS_WITH_CLAUDE_CODE = {
    "enrich": {
        "provider": "claude_code",
        "profiles": {
            "claude_code": {
                "endpoint": "",
                "api_key": "",
                "model": "sonnet",
                "models": ["sonnet"],
            }
        },
    }
}


class _FakeAdapter:
    """Captures the prompt and config passed to enrich()."""

    def __init__(self, response="Summary text"):
        self.last_prompt = ""
        self.last_config = {}
        self.response = response

    def test_connection(self, config):
        return True, "ok"

    def list_models(self, config):
        return ["model-a"]

    def enrich(self, prompt, config):
        self.last_prompt = prompt
        self.last_config = config
        return self.response


# ===========================================================================
# 1. ClaudeCodeAdapter
# ===========================================================================


class TestClaudeCodeAdapterTestConnection(unittest.TestCase):
    """test_connection when CLI is found / not found."""

    def test_cli_found(self):
        adapter = enricher.ClaudeCodeAdapter()
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="1.2.3\n", stderr="")
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/local/bin/claude"), \
             mock.patch("subprocess.run", return_value=proc):
            ok, msg = adapter.test_connection({})
        self.assertTrue(ok)
        self.assertIn("1.2.3", msg)

    def test_cli_not_found(self):
        adapter = enricher.ClaudeCodeAdapter()
        with mock.patch.object(adapter, "_find_cli", return_value=None):
            ok, msg = adapter.test_connection({})
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_cli_subprocess_exception(self):
        adapter = enricher.ClaudeCodeAdapter()
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/local/bin/claude"), \
             mock.patch("subprocess.run", side_effect=OSError("boom")):
            ok, msg = adapter.test_connection({})
        self.assertFalse(ok)
        self.assertIn("boom", msg)


class TestClaudeCodeAdapterListModels(unittest.TestCase):
    def test_returns_fixed_list(self):
        adapter = enricher.ClaudeCodeAdapter()
        models = adapter.list_models({})
        self.assertEqual(models, ["sonnet", "opus", "haiku"])


class TestClaudeCodeAdapterEnrich(unittest.TestCase):
    """enrich(): subprocess calls, non-zero exit, empty output, timeout."""

    def test_success(self):
        adapter = enricher.ClaudeCodeAdapter()
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="LLM response\n", stderr="")
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/bin/claude"), \
             mock.patch("subprocess.run", return_value=proc) as run_mock:
            result = adapter.enrich("my prompt", {"model": "opus"})
        self.assertEqual(result, "LLM response")
        args = run_mock.call_args[0][0]
        self.assertIn("-p", args)
        self.assertIn("my prompt", args)
        self.assertIn("opus", args)

    def test_cli_not_found_raises(self):
        adapter = enricher.ClaudeCodeAdapter()
        with mock.patch.object(adapter, "_find_cli", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.enrich("prompt", {"model": "sonnet"})
        self.assertIn("not found", str(ctx.exception))

    def test_non_zero_exit_raises(self):
        adapter = enricher.ClaudeCodeAdapter()
        proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="auth failed")
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/bin/claude"), \
             mock.patch("subprocess.run", return_value=proc):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.enrich("prompt", {"model": "sonnet"})
        self.assertIn("auth failed", str(ctx.exception))

    def test_empty_output_raises(self):
        adapter = enricher.ClaudeCodeAdapter()
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/bin/claude"), \
             mock.patch("subprocess.run", return_value=proc):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.enrich("prompt", {"model": "sonnet"})
        self.assertIn("empty", str(ctx.exception).lower())

    def test_timeout_propagates(self):
        adapter = enricher.ClaudeCodeAdapter()
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/bin/claude"), \
             mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=120)):
            with self.assertRaises(subprocess.TimeoutExpired):
                adapter.enrich("prompt", {"model": "sonnet"})

    def test_default_model_is_sonnet(self):
        adapter = enricher.ClaudeCodeAdapter()
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")
        with mock.patch.object(adapter, "_find_cli", return_value="/usr/bin/claude"), \
             mock.patch("subprocess.run", return_value=proc) as run_mock:
            adapter.enrich("prompt", {})
        args = run_mock.call_args[0][0]
        model_idx = args.index("--model")
        self.assertEqual(args[model_idx + 1], "sonnet")


# ===========================================================================
# 2. _load_prompt
# ===========================================================================


class TestLoadPrompt(unittest.TestCase):
    def test_falls_back_to_default_when_no_file(self):
        with mock.patch.object(Path, "exists", return_value=False):
            result = enricher._load_prompt("weekly", "DEFAULT PROMPT")
        self.assertEqual(result, "DEFAULT PROMPT")

    def test_loads_from_file_when_exists(self):
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(Path, "read_text", return_value="CUSTOM PROMPT"):
            result = enricher._load_prompt("weekly", "DEFAULT PROMPT")
        self.assertEqual(result, "CUSTOM PROMPT")


# ===========================================================================
# 3. generate_weekly_summary
# ===========================================================================


class TestGenerateWeeklySummary(unittest.TestCase):
    def _patch_settings(self):
        return mock.patch("trnscrb.settings.load", return_value=_SETTINGS_WITH_CLAUDE_CODE)

    def test_combines_transcripts_and_calls_adapter(self):
        fake = _FakeAdapter(response="Weekly summary output")
        transcripts = [
            {"name": "standup.txt", "text": "Standup notes"},
            {"name": "retro.txt", "text": "Retro notes"},
        ]
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}):
            result = enricher.generate_weekly_summary(
                transcripts, "2026-03-23", "2026-03-27",
            )
        self.assertEqual(result, "Weekly summary output")
        self.assertIn("standup.txt", fake.last_prompt)
        self.assertIn("retro.txt", fake.last_prompt)
        self.assertIn("Standup notes", fake.last_prompt)
        self.assertIn("Retro notes", fake.last_prompt)

    def test_uses_correct_week_dates_in_prompt(self):
        fake = _FakeAdapter()
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}):
            enricher.generate_weekly_summary(
                [{"name": "a.txt", "text": "t"}], "2026-03-23", "2026-03-27",
            )
        self.assertIn("2026-03-23", fake.last_prompt)
        self.assertIn("2026-03-27", fake.last_prompt)

    def test_prompt_override(self):
        fake = _FakeAdapter()
        custom = "Custom: {week_start} {week_end} {transcripts}"
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}):
            enricher.generate_weekly_summary(
                [{"name": "a.txt", "text": "content"}],
                "2026-03-23", "2026-03-27",
                prompt_override=custom,
            )
        self.assertTrue(fake.last_prompt.startswith("Custom:"))
        self.assertIn("content", fake.last_prompt)

    def test_loads_custom_prompt_file_when_exists(self):
        """If no prompt_override, _load_prompt should be consulted."""
        fake = _FakeAdapter()
        custom_template = "FILE: {week_start} to {week_end}\n{transcripts}"
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}), \
             mock.patch("trnscrb.enricher._load_prompt", return_value=custom_template) as lp:
            enricher.generate_weekly_summary(
                [{"name": "a.txt", "text": "x"}], "2026-01-01", "2026-01-05",
            )
        lp.assert_called_once_with("weekly", enricher._DEFAULT_WEEKLY_PROMPT)
        self.assertTrue(fake.last_prompt.startswith("FILE:"))


# ===========================================================================
# 4. generate_annual_summary
# ===========================================================================


class TestGenerateAnnualSummary(unittest.TestCase):
    def _patch_settings(self):
        return mock.patch("trnscrb.settings.load", return_value=_SETTINGS_WITH_CLAUDE_CODE)

    def test_passes_summaries_and_year(self):
        fake = _FakeAdapter(response="Annual output")
        weekly_text = "Week 1 summary\nWeek 2 summary"
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}):
            result = enricher.generate_annual_summary(weekly_text, "2026")
        self.assertEqual(result, "Annual output")
        self.assertIn("Week 1 summary", fake.last_prompt)
        self.assertIn("2026", fake.last_prompt)

    def test_prompt_override(self):
        fake = _FakeAdapter()
        custom = "Annual custom: {summaries} for {year}"
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}):
            enricher.generate_annual_summary("data", "2026", prompt_override=custom)
        self.assertTrue(fake.last_prompt.startswith("Annual custom:"))

    def test_loads_custom_prompt_file_when_exists(self):
        fake = _FakeAdapter()
        custom_template = "FROM FILE: {summaries} ({year})"
        with self._patch_settings(), \
             mock.patch.dict(enricher._ADAPTERS, {"claude_code": fake}), \
             mock.patch("trnscrb.enricher._load_prompt", return_value=custom_template) as lp:
            enricher.generate_annual_summary("data", "2026")
        lp.assert_called_once_with("annual", enricher._DEFAULT_ANNUAL_PROMPT)


# ===========================================================================
# 5. MCP tools: get_weekly_transcripts
# ===========================================================================


class TestMCPGetWeeklyTranscripts(unittest.TestCase):
    """Tests for mcp_server.get_weekly_transcripts."""

    def setUp(self):
        import tempfile, shutil
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = mock.patch("trnscrb.storage.NOTES_DIR", Path(self._tmpdir))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, name, text):
        p = Path(self._tmpdir) / name
        p.write_text(text, encoding="utf-8")

    def test_filters_by_date_range(self):
        from trnscrb.mcp_server import get_weekly_transcripts
        self._write("2026-03-23_09-00-00_standup.txt", "standup text")
        self._write("2026-03-25_14-00-00_retro.txt", "retro text")
        self._write("2026-03-30_10-00-00_other.txt", "other text")
        result = get_weekly_transcripts("2026-W13")
        self.assertIn("standup text", result)
        self.assertIn("retro text", result)
        self.assertNotIn("other text", result)
        self.assertIn("Transcripts: 2", result)

    def test_skips_weekly_and_annual_files(self):
        from trnscrb.mcp_server import get_weekly_transcripts
        self._write("weekly-2026-W13.txt", "weekly summary")
        self._write("annual-2026.txt", "annual summary")
        self._write("2026-03-23_09-00-00_standup.txt", "standup text")
        result = get_weekly_transcripts("2026-W13")
        self.assertIn("standup text", result)
        self.assertNotIn("weekly summary", result)
        self.assertNotIn("annual summary", result)

    def test_no_transcripts_found(self):
        from trnscrb.mcp_server import get_weekly_transcripts

        with mock.patch("trnscrb.storage.NOTES_DIR") as mock_dir:
            mock_dir.glob.return_value = []
            result = get_weekly_transcripts("2026-W13")

        self.assertIn("No transcripts found", result)

    def test_invalid_week_format(self):
        from trnscrb.mcp_server import get_weekly_transcripts
        result = get_weekly_transcripts("bad-week")
        self.assertIn("Invalid week format", result)


# ===========================================================================
# 6. MCP tools: search_transcripts
# ===========================================================================


class TestMCPSearchTranscripts(unittest.TestCase):
    """Tests for mcp_server.search_transcripts."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = mock.patch("trnscrb.storage.NOTES_DIR", Path(self._tmpdir))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, name, text):
        p = Path(self._tmpdir) / name
        p.write_text(text, encoding="utf-8")

    def test_returns_matches_with_context(self):
        from trnscrb.mcp_server import search_transcripts
        self._write("2026-03-23_standup.txt", "line1\nline2\nfind this keyword here\nline4\nline5")
        result = search_transcripts("keyword")
        self.assertIn("keyword", result)
        self.assertIn("line2", result)
        self.assertIn("line4", result)
        self.assertIn("2026-03-23_standup.txt", result)

    def test_no_matches(self):
        from trnscrb.mcp_server import search_transcripts
        self._write("2026-03-23_standup.txt", "nothing here")
        result = search_transcripts("nonexistent")
        self.assertIn("No matches", result)

    def test_no_files(self):
        from trnscrb.mcp_server import search_transcripts
        result = search_transcripts("anything")
        self.assertIn("No transcripts found", result)

    def test_matches_across_multiple_files(self):
        from trnscrb.mcp_server import search_transcripts
        self._write("file1.txt", "alpha keyword beta")
        self._write("file2.txt", "gamma keyword delta")
        result = search_transcripts("keyword")
        self.assertIn("file1.txt", result)
        self.assertIn("file2.txt", result)


if __name__ == "__main__":
    unittest.main()
