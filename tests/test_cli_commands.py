"""Tests for CLI commands (search, weekly) and storage helpers."""

import os
import tempfile
import unittest
from datetime import datetime, date
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from trnscrb.cli import cli
from trnscrb import storage


class _TmpNotesMixin:
    """Create a real temp directory for NOTES_DIR so sorted(glob()) works."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = mock.patch("trnscrb.storage.NOTES_DIR", Path(self._tmpdir))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, name: str, text: str) -> Path:
        p = Path(self._tmpdir) / name
        p.write_text(text, encoding="utf-8")
        return p


# ===========================================================================
# 1. CLI search command
# ===========================================================================


class TestCLISearchCommand(_TmpNotesMixin, unittest.TestCase):
    """Tests for `trnscrb search`."""

    def test_finds_matches_across_multiple_files(self):
        self._write("file1.txt", "alpha keyword beta")
        self._write("file2.txt", "gamma keyword delta")
        result = CliRunner().invoke(cli, ["search", "keyword"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("file1.txt", result.output)
        self.assertIn("file2.txt", result.output)
        self.assertIn("2 match(es)", result.output)

    def test_respects_context_lines(self):
        self._write("test.txt", "line0\nline1\nline2\nmatch_here\nline4\nline5\nline6")
        result = CliRunner().invoke(cli, ["search", "match_here", "-n", "2"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("line2", result.output)
        self.assertIn("line5", result.output)

    def test_no_match_case(self):
        self._write("test.txt", "nothing relevant")
        result = CliRunner().invoke(cli, ["search", "nonexistent"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No matches for 'nonexistent'", result.output)

    def test_no_files(self):
        result = CliRunner().invoke(cli, ["search", "test"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No transcripts found", result.output)


# ===========================================================================
# 2. CLI weekly command
# ===========================================================================


class TestCLIWeeklyCommand(_TmpNotesMixin, unittest.TestCase):
    """Tests for `trnscrb weekly`."""

    def test_filters_transcripts_by_date_range(self):
        # 2026-W13: Monday 2026-03-23 to Friday 2026-03-27
        self._write("2026-03-23_09-00-00_standup.txt", "standup content")
        self._write("2026-03-30_09-00-00_other.txt", "other content")

        with mock.patch("trnscrb.enricher.generate_weekly_summary", return_value="WEEKLY SUMMARY") as gen_mock, \
             mock.patch("trnscrb.enricher.get_active_provider_config",
                        return_value=("claude_code", {"model": "sonnet"})), \
             mock.patch("trnscrb.enricher.provider_label", return_value="Claude Code"), \
             mock.patch("trnscrb.storage.save_transcript"):
            result = CliRunner().invoke(cli, ["weekly", "--week", "2026-W13"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("WEEKLY SUMMARY", result.output)
        transcripts = gen_mock.call_args[0][0]
        self.assertEqual(len(transcripts), 1)
        self.assertIn("standup", transcripts[0]["name"])

    def test_invalid_week_format(self):
        result = CliRunner().invoke(cli, ["weekly", "--week", "bad"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid week format", result.output)

    def test_no_transcripts_found(self):
        result = CliRunner().invoke(cli, ["weekly", "--week", "2026-W13"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No transcripts found", result.output)

    def test_prompt_flag(self):
        self._write("2026-03-23_09-00-00_standup.txt", "standup content")
        custom_template = "CUSTOM: {week_start} {week_end} {transcripts}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(custom_template)
            tmp_path = tmp.name

        try:
            with mock.patch("trnscrb.enricher.generate_weekly_summary", return_value="RESULT") as gen_mock, \
                 mock.patch("trnscrb.enricher.get_active_provider_config",
                            return_value=("claude_code", {"model": "sonnet"})), \
                 mock.patch("trnscrb.enricher.provider_label", return_value="Claude Code"), \
                 mock.patch("trnscrb.storage.save_transcript"):
                result = CliRunner().invoke(cli, ["weekly", "--week", "2026-W13", "--prompt", tmp_path])

            self.assertEqual(result.exit_code, 0)
            call_kwargs = gen_mock.call_args[1]
            self.assertEqual(call_kwargs["prompt_override"], custom_template)
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# 3. storage.get_transcript_path
# ===========================================================================


class TestGetTranscriptPath(unittest.TestCase):
    def test_includes_seconds_in_filename(self):
        dt = datetime(2026, 3, 23, 14, 30, 45)
        with mock.patch("trnscrb.storage.NOTES_DIR", Path("/tmp/test-notes")), \
             mock.patch("trnscrb.storage.ensure_notes_dir"):
            path = storage.get_transcript_path("standup", dt)
        self.assertIn("2026-03-23_14-30-45", path.name)
        self.assertIn("standup", path.name)

    def test_sanitizes_meeting_name(self):
        dt = datetime(2026, 3, 23, 14, 30, 0)
        with mock.patch("trnscrb.storage.NOTES_DIR", Path("/tmp/test-notes")), \
             mock.patch("trnscrb.storage.ensure_notes_dir"):
            path = storage.get_transcript_path("My Meeting! @home", dt)
        self.assertNotIn("!", path.name)
        self.assertNotIn("@", path.name)
        self.assertNotIn(" ", path.name)

    def test_truncates_long_names(self):
        dt = datetime(2026, 3, 23, 14, 30, 0)
        with mock.patch("trnscrb.storage.NOTES_DIR", Path("/tmp/test-notes")), \
             mock.patch("trnscrb.storage.ensure_notes_dir"):
            path = storage.get_transcript_path("a" * 100, dt)
        stem = path.stem
        name_part = stem.split("_", 3)[-1]
        self.assertLessEqual(len(name_part), 50)


# ===========================================================================
# 4. storage.format_transcript
# ===========================================================================


class TestFormatTranscript(unittest.TestCase):
    def test_duration_spacing(self):
        segments = [{"start": 0.0, "end": 65.0, "text": "Hello", "speaker": "Alice"}]
        text = storage.format_transcript(segments, datetime(2026, 3, 23, 14, 30, 0), "standup")
        self.assertIn("Duration: 01:05", text)

    def test_empty_segments(self):
        text = storage.format_transcript([], datetime(2026, 1, 1, 0, 0), "empty")
        self.assertIn("Duration: 00:00", text)

    def test_speaker_grouping(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "Alice"},
            {"start": 5.0, "end": 10.0, "text": "World", "speaker": "Alice"},
            {"start": 10.0, "end": 15.0, "text": "Hi", "speaker": "Bob"},
        ]
        text = storage.format_transcript(segments, datetime(2026, 1, 1, 0, 0, 0), "test")
        self.assertEqual(text.count("[Alice]"), 1)
        self.assertEqual(text.count("[Bob]"), 1)


class SearchSpecialCharsTest(_TmpNotesMixin, unittest.TestCase):
    """Test search with regex special characters (re.escape should handle them)."""

    def test_search_cplusplus(self):
        """Searching for 'C++' should not crash (+ is a regex metachar)."""
        self._write("test.txt", "We discussed C++ performance improvements")
        result = CliRunner().invoke(cli, ["search", "C++"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("C++", result.output)
        self.assertIn("1 match(es)", result.output)

    def test_search_file_dot_txt(self):
        """Searching for 'file.txt' should treat the dot literally."""
        self._write("notes.txt", "Please update file.txt before the release")
        result = CliRunner().invoke(cli, ["search", "file.txt"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("file.txt", result.output)
        self.assertIn("1 match(es)", result.output)

    def test_search_parentheses(self):
        """Parentheses in search should be escaped."""
        self._write("notes.txt", "Called func(arg) in the handler")
        result = CliRunner().invoke(cli, ["search", "func(arg)"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("func(arg)", result.output)

    def test_search_brackets(self):
        """Square brackets should be escaped."""
        self._write("notes.txt", "Array access arr[0] was discussed")
        result = CliRunner().invoke(cli, ["search", "arr[0]"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("arr[0]", result.output)


class AnnualCommandTest(_TmpNotesMixin, unittest.TestCase):
    """Test the `trnscrb annual` command."""

    def test_no_weekly_files_shows_message(self):
        """If no weekly summaries exist, should show an informative message."""
        result = CliRunner().invoke(cli, ["annual", "--year", "2026"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No weekly summaries found", result.output)

    def test_annual_with_files_calls_generate(self):
        """If weekly files exist, generate_annual_summary should be called."""
        self._write("weekly-2026-W10.txt", "Week 10 summary content")
        self._write("weekly-2026-W11.txt", "Week 11 summary content")

        with mock.patch("trnscrb.enricher.generate_annual_summary", return_value="ANNUAL SUMMARY") as gen_mock, \
             mock.patch("trnscrb.enricher.get_active_provider_config",
                        return_value=("claude_code", {"model": "sonnet"})), \
             mock.patch("trnscrb.enricher.provider_label", return_value="Claude Code"), \
             mock.patch("trnscrb.storage.save_transcript"):
            result = CliRunner().invoke(cli, ["annual", "--year", "2026"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("ANNUAL SUMMARY", result.output)
        gen_mock.assert_called_once()
        # Check both weekly files were included in the combined text
        call_args = gen_mock.call_args[0]
        self.assertIn("Week 10 summary content", call_args[0])
        self.assertIn("Week 11 summary content", call_args[0])
        self.assertEqual(call_args[1], "2026")


class PathTraversalTest(unittest.TestCase):
    """Test that storage.read_transcript blocks path traversal."""

    def test_dotdot_in_transcript_id_returns_none(self):
        """A transcript_id containing '../' should return None."""
        with mock.patch("trnscrb.storage.NOTES_DIR", Path("/tmp/test-notes")):
            result = storage.read_transcript("../../etc/passwd")
        self.assertIsNone(result)

    def test_dotdot_relative_traversal_blocked(self):
        """Even a subtle traversal like 'foo/../../../etc/passwd' should be blocked."""
        with mock.patch("trnscrb.storage.NOTES_DIR", Path("/tmp/test-notes")):
            result = storage.read_transcript("foo/../../../etc/passwd")
        self.assertIsNone(result)

    def test_normal_id_works(self):
        """A normal transcript_id should work (if file exists)."""
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            p = Path(tmpdir) / "2026-03-23_standup.txt"
            p.write_text("content", encoding="utf-8")
            with mock.patch("trnscrb.storage.NOTES_DIR", Path(tmpdir)):
                result = storage.read_transcript("2026-03-23_standup")
            self.assertEqual(result, "content")
        finally:
            shutil.rmtree(tmpdir)


class DevicesCommandTest(unittest.TestCase):
    """Test `trnscrb devices` command.

    The `devices` CLI command does::

        from trnscrb.recorder import Recorder
        devs = Recorder.list_input_devices()

    Because ``sounddevice`` (a C-extension dependency of ``trnscrb.recorder``)
    may not be installed in the test environment, we insert a stub into
    ``sys.modules`` *before* the module is first imported.  Once the module is
    loaded the ``Recorder`` class can be patched normally.
    """

    @classmethod
    def setUpClass(cls):
        """Ensure trnscrb.recorder can be imported even without sounddevice."""
        import sys
        if "sounddevice" not in sys.modules:
            sys.modules["sounddevice"] = mock.MagicMock()
            cls._sd_injected = True
        else:
            cls._sd_injected = False
        # Now import so the module is in sys.modules for patching
        import trnscrb.recorder  # noqa: F401

    @classmethod
    def tearDownClass(cls):
        if cls._sd_injected:
            import sys
            sys.modules.pop("sounddevice", None)

    def _run_devices(self, device_list):
        mock_rec = mock.MagicMock()
        mock_rec.list_input_devices.return_value = device_list
        with mock.patch("trnscrb.recorder.Recorder", mock_rec):
            return CliRunner().invoke(cli, ["devices"])

    def test_no_devices(self):
        """When no input devices are found, should show a message."""
        result = self._run_devices([])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No input devices found", result.output)

    def test_with_devices(self):
        """When devices are found, should list them."""
        devices = [
            {"index": 0, "name": "Built-in Microphone", "channels": 1},
            {"index": 1, "name": "BlackHole 2ch", "channels": 2},
        ]
        result = self._run_devices(devices)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Built-in Microphone", result.output)
        self.assertIn("BlackHole 2ch", result.output)
        self.assertIn("(BlackHole)", result.output)
        self.assertIn("[0]", result.output)
        self.assertIn("[1]", result.output)

    def test_no_blackhole_tag_for_regular_device(self):
        """Regular devices should not have the (BlackHole) tag."""
        devices = [
            {"index": 0, "name": "USB Microphone", "channels": 1},
        ]
        result = self._run_devices(devices)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("USB Microphone", result.output)
        self.assertNotIn("(BlackHole)", result.output)


if __name__ == "__main__":
    unittest.main()
