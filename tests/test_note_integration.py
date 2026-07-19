"""Tests for Claude Code note integration (menu_bar helpers)."""

import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb import menu_bar


def _setting(values):
    """Return a get_setting stand-in backed by a dict."""
    return lambda key: values.get(key)


class FindClaudeCliTest(unittest.TestCase):
    def test_prefers_path_lookup(self):
        with patch("shutil.which", return_value="/somewhere/claude"):
            self.assertEqual(menu_bar._find_claude_cli(), "/somewhere/claude")

    def test_none_when_absent_everywhere(self):
        with (
            patch("shutil.which", return_value=None),
            patch.object(Path, "is_file", return_value=False),
        ):
            self.assertIsNone(menu_bar._find_claude_cli())


class IntegrateNotesTest(unittest.TestCase):
    SETTINGS = {
        "integrate_prompt": "Integrate {transcript_path} into notes.",
        "integrate_allowed_tools": "Read,Write",
    }

    def test_launches_claude_with_prompt_and_tools(self):
        with (
            patch.object(menu_bar, "_find_claude_cli", return_value="/bin/claude"),
            patch.object(menu_bar, "get_setting", _setting(self.SETTINGS)),
            patch.object(menu_bar.subprocess, "Popen") as popen,
        ):
            menu_bar._integrate_notes(Path("/tmp/x.md"))

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:2], ["/bin/claude", "-p"])
        self.assertIn("/tmp/x.md", cmd[2])
        self.assertEqual(cmd[3:], ["--allowedTools", "Read,Write"])

    def test_empty_allowed_tools_omits_flag(self):
        settings = {**self.SETTINGS, "integrate_allowed_tools": ""}
        with (
            patch.object(menu_bar, "_find_claude_cli", return_value="/bin/claude"),
            patch.object(menu_bar, "get_setting", _setting(settings)),
            patch.object(menu_bar.subprocess, "Popen") as popen,
        ):
            menu_bar._integrate_notes(Path("/tmp/x.md"))

        self.assertNotIn("--allowedTools", popen.call_args.args[0])

    def test_skips_when_cli_missing(self):
        with (
            patch.object(menu_bar, "_find_claude_cli", return_value=None),
            patch.object(menu_bar.subprocess, "Popen") as popen,
        ):
            menu_bar._integrate_notes(Path("/tmp/x.md"))

        popen.assert_not_called()

    def test_bad_template_does_not_launch(self):
        settings = {**self.SETTINGS, "integrate_prompt": "bad {nope} template"}
        with (
            patch.object(menu_bar, "_find_claude_cli", return_value="/bin/claude"),
            patch.object(menu_bar, "get_setting", _setting(settings)),
            patch.object(menu_bar.subprocess, "Popen") as popen,
        ):
            menu_bar._integrate_notes(Path("/tmp/x.md"))

        popen.assert_not_called()

    def test_default_settings_produce_valid_invocation(self):
        """The shipped default prompt/tools must work as-is."""
        from trnscrb.settings import _DEFAULTS

        defaults = {
            "integrate_prompt": _DEFAULTS["integrate_prompt"],
            "integrate_allowed_tools": _DEFAULTS["integrate_allowed_tools"],
        }
        with (
            patch.object(menu_bar, "_find_claude_cli", return_value="/bin/claude"),
            patch.object(menu_bar, "get_setting", _setting(defaults)),
            patch.object(menu_bar.subprocess, "Popen") as popen,
        ):
            menu_bar._integrate_notes(Path("/tmp/meeting.md"))

        cmd = popen.call_args.args[0]
        self.assertIn("/tmp/meeting.md", cmd[2])
        self.assertFalse(_DEFAULTS["auto_integrate"], "auto_integrate must default to off")


class OnBatteryTest(unittest.TestCase):
    def _pmset(self, stdout):
        from unittest.mock import MagicMock

        return MagicMock(stdout=stdout)

    def test_ac_power_is_false(self):
        with patch.object(
            menu_bar.subprocess, "run", return_value=self._pmset("Now drawing from 'AC Power'\n")
        ):
            self.assertFalse(menu_bar._on_battery())

    def test_battery_power_is_true(self):
        with patch.object(
            menu_bar.subprocess,
            "run",
            return_value=self._pmset("Now drawing from 'Battery Power'\n -InternalBattery-0"),
        ):
            self.assertTrue(menu_bar._on_battery())

    def test_error_defaults_to_false(self):
        with patch.object(menu_bar.subprocess, "run", side_effect=OSError("no pmset")):
            self.assertFalse(menu_bar._on_battery())


class ModelUnloadTest(unittest.TestCase):
    def test_transcriber_unload_clears_all_models(self):
        from trnscrb import transcriber

        sentinel = object()
        with (
            patch.object(transcriber, "_whisper_model", sentinel),
            patch.object(transcriber, "_parakeet_model", sentinel),
            patch.object(transcriber, "_voxtral_pipeline", sentinel),
        ):
            transcriber.unload_models()
            self.assertIsNone(transcriber._whisper_model)
            self.assertIsNone(transcriber._parakeet_model)
            self.assertIsNone(transcriber._voxtral_pipeline)

    def test_diarizer_unload_clears_pipeline(self):
        from trnscrb import diarizer

        with patch.object(diarizer, "_pipeline", object()):
            diarizer.unload_pipeline()
            self.assertIsNone(diarizer._pipeline)


if __name__ == "__main__":
    unittest.main()
