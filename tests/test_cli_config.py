"""Tests for the `trnscrb config` get/set/list command."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from trnscrb import cli, settings


class ConfigCommandTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patch.object(settings, "_SETTINGS_FILE", Path(self._tmp.name) / "s.json").start()
        self.addCleanup(patch.stopall)

    def test_set_string(self):
        r = self.runner.invoke(cli.cli, ["config", "set", "user_name", "Jonathan"])
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(settings.get("user_name"), "Jonathan")

    def test_set_bool_coercion(self):
        self.runner.invoke(cli.cli, ["config", "set", "auto_enrich", "false"])
        self.assertIs(settings.get("auto_enrich"), False)
        self.runner.invoke(cli.cli, ["config", "set", "auto_enrich", "on"])
        self.assertIs(settings.get("auto_enrich"), True)

    def test_set_int_coercion(self):
        self.runner.invoke(cli.cli, ["config", "set", "mlx_cache_limit_mb", "256"])
        self.assertEqual(settings.get("mlx_cache_limit_mb"), 256)

    def test_bad_bool_is_rejected(self):
        r = self.runner.invoke(cli.cli, ["config", "set", "auto_enrich", "maybe"])
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("boolean", r.output)

    def test_unknown_key_rejected(self):
        r = self.runner.invoke(cli.cli, ["config", "set", "nope", "x"])
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("Unknown setting", r.output)

    def test_get_and_list(self):
        self.runner.invoke(cli.cli, ["config", "set", "user_name", "Jonathan"])
        got = self.runner.invoke(cli.cli, ["config", "get", "user_name"])
        self.assertIn("Jonathan", got.output)
        listed = self.runner.invoke(cli.cli, ["config", "list"])
        self.assertIn("user_name", listed.output)
        self.assertNotIn("  enrich =", listed.output)  # nested dict settings are excluded


if __name__ == "__main__":
    unittest.main()
