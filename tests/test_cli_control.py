"""Tests for the permission-free CLI control commands (toggle recording)."""

import signal
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from trnscrb import cli


class ToggleCommandTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_errors_when_app_not_running(self):
        with patch.object(cli, "_running_app_pid", return_value=None):
            result = self.runner.invoke(cli.cli, ["toggle"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("isn't running", result.output)

    def test_sends_sigusr1_when_running(self):
        with (
            patch.object(cli, "_running_app_pid", return_value=4321),
            patch("os.kill") as kill,
        ):
            result = self.runner.invoke(cli.cli, ["toggle"])
        self.assertEqual(result.exit_code, 0)
        kill.assert_called_once_with(4321, signal.SIGUSR1)

    def test_reports_signal_failure(self):
        with (
            patch.object(cli, "_running_app_pid", return_value=4321),
            patch("os.kill", side_effect=OSError("no such process")),
        ):
            result = self.runner.invoke(cli.cli, ["toggle"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Could not signal", result.output)


class RunningAppPidTest(unittest.TestCase):
    def test_none_when_no_lock_holder(self):
        with patch("trnscrb.single_instance.SingleInstance.holder_pid", return_value=None):
            self.assertIsNone(cli._running_app_pid())

    def test_none_when_pid_is_dead(self):
        with (
            patch("trnscrb.single_instance.SingleInstance.holder_pid", return_value=999999),
            patch("os.kill", side_effect=OSError("dead")),
        ):
            self.assertIsNone(cli._running_app_pid())

    def test_returns_live_pid(self):
        with (
            patch("trnscrb.single_instance.SingleInstance.holder_pid", return_value=4321),
            patch("os.kill", return_value=None),
        ):
            self.assertEqual(cli._running_app_pid(), 4321)


if __name__ == "__main__":
    unittest.main()
