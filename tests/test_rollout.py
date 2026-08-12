"""Tests for surviving an upgrade that deletes the running install.

`brew upgrade` removes the Cellar tree a running trnscrb executes from. The
process keeps working on already-imported modules and fails only at the next
import — which, because heavy dependencies load lazily, lands on a feature
mid-meeting rather than on the app. It cost a silent day of lost speaker
labels and voiceprints.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trnscrb import rollout


class IsStaleTest(unittest.TestCase):
    def test_existing_install_is_not_stale(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(rollout, "install_root", return_value=Path(d)):
                self.assertFalse(rollout.is_stale())

    def test_deleted_install_is_stale(self):
        with tempfile.TemporaryDirectory() as d:
            gone = Path(d) / "0.36.0" / "libexec" / "venv"
        with mock.patch.object(rollout, "install_root", return_value=gone):
            self.assertTrue(rollout.is_stale())

    def test_real_process_is_not_stale(self):
        """This interpreter's own prefix exists, so nothing false-positives."""
        self.assertFalse(rollout.is_stale())


class RestartTest(unittest.TestCase):
    def _no_agent(self):
        return mock.patch.object(Path, "is_file", return_value=False)

    def test_prefers_launchd_when_the_agent_exists(self):
        with (
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(rollout.subprocess, "run") as run,
        ):
            self.assertTrue(rollout.restart())
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["launchctl", "kickstart", "-k"])
        self.assertIn("io.trnscrb.app", argv[3])

    def test_falls_back_to_launching_the_binary(self):
        with (
            self._no_agent(),
            mock.patch.object(rollout.shutil, "which", return_value="/opt/homebrew/bin/trnscrb"),
            mock.patch.object(rollout.subprocess, "Popen") as popen,
        ):
            self.assertTrue(rollout.restart())
        self.assertEqual(popen.call_args.args[0], ["/opt/homebrew/bin/trnscrb", "start"])

    def test_launchd_failure_falls_back(self):
        with (
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(
                rollout.subprocess, "run", side_effect=subprocess.SubprocessError("boom")
            ),
            mock.patch.object(rollout.shutil, "which", return_value="/bin/trnscrb"),
            mock.patch.object(rollout.subprocess, "Popen") as popen,
        ):
            self.assertTrue(rollout.restart())
        popen.assert_called_once()

    def test_reports_failure_when_nothing_can_be_launched(self):
        with (
            self._no_agent(),
            mock.patch.object(rollout.shutil, "which", return_value=None),
        ):
            self.assertFalse(rollout.restart())


class CheckRolloutTest(unittest.TestCase):
    """The app must never restart out from under a meeting."""

    def _app(self, *, recording=False, processing=False):
        from trnscrb.menu_bar import TrnscrbApp

        app = TrnscrbApp.__new__(TrnscrbApp)
        app._stale_notified = False
        app._rollout_timer = mock.Mock()
        app._recorder = mock.Mock(is_recording=True) if recording else None
        app._process_thread = mock.Mock(**{"is_alive.return_value": True}) if processing else None
        app._shutdown = mock.Mock()
        return app

    def _run(self, app, stale=True):
        with (
            mock.patch.object(rollout, "is_stale", return_value=stale),
            mock.patch.object(rollout, "restart", return_value=True) as restart,
            mock.patch("trnscrb.menu_bar.rumps.quit_application") as quit_app,
            mock.patch("trnscrb.menu_bar._notify"),
        ):
            app._check_rollout()
        return restart, quit_app

    def test_idle_and_stale_restarts(self):
        restart, quit_app = self._run(self._app())
        restart.assert_called_once()
        quit_app.assert_called_once()

    def test_recording_defers_the_restart(self):
        restart, quit_app = self._run(self._app(recording=True))
        restart.assert_not_called()
        quit_app.assert_not_called()

    def test_transcribing_defers_the_restart(self):
        restart, _ = self._run(self._app(processing=True))
        restart.assert_not_called()

    def test_healthy_install_does_nothing(self):
        restart, _ = self._run(self._app(), stale=False)
        restart.assert_not_called()

    def test_in_progress_recording_is_saved_before_quitting(self):
        app = self._app()
        self._run(app)
        app._shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
