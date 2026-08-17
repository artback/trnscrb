"""Tests for the launch-at-login plist.

The plist injected DYLD_LIBRARY_PATH into the app, which made AppKit and
ImageIO load Homebrew's dylibs instead of the system ones. The app died three
seconds into every launch — but only when launched from the LaunchAgent, so
it read as a launchd fault rather than a plist one.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trnscrb import cli


class PlistContentTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.plist = Path(tmp.name) / "io.trnscrb.app.plist"
        patcher = mock.patch.object(cli, "_PLIST_PATH", self.plist)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self):
        with (
            mock.patch("trnscrb.app_bundle.ensure_bundle", return_value="/Apps/Trnscrb.app/x"),
            mock.patch.object(cli.subprocess, "run"),
        ):
            self.assertTrue(cli._setup_login_item("/opt/homebrew/bin/trnscrb"))
        return self.plist.read_text()

    def test_no_dyld_override_is_written(self):
        self.assertNotIn("DYLD_LIBRARY_PATH", self._write())

    def test_it_still_launches_through_the_bundle(self):
        """The bundle is what owns the Screen Recording grant."""
        self.assertIn("Trnscrb.app", self._write())

    def test_it_keeps_the_restart_policy(self):
        content = self._write()
        self.assertIn("SuccessfulExit", content)
        self.assertIn("RunAtLoad", content)


class NeedsUpdateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.plist = Path(tmp.name) / "io.trnscrb.app.plist"
        patcher = mock.patch.object(cli, "_PLIST_PATH", self.plist)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _plist(self, body):
        self.plist.write_text(body)

    def test_a_plist_with_the_dyld_override_needs_rewriting(self):
        self._plist("SuccessfulExit Trnscrb.app DYLD_LIBRARY_PATH")
        self.assertTrue(cli._login_item_needs_update())

    def test_a_current_plist_is_left_alone(self):
        self._plist("SuccessfulExit Trnscrb.app")
        self.assertFalse(cli._login_item_needs_update())

    def test_a_plist_without_the_restart_policy_needs_rewriting(self):
        self._plist("Trnscrb.app")
        self.assertTrue(cli._login_item_needs_update())

    def test_a_plist_not_using_the_bundle_needs_rewriting(self):
        self._plist("SuccessfulExit")
        self.assertTrue(cli._login_item_needs_update())

    def test_a_missing_plist_is_not_an_update(self):
        self.assertFalse(cli._login_item_needs_update())


if __name__ == "__main__":
    unittest.main()
