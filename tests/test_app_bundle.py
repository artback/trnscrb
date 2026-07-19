"""Tests for the Trnscrb.app wrapper bundle."""

import plistlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb import app_bundle


class AppBundleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = patch.object(app_bundle, "bundle_path", lambda: self.root / "Trnscrb.app")
        patcher.start()
        self.addCleanup(patcher.stop)

        # A fake trnscrb binary that records how it was invoked, then exits 7.
        self.target = self.root / "trnscrb"
        self.record = self.root / "invoked.txt"
        self.target.write_text(
            f'#!/bin/sh\necho "$1 bundle=$TRNSCRB_IN_BUNDLE" > "{self.record}"\nexit 7\n'
        )
        self.target.chmod(0o755)

    def test_ensure_bundle_creates_structure(self):
        executable = app_bundle.ensure_bundle(str(self.target))
        self.assertIsNotNone(executable)
        self.assertTrue(executable.exists())

        info = plistlib.loads((executable.parent.parent / "Info.plist").read_bytes())
        self.assertEqual(info["CFBundleIdentifier"], app_bundle.BUNDLE_ID)
        self.assertEqual(info["CFBundleExecutable"], "Trnscrb")
        self.assertTrue(info["LSUIElement"])
        self.assertIn("NSMicrophoneUsageDescription", info)
        self.assertIn("NSAppleEventsUsageDescription", info)

    @unittest.skipUnless(shutil.which("cc") or shutil.which("clang"), "needs a C compiler")
    def test_launcher_spawns_target_and_propagates_exit_code(self):
        executable = app_bundle.ensure_bundle(str(self.target))
        result = subprocess.run([str(executable)], timeout=30)
        self.assertEqual(result.returncode, 7, "child exit code must propagate")
        self.assertEqual(self.record.read_text().strip(), "start bundle=1")

    def test_script_fallback_without_compiler(self):
        with patch.object(app_bundle.shutil, "which", return_value=None):
            executable = app_bundle.ensure_bundle(str(self.target))
        self.assertIsNotNone(executable)
        self.assertIn("#!/bin/sh", executable.read_text())
        result = subprocess.run([str(executable)], timeout=30)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.record.read_text().strip(), "start bundle=1")

    def test_idempotent_when_current(self):
        first = app_bundle.ensure_bundle(str(self.target))
        mtime = first.stat().st_mtime_ns
        second = app_bundle.ensure_bundle(str(self.target))
        self.assertEqual(first, second)
        self.assertEqual(second.stat().st_mtime_ns, mtime, "must not rebuild when current")

    def test_rebuilds_when_target_changes(self):
        app_bundle.ensure_bundle(str(self.target))
        other = self.root / "other-trnscrb"
        shutil.copy(self.target, other)
        other.chmod(0o755)
        self.assertFalse(app_bundle.is_current(str(other)))
        app_bundle.ensure_bundle(str(other))
        self.assertTrue(app_bundle.is_current(str(other)))

    def test_returns_none_for_missing_binary(self):
        self.assertIsNone(app_bundle.ensure_bundle(str(self.root / "nope")))


if __name__ == "__main__":
    unittest.main()
