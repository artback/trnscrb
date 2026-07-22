"""Tests for the bundled sck-capture helper and its packaging.

The helper is the process that holds the Screen Recording permission, so the
properties tested here are what keep that grant scoped to Trnscrb.
"""

import os
import plistlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb import app_bundle, sck_helper


class HelperSourceTest(unittest.TestCase):
    def test_swift_source_ships_with_the_package(self):
        self.assertTrue(
            app_bundle._HELPER_SOURCE.exists(),
            f"helper source missing at {app_bundle._HELPER_SOURCE}",
        )

    def test_source_declares_the_modes_python_relies_on(self):
        src = app_bundle._HELPER_SOURCE.read_text()
        self.assertIn("--check", src)
        self.assertIn("READY", src, "start-up handshake the Python side waits for")


class BuildHelperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name) / "sck-capture"

    def test_returns_false_without_swift(self):
        with patch.object(app_bundle.shutil, "which", return_value=None):
            self.assertFalse(app_bundle.build_helper(self.dest))
        self.assertFalse(self.dest.exists())

    def test_returns_false_when_source_missing(self):
        with patch.object(app_bundle, "_HELPER_SOURCE", Path("/nonexistent.swift")):
            self.assertFalse(app_bundle.build_helper(self.dest))

    @unittest.skipUnless(shutil.which("swiftc"), "needs the Swift toolchain")
    def test_compiles_a_working_helper(self):
        self.assertTrue(app_bundle.build_helper(self.dest))
        self.assertTrue(os.access(self.dest, os.X_OK))
        # --check answers about permission and must never hang or crash.
        result = subprocess.run([str(self.dest), "--check"], timeout=60)
        self.assertIn(result.returncode, (0, 1))


class BundleIncludesHelperTest(unittest.TestCase):
    """A missing or wrongly-signed helper silently costs system audio."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.target = self.root / "trnscrb"
        self.target.write_text("#!/bin/sh\nexit 0\n")
        self.target.chmod(0o755)
        icon_patch = patch("trnscrb.app_icon.build_icns", return_value=False)
        icon_patch.start()
        self.addCleanup(icon_patch.stop)

    @unittest.skipUnless(shutil.which("swiftc"), "needs the Swift toolchain")
    def test_bundle_contains_helper(self):
        bundle = self.root / "Trnscrb.app"
        app_bundle.build_bundle(bundle, str(self.target))
        helper = bundle / "Contents" / "MacOS" / app_bundle.HELPER_NAME
        self.assertTrue(helper.exists(), "helper missing from the built bundle")

    @unittest.skipUnless(
        shutil.which("swiftc") and shutil.which("codesign"), "needs Swift and codesign"
    )
    def test_helper_carries_the_app_identity_and_seal_holds(self):
        """TCC identifies the helper by its signing identifier — if it is not
        the app's, the grant lands on a separate subject and never applies."""
        bundle = self.root / "Trnscrb.app"
        app_bundle.build_bundle(bundle, str(self.target))
        helper = bundle / "Contents" / "MacOS" / app_bundle.HELPER_NAME

        info = subprocess.run(
            ["codesign", "-dv", str(helper)], capture_output=True, text=True, timeout=60
        )
        self.assertIn(f"Identifier={app_bundle.BUNDLE_ID}", info.stderr)

        verify = subprocess.run(
            ["codesign", "--verify", "--strict", str(bundle)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(verify.returncode, 0, f"seal broken: {verify.stderr.strip()}")

    def test_missing_swift_still_produces_a_usable_bundle(self):
        """No Swift toolchain must degrade to mic-only, not fail the install."""
        bundle = self.root / "Trnscrb.app"
        with patch.object(app_bundle, "build_helper", return_value=False):
            executable = app_bundle.build_bundle(bundle, str(self.target))
        self.assertIsNotNone(executable)
        self.assertTrue(executable.exists())
        info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
        self.assertEqual(info["CFBundleIdentifier"], app_bundle.BUNDLE_ID)


class HelperDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_none_when_no_bundle_present(self):
        with (
            patch("trnscrb.app_bundle.bundle_path", lambda: self.root / "Nope.app"),
            patch("shutil.which", return_value=None),
        ):
            self.assertIsNone(sck_helper.helper_path())

    def test_finds_helper_in_installed_bundle(self):
        bundle = self.root / "Trnscrb.app"
        macos = bundle / "Contents" / "MacOS"
        macos.mkdir(parents=True)
        (macos / app_bundle.HELPER_NAME).write_text("#!/bin/sh\nexit 0\n")
        with patch("trnscrb.app_bundle.bundle_path", lambda: bundle):
            self.assertEqual(sck_helper.helper_path(), macos / app_bundle.HELPER_NAME)

    def test_permission_check_is_none_without_helper(self):
        with patch.object(sck_helper, "helper_path", return_value=None):
            self.assertIsNone(sck_helper.has_permission())

    def test_capture_available_reflects_helper_presence(self):
        with patch.object(sck_helper, "helper_path", return_value=None):
            self.assertFalse(sck_helper.HelperCapture.available())
        with patch.object(sck_helper, "helper_path", return_value=Path("/x/sck-capture")):
            self.assertTrue(sck_helper.HelperCapture.available())


if __name__ == "__main__":
    unittest.main()
