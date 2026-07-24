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
            app_bundle._APP_SOURCE.exists(),
            f"unified app source missing at {app_bundle._APP_SOURCE}",
        )

    def test_source_declares_the_modes_python_relies_on(self):
        src = app_bundle._APP_SOURCE.read_text()
        self.assertIn("--check", src)
        self.assertIn("--sck-capture", src, "capture mode the recorder spawns")
        self.assertIn("@LAUNCH_TARGET@", src, "launch target placeholder")
        self.assertIn("READY", src, "start-up handshake the Python side waits for")


class BuildMainExecutableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name) / "Trnscrb"

    def test_shell_fallback_without_any_compiler(self):
        with patch.object(app_bundle.shutil, "which", return_value=None):
            kind = app_bundle._build_main_executable("/usr/bin/true", self.dest)
        self.assertEqual(kind, "script")
        self.assertTrue(self.dest.exists())

    @unittest.skipUnless(shutil.which("swiftc"), "needs the Swift toolchain")
    def test_swift_build_produces_capture_capable_binary(self):
        kind = app_bundle._build_main_executable("/usr/bin/true", self.dest)
        self.assertEqual(kind, "swift")
        self.assertTrue(os.access(self.dest, os.X_OK))
        # --check answers about permission and must never hang or launch.
        result = subprocess.run([str(self.dest), "--check"], timeout=60)
        self.assertIn(result.returncode, (0, 1))

    @unittest.skipUnless(shutil.which("swiftc"), "needs the Swift toolchain")
    def test_launch_target_is_baked_in(self):
        app_bundle._build_main_executable("/opt/homebrew/opt/trnscrb/x", self.dest)
        self.assertIn(b"/opt/homebrew/opt/trnscrb/x", self.dest.read_bytes())


class BundleIncludesHelperTest(unittest.TestCase):
    """The single main executable is also the capturer; a missing Swift
    toolchain silently costs system audio but must not break the install."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.target = self.root / "trnscrb"
        self.target.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
        self.target.chmod(0o755)
        icon_patch = patch("trnscrb.app_icon.build_icns", return_value=False)
        icon_patch.start()
        self.addCleanup(icon_patch.stop)

    @unittest.skipUnless(shutil.which("swiftc"), "needs the Swift toolchain")
    def test_bundle_has_single_executable_no_separate_helper(self):
        bundle = self.root / "Trnscrb.app"
        app_bundle.build_bundle(bundle, str(self.target))
        macos = bundle / "Contents" / "MacOS"
        self.assertTrue((macos / "Trnscrb").exists())
        self.assertFalse((macos / "sck-capture").exists(), "no separate helper binary")

    @unittest.skipUnless(
        shutil.which("swiftc") and shutil.which("codesign"), "needs Swift and codesign"
    )
    def test_executable_carries_app_identity_and_seal_holds(self):
        """The one granted cdhash is the one that captures. It must be signed
        with the app identifier and the seal must verify."""
        bundle = self.root / "Trnscrb.app"
        app_bundle.build_bundle(bundle, str(self.target))
        exe = bundle / "Contents" / "MacOS" / "Trnscrb"

        info = subprocess.run(
            ["codesign", "-dv", str(exe)], capture_output=True, text=True, timeout=60
        )
        self.assertIn(f"Identifier={app_bundle.BUNDLE_ID}", info.stderr)

        verify = subprocess.run(
            ["codesign", "--verify", "--strict", str(bundle)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(verify.returncode, 0, f"seal broken: {verify.stderr.strip()}")

    def test_no_toolchain_still_produces_a_usable_bundle(self):
        """No swiftc and no cc must degrade to a launch-only shell stub, not
        fail the install — recording still works, mic-only."""
        bundle = self.root / "Trnscrb.app"
        with patch.object(app_bundle.shutil, "which", return_value=None):
            executable = app_bundle.build_bundle(bundle, str(self.target))
        self.assertIsNotNone(executable)
        self.assertTrue(executable.exists())
        self.assertIn("#!/bin/sh", executable.read_text())
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
        (macos / app_bundle.HELPER_NAME).chmod(0o755)
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
