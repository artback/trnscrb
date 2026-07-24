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
        # Icon rendering is slow and covered by its own test.
        icon_patch = patch("trnscrb.app_icon.build_icns", return_value=False)
        icon_patch.start()
        self.addCleanup(icon_patch.stop)

        # A fake trnscrb entrypoint the main executable execs directly. It must
        # be a Python-shebang script so _python_script passes it through
        # unchanged (a shell shebang is treated as a brew wrapper and swapped).
        self.target = self.root / "trnscrb"
        self.record = self.root / "invoked.txt"
        self.target.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            f"open({str(self.record)!r}, 'w').write("
            "sys.argv[1] + ' bundle=' + os.environ.get('TRNSCRB_IN_BUNDLE', ''))\n"
            "sys.exit(7)\n"
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

    @unittest.skipUnless(shutil.which("swiftc") or shutil.which("cc"), "needs swiftc or cc")
    def test_main_executable_launches_target_and_propagates_exit_code(self):
        executable = app_bundle.ensure_bundle(str(self.target))
        result = subprocess.run([str(executable)], timeout=90)
        self.assertEqual(result.returncode, 7, "child exit code must propagate")
        self.assertEqual(self.record.read_text().strip(), "start bundle=1")

    @unittest.skipUnless(shutil.which("swiftc"), "needs swiftc")
    def test_main_executable_is_single_binary_with_capture_flag(self):
        """The one binary handles both launching and --sck-capture, so the
        granted cdhash is the capturing cdhash. --check must not launch Python."""
        executable = app_bundle.ensure_bundle(str(self.target))
        self.assertFalse((executable.parent / "sck-capture").exists(), "no separate helper binary")
        result = subprocess.run([str(executable), "--check"], timeout=30)
        self.assertIn(result.returncode, (0, 1), "--check exits 0/1, does not launch")
        self.assertFalse(self.record.exists(), "--check must not run the launch target")

    @unittest.skipUnless(shutil.which("codesign"), "needs codesign")
    def test_bundle_signature_seal_is_valid(self):
        """An invalid seal stops macOS persisting the Screen Recording grant.

        launcher.txt in Resources/ must be signed, not added afterward.
        """
        app_bundle.ensure_bundle(str(self.target))
        bundle = app_bundle.bundle_path()
        verify = subprocess.run(
            ["codesign", "--verify", "--strict", str(bundle)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(verify.returncode, 0, f"sealed resource invalid: {verify.stderr.strip()}")

    def test_script_fallback_without_any_compiler(self):
        """No swiftc and no cc → launch-only shell stub; the app still runs
        (mic-only), it just can't capture system audio."""
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


class PackagedBundleTest(unittest.TestCase):
    """When the package ships a prebuilt bundle (Homebrew), it is copied, not rebuilt."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = patch.object(app_bundle, "bundle_path", lambda: self.root / "Trnscrb.app")
        patcher.start()
        self.addCleanup(patcher.stop)

        # Simulated Homebrew prefix: prefix/bin/trnscrb + prefix/Trnscrb.app
        self.prefix = self.root / "Cellar" / "trnscrb" / "0.10.0"
        bin_dir = self.prefix / "bin"
        bin_dir.mkdir(parents=True)
        self.binary = bin_dir / "trnscrb"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        self._make_packaged(version="0.10.0")

    def _make_packaged(self, version, launcher_body="#!/bin/sh\nexit 0\n", marker="2\n/opt/x\n"):
        packaged = self.prefix / "Trnscrb.app"
        macos_dir = packaged / "Contents" / "MacOS"
        resources = packaged / "Contents" / "Resources"
        macos_dir.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)
        launcher = macos_dir / "Trnscrb"
        launcher.write_text(launcher_body)
        launcher.chmod(0o755)
        (resources / "launcher.txt").write_text(marker)
        with open(packaged / "Contents" / "Info.plist", "wb") as f:
            plistlib.dump(
                {"CFBundleIdentifier": app_bundle.BUNDLE_ID, "CFBundleVersion": version}, f
            )
        return packaged

    def test_packaged_bundle_is_discovered(self):
        found = app_bundle._packaged_bundle(self.binary)
        self.assertEqual(found, (self.prefix / "Trnscrb.app").resolve())

    def test_ensure_bundle_copies_packaged(self):
        executable = app_bundle.ensure_bundle(str(self.binary))
        self.assertEqual(executable, app_bundle.executable_path())
        self.assertTrue(executable.exists())
        installed_plist = app_bundle.bundle_path() / "Contents" / "Info.plist"
        self.assertEqual(plistlib.loads(installed_plist.read_bytes())["CFBundleVersion"], "0.10.0")

    def test_same_marker_is_not_recopied(self):
        first = app_bundle.ensure_bundle(str(self.binary))
        mtime = first.stat().st_mtime_ns
        second = app_bundle.ensure_bundle(str(self.binary))
        self.assertEqual(second.stat().st_mtime_ns, mtime)

    def test_version_bump_with_same_marker_keeps_installed_bundle(self):
        """Replacing the bundle invalidates the TCC grant — a pure version
        bump must leave the installed bundle (and its signature) untouched."""
        first = app_bundle.ensure_bundle(str(self.binary))
        mtime = first.stat().st_mtime_ns
        self._make_packaged(version="0.11.0")  # same identity marker
        second = app_bundle.ensure_bundle(str(self.binary))
        self.assertEqual(second.stat().st_mtime_ns, mtime, "must not re-copy")
        installed_plist = app_bundle.bundle_path() / "Contents" / "Info.plist"
        self.assertEqual(
            plistlib.loads(installed_plist.read_bytes())["CFBundleVersion"],
            "0.10.0",
            "old Info.plist kept on purpose — cosmetic staleness beats losing the grant",
        )

    def test_changed_marker_replaces_installed(self):
        app_bundle.ensure_bundle(str(self.binary))
        self._make_packaged(
            version="0.11.0", launcher_body="#!/bin/sh\nexit 1\n", marker="3\n/opt/x\n"
        )
        executable = app_bundle.ensure_bundle(str(self.binary))
        self.assertEqual(executable.read_text(), "#!/bin/sh\nexit 1\n")
        installed_plist = app_bundle.bundle_path() / "Contents" / "Info.plist"
        self.assertEqual(plistlib.loads(installed_plist.read_bytes())["CFBundleVersion"], "0.11.0")

    def test_is_installed_tracks_bundle_identity(self):
        self.assertFalse(app_bundle.is_installed(str(self.binary)))
        app_bundle.ensure_bundle(str(self.binary))
        self.assertTrue(app_bundle.is_installed(str(self.binary)))
        self._make_packaged(version="0.11.0")  # same marker — still installed
        self.assertTrue(app_bundle.is_installed(str(self.binary)))
        self._make_packaged(version="0.12.0", marker="3\n/opt/x\n")
        self.assertFalse(app_bundle.is_installed(str(self.binary)))


class AppIconTest(unittest.TestCase):
    def _pillow_and_iconutil(self):
        if not shutil.which("iconutil"):
            return False
        try:
            import PIL  # noqa: F401

            return True
        except ImportError:
            return False

    def test_build_icns_produces_valid_icon(self):
        if not self._pillow_and_iconutil():
            self.skipTest("needs Pillow and iconutil")
        from trnscrb import app_icon

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "Trnscrb.icns"
            with patch.object(app_icon, "_MASTER", 1024):  # keep the test fast
                self.assertTrue(app_icon.build_icns(dest))
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes()[:4], b"icns", "must be a valid icns file")

    def test_bundle_gets_icon_and_plist_entry(self):
        if not self._pillow_and_iconutil():
            self.skipTest("needs Pillow and iconutil")
        from trnscrb import app_icon

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "trnscrb"
            target.write_text("#!/bin/sh\nexit 0\n")
            target.chmod(0o755)
            with patch.object(app_icon, "_MASTER", 1024):
                executable = app_bundle.build_bundle(root / "Trnscrb.app", str(target))
            self.assertIsNotNone(executable)
            bundle = root / "Trnscrb.app"
            self.assertTrue((bundle / "Contents" / "Resources" / "Trnscrb.icns").exists())
            info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
            self.assertEqual(info["CFBundleIconFile"], "Trnscrb")
            self.assertEqual(info["LSApplicationCategoryType"], "public.app-category.productivity")


if __name__ == "__main__":
    unittest.main()
