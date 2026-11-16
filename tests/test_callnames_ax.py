"""Tests for callnames.ax — headless import + no-op behaviour.

These tests verify that:
- The module imports on headless Linux (no PyObjC needed).
- sample() returns ok=False when AX is unavailable.
- has_permission() returns False headless.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from unittest.mock import patch


class TestAxHeadless(unittest.TestCase):
    """ax.py must import everywhere, even without PyObjC."""

    def test_import_without_pyobjc(self):
        """Import the module by fully qualifying its path."""
        import trnscrb.callnames.ax as ax_mod
        # Even without PyObjC, the module must be importable.
        # AX_AVAILABLE is set to False via the try/except.
        self.assertFalse(ax_mod.AX_AVAILABLE)

    def test_sample_headless(self):
        """sample() on headless → ok=False, never raises."""
        import trnscrb.callnames.ax as ax_mod
        result = ax_mod.sample(bundle_id="us.zoom.xos")
        self.assertFalse(result["ok"])
        self.assertEqual(result["app"], "us.zoom.xos")

    def test_sample_no_bundle(self):
        """sample() with no bundle → ok=False."""
        import trnscrb.callnames.ax as ax_mod
        result = ax_mod.sample()
        self.assertFalse(result["ok"])

    def test_has_permission_headless(self):
        """has_permission() on headless → False."""
        import trnscrb.callnames.ax as ax_mod
        self.assertFalse(ax_mod.has_permission())

    def test_bundle_map(self):
        """Bundle ID mapping for known apps."""
        import trnscrb.callnames.ax as ax_mod
        self.assertEqual(ax_mod._bundle_for_app("Zoom"), "us.zoom.xos")
        self.assertEqual(ax_mod._bundle_for_app("Microsoft Teams"), "com.microsoft.teams2")
        self.assertEqual(ax_mod._bundle_for_app("Slack Huddle"), "com.tinyspeck.slackmacgap")
        self.assertIsNone(ax_mod._bundle_for_app("Unknown"))


class TestPickBundle(unittest.TestCase):
    """pick_bundle: frontmost-first among the running candidates."""

    def _pick(self, frontmost, running, candidates):
        import trnscrb.callnames.ax as ax_mod
        with patch.object(ax_mod, "AX_AVAILABLE", True), \
             patch.object(ax_mod, "_frontmost_bundle", return_value=frontmost), \
             patch.object(ax_mod, "_get_pid",
                          side_effect=lambda b: 1 if b in running else None):
            return ax_mod.pick_bundle(candidates)

    def test_frontmost_wins(self):
        """User's browser is Safari, not the first candidate (Chrome)."""
        candidates = ["com.google.Chrome", "com.apple.Safari"]
        self.assertEqual(
            self._pick("com.apple.Safari",
                       {"com.apple.Safari", "com.google.Chrome"}, candidates),
            "com.apple.Safari",
        )

    def test_first_running_when_frontmost_is_other(self):
        candidates = ["com.google.Chrome", "com.apple.Safari"]
        self.assertEqual(
            self._pick("com.apple.Terminal", {"com.google.Chrome"}, candidates),
            "com.google.Chrome",
        )

    def test_falls_back_to_first_candidate(self):
        candidates = ["com.google.Chrome", "com.apple.Safari"]
        self.assertEqual(
            self._pick("com.apple.Terminal", set(), candidates),
            "com.google.Chrome",
        )

    def test_headless_keeps_input_order(self):
        import trnscrb.callnames.ax as ax_mod
        candidates = ["com.google.Chrome", "com.apple.Safari"]
        with patch.object(ax_mod, "AX_AVAILABLE", False):
            self.assertEqual(ax_mod.pick_bundle(candidates), "com.google.Chrome")

    def test_empty(self):
        import trnscrb.callnames.ax as ax_mod
        self.assertIsNone(ax_mod.pick_bundle([]))


class TestAxImportIsolation(unittest.TestCase):
    """Ensure ax.py imports don't leak to other callnames modules."""

    def test_parse_imports_clean(self):
        """parse.py imports should not require PyObjC."""
        import trnscrb.callnames.parse as parse_mod
        # Must be importable regardless of AX_AVAILABLE.
        self.assertTrue(hasattr(parse_mod, "zoom"))
        self.assertTrue(hasattr(parse_mod, "teams"))
        self.assertTrue(hasattr(parse_mod, "meet"))
        self.assertTrue(hasattr(parse_mod, "webex"))

    def test_attribute_imports_clean(self):
        """attribute.py must import cleanly."""
        import trnscrb.callnames.attribute as attr_mod
        self.assertTrue(hasattr(attr_mod, "name_for"))
        self.assertTrue(hasattr(attr_mod, "attributed_meeting"))
        self.assertTrue(hasattr(attr_mod, "Sample"))


class TestAxWithMockedAX(unittest.TestCase):
    """Tests when AX is available (mocked)."""

    def test_bundle_for_app_known(self):
        """Bundle mapping works regardless of AX_AVAILABLE."""
        import trnscrb.callnames.ax as ax_mod
        self.assertEqual(ax_mod._bundle_for_app("Zoom"), "us.zoom.xos")
        self.assertEqual(ax_mod._bundle_for_app("Microsoft Teams"), "com.microsoft.teams2")


def _importer(mod: str):
    """Get an imported module by name (for patching)."""
    return __import__(mod, fromlist=[""])


if __name__ == "__main__":
    unittest.main()