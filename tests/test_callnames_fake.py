"""Tests for callnames.fake — fake mode loading."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from trnscrb.callnames import attribute as _attr
from trnscrb.callnames import fake as _fake


class TestFakeMode(unittest.TestCase):
    def test_no_fake_mode(self):
        """Without env var → _fake_mode() returns False."""
        os.environ.pop("TRNSCRB_FAKE_CALL_NAMES", None)
        self.assertFalse(_fake._fake_mode())

    def test_fake_mode_env(self):
        """Env var set → True."""
        old = os.environ.get("TRNSCRB_FAKE_CALL_NAMES")
        try:
            os.environ["TRNSCRB_FAKE_CALL_NAMES"] = "1"
            self.assertTrue(_fake._fake_mode())
            os.environ["TRNSCRB_FAKE_CALL_NAMES"] = "true"
            self.assertTrue(_fake._fake_mode())
            os.environ["TRNSCRB_FAKE_CALL_NAMES"] = "false"
            self.assertFalse(_fake._fake_mode())
        finally:
            if old is not None:
                os.environ["TRNSCRB_FAKE_CALL_NAMES"] = old
            else:
                os.environ.pop("TRNSCRB_FAKE_CALL_NAMES", None)

    def test_load_fake_no_env(self):
        """No env var → None."""
        os.environ.pop("TRNSCRB_FAKE_CALL_NAMES", None)
        result = _fake.load_fake_samples()
        self.assertIsNone(result)

    def test_load_fake_builtin(self):
        """TRNSCRB_FAKE_CALL_NAMES=1 → canned fixture samples."""
        old = os.environ.get("TRNSCRB_FAKE_CALL_NAMES")
        try:
            os.environ["TRNSCRB_FAKE_CALL_NAMES"] = "1"
            samples = _fake.load_fake_samples()
            self.assertIsNotNone(samples)
            self.assertGreater(len(samples), 0)
            # Check first sample has the right shape
            s = samples[0]
            self.assertIsInstance(s, _attr.Sample)
            self.assertIn("Ann Lee", s.participants)
        finally:
            if old is not None:
                os.environ["TRNSCRB_FAKE_CALL_NAMES"] = old
            else:
                os.environ.pop("TRNSCRB_FAKE_CALL_NAMES", None)

    def test_load_fake_file(self):
        """Env var = file path → samples loaded from JSON."""
        old = os.environ.get("TRNSCRB_FAKE_CALL_NAMES")
        data = [
            {
                "ts": 1000.0,
                "speaking": ["Ann"],
                "participants": ["Ann", "Bo"],
                "me": "Bo",
                "ok": True,
            },
        ]
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(data, f)
                fpath = f.name
            os.environ["TRNSCRB_FAKE_CALL_NAMES"] = fpath
            samples = _fake.load_fake_samples()
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].ts, 1000.0)
            self.assertEqual(samples[0].speaking, ["Ann"])
            self.assertEqual(samples[0].me, "Bo")
        finally:
            if old is not None:
                os.environ["TRNSCRB_FAKE_CALL_NAMES"] = old
            else:
                os.environ.pop("TRNSCRB_FAKE_CALL_NAMES", None)

    def test_load_fake_bad_file(self):
        """Non-existent file → None (never raises)."""
        old = os.environ.get("TRNSCRB_FAKE_CALL_NAMES")
        try:
            os.environ["TRNSCRB_FAKE_CALL_NAMES"] = "/nonexistent/path.json"
            result = _fake.load_fake_samples()
            self.assertIsNone(result)
        finally:
            if old is not None:
                os.environ["TRNSCRB_FAKE_CALL_NAMES"] = old
            else:
                os.environ.pop("TRNSCRB_FAKE_CALL_NAMES", None)


if __name__ == "__main__":
    unittest.main()