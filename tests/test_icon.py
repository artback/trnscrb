"""Tests for atomic icon generation (no torn reads → no ImageIO crash-loop)."""

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PIL import Image  # noqa: F401

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from trnscrb import icon


@unittest.skipUnless(_HAS_PIL, "Pillow not installed")
class IconAtomicWriteTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_writes_a_valid_complete_png(self):
        from PIL import Image

        out = self.dir / "mic.png"
        icon._make_mic(out, fill=(0, 0, 0, 255))
        self.assertTrue(out.exists())
        with Image.open(out) as img:
            img.verify()  # raises if the file is truncated/torn

    def test_leaves_no_temp_files_behind(self):
        icon._make_mic(self.dir / "mic.png", fill=(0, 0, 0, 255))
        # Only the final PNG should remain — no mkstemp leftovers.
        self.assertEqual([p.name for p in self.dir.iterdir()], ["mic.png"])

    def test_cleans_up_temp_on_failure(self):
        target = self.dir / "mic.png"
        with patch("trnscrb.icon.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                icon._make_mic(target, fill=(0, 0, 0, 255))
        # The temp file must not linger after a failed write.
        self.assertEqual(list(self.dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
