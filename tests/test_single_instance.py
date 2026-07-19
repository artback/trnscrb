"""Tests for the single-instance flock guard."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb.single_instance import SingleInstance


class SingleInstanceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = patch("trnscrb.single_instance._LOCK_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_acquire_succeeds_when_free(self):
        lock = SingleInstance("t")
        self.assertTrue(lock.acquire())
        lock.release()

    def test_acquire_is_idempotent_for_holder(self):
        lock = SingleInstance("t")
        self.assertTrue(lock.acquire())
        self.assertTrue(lock.acquire())
        lock.release()

    def test_second_instance_is_rejected(self):
        a = SingleInstance("t")
        b = SingleInstance("t")
        self.assertTrue(a.acquire())
        self.assertFalse(b.acquire())
        a.release()

    def test_release_allows_reacquire(self):
        a = SingleInstance("t")
        b = SingleInstance("t")
        self.assertTrue(a.acquire())
        a.release()
        self.assertTrue(b.acquire())
        b.release()

    def test_holder_pid_recorded(self):
        lock = SingleInstance("t")
        lock.acquire()
        self.assertEqual(lock.holder_pid(), os.getpid())
        lock.release()

    def test_holder_pid_none_without_lockfile(self):
        self.assertIsNone(SingleInstance("never-created").holder_pid())

    def test_different_names_are_independent(self):
        a = SingleInstance("one")
        b = SingleInstance("two")
        self.assertTrue(a.acquire())
        self.assertTrue(b.acquire())
        a.release()
        b.release()

    def test_other_process_is_rejected(self):
        lock = SingleInstance("t")
        self.assertTrue(lock.acquire())
        code = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from unittest.mock import patch\n"
                "from pathlib import Path\n"
                "import trnscrb.single_instance as si\n"
                f"si._LOCK_DIR = Path({self._tmp.name!r})\n"
                "sys.exit(0 if not si.SingleInstance('t').acquire() else 1)",
            ],
            cwd=Path(__file__).resolve().parent.parent,
        ).returncode
        self.assertEqual(code, 0, "another process must not acquire a held lock")
        lock.release()


if __name__ == "__main__":
    unittest.main()
