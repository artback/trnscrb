"""Tests for the push ledger, the automatic resync loop, and the remote CLI."""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from trnscrb import cli, push, settings, storage


def _note(name: str, body: str = "meeting text") -> Path:
    path = storage.NOTES_DIR / name
    path.write_text(body, encoding="utf-8")
    return path


class _Resp:
    """Just enough of an httpx response for push_file / fetch_transcript."""

    status_code = 200
    text = "remote body"

    def json(self):
        return {}


class PushLedgerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        patch.object(settings, "_SETTINGS_FILE", tmp / "settings.json").start()
        patch.object(push, "_STATE_FILE", tmp / "push_state.json").start()
        patch.object(storage, "NOTES_DIR", tmp / "meeting-notes").start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self._tmp.cleanup)
        storage.NOTES_DIR.mkdir(parents=True)
        settings.put("remote_url", "http://127.0.0.1:1")
        settings.put("remote_token", "test-token")

    def test_new_file_is_pending(self):
        _note("2026-01-01_10-00_mtg.txt")
        self.assertEqual([f.name for f in push.pending_files()], ["2026-01-01_10-00_mtg.txt"])

    def test_recorded_push_is_no_longer_pending(self):
        p = _note("2026-01-01_10-00_mtg.txt")
        push.record_pushed(p)
        self.assertEqual(push.pending_files(), [])
        self.assertIn(p.name, push.load_state())

    def test_changed_file_becomes_pending_again(self):
        p = _note("2026-01-01_10-00_mtg.txt")
        push.record_pushed(p)
        p.write_text("meeting text plus a new line\n", encoding="utf-8")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
        self.assertEqual([f.name for f in push.pending_files()], [p.name])

    def test_push_file_records_state_on_success(self):
        p = _note("2026-01-01_10-00_mtg.txt")
        with patch.object(push.httpx, "post", return_value=_Resp()):
            self.assertTrue(push.push_file(p))
        self.assertIn(p.name, push.load_state())

    def test_resync_pushes_only_pending(self):
        a = _note("a.txt")
        _note("b.txt")
        push.record_pushed(a)  # a is in sync; b has never been pushed
        calls = []

        def fake_push_file(path, **kw):
            calls.append(path.name)
            return True

        with patch.object(push, "push_file", side_effect=fake_push_file):
            ok, failed = push.resync()
        self.assertEqual((ok, failed), (1, 0))
        self.assertEqual(calls, ["b.txt"])

    def test_resync_unconfigured_is_noop(self):
        _note("a.txt")
        settings.put("remote_url", "")
        self.assertEqual(push.resync(), (0, 0))

    def test_last_push_reports_newest(self):
        a = _note("a.txt")
        b = _note("b.txt")
        push.record_pushed(a)
        time.sleep(0.01)
        push.record_pushed(b)
        self.assertEqual(push.last_push()[0], "b.txt")


class ResyncLoopTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        patch.object(settings, "_SETTINGS_FILE", tmp / "settings.json").start()
        patch.object(push, "_STATE_FILE", tmp / "push_state.json").start()
        patch.object(storage, "NOTES_DIR", tmp / "meeting-notes").start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self._tmp.cleanup)
        storage.NOTES_DIR.mkdir(parents=True)
        settings.put("remote_url", "http://127.0.0.1:1")
        settings.put("remote_token", "test-token")

    def test_loop_runs_repeatedly_until_stopped(self):
        cycles = []

        def fake_resync(progress=None, **kw):
            cycles.append(1)
            return (1, 0)

        stop = threading.Event()
        with patch.object(push, "resync", side_effect=fake_resync):
            t = threading.Thread(
                target=push.resync_loop, args=(stop,), kwargs={"interval_secs": 0.02}
            )
            t.start()
            time.sleep(0.12)
            stop.set()
            t.join(timeout=5)
        self.assertGreaterEqual(len(cycles), 2)

    def test_loop_stays_quiet_when_disabled(self):
        settings.put("remote_resync_interval_minutes", 0)
        stop = threading.Event()
        with patch.object(push, "resync") as fake:
            t = threading.Thread(target=push.resync_loop, args=(stop,))
            t.start()
            time.sleep(0.1)
            stop.set()
            t.join(timeout=5)
        fake.assert_not_called()


class RemoteCliTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        patch.object(settings, "_SETTINGS_FILE", tmp / "settings.json").start()
        patch.object(push, "_STATE_FILE", tmp / "push_state.json").start()
        patch.object(storage, "NOTES_DIR", tmp / "meeting-notes").start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self._tmp.cleanup)
        storage.NOTES_DIR.mkdir(parents=True)
        settings.put("remote_url", "http://127.0.0.1:1")
        settings.put("remote_token", "test-token")

    def test_sync_pushes_pending_by_default(self):
        _note("a.txt")
        _note("b.txt")
        with (
            patch.object(push, "resync", return_value=(2, 0)) as resync,
            patch.object(push, "sync_all") as sync_all,
        ):
            r = self.runner.invoke(cli.cli, ["sync"])
        self.assertEqual(r.exit_code, 0, r.output)
        resync.assert_called_once()
        sync_all.assert_not_called()
        self.assertIn("Pushed 2", r.output)

    def test_sync_all_forces_full_push(self):
        with (
            patch.object(push, "sync_all", return_value=(3, 0)) as sync_all,
            patch.object(push, "resync") as resync,
        ):
            r = self.runner.invoke(cli.cli, ["sync", "--all"])
        self.assertEqual(r.exit_code, 0, r.output)
        sync_all.assert_called_once()
        resync.assert_not_called()

    def test_sync_not_configured_fails(self):
        settings.put("remote_url", "")
        r = self.runner.invoke(cli.cli, ["sync"])
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("remote_url is not set", r.output)

    def test_remote_status_reachable(self):
        _note("a.txt")
        health = {"version": "0.70.0", "transcript_count": 42}
        with (
            patch.object(push, "status", return_value=health),
            patch.object(push, "last_push", return_value=("a.txt", "2026-01-01T00:00:00")),
        ):
            r = self.runner.invoke(cli.cli, ["remote", "status"])
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("reachable", r.output)
        self.assertIn("42 transcript(s) on server", r.output)
        self.assertIn("1 transcript(s) pending", r.output)
        self.assertIn("last push: a.txt", r.output)

    def test_remote_status_unreachable(self):
        with patch.object(push, "status", return_value=None):
            r = self.runner.invoke(cli.cli, ["remote", "status"])
        self.assertIn("unreachable", r.output)

    def test_remote_list_marks_local_files(self):
        _note("a.txt")
        remote = [
            {"id": "a", "name": "a.txt", "modified": "2026-01-02T00:00:00"},
            {"id": "b", "name": "b.txt", "modified": "2026-01-01T00:00:00"},
        ]
        with patch.object(push, "remote_transcripts", return_value=remote):
            r = self.runner.invoke(cli.cli, ["remote", "list"])
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("✓ a", r.output)
        self.assertIn("− b", r.output)
        self.assertIn("2 transcript(s) on the remote server", r.output)

    def test_remote_list_unreachable(self):
        with patch.object(push, "remote_transcripts", return_value=None):
            r = self.runner.invoke(cli.cli, ["remote", "list"])
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("unreachable", r.output)

    def test_remote_pull_saves_locally(self):
        with patch.object(push, "fetch_transcript", return_value="remote text"):
            r = self.runner.invoke(cli.cli, ["remote", "pull", "2026-01-01_10-00_mtg"])
        self.assertEqual(r.exit_code, 0, r.output)
        path = storage.NOTES_DIR / "2026-01-01_10-00_mtg.txt"
        self.assertEqual(path.read_text(encoding="utf-8"), "remote text")

    def test_remote_pull_refuses_to_clobber_different_local(self):
        local = _note("2026-01-01_10-00_mtg.txt", "local text")
        with patch.object(push, "fetch_transcript", return_value="remote text"):
            r = self.runner.invoke(cli.cli, ["remote", "pull", "2026-01-01_10-00_mtg"])
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("different content", r.output)
        self.assertEqual(local.read_text(encoding="utf-8"), "local text")
        with patch.object(push, "fetch_transcript", return_value="remote text"):
            r = self.runner.invoke(cli.cli, ["remote", "pull", "2026-01-01_10-00_mtg", "--force"])
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(local.read_text(encoding="utf-8"), "remote text")

    def test_remote_pull_missing_on_server(self):
        with patch.object(push, "fetch_transcript", return_value=None):
            r = self.runner.invoke(cli.cli, ["remote", "pull", "nope"])
        self.assertNotEqual(r.exit_code, 0)

    def test_fetch_transcript_strips_txt_suffix(self):
        with patch.object(push.httpx, "get", return_value=_Resp()) as g:
            self.assertEqual(push.fetch_transcript("abc.txt"), "remote body")
        g.assert_called_once()
        self.assertIn("/api/transcript/abc", g.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
