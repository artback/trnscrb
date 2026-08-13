"""Tests for the MCP server surviving an upgrade underneath it.

The server is a stdio child of its client, so it cannot usefully restart
itself — exiting would drop the connection mid-conversation. It can load
what it needs before an upgrade removes it, and say so when one has.
"""

import unittest
from unittest import mock

from trnscrb import mcp_server, rollout


class StaleNoticeTest(unittest.TestCase):
    def test_healthy_install_says_nothing(self):
        with mock.patch.object(rollout, "is_stale", return_value=False):
            self.assertEqual(mcp_server._stale_notice(), "")

    def test_stale_install_names_the_remedy(self):
        with mock.patch.object(rollout, "is_stale", return_value=True):
            notice = mcp_server._stale_notice()
        self.assertIn("upgraded", notice)
        self.assertIn("Restart Claude Desktop", notice)

    def test_status_carries_the_notice_while_recording(self):
        with (
            mock.patch.object(rollout, "is_stale", return_value=True),
            mock.patch.object(mcp_server, "_recorder", mock.Mock(is_recording=True)),
            mock.patch.object(mcp_server, "_recording_started_at", mcp_server.datetime.now()),
        ):
            out = mcp_server.recording_status()
        self.assertIn("Restart Claude Desktop", out)
        self.assertIn("Recording in progress", out)

    def test_status_is_clean_when_healthy(self):
        with (
            mock.patch.object(rollout, "is_stale", return_value=False),
            mock.patch.object(mcp_server, "_recorder", mock.Mock(is_recording=True)),
            mock.patch.object(mcp_server, "_recording_started_at", mcp_server.datetime.now()),
        ):
            out = mcp_server.recording_status()
        self.assertNotIn("upgraded", out)


class PreloadForRecordingTest(unittest.TestCase):
    def test_both_stacks_are_loaded(self):
        with (
            mock.patch.object(mcp_server.settings, "get", return_value="auto"),
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(mcp_server.transcriber, "preload") as t,
            mock.patch.object(mcp_server.diarizer, "preload", return_value=True) as d,
        ):
            mcp_server._preload_for_recording()
        t.assert_called_once()
        d.assert_called_once_with("hf_x")

    def test_diarizer_still_loads_when_transcription_fails(self):
        with (
            mock.patch.object(mcp_server.settings, "get", return_value="auto"),
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(mcp_server.transcriber, "preload", side_effect=RuntimeError("x")),
            mock.patch.object(mcp_server.diarizer, "preload", return_value=True) as d,
        ):
            mcp_server._preload_for_recording()
        d.assert_called_once()

    def test_failures_never_raise(self):
        with (
            mock.patch.object(mcp_server.settings, "get", return_value="auto"),
            mock.patch("trnscrb.settings.read_hf_token", return_value="hf_x"),
            mock.patch.object(mcp_server.transcriber, "preload", side_effect=RuntimeError("x")),
            mock.patch.object(mcp_server.diarizer, "preload", side_effect=RuntimeError("y")),
        ):
            mcp_server._preload_for_recording()


if __name__ == "__main__":
    unittest.main()
