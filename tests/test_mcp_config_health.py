"""Tests for detecting and repairing a stale Claude Desktop MCP command path.

A dead command path (e.g. a ~/.local/bin binary left behind after moving to
Homebrew) makes Claude Desktop spawn a missing executable, which surfaces to
the user as the trnscrb MCP server repeatedly disconnecting.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trnscrb import cli


class McpConfigHealthTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = self.root / "claude_desktop_config.json"
        patcher = patch.object(cli, "_CLAUDE_CONFIG", self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, command):
        self.config.write_text(
            json.dumps({"mcpServers": {"trnscrb": {"command": command, "args": ["server"]}}})
        )

    def _real_binary(self):
        binary = self.root / "trnscrb"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        return str(binary)

    def test_healthy_when_command_exists(self):
        self._write(self._real_binary())
        self.assertTrue(cli._mcp_config_healthy())

    def test_unhealthy_when_command_missing(self):
        self._write(str(self.root / "gone" / "trnscrb"))
        self.assertTrue(cli._mcp_configured(), "still configured, just broken")
        self.assertFalse(cli._mcp_config_healthy())

    def test_unhealthy_when_command_not_executable(self):
        plain = self.root / "trnscrb"
        plain.write_text("x")
        plain.chmod(0o644)
        self._write(str(plain))
        self.assertFalse(cli._mcp_config_healthy())

    def test_no_config_is_unhealthy_and_unconfigured(self):
        self.assertFalse(cli._mcp_configured())
        self.assertFalse(cli._mcp_config_healthy())

    def test_command_path_is_read_back(self):
        self._write("/opt/homebrew/bin/trnscrb")
        self.assertEqual(cli._mcp_command_path(), "/opt/homebrew/bin/trnscrb")

    def test_write_config_repairs_a_stale_path(self):
        self._write(str(self.root / "old" / ".local" / "trnscrb"))
        self.assertFalse(cli._mcp_config_healthy())
        good = self._real_binary()
        with patch("shutil.which", return_value=good):
            cli._write_mcp_config()
        self.assertEqual(cli._mcp_command_path(), good)
        self.assertTrue(cli._mcp_config_healthy())


if __name__ == "__main__":
    unittest.main()
