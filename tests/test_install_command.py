"""`trnscrb install` on an already fully-configured machine.

Regression: commit 15a4d43 reworked the PTT section of install() and
dropped the `changed = False` initialization. Every remaining assignment
to `changed` sits inside a conditional, so a re-run of install on a
machine where PTT, auto_record, the backend and the model id are all
stored (e.g. after a version upgrade) skipped them all and the final
`if changed:` raised UnboundLocalError.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from trnscrb import settings as settings_mod
from trnscrb import storage
from trnscrb.cli import cli

FULLY_CONFIGURED = {
    "auto_record": True,
    "transcription_backend": "parakeet",
    "parakeet_model_id": "redparakeet/parakeet-mlx-s",
    "dictation_ptt_key": "ctrl+alt+f8",
}


class InstallFullyConfiguredTest(unittest.TestCase):
    """install must complete cleanly when nothing needs changing."""

    def _invoke_install(self, settings_path: Path):
        tmp = Path(tempfile.mkdtemp(prefix="trnscrb-install-test-"))
        with (
            mock.patch.object(settings_mod, "_SETTINGS_FILE", settings_path),
            # Every dependency that would prompt, install packages, or touch
            # macOS-only machinery — the point is the settings branches only.
            mock.patch("trnscrb.cli._pkg_installed", return_value=True),
            mock.patch("trnscrb.cli._system_audio_ready", return_value=(True, "app")),
            mock.patch("trnscrb.cli._parakeet_model_cached", return_value=True),
            mock.patch("trnscrb.cli._get_hf_token", return_value="hf_dummy"),
            mock.patch("trnscrb.cli._request_mic_permission"),
            mock.patch("trnscrb.cli._request_calendar_permission"),
            mock.patch("trnscrb.cli._login_item_exists", return_value=True),
            mock.patch("trnscrb.cli._login_item_needs_update", return_value=False),
            mock.patch("trnscrb.app_bundle.is_installed", return_value=True),
            mock.patch("shutil.which", return_value=None),
            mock.patch("trnscrb.cli._OPENCODE_CONFIG", tmp / "opencode-missing.json"),
            mock.patch.object(storage, "NOTES_DIR", tmp / "notes"),
        ):
            return CliRunner().invoke(cli, ["install"])

    def test_reinstall_is_clean_when_everything_is_configured(self):
        """A re-run of install (e.g. after `brew upgrade`) must not crash.

        Every settings branch is skipped — PTT key stored, auto_record on,
        backend and model id set — so `changed` must already be False before
        any of them could assign to it.
        """
        tmp = Path(tempfile.mkdtemp(prefix="trnscrb-install-test-"))
        settings_path = tmp / ".config" / "trnscrb" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(FULLY_CONFIGURED), encoding="utf-8")

        result = self._invoke_install(settings_path)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Setup complete!", result.output)
        # Nothing was re-saved: the stored settings come back untouched.
        self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8")), FULLY_CONFIGURED)


if __name__ == "__main__":
    unittest.main()
