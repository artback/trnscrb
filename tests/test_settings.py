import json
import tempfile
import unittest
from pathlib import Path

from trnscrb import settings


class SettingsTests(unittest.TestCase):
    def test_defaults_include_parakeet_backend_config(self):
        self.assertEqual(settings._DEFAULTS["transcription_backend"], "parakeet")
        self.assertEqual(
            settings._DEFAULTS["parakeet_model_id"],
            "mlx-community/parakeet-tdt-0.6b-v3",
        )

    def test_load_backfills_missing_backend_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_settings = Path(tmpdir) / "settings.json"
            tmp_settings.write_text(json.dumps({"auto_record": False, "model_size": "small"}))
            original = settings._SETTINGS_FILE
            settings._SETTINGS_FILE = tmp_settings
            try:
                loaded = settings.load()
            finally:
                settings._SETTINGS_FILE = original

        self.assertEqual(loaded["auto_record"], False)
        self.assertEqual(loaded["model_size"], "small")
        self.assertEqual(loaded["transcription_backend"], "parakeet")
        self.assertEqual(loaded["parakeet_model_id"], "mlx-community/parakeet-tdt-0.6b-v3")


if __name__ == "__main__":
    unittest.main()
