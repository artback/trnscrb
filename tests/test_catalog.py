"""Tests for the model catalog: data integrity, hardware recommendation,
download verification, and the `trnscrb models` command."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from trnscrb import catalog
from trnscrb.cli import cli

REQUIRED_FIELDS = {"id", "backend", "source", "approx_size_gb", "sha256", "requirements", "notes"}
KNOWN_BACKENDS = {"parakeet", "qwen3", "whisper", "voxtral", "diarization", "semantic_search"}
TRANSCRIPTION_BACKENDS = {"parakeet", "qwen3", "whisper", "voxtral"}
STATUSES = {catalog.MISSING, catalog.OK, catalog.MISMATCH, catalog.NOT_VERIFIABLE}


class CatalogDataTest(unittest.TestCase):
    """The catalog is declarative data — it must stay internally consistent."""

    def test_ids_are_unique(self):
        ids = [e["id"] for e in catalog.CATALOG]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_present(self):
        for entry in catalog.CATALOG:
            with self.subTest(entry=entry["id"]):
                self.assertTrue(REQUIRED_FIELDS.issubset(entry))

    def test_sizes_parse_and_are_positive(self):
        for entry in catalog.CATALOG:
            with self.subTest(entry=entry["id"]):
                self.assertGreater(float(entry["approx_size_gb"]), 0.0)

    def test_backends_are_known(self):
        for entry in catalog.CATALOG:
            self.assertIn(entry["backend"], KNOWN_BACKENDS)

    def test_sources_are_hf_repo_ids(self):
        for entry in catalog.CATALOG:
            self.assertIn("/", entry["source"])

    def test_a_pinned_checksum_names_its_file(self):
        # A sha256 without a `file` would verify against the first of
        # several artifacts — meaningless for a multi-file repo.
        for entry in catalog.CATALOG:
            if entry["sha256"] is not None:
                self.assertIn("file", entry)
                self.assertTrue(entry["file"])

    def test_whisper_entries_carry_their_size(self):
        for entry in catalog.CATALOG:
            if entry["backend"] == "whisper":
                self.assertTrue(entry.get("model_size"))

    def test_catalog_is_machine_readable(self):
        json.dumps(catalog.CATALOG)


class RecommendTest(unittest.TestCase):
    """recommend() against fake hardware profiles: the pick must fit."""

    def _rec(self, **facts):
        return catalog.recommend(facts)

    def test_apple_silicon_8gb_gets_parakeet(self):
        r = self._rec(apple_silicon=True, ram_gb=8)
        self.assertEqual(r["backend"], "parakeet")
        self.assertEqual(r["model"], "mlx-community/parakeet-tdt-0.6b-v3")

    def test_apple_silicon_16gb_gets_parakeet(self):
        self.assertEqual(self._rec(apple_silicon=True, ram_gb=16)["backend"], "parakeet")

    def test_apple_silicon_32gb_keeps_the_smaller_best_model(self):
        # qwen3-1.7b ties parakeet on quality; the size tie-break keeps the
        # smaller download for the live-transcription loop.
        self.assertEqual(self._rec(apple_silicon=True, ram_gb=32)["backend"], "parakeet")

    def test_intel_8gb_gets_whisper_turbo(self):
        r = self._rec(apple_silicon=False, ram_gb=8)
        self.assertEqual(r["backend"], "whisper")
        self.assertEqual(r["model_size"], "turbo")

    def test_intel_4gb_gets_whisper_small(self):
        r = self._rec(apple_silicon=False, ram_gb=4)
        self.assertEqual(r["backend"], "whisper")
        self.assertEqual(r["model_size"], "small")

    def test_intel_32gb_breaks_the_turbo_large_tie_toward_speed(self):
        r = self._rec(apple_silicon=False, ram_gb=32)
        self.assertEqual(r["model_size"], "turbo")

    def test_a_machine_that_fits_nothing_gets_the_smallest(self):
        r = self._rec(apple_silicon=False, ram_gb=2)
        self.assertEqual(r["model_size"], "tiny")

    def test_missing_facts_keys_are_safe(self):
        # An empty facts dict must not raise: recommend() degrades to its
        # own machine_facts(), which has its own safe defaults.
        r = catalog.recommend({})
        self.assertIn(r["backend"], TRANSCRIPTION_BACKENDS)

    def test_headless_machine_facts_recommend_something(self):
        # machine_facts() on this (headless) host: no Apple, real RAM.
        facts = catalog.machine_facts()
        self.assertFalse(facts["apple_silicon"])
        self.assertGreater(facts["ram_gb"], 0)
        r = catalog.recommend(facts)
        self.assertIn(r["backend"], TRANSCRIPTION_BACKENDS)
        self.assertIn("reason", r)

    def test_configured_entries_follow_the_backend(self):
        ids = [e["id"] for e in catalog.configured_entries({"transcription_backend": "auto"})]
        # The fallback chain: every backend falls back to Whisper, and the
        # auto backend routes non-English meetings to qwen3.
        self.assertIn("parakeet-tdt-0.6b-v3", ids)
        self.assertIn("qwen3-asr-0.6b", ids)
        self.assertIn("whisper-small", ids)

    def test_configured_entries_honor_a_custom_whisper_size(self):
        ids = [e["id"] for e in catalog.configured_entries(
            {"transcription_backend": "whisper", "model_size": "medium"}
        )]
        self.assertEqual(ids, ["whisper-medium"])

    def test_configured_entries_honor_a_custom_parakeet_id(self):
        entries = catalog.configured_entries(
            {"transcription_backend": "parakeet", "parakeet_model_id": "custom/parakeet"}
        )
        self.assertEqual(entries[0]["source"], "custom/parakeet")


class VerifyTest(unittest.TestCase):
    """verify() against fixture files with known hashes."""

    def _fixture(self, root: Path, repo: str, filename: str, content: bytes) -> None:
        snap = root / f"models--{repo.replace('/', '--')}" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / filename).write_bytes(content)

    def test_missing_when_nothing_downloaded(self):
        entry = {"id": "x", "source": "org/repo", "sha256": None}
        with tempfile.TemporaryDirectory() as tmp:
            v = catalog.verify(entry, cache_root=tmp)
        self.assertEqual(v["status"], catalog.MISSING)
        self.assertEqual(v["size_bytes"], 0)

    def test_ok_when_checksum_matches(self):
        content = b"RIFF"
        entry = {
            "id": "x",
            "source": "org/repo",
            "sha256": hashlib.sha256(content).hexdigest(),
            "file": "model.bin",
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._fixture(Path(tmp), "org/repo", "model.bin", content)
            v = catalog.verify(entry, cache_root=tmp)
        self.assertEqual(v["status"], catalog.OK)

    def test_mismatch_when_checksum_differs(self):
        entry = {
            "id": "x",
            "source": "org/repo",
            "sha256": hashlib.sha256(b"RIFF").hexdigest(),
            "file": "model.bin",
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._fixture(Path(tmp), "org/repo", "model.bin", b"RIFFX")
            v = catalog.verify(entry, cache_root=tmp)
        self.assertEqual(v["status"], catalog.MISMATCH)
        self.assertIn("sha256", v["detail"])

    def test_not_verifiable_when_downloaded_without_pinned_hash(self):
        entry = {"id": "x", "source": "org/repo", "sha256": None}
        with tempfile.TemporaryDirectory() as tmp:
            self._fixture(Path(tmp), "org/repo", "model.bin", b"RIFF")
            v = catalog.verify(entry, cache_root=tmp)
        self.assertEqual(v["status"], catalog.NOT_VERIFIABLE)
        self.assertEqual(v["size_bytes"], len(b"RIFF"))

    def test_missing_when_the_pinned_file_is_absent(self):
        # A partial download: the repo is cached but the pinned artifact is
        # not, which must read as missing, not as not-verifiable.
        entry = {
            "id": "x",
            "source": "org/repo",
            "sha256": hashlib.sha256(b"RIFF").hexdigest(),
            "file": "model.bin",
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._fixture(Path(tmp), "org/repo", "other.bin", b"RIFF")
            v = catalog.verify(entry, cache_root=tmp)
        self.assertEqual(v["status"], catalog.MISSING)
        self.assertIn("partial", v["detail"])

    def test_disk_size_sums_the_snapshot(self):
        entry = {"id": "x", "source": "org/repo", "sha256": None}
        with tempfile.TemporaryDirectory() as tmp:
            self._fixture(Path(tmp), "org/repo", "a.bin", b"\x00" * 100)
            self._fixture(Path(tmp), "org/repo", "b.bin", b"\x00" * 4)
            self.assertEqual(catalog.disk_size(entry, cache_root=tmp), 104)

    def test_verify_all_covers_the_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = catalog.verify_all(cache_root=tmp)
        self.assertEqual(len(results), len(catalog.CATALOG))
        for v in results:
            self.assertIn(v["status"], STATUSES)
            self.assertIsInstance(v["size_bytes"], int)


class ModelsCommandTest(unittest.TestCase):
    """`trnscrb models`: the catalog with download state and the pick."""

    def test_lists_the_catalog_and_the_recommendation(self):
        with mock.patch(
            "trnscrb.catalog.machine_facts", return_value={"apple_silicon": True, "ram_gb": 16}
        ):
            result = CliRunner().invoke(cli, ["models"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("parakeet-tdt-0.6b-v3", result.output)
        self.assertIn("whisper-small", result.output)
        self.assertIn("pyannote-community-1", result.output)
        self.assertIn("Recommended:", result.output)
        self.assertIn("parakeet", result.output)

    def test_recommendation_tracks_the_machine(self):
        with mock.patch(
            "trnscrb.catalog.machine_facts", return_value={"apple_silicon": False, "ram_gb": 4}
        ):
            result = CliRunner().invoke(cli, ["models"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Recommended: whisper", result.output)

    def test_checksum_mismatch_is_flagged(self):
        bad = {
            "id": "whisper-small",
            "status": catalog.MISMATCH,
            "size_bytes": 486_000_000,
            "detail": "sha256 deadbeef != pinned cafebabe",
        }
        with mock.patch(
            "trnscrb.catalog.machine_facts", return_value={"apple_silicon": True, "ram_gb": 16}
        ), mock.patch(
            "trnscrb.catalog.verify", side_effect=lambda e, **kw: bad if e["id"] == "whisper-small" else {"id": e["id"], "status": catalog.MISSING, "size_bytes": 0, "detail": ""}
        ):
            result = CliRunner().invoke(cli, ["models"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("CHECKSUM MISMATCH", result.output)


if __name__ == "__main__":
    unittest.main()