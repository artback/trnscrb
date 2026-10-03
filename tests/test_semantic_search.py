"""Tests for offline semantic search — chunking, incremental index, ranking.

The embedding model is mocked so these run without sentence-transformers.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from trnscrb import semantic_search as sem

_VOCAB = ["cloud", "budget", "kubernetes"]


def _fake_embed(texts):
    """A tiny keyword-presence embedding so ranking is deterministic."""
    out = []
    for t in texts:
        low = t.lower()
        v = np.array([1.0 if w in low else 0.0 for w in _VOCAB], dtype="float32")
        norm = np.linalg.norm(v)
        out.append(v / norm if norm else v)
    return np.array(out, dtype="float32")


def _transcript(topic: str) -> str:
    return (
        "Meeting: Sync\n"
        "Date:    2026-07-30 11:00\n"
        "============================================================\n"
        "[Them]\n"
        f"  00:11  Let us talk about the {topic} plan for the {topic} rollout\n"
    )


class ChunkingTest(unittest.TestCase):
    def test_chunks_capture_timestamp_and_text(self):
        body = _transcript("cloud")
        chunks = list(sem._chunks(body))
        self.assertEqual(len(chunks), 1)
        ts, text = chunks[0]
        self.assertEqual(ts, "00:11")
        self.assertIn("cloud", text)
        self.assertNotIn("Meeting:", text)  # header lines are skipped

    def test_long_body_splits_into_multiple_passages(self):
        lines = ["Meeting: X", "Date:    2026-07-30 11:00", "=" * 20]
        for i in range(40):
            lines.append(f"  00:{i:02d}  word one two three four five")
        chunks = list(sem._chunks("\n".join(lines)))
        self.assertGreater(len(chunks), 1)

    def test_date_from_header(self):
        self.assertEqual(sem._date_from(_transcript("cloud"), "x"), "2026-07-30 11:00")

    def test_date_from_stem_fallback(self):
        self.assertEqual(sem._date_from("no date line", "2026-07-30_11-00_Meet"), "2026-07-30")


class IndexAndSearchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name) / "notes"
        self.notes.mkdir()
        self.cfg = Path(self._tmp.name) / "cfg"
        self.cfg.mkdir()

        # Redirect index files and the notes dir into the temp tree.
        patch.object(sem, "_INDEX_DIR", self.cfg).start()
        patch.object(sem, "_VECTORS_FILE", self.cfg / "v.npy").start()
        patch.object(sem, "_META_FILE", self.cfg / "m.json").start()
        self._embed = patch.object(sem, "_embed", side_effect=_fake_embed).start()
        patch("trnscrb.storage.NOTES_DIR", self.notes).start()
        self.addCleanup(patch.stopall)

    def _write(self, name, topic):
        (self.notes / name).write_text(_transcript(topic), encoding="utf-8")

    def test_builds_and_ranks_by_meaning(self):
        self._write("2026-07-30_10-00_a.txt", "cloud")
        self._write("2026-07-30_11-00_b.txt", "budget")
        n = sem.build_index()
        self.assertEqual(n, 2)  # one passage each

        hits = sem.search("cloud strategy")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["transcript_id"], "2026-07-30_10-00_a")
        self.assertIn("timestamp", hits[0])
        self.assertGreater(hits[0]["score"], hits[-1]["score"] if len(hits) > 1 else -1)

    def test_incremental_only_reembeds_changed(self):
        self._write("2026-07-30_10-00_a.txt", "cloud")
        self._write("2026-07-30_11-00_b.txt", "budget")
        sem.build_index()
        self._embed.reset_mock()

        # Nothing changed → no embedding work.
        sem.build_index()
        self._embed.assert_not_called()

        # Add one transcript → only its passages are embedded.
        self._write("2026-07-30_12-00_c.txt", "kubernetes")
        sem.build_index()
        self._embed.assert_called_once()
        embedded_texts = self._embed.call_args[0][0]
        self.assertTrue(all("kubernetes" in t.lower() for t in embedded_texts))

    def test_removed_transcript_drops_from_index(self):
        self._write("2026-07-30_10-00_a.txt", "cloud")
        self._write("2026-07-30_11-00_b.txt", "budget")
        sem.build_index()
        (self.notes / "2026-07-30_11-00_b.txt").unlink()
        n = sem.build_index()
        self.assertEqual(n, 1)
        ids = {h["transcript_id"] for h in sem.search("budget")}
        self.assertNotIn("2026-07-30_11-00_b", ids)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(sem.search("   "), [])


class WarmupTest(unittest.TestCase):
    """Tests for the model warm/unwarm lifecycle (preload, unload, is_warmed)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name) / "notes"
        self.notes.mkdir()
        self.cfg = Path(self._tmp.name) / "cfg"
        self.cfg.mkdir()

        patch.object(sem, "_INDEX_DIR", self.cfg).start()
        patch.object(sem, "_VECTORS_FILE", self.cfg / "v.npy").start()
        patch.object(sem, "_META_FILE", self.cfg / "m.json").start()
        patch("trnscrb.storage.NOTES_DIR", self.notes).start()
        self._embed = patch.object(sem, "_embed", side_effect=_fake_embed).start()
        # _get_model sets module-level _model; side_effect must mirror that
        # behavior (assigning to sem._model) so the mock is realistic.
        self._mock_get = patch.object(sem, "_get_model").start()
        self.addCleanup(patch.stopall)

    def test_is_warmed_returns_false_before_preload(self):
        sem._model = None
        self.assertFalse(sem.is_warmed())

    def test_is_warmed_returns_true_after_preload(self):
        instance = object()
        self._mock_get.side_effect = lambda: setattr(sem, "_model", instance) or instance
        self.assertTrue(sem.preload())
        self.assertTrue(sem.is_warmed())
        self.assertIs(sem._model, instance)

    def test_preload_is_idempotent(self):
        instance = object()
        self._mock_get.side_effect = lambda: setattr(sem, "_model", instance) or instance
        sem.preload()
        first = sem._model
        sem.preload()
        sem.preload()
        # model stays the same after warm (singleton)
        self.assertTrue(sem.is_warmed())
        self.assertIs(sem._model, first)

    def test_unload_clears_model(self):
        instance = object()
        self._mock_get.side_effect = lambda: setattr(sem, "_model", instance) or instance
        sem.preload()
        self.assertTrue(sem.is_warmed())
        sem.unload()
        self.assertFalse(sem.is_warmed())
        self.assertIsNone(sem._model)

    def test_unload_is_safe_when_not_loaded(self):
        sem._model = None
        # Should not raise
        sem.unload()
        self.assertIsNone(sem._model)

    def test_reload_after_unload_works(self):
        instance = object()
        self._mock_get.side_effect = lambda: setattr(sem, "_model", instance) or instance
        sem.preload()
        self.assertTrue(sem.is_warmed())
        sem.unload()
        self.assertFalse(sem.is_warmed())
        # Subsequent preload re-loads
        sem.preload()
        self.assertTrue(sem.is_warmed())
        self.assertIs(sem._model, instance)

    def test_preload_false_when_model_unavailable(self):
        self._mock_get.side_effect = ImportError("No module named 'sentence_transformers'")
        result = sem.preload()
        self.assertFalse(result)

    def test_preload_false_on_runtime_error(self):
        self._mock_get.side_effect = RuntimeError("Failed to load model")
        result = sem.preload()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
