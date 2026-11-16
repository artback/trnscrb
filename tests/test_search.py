"""Tests for hybrid search — RRF fusion, rankers, filters, citations, degradation.

The embedding backend is mocked throughout; nothing here loads a model or
touches the real notes dir (the fixture redirects storage.NOTES_DIR).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trnscrb import mcp_server, search, storage
from trnscrb import semantic_search as sem


# Transcript shape: storage.format_transcript header, then [Speaker] lines
# over timestamped lines. With one turn, the spoken line is line 6.
def _body(date="2026-07-30 10:00", turns=()):
    lines = ["Meeting: Sync", f"Date:    {date}", "=" * 60, ""]
    for speaker, ts, text in turns:
        lines.append(f"[{speaker}]")
        lines.append(f"  {ts}  {text}")
    return "\n".join(lines) + "\n"


_SYNC_NAME = "2026-07-30_10-00_sync.txt"
_SYNC_TURN = ("Ann", "00:05", "we will ship the hybrid search today")


def _record(stem, text, ts="00:05"):
    """A semantic_search hit record, as the real one is shaped."""
    return {
        "transcript_id": stem,
        "when": "2026-07-30 10:00",
        "timestamp": ts,
        "text": text,
        "score": 0.9,
    }


class _NotesFixture:
    """Redirects storage.NOTES_DIR into a temp tree per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.notes = Path(self._tmp.name) / "notes"
        self.notes.mkdir()
        self._patch = mock.patch.object(storage, "NOTES_DIR", self.notes)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _write(self, rel, date="2026-07-30 10:00", turns=()):
        path = self.notes / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_body(date, turns), encoding="utf-8")
        return path

    def _write_raw(self, name, text):
        path = self.notes / name
        path.write_text(text, encoding="utf-8")
        return path

    def _passage(self, rel):
        """The vector index's passage for a file, re-derived the same way."""
        body = (self.notes / rel).read_text(encoding="utf-8")
        passages = [t for _, _, _, t in search._chunks_with_lines(body)]
        self.assertTrue(passages)
        return passages[0]


class RRFMathTest(unittest.TestCase):
    def test_hand_computed_scores(self):
        fused = search.rrf([["a", "b", "c"], ["b", "c", "a"]])
        self.assertAlmostEqual(fused["a"][0], 1 / 61 + 1 / 63)
        self.assertAlmostEqual(fused["b"][0], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused["c"][0], 1 / 63 + 1 / 62)
        # all three appear in both rankers' lists
        for item in ("a", "b", "c"):
            self.assertEqual(fused[item][1], [0, 1])

    def test_ranker_positions_recorded_per_list(self):
        fused = search.rrf([["a"], ["b"]])
        self.assertEqual(fused["a"][1], [0])
        self.assertEqual(fused["b"][1], [1])

    def test_ranking_order(self):
        fused = search.rrf([["a", "b", "c"], ["b", "c", "a"]])
        order = [k for k, _ in sorted(fused.items(), key=lambda kv: -kv[1][0])]
        self.assertEqual(order, ["b", "a", "c"])

    def test_single_list_keeps_its_order(self):
        fused = search.rrf([["x", "y"]])
        self.assertGreater(fused["x"][0], fused["y"][0])
        self.assertEqual(fused["x"][1], [0])

    def test_k_is_a_parameter(self):
        fused = search.rrf([["a"]], k=0)
        self.assertAlmostEqual(fused["a"][0], 1.0)


class FuseTest(unittest.TestCase):
    def test_overlapping_hits_merge_to_both(self):
        kw = [search.Hit(name="a.txt", line_start=5, line_end=6, text="kw context")]
        vec = [
            search.Hit(
                name="a.txt",
                line_start=6,
                line_end=14,
                text="passage",
                source="vector",
                timestamp="00:31",
            )
        ]
        fused = search.fuse(kw, vec)
        self.assertEqual(len(fused), 1)
        hit = fused[0]
        self.assertEqual(hit.source, "both")
        self.assertAlmostEqual(hit.score, 2 / 61)
        self.assertEqual(hit.text, "kw context")  # the region keeps the display
        self.assertEqual(hit.timestamp, "00:31")  # inherited from the passage
        self.assertNotEqual(kw[0].timestamp, "00:31")  # inputs are not mutated

    def test_disjoint_hits_stay_separate(self):
        kw = [search.Hit(name="a.txt", line_start=5, line_end=6, text="kw")]
        vec = [search.Hit(name="a.txt", line_start=30, line_end=40, text="vec", source="vector")]
        fused = search.fuse(kw, vec)
        self.assertEqual(len(fused), 2)
        self.assertEqual({h.source for h in fused}, {"keyword", "vector"})
        self.assertAlmostEqual(fused[0].score, 1 / 61)
        self.assertEqual(fused[0].source, "keyword")  # tie broken by line position

    def test_vector_hit_without_lines_survives(self):
        vec = [search.Hit(name="a.txt", text="passage", source="vector", timestamp="00:31")]
        fused = search.fuse([], vec)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].source, "vector")
        self.assertEqual(fused[0].citation, "a.txt @ 00:31")

    def test_different_files_never_merge(self):
        kw = [search.Hit(name="a.txt", line_start=5, line_end=6, text="kw")]
        vec = [search.Hit(name="b.txt", line_start=5, line_end=6, text="vec", source="vector")]
        fused = search.fuse(kw, vec)
        self.assertEqual(len(fused), 2)
        self.assertEqual({h.source for h in fused}, {"keyword", "vector"})


class CitationTest(unittest.TestCase):
    def test_range_citation(self):
        hit = search.Hit(name="2026-11-12_14-30_standup.txt", line_start=142, line_end=148)
        self.assertEqual(hit.citation, "2026-11-12_14-30_standup.txt:142-148")

    def test_single_line_citation(self):
        hit = search.Hit(name="a.txt", line_start=142, line_end=142)
        self.assertEqual(hit.citation, "a.txt:142")

    def test_timestamp_appended(self):
        hit = search.Hit(name="a.txt", line_start=142, line_end=148, timestamp="23:05")
        self.assertEqual(hit.citation, "a.txt:142-148 @ 23:05")

    def test_unlocated_vector_hit_cites_timestamp_only(self):
        hit = search.Hit(name="a.txt", source="vector", timestamp="23:05")
        self.assertEqual(hit.citation, "a.txt @ 23:05")

    def test_nested_name_is_the_relative_path(self):
        hit = search.Hit(name="work/2026/a.txt", line_start=5, line_end=7)
        self.assertEqual(hit.citation, "work/2026/a.txt:5-7")


class ChunkParityTest(unittest.TestCase):
    def test_passages_match_the_semantic_search_chunker(self):
        # Invariant: a passage re-derived from a file must equal the one stored
        # in the index, or vector line citations silently go stale.
        body = _body(turns=[("Ann", "00:01", " ".join(f"word{i}" for i in range(100)))])
        self.assertEqual(
            [t for _, t in sem._chunks(body)],
            [t for _, _, _, t in search._chunks_with_lines(body)],
        )

    def test_line_ranges_carry_the_chunk(self):
        body = _body(turns=[("Ann", "00:01", " ".join(f"word{i}" for i in range(100)))])
        chunks = list(search._chunks_with_lines(body))
        _ts, first, last, text = chunks[0]
        # one line over the 80-word threshold is one whole-line chunk
        self.assertEqual(first, 6)
        self.assertEqual(last, 6)
        self.assertEqual(len(text.split()), 100)
        self.assertEqual(len(chunks), 1)


class KeywordRankTest(_NotesFixture, unittest.TestCase):
    def test_ranks_files_by_match_count_and_carries_provenance(self):
        # Two matches in a (regions 5-7 and 10-12), one in b.
        self._write_raw(
            "2026-01-02_10-00_a.txt",
            _body(
                "2026-01-02 10:00",
                [
                    ("Ann", "00:01", "we will release the new thing"),
                    ("Ann", "00:02", "chatter in between"),
                    ("Ann", "00:03", "chatter again to widen the gap"),
                    ("Ann", "00:04", "another release later on"),
                ],
            ),
        )
        self._write_raw(
            "2026-01-03_10-00_b.txt",
            _body("2026-01-03 10:00", [("Bob", "00:01", "one release mention")]),
        )
        hits = search.keyword_rank("release", search.transcript_files())
        self.assertEqual([h.name for h in hits][:2], ["2026-01-02_10-00_a.txt"] * 2)
        first = hits[0]
        self.assertEqual((first.line_start, first.line_end), (5, 7))
        self.assertIn("[Ann]", first.text)  # context lines are kept
        self.assertEqual(first.timestamp, "00:01")
        self.assertEqual(first.when, "2026-01-02")
        self.assertEqual(hits[2].name, "2026-01-03_10-00_b.txt")

    def test_case_insensitive(self):
        self._write_raw("a.txt", _body(turns=[("Ann", "00:01", "RELEASE the plan")]))
        hits = search.keyword_rank("release", search.transcript_files())
        self.assertEqual(len(hits), 1)

    def test_no_match_or_empty_query(self):
        self._write_raw("a.txt", _body(turns=[("Ann", "00:01", "nothing here")]))
        self.assertEqual(search.keyword_rank("zzz", search.transcript_files()), [])
        self.assertEqual(search.keyword_rank("  ", search.transcript_files()), [])


class HybridTest(_NotesFixture, unittest.TestCase):
    def test_fuses_keyword_and_vector(self):
        self._write(_SYNC_NAME, turns=[_SYNC_TURN])
        passage = self._passage(_SYNC_NAME)
        with (
            mock.patch.object(sem, "available", return_value=True),
            mock.patch.object(
                sem, "search", return_value=[_record("2026-07-30_10-00_sync", passage)]
            ),
        ):
            result = search.hybrid("hybrid search")
        self.assertTrue(result.vector_available)
        self.assertTrue(result.vector_used)
        top = result.results[0]
        self.assertEqual(top.source, "both")
        self.assertAlmostEqual(top.score, 2 / 61)
        self.assertEqual(top.citation, "2026-07-30_10-00_sync.txt:5-6 @ 00:05")

    def test_degrades_when_backend_missing(self):
        self._write(_SYNC_NAME, turns=[_SYNC_TURN])
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("hybrid")
        self.assertFalse(result.vector_available)
        self.assertFalse(result.vector_used)
        self.assertTrue(result.results)
        self.assertEqual(result.results[0].source, "keyword")

    def test_degrades_when_lookup_fails(self):
        self._write(_SYNC_NAME, turns=[_SYNC_TURN])
        with (
            mock.patch.object(sem, "available", return_value=True),
            mock.patch.object(sem, "search", side_effect=RuntimeError("no torch")),
        ):
            result = search.hybrid("hybrid")
        self.assertFalse(result.vector_available)
        self.assertEqual(result.results[0].source, "keyword")

    def test_vector_hits_are_scoped_to_the_filtered_files(self):
        self._write(
            "2026-01-15_10-00_old.txt",
            date="2026-01-15 10:00",
            turns=[("Ann", "00:01", "budget talk")],
        )
        self._write(
            "2026-06-01_10-00_new.txt",
            date="2026-06-01 10:00",
            turns=[("Ann", "00:01", "budget talk")],
        )
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget", since="2026-03-01")
        self.assertEqual([h.name for h in result.results], ["2026-06-01_10-00_new.txt"])
        self.assertEqual(result.files_searched, 1)

    def test_empty_query_yields_nothing(self):
        self._write("a.txt")
        result = search.hybrid("   ")
        self.assertEqual(result.results, [])


class FiltersTest(_NotesFixture, unittest.TestCase):
    def _two_dated(self):
        self._write(
            "2026-01-15_10-00_old.txt",
            date="2026-01-15 10:00",
            turns=[("Ann", "00:01", "budget talk")],
        )
        self._write(
            "2026-06-01_10-00_new.txt",
            date="2026-06-01 10:00",
            turns=[("Ann", "00:01", "budget talk")],
        )

    def test_since_keeps_newer(self):
        self._two_dated()
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget", since="2026-03-01")
        self.assertEqual([h.name for h in result.results], ["2026-06-01_10-00_new.txt"])

    def test_until_keeps_older(self):
        self._two_dated()
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget", until="2026-03-01")
        self.assertEqual([h.name for h in result.results], ["2026-01-15_10-00_old.txt"])

    def test_boundaries_are_inclusive(self):
        self._two_dated()
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget", since="2026-01-15", until="2026-06-01")
        self.assertEqual(len(result.results), 2)

    def test_undated_transcript_is_kept(self):
        self._write_raw("scratch.txt", "budget mention without a date\n")
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget", since="2026-01-01")
        self.assertEqual([h.name for h in result.results], ["scratch.txt"])

    def test_speaker_filter(self):
        self._write_raw(
            "2026-07-30_10-00_sync.txt",
            _body(
                turns=[
                    ("Ann", "00:01", "the budget is fine"),
                    ("Bob", "00:02", "something unrelated happens"),
                    ("Bob", "00:03", "and more unrelated chatter"),
                    ("Bob", "00:04", "the budget is not fine"),
                ]
            ),
        )
        with mock.patch.object(sem, "available", return_value=False):
            ann = search.hybrid("budget", speaker="Ann")
            bob = search.hybrid("budget", speaker="bob")
            sub = search.hybrid("budget", speaker="bo")
            none = search.hybrid("budget", speaker="carol")
        self.assertEqual([h.line_start for h in ann.results], [5])
        self.assertEqual([h.line_start for h in bob.results], [11])
        self.assertEqual(bob.results[0].citation, "2026-07-30_10-00_sync.txt:11-12 @ 00:04")
        self.assertEqual([h.line_start for h in sub.results], [11])
        self.assertEqual(none.results, [])

    def test_path_scopes_to_a_subfolder(self):
        self._write("2026-07-30_10-00_top.txt", turns=[("Ann", "00:01", "budget top")])
        self._write("work/2026-07-30_10-00_nested.txt", turns=[("Ann", "00:01", "budget nested")])
        with mock.patch.object(sem, "available", return_value=False):
            scoped = search.hybrid("budget", path="work")
            all_files = search.hybrid("budget")
            missing = search.hybrid("budget", path="nowhere")
            slashed = search.hybrid("budget", path="work/")
        self.assertEqual([h.name for h in scoped.results], ["work/2026-07-30_10-00_nested.txt"])
        self.assertEqual(len(scoped.results), len(slashed.results))
        self.assertEqual(missing.results, [])
        self.assertEqual(missing.files_searched, 0)
        self.assertEqual(len(all_files.results), 2)

    def test_subfolders_are_walked_without_a_path_filter(self):
        self._write(
            "work/2026/2026-07-30_10-00_nested.txt", turns=[("Ann", "00:01", "budget deep")]
        )
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget")
        self.assertEqual(result.results[0].name, "work/2026/2026-07-30_10-00_nested.txt")
        self.assertEqual(
            result.results[0].citation, "work/2026/2026-07-30_10-00_nested.txt:5-6 @ 00:01"
        )

    def test_summary_files_are_skipped(self):
        self._write_raw("weekly-2026-W31.txt", "budget summary\n")
        with mock.patch.object(sem, "available", return_value=False):
            result = search.hybrid("budget")
        self.assertEqual(result.results, [])
        self.assertEqual(result.files_searched, 0)

    def test_invalid_date_raises(self):
        with self.assertRaises(ValueError):
            search.hybrid("x", since="last week")


class McpSearchTest(_NotesFixture, unittest.TestCase):
    def _sync_file(self):
        self._write(_SYNC_NAME, turns=[_SYNC_TURN])

    def test_shape_citation_and_vector_note(self):
        self._sync_file()
        with mock.patch.object(sem, "available", return_value=False):
            out = mcp_server.search_transcripts("hybrid")
        self.assertTrue(out.startswith("note: vector search unavailable — keyword results only."))
        self.assertIn("## 2026-07-30_10-00_sync.txt", out)
        self.assertIn("2026-07-30_10-00_sync.txt:5-6 @ 00:05", out)
        self.assertIn("we will ship the hybrid search today", out)

    def test_note_omitted_when_vector_used(self):
        self._sync_file()
        passage = self._passage(_SYNC_NAME)
        with (
            mock.patch.object(sem, "available", return_value=True),
            mock.patch.object(
                sem, "search", return_value=[_record("2026-07-30_10-00_sync", passage)]
            ),
        ):
            out = mcp_server.search_transcripts("hybrid")
        self.assertNotIn("vector search unavailable", out)
        self.assertIn("## 2026-07-30_10-00_sync.txt", out)

    def test_backward_compatible_query_only(self):
        self._sync_file()
        with mock.patch.object(sem, "available", return_value=False):
            out = mcp_server.search_transcripts("hybrid")
        self.assertIn("## 2026-07-30_10-00_sync.txt", out)

    def test_no_matches(self):
        self._sync_file()
        with mock.patch.object(sem, "available", return_value=False):
            self.assertEqual(mcp_server.search_transcripts("zzz"), "No matches for 'zzz'.")

    def test_no_transcripts(self):
        with mock.patch.object(sem, "available", return_value=False):
            self.assertEqual(
                mcp_server.search_transcripts("x"),
                "No transcripts found in ~/meeting-notes.",
            )

    def test_invalid_since_message(self):
        self.assertEqual(
            mcp_server.search_transcripts("x", since="last week"),
            "invalid since date 'last week' — use YYYY-MM-DD",
        )


if __name__ == "__main__":
    unittest.main()
