"""Tests for bounding MLX's GPU buffer cache.

MLX caches every GPU buffer it allocates and never shrinks the cache on its
own, so a long-running menu bar app accumulates gigabytes of Metal memory
between meetings unless it is explicitly capped and trimmed.
"""

import sys
import types
import unittest
from unittest import mock

from trnscrb import transcriber


def _fake_mlx(cache_bytes=2_000_000_000):
    """Stand-in for mlx.core tracking cache-limit and clear calls."""
    state = {"limit": None, "cache": cache_bytes, "cleared": 0}

    def set_cache_limit(n):
        state["limit"] = n

    def clear_cache():
        state["cleared"] += 1
        state["cache"] = 0

    module = types.SimpleNamespace(
        set_cache_limit=set_cache_limit,
        clear_cache=clear_cache,
        get_cache_memory=lambda: state["cache"],
    )
    return module, state


class TrimMlxCacheTest(unittest.TestCase):
    def test_noop_when_mlx_not_imported(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("mlx.core", None)
            self.assertEqual(transcriber.trim_mlx_cache(), 0.0)

    def test_never_imports_mlx_itself(self):
        """Whisper-only users must not pay for importing MLX."""
        real_import = __import__

        def guard(name, *args, **kwargs):
            if name.startswith("mlx"):
                raise AssertionError("trim_mlx_cache must not import mlx")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("mlx.core", None)
            with mock.patch("builtins.__import__", side_effect=guard):
                self.assertEqual(transcriber.trim_mlx_cache(), 0.0)

    def test_releases_cache_and_reports_mb(self):
        module, state = _fake_mlx(cache_bytes=2_000_000_000)
        with mock.patch.dict(sys.modules, {"mlx.core": module}):
            freed = transcriber.trim_mlx_cache()
        self.assertEqual(state["cleared"], 1)
        self.assertAlmostEqual(freed, 2000.0, places=0)

    def test_survives_mlx_errors(self):
        broken = types.SimpleNamespace(
            get_cache_memory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            clear_cache=lambda: None,
        )
        with mock.patch.dict(sys.modules, {"mlx.core": broken}):
            self.assertEqual(transcriber.trim_mlx_cache(), 0.0)


class BoundMlxCacheTest(unittest.TestCase):
    def test_applies_configured_limit(self):
        module, state = _fake_mlx()
        with mock.patch.dict(sys.modules, {"mlx.core": module}):
            with mock.patch("trnscrb.settings.get", return_value=256):
                transcriber._bound_mlx_cache()
        self.assertEqual(state["limit"], 256 * 1024 * 1024)

    def test_default_limit_when_unset(self):
        module, state = _fake_mlx()
        with mock.patch.dict(sys.modules, {"mlx.core": module}):
            with mock.patch("trnscrb.settings.get", return_value=None):
                transcriber._bound_mlx_cache()
        self.assertEqual(state["limit"], transcriber._MLX_CACHE_LIMIT_MB * 1024 * 1024)

    def test_zero_disables_the_cap(self):
        module, state = _fake_mlx()
        with mock.patch.dict(sys.modules, {"mlx.core": module}):
            with mock.patch("trnscrb.settings.get", return_value=0):
                transcriber._bound_mlx_cache()
        self.assertIsNone(state["limit"], "0 must leave MLX at its own default")


class UnloadReleasesGpuMemoryTest(unittest.TestCase):
    def test_unload_models_trims_cache(self):
        """Dropping Python references is not enough — the GPU cache holds the
        weights until it is explicitly cleared."""
        module, state = _fake_mlx(cache_bytes=1_500_000_000)
        with mock.patch.dict(sys.modules, {"mlx.core": module}):
            transcriber.unload_models()
        self.assertGreaterEqual(state["cleared"], 1)


if __name__ == "__main__":
    unittest.main()
