"""Retriever factory, eval-harness helpers, and the reference-DB build.

Run from argus-localization/:  python -m unittest discover -s tests -t .
No model weights or GPU needed.
"""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from core.types import GeoTile
from database.reference_database import ReferenceDatabase, dedup_search
from index import FlatIndex
from index.numpy_index import NumpyFlatIndex
from retrievers.factory import build_retriever, retriever_id, retriever_settings
from scripts.evaluate import (
    db_cache_dir,
    parse_retriever_opts,
    recalls_from_ranks,
    smoke_subset,
)


def square_tile(tile_id: str, lat: float, lon: float, half: float = 1.0) -> GeoTile:
    corners = np.array(
        [[lat - half, lon - half], [lat + half, lon - half], [lat + half, lon + half], [lat - half, lon + half]]
    )
    return GeoTile(tile_id, f"{tile_id}.jpg", corners, meta={"nadir_lat": lat, "nadir_lon": lon})


class RetrieverSettingsTest(unittest.TestCase):
    def test_config_then_overrides_win(self):
        config = {"retriever": {"remoteclip": {"model_name": "ViT-L-14", "image_size": 336}}}
        settings = retriever_settings("remoteclip", config, {"image_size": 224})
        self.assertEqual(settings["model_name"], "ViT-L-14")
        self.assertEqual(settings["image_size"], 224)
        self.assertEqual(settings["resize_mode"], "squash")  # untouched default

    def test_unknown_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            retriever_settings("remoteclip", {}, {"imagesize": 224})

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            retriever_settings("clip", {})

    def test_config_sections_for_other_kinds_are_ignored(self):
        config = {"retriever": {"kind": "earthloc", "descriptor_dim": 512, "head": "gem"}}
        self.assertEqual(retriever_settings("earthloc", config), {})


class RetrieverIdTest(unittest.TestCase):
    def test_embedding_relevant_settings_change_the_id(self):
        base = retriever_settings("remoteclip", {})
        ids = {
            retriever_id("remoteclip", base),
            retriever_id("remoteclip", {**base, "model_name": "ViT-L-14"}),
            retriever_id("remoteclip", {**base, "image_size": 320}),
            retriever_id("remoteclip", {**base, "quick_gelu": True}),
            retriever_id("remoteclip", {**base, "resize_mode": "crop"}),
        }
        self.assertEqual(len(ids), 5)

    def test_batch_size_and_backend_do_not(self):
        base = retriever_settings("remoteclip", {})
        self.assertEqual(
            retriever_id("remoteclip", base),
            retriever_id("remoteclip", {**base, "max_batch": 8, "backend": "timm"}),
        )

    def test_qwen_ids(self):
        base = retriever_settings("qwen3vl_embedding", {})
        self.assertEqual(retriever_id("qwen3vl_embedding", base), "Qwen3-VL-Embedding-2B-320")
        self.assertEqual(
            retriever_id("qwen3vl_embedding", {**base, "dim": 512}), "Qwen3-VL-Embedding-2B-320-d512"
        )
        with_instruction = retriever_id("qwen3vl_embedding", {**base, "instruction": "Find the same area."})
        self.assertTrue(with_instruction.startswith("Qwen3-VL-Embedding-2B-320-instr"))
        self.assertEqual(with_instruction, retriever_id("qwen3vl_embedding", {**base, "instruction": "Find the same area."}))


class BuildRetrieverTest(unittest.TestCase):
    def test_missing_weights_fail_loudly_instead_of_downloading(self):
        with tempfile.TemporaryDirectory() as empty:
            user_config = {"remoteclip_checkpoint_dir": empty, "qwen3vl_embedding_dir": empty}
            with self.assertRaisesRegex(FileNotFoundError, "RemoteCLIP-ViT-B-32.pt"):
                build_retriever("remoteclip", retriever_settings("remoteclip", {}), user_config, "cpu")
            with self.assertRaisesRegex(FileNotFoundError, "Qwen3-VL-Embedding-2B"):
                build_retriever(
                    "qwen3vl_embedding", retriever_settings("qwen3vl_embedding", {}), user_config, "cpu"
                )


class EvalHelpersTest(unittest.TestCase):
    def test_recalls_from_ranks(self):
        ranks = [("a", 1), ("b", 3), ("c", None), ("d", 12)]
        self.assertEqual(recalls_from_ranks(ranks, [1, 5, 15]), {1: 25.0, 5: 50.0, 15: 75.0})
        self.assertEqual(recalls_from_ranks([], [1]), {1: 0.0})

    def test_parse_retriever_opts_uses_yaml_types(self):
        opts = parse_retriever_opts(["image_size=320", "quick_gelu=true", "dim=null", "model_name=ViT-L-14"])
        self.assertEqual(opts, {"image_size": 320, "quick_gelu": True, "dim": None, "model_name": "ViT-L-14"})
        with self.assertRaises(SystemExit):
            parse_retriever_opts(["image_size"])

    def test_cache_dirs(self):
        # EarthLoc keeps the pre-existing cache name, so old caches stay valid.
        self.assertEqual(db_cache_dir("cache", "earthloc", "earthloc", "Alps", False), os.path.join("cache", "db_Alps"))
        self.assertEqual(
            db_cache_dir("cache", "remoteclip", "remoteclip-ViT-B-32-224-squash", "Toshka Lakes", True),
            os.path.join("cache", "db_remoteclip-ViT-B-32-224-squash_Toshka_Lakes_smoke"),
        )

    def test_smoke_subset_keeps_every_sampled_querys_true_tiles(self):
        tiles = [square_tile(f"t{i}", lat=float(i), lon=0.0, half=0.5) for i in range(40)]
        queries = [square_tile(f"q{i}", lat=float(i), lon=0.0, half=0.5) for i in range(0, 40, 2)]
        sampled, tiles_fn = smoke_subset(queries, lambda: tiles, 0.2, seed=0, num_queries=5, num_tiles=12)
        chosen_ids = {t.tile_id for t in tiles_fn()}
        self.assertEqual(len(sampled), 5)
        self.assertEqual(len(chosen_ids), 12)
        for query in sampled:
            self.assertIn(query.tile_id.replace("q", "t"), chosen_ids)


class _MeanColorRetriever:
    """Deterministic fake: descriptor = normalized mean RGB (+ a constant)."""

    descriptor_dim = 4

    def embed(self, image):
        return self.embed_batch([image])[0]

    def embed_batch(self, images):
        feats = np.array([[*np.asarray(img, dtype=np.float32).mean(axis=(0, 1)), 1.0] for img in images])
        return (feats / np.linalg.norm(feats, axis=1, keepdims=True)).astype(np.float32)


class ReferenceDatabaseBuildTest(unittest.TestCase):
    def test_prefetching_build_matches_a_sequential_embed(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            tiles = []
            for i in range(11):  # not a multiple of the batch size
                path = os.path.join(tmp, f"t{i}.png")
                Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(path)
                tiles.append(GeoTile(f"t{i}", path, np.zeros((4, 2))))

            db = ReferenceDatabase(_MeanColorRetriever(), FlatIndex(4))
            db.build(tiles, batch_size=4, num_workers=3)

            self.assertEqual(len(db.index.tile_ids), 11 * 4)
            self.assertEqual(db.index.tile_ids[:5], ["t0::rot0", "t0::rot90", "t0::rot180", "t0::rot270", "t1::rot0"])
            query = np.array(Image.open(tiles[7].image_path).convert("RGB"))
            best_tile, _ = dedup_search(db.index, _MeanColorRetriever().embed(query), 1)[0]
            self.assertEqual(best_tile, "t7")


class NumpyFlatIndexTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((500, 16)).astype(np.float32)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.ids = [f"t{i}" for i in range(500)]
        self.queries = self.vectors[[3, 77, 400]] + 0.05

    def test_matches_faiss(self):
        try:
            from index.faiss_index import FaissFlatIndex
        except ImportError:
            self.skipTest("faiss not installed")
        exact, approx = FaissFlatIndex(16), NumpyFlatIndex(16)
        for index in (exact, approx):
            index.add(self.ids[:200], self.vectors[:200])  # two adds, like a batched build
            index.add(self.ids[200:], self.vectors[200:])
        for q in self.queries:
            expected, got = exact.search(q, 25), approx.search(q, 25)
            self.assertEqual([t for t, _ in got], [t for t, _ in expected])
            np.testing.assert_allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)
            self.assertEqual(
                [t for t, _ in dedup_search(approx, q, 10)], [t for t, _ in dedup_search(exact, q, 10)]
            )

    def test_save_load_and_edge_cases(self):
        index = NumpyFlatIndex(16)
        self.assertEqual(index.search(self.queries[0], 5), [])
        index.add(self.ids, self.vectors)
        self.assertEqual(len(index.search(self.queries[0], 10_000)), 500)
        with tempfile.TemporaryDirectory() as tmp:
            index.save(os.path.join(tmp, "index"))
            loaded = NumpyFlatIndex(1)
            loaded.load(os.path.join(tmp, "index"))
            self.assertEqual(loaded.descriptor_dim, 16)
            self.assertEqual(loaded.search(self.queries[1], 5), index.search(self.queries[1], 5))
            with self.assertRaises(FileNotFoundError):
                NumpyFlatIndex(16).load(os.path.join(tmp, "missing"))


if __name__ == "__main__":
    unittest.main()
