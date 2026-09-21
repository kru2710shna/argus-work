"""Retrieval and matching eval harness. See docs/argus_localization_spec.md section 7.

Retrieval: report recall@1, recall@5, recall@k. A retrieved tile counts as
correct if its footprint overlaps the query ground-truth footprint above
IoU 0.2 (configurable). recall@k is the metric that matters, not R@1
(docs/argus_localization_design.md section 6).

Matching: per query, best-candidate num_inliers, fix rate (fraction with
num_inliers >= min_inliers), median localization error in km.

Retrieval and matching are reported separately, they fail for different
reasons.

EarthLoc queries are oblique and cover about 25,000 sq km each, nothing
like an Argus frame, so these numbers validate plumbing, not Argus
performance (docs/argus_localization_spec.md section 6, section 10).
"""

import argparse
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import numpy as np
import yaml
from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import Retriever
from core.types import GeoTile
from data_loading.earthloc_loader import load_query_set, parse_geotile_filename
from database.reference_database import ReferenceDatabase, dedup_search
from index import FlatIndex
from retrievers.factory import RETRIEVER_KINDS, build_retriever, retriever_id, retriever_settings

if TYPE_CHECKING:
    # Imported for real only in run_matching(), so retrieval-only runs don't
    # need LightGlue installed.
    from core.pipeline import LocalizationPipeline

# Same six eval regions as EarthLoc's eval.py (center_lat, center_lon).
REGIONS = {
    "Alps": (45, 10),
    "Texas": (30, -95),
    "Toshka Lakes": (23, 30),
    "Amazon": (-3, -60),
    "Napa": (38, -122),
    "Gobi": (40, 105),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def footprint_polygon(tile: GeoTile) -> Polygon:
    # corners_latlon order is BL,TL,TR,BR (see core/types.py). Polygon wants
    # (x, y) i.e. (lon, lat), and this order already traces the quad in a
    # consistent (non-self-intersecting) winding.
    return Polygon([(lon, lat) for lat, lon in tile.corners_latlon])


def footprint_bbox(tile: GeoTile) -> tuple[float, float, float, float]:
    lats = tile.corners_latlon[:, 0]
    lons = tile.corners_latlon[:, 1]
    return lats.min(), lats.max(), lons.min(), lons.max()


def footprint_iou(tile_a: GeoTile, tile_b: GeoTile) -> float:
    poly_a, poly_b = footprint_polygon(tile_a), footprint_polygon(tile_b)
    if not poly_a.is_valid or not poly_b.is_valid or not poly_a.intersects(poly_b):
        return 0.0
    union = poly_a.union(poly_b).area
    if union == 0.0:
        return 0.0
    return poly_a.intersection(poly_b).area / union


def find_positive_tile_ids(
    query: GeoTile, db_tiles: list[GeoTile], db_bboxes: np.ndarray, iou_threshold: float
) -> list[str]:
    q_lat_min, q_lat_max, q_lon_min, q_lon_max = footprint_bbox(query)
    # Cheap bounding-box prefilter (vectorized) before the exact, slower polygon IoU.
    candidate_mask = (
        (db_bboxes[:, 2] <= q_lon_max)
        & (db_bboxes[:, 3] >= q_lon_min)
        & (db_bboxes[:, 0] <= q_lat_max)
        & (db_bboxes[:, 1] >= q_lat_min)
    )
    positives = []
    for idx in np.nonzero(candidate_mask)[0]:
        if footprint_iou(query, db_tiles[idx]) >= iou_threshold:
            positives.append(db_tiles[idx].tile_id)
    return positives


def load_image_array(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_scoped_db_tiles(
    database_dir: str, center_lat: float, center_lon: float, year: int, dist_km: float
) -> list[GeoTile]:
    import glob

    month_dirs = sorted(
        d for d in os.listdir(database_dir) if d.startswith(f"{year}_") and os.path.isdir(os.path.join(database_dir, d))
    )
    paths = []
    for month_dir in month_dirs:
        paths += glob.glob(os.path.join(database_dir, month_dir, "**", "*.jpg"), recursive=True)
    tiles = [parse_geotile_filename(p) for p in paths]
    return [
        t
        for t in tiles
        if haversine_km(center_lat, center_lon, t.meta["nadir_lat"], t.meta["nadir_lon"]) < dist_km
    ]


def load_scoped_queries(
    queries_dir: str, center_lat: float, center_lon: float, dist_km: float
) -> list[GeoTile]:
    queries = load_query_set(queries_dir)
    return [
        q
        for q in queries
        if haversine_km(center_lat, center_lon, q.meta["nadir_lat"], q.meta["nadir_lon"]) < dist_km
    ]


def rank_queries(
    db: ReferenceDatabase,
    queries: list[GeoTile],
    max_k: int,
    iou_threshold: float = 0.2,
    batch_size: int = 64,
) -> list[tuple[str, int | None]]:
    """(query tile_id, 1-based rank of the first correct tile or None) for every
    query that has a ground-truth positive. Recall@k for any k <= max_k follows
    from these, and per-query ranks allow paired tests between retrievers."""
    db_tiles = list(db.tiles.values())
    db_bboxes = np.array([footprint_bbox(t) for t in db_tiles])

    # Ground-truth positives need the (cheap) bbox-prefiltered IoU check per query, but
    # embedding is the expensive part, so batch all queries with a ground truth positive
    # through the retriever instead of the one-image-at-a-time db.retrieve() per query.
    scored_queries = []
    for query in queries:
        positives = find_positive_tile_ids(query, db_tiles, db_bboxes, iou_threshold)
        if positives:
            scored_queries.append((query, set(positives)))
        else:
            logging.debug(f"Query {query.tile_id} has no positives, skipping (probably over the sea).")

    ranks = []
    for start in range(0, len(scored_queries), batch_size):
        chunk = scored_queries[start : start + batch_size]
        images = [load_image_array(query.image_path) for query, _ in chunk]
        descriptors = db.retriever.embed_batch(images)
        for (query, positives_set), descriptor in zip(chunk, descriptors):
            retrieved = dedup_search(db.index, descriptor, max_k)
            first_hit = next(
                (rank for rank, (tile_id, _) in enumerate(retrieved, 1) if tile_id in positives_set),
                None,
            )
            ranks.append((query.tile_id, first_hit))
    return ranks


def recalls_from_ranks(ranks: list[tuple[str, int | None]], k_values: list[int]) -> dict[int, float]:
    if not ranks:
        return {k: 0.0 for k in k_values}
    return {
        k: 100.0 * sum(1 for _, rank in ranks if rank is not None and rank <= k) / len(ranks)
        for k in k_values
    }


def evaluate_retrieval(
    db: ReferenceDatabase,
    queries: list[GeoTile],
    k_values: list[int],
    iou_threshold: float = 0.2,
    batch_size: int = 64,
) -> tuple[dict[int, float], int]:
    ranks = rank_queries(db, queries, max(k_values), iou_threshold, batch_size)
    return recalls_from_ranks(ranks, k_values), len(ranks)


def evaluate_matching(pipeline: "LocalizationPipeline", queries: list[GeoTile]) -> dict:
    num_inliers_list = []
    num_fix = 0
    errors_km = []
    for query in queries:
        query_image = load_image_array(query.image_path)
        result = pipeline.localize(query_image)
        num_inliers_list.append(result.confidence)
        if result.status == "fix":
            num_fix += 1
            gt_center = query.corners_latlon.mean(axis=0)
            if result.query_footprint_latlon is not None:
                est_center = result.query_footprint_latlon.mean(axis=0)
            else:
                est_center = np.array([[tp.lat, tp.lon] for tp in result.tie_points]).mean(axis=0)
            errors_km.append(haversine_km(gt_center[0], gt_center[1], est_center[0], est_center[1]))

    return {
        "num_queries": len(queries),
        "mean_num_inliers": statistics.fmean(num_inliers_list) if num_inliers_list else 0.0,
        "fix_rate": 100.0 * num_fix / len(queries) if queries else 0.0,
        "median_localization_error_km": statistics.median(errors_km) if errors_km else None,
    }


def parse_retriever_opts(opts: list[str]) -> dict:
    """--retriever-opt key=value pairs; values are parsed as YAML (224 -> int, true -> bool)."""
    overrides = {}
    for opt in opts:
        key, sep, value = opt.partition("=")
        if not sep:
            raise SystemExit(f"--retriever-opt expects key=value, got {opt!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def db_cache_dir(cache_root: str, kind: str, rid: str, region: str, smoke: bool) -> str:
    region_slug = region.replace(" ", "_")
    # EarthLoc keeps the original db_<region> name so existing caches stay valid.
    name = f"db_{region_slug}" if kind == "earthloc" else f"db_{rid}_{region_slug}"
    return os.path.join(cache_root, name + ("_smoke" if smoke else ""))


def smoke_subset(
    queries: list[GeoTile],
    scope_db_tiles: Callable[[], list[GeoTile]],
    iou_threshold: float,
    seed: int,
    num_queries: int = 50,
    num_tiles: int = 500,
) -> tuple[list[GeoTile], Callable[[], list[GeoTile]]]:
    """A small query sample plus a DB of their true tiles and random distractors,
    so a plumbing check exercises real retrieval hits in minutes. The tile list
    is only computed when the (smoke) DB actually has to be built."""
    rng = random.Random(seed)
    queries = rng.sample(queries, min(num_queries, len(queries)))

    def tiles() -> list[GeoTile]:
        all_tiles = scope_db_tiles()
        bboxes = np.array([footprint_bbox(t) for t in all_tiles])
        keep = {
            tile_id
            for query in queries
            for tile_id in find_positive_tile_ids(query, all_tiles, bboxes, iou_threshold)
        }
        chosen = [t for t in all_tiles if t.tile_id in keep]
        others = [t for t in all_tiles if t.tile_id not in keep]
        return chosen + rng.sample(others, max(0, min(num_tiles - len(chosen), len(others))))

    return queries, tiles


def load_or_build_db(
    retriever: Retriever,
    rid: str,
    cache_dir: str,
    rebuild: bool,
    scope_db_tiles: Callable[[], list[GeoTile]],
) -> tuple[ReferenceDatabase, float | None]:
    """Returns the database and its build time in seconds (None when loaded from cache)."""
    index = FlatIndex(retriever.descriptor_dim)
    meta_path = os.path.join(cache_dir, "retriever.json")
    if not rebuild and os.path.exists(os.path.join(cache_dir, "tiles.json")):
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                cached_rid = json.load(f)["retriever_id"]
            if cached_rid != rid:
                raise SystemExit(f"{cache_dir} was built by {cached_rid}, not {rid}; pass --rebuild-db")
        logging.info(f"Loading cached reference database from {cache_dir}")
        db = ReferenceDatabase.load(cache_dir, retriever, index)
        if db.index.descriptor_dim != retriever.descriptor_dim:
            raise SystemExit(
                f"{cache_dir} holds {db.index.descriptor_dim}-d descriptors but {rid} makes "
                f"{retriever.descriptor_dim}-d ones; pass --rebuild-db"
            )
        return db, None

    db_tiles = scope_db_tiles()
    logging.info(f"Embedding {len(db_tiles)} reference tiles x 4 rotations with {rid}")
    db = ReferenceDatabase(retriever, index)
    t0 = time.time()
    db.build(db_tiles)
    build_seconds = time.time() - t0
    logging.info(
        f"Built and embedded reference database in {build_seconds:.1f}s "
        f"({4 * len(db_tiles) / max(build_seconds, 1e-9):.0f} img/s)"
    )
    db.save(cache_dir)
    with open(meta_path, "w") as f:
        json.dump({"retriever_id": rid, "num_tiles": len(db_tiles), "build_seconds": build_seconds}, f)
    return db, build_seconds


def main():
    parser = argparse.ArgumentParser(description="Evaluate the retrieve-then-match pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--user-config", default="user_config.yaml")
    parser.add_argument("--region", default="Alps", choices=list(REGIONS.keys()))
    parser.add_argument(
        "--retriever", choices=RETRIEVER_KINDS, default=None,
        help="retriever kind (default: retriever.kind in config.yaml)",
    )
    parser.add_argument(
        "--retriever-opt", action="append", default=[], metavar="KEY=VALUE",
        help="override one retriever.<kind> setting, e.g. model_name=ViT-L-14 (repeatable)",
    )
    parser.add_argument("--rebuild-db", action="store_true", help="ignore any cached reference database")
    parser.add_argument("--skip-matching", action="store_true", help="retrieval recall only, no SIFT+LightGlue")
    parser.add_argument(
        "--smoke", action="store_true",
        help="plumbing check: 50 queries against their true tiles plus distractors (~500 tiles), "
        "separate cache; recall numbers are meaningless",
    )
    parser.add_argument("--results-dir", default=None, help="default: <output_dir>/results from user_config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)

    device = config.get("device", "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            logging.warning("CUDA is not available, falling back to CPU")
            device = "cpu"
    center_lat, center_lon = REGIONS[args.region]

    kind = args.retriever or config["retriever"]["kind"]
    settings = retriever_settings(kind, config, parse_retriever_opts(args.retriever_opt))
    rid = retriever_id(kind, settings)
    logging.info(f"Retriever {rid}: {settings}")
    retriever = build_retriever(kind, settings, user_config, device)

    queries = load_scoped_queries(
        user_config["queries_dir"], center_lat, center_lon, config["eval"]["query_dist_km"]
    )
    logging.info(f"{len(queries)} queries within {config['eval']['query_dist_km']} km of {args.region}")

    def scope_db_tiles() -> list[GeoTile]:
        tiles = load_scoped_db_tiles(
            user_config["database_dir"],
            center_lat,
            center_lon,
            config["eval"]["db_year"],
            config["eval"]["db_dist_km"],
        )
        logging.info(f"{len(tiles)} reference tiles within {config['eval']['db_dist_km']} km of {args.region}")
        return tiles

    if args.smoke:
        queries, scope_db_tiles = smoke_subset(
            queries, scope_db_tiles, config["eval"]["iou_threshold"], args.seed
        )

    cache_dir = db_cache_dir(user_config["cache_dir"], kind, rid, args.region, args.smoke)
    db, build_seconds = load_or_build_db(retriever, rid, cache_dir, args.rebuild_db, scope_db_tiles)

    k_values = config["eval"]["recall_k_values"]
    logging.info("Evaluating retrieval...")
    t0 = time.time()
    ranks = rank_queries(db, queries, max(k_values), config["eval"]["iou_threshold"])
    retrieval_seconds = time.time() - t0
    recalls = recalls_from_ranks(ranks, k_values)
    num_evaluated = len(ranks)
    logging.info(f"Retrieval eval took {retrieval_seconds:.1f}s over {num_evaluated} queries with ground truth")
    recalls_str = ", ".join(f"R@{k}: {v:.1f}" for k, v in recalls.items())
    print(f"\n=== Retrieval ({args.region}, {rid}, {num_evaluated} queries with a ground-truth positive) ===")
    print(recalls_str)

    results = {
        "region": args.region,
        "retriever_kind": kind,
        "retriever_id": rid,
        "retriever_settings": settings,
        "descriptor_dim": retriever.descriptor_dim,
        "smoke": args.smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_db_tiles": len(db.tiles),
        "db_build_seconds": build_seconds,
        "num_queries_scoped": len(queries),
        "num_queries_evaluated": num_evaluated,
        "retrieval_seconds": retrieval_seconds,
        "recalls": {f"R@{k}": v for k, v in recalls.items()},
        "first_hit_rank": dict(ranks),
    }

    if not args.skip_matching:
        results["matching"] = run_matching(db, queries, config, device, args.region)

    results_dir = args.results_dir or os.path.join(user_config["output_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    results_path = os.path.join(results_dir, f"{args.region.replace(' ', '_')}_{rid}{suffix}_{stamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Wrote {results_path}")


def run_matching(
    db: ReferenceDatabase, queries: list[GeoTile], config: dict, device: str, region: str
) -> dict:
    from core.pipeline import LocalizationPipeline
    from core.types import PipelineConfig
    from georeference.georeferencer import Georeferencer
    from matchers.sift_lightglue_matcher import SiftLightGlueMatcher

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=config["pipeline"]["min_inliers"],
        device=device,
    )
    georef = Georeferencer()
    pipeline_config = PipelineConfig(**config["pipeline"])
    pipeline = LocalizationPipeline(db, matcher, georef, pipeline_config)

    max_matching = config["eval"]["max_matching_queries"]
    matching_queries = (
        random.sample(queries, max_matching) if len(queries) > max_matching else queries
    )
    logging.info(f"Evaluating matching on {len(matching_queries)} of {len(queries)} queries...")
    t0 = time.time()
    matching_stats = evaluate_matching(pipeline, matching_queries)
    logging.info(f"Matching eval took {time.time() - t0:.1f}s")

    print(f"\n=== Matching ({region}, {matching_stats['num_queries']} queries) ===")
    print(f"mean best-candidate num_inliers: {matching_stats['mean_num_inliers']:.1f}")
    print(f"fix rate (num_inliers >= {config['pipeline']['min_inliers']}): {matching_stats['fix_rate']:.1f}%")
    median_error = matching_stats["median_localization_error_km"]
    print(f"median localization error: {median_error:.1f} km" if median_error is not None else "median localization error: n/a (no fixes)")
    return matching_stats


if __name__ == "__main__":
    main()
