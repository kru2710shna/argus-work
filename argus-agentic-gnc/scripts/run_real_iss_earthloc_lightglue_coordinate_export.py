"""Run real ISS images through EarthLoc + LightGlue and export coordinates.

WHAT THIS FILE DOES
-------------------
This is the real-visual-data step of the Navigation Measurement Builder.

It runs:

    12 real ISS query images
    + 18 nearby Earth reference tiles
    + 24 distractor reference tiles
        -> EarthLoc retrieval (top-k)
        -> SIFT + LightGlue matching
        -> RANSAC geometric verification
        -> verified image-to-Earth tie points

It writes one JSON record per real ISS image. A successful record contains:

    query image pixel [u, v]
    + georeferenced Earth latitude/longitude [lat, lon]
    + inlier count
    + matched Earth map tile

The next navigation layer converts these records into:

    pixel -> body-frame bearing
    latitude/longitude -> ECEF -> ECI landmark
    -> FSW-Payload landmark observations

IMPORTANT LIMITATION
--------------------
These ISS images do not include synchronized flight IMU, precise camera
calibration, or per-image timestamps. Those will be supplied by the later
Basilisk hybrid simulation. This script proves the REAL visual side only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCALIZATION_ROOT = REPOSITORY_ROOT / "argus-localization"

# Reuse the team's existing localization implementation. Do not duplicate it.
sys.path.insert(0, str(LOCALIZATION_ROOT))

from core.pipeline import LocalizationPipeline
from core.types import LocalizationResult, PipelineConfig
from data_loading.earthloc_loader import parse_geotile_filename
from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from retrievers.earthloc_retriever import EarthLocRetriever


def load_rgb_image(path: Path) -> np.ndarray:
    """Load one RGB image as the uint8 [height, width, 3] array pipeline input."""
    return np.array(Image.open(path).convert("RGB"))


def resolve_device(requested_device: str) -> str:
    """Resolve auto/cuda/cpu without silently claiming CUDA exists."""
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Run this real visual experiment in Colab with GPU enabled."
        )

    return requested_device


def require_files(data_root: Path, relative_paths: list[str]) -> None:
    """Fail early with a clear missing-data message."""
    missing = [
        str(data_root / relative_path)
        for relative_path in relative_paths
        if not (data_root / relative_path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} required EarthLoc data files are missing. "
            f"First missing file: {missing[0]}"
        )


def result_to_coordinate_record(
    *,
    frame_index: int,
    query_relative_path: str,
    query_tile_id: str,
    result: LocalizationResult,
    positive_tile_ids: set[str],
) -> dict:
    """Convert one localization result into JSON-safe coordinate output.

    `known_overlap_retrieved_top_k` is evaluation-only. It is calculated
    after localization and is never passed into EarthLoc or LightGlue.
    """
    candidates = result.debug.get("candidates", []) if result.debug else []
    retrieved_tile_ids = [candidate["tile_id"] for candidate in candidates]

    return {
        "frame_index": frame_index,
        "frame_id": query_tile_id,
        "frame_path": query_relative_path,
        "status": result.status,
        "confidence_num_inliers": float(result.confidence),
        "matched_tile_id": result.matched_tile_id,
        "tie_points": [
            {
                "u": float(tie_point.u),
                "v": float(tie_point.v),
                "lat": float(tie_point.lat),
                "lon": float(tie_point.lon),
            }
            for tie_point in result.tie_points
        ],
        "query_footprint_latlon": (
            result.query_footprint_latlon.tolist()
            if result.query_footprint_latlon is not None
            else None
        ),
        "retrieval_candidates": [
            {
                "tile_id": candidate["tile_id"],
                "similarity": float(candidate["retrieval_similarity"]),
                "num_inliers": int(candidate["num_inliers"]),
            }
            for candidate in candidates
        ],
        "evaluation_only": {
            "known_overlap_retrieved_top_k": any(
                tile_id in positive_tile_ids for tile_id in retrieved_tile_ids
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real ISS imagery through EarthLoc retrieval and LightGlue "
            "geometric verification, then export coordinate records."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="EarthLoc data root containing queries/, database/, and checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "argus-agentic-gnc"
        / "configs"
        / "earthloc_real_episode.json",
        help="Fixed 12-image ISS episode manifest.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <data-root>/best_trained_model.pt.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="EarthLoc candidates passed to LightGlue.",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=30,
        help="Minimum verified correspondences required for status='fix'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of ISS frames; use 1 for a smoke test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "argus-agentic-gnc"
        / "outputs"
        / "real_iss_earthloc_lightglue_coordinate_records.json",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    if args.min_inliers < 4:
        raise ValueError("--min-inliers must be at least 4 for homography verification")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else data_root / "best_trained_model.pt"
    )
    device = resolve_device(args.device)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"EarthLoc checkpoint does not exist: {checkpoint_path}")

    manifest = json.loads(manifest_path.read_text())

    query_relative_paths = list(manifest["queries"])
    reference_relative_paths = [
        *manifest["database_tiles"],
        *manifest["distractor_tiles"],
    ]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        query_relative_paths = query_relative_paths[: args.limit]

    require_files(
        data_root,
        [*query_relative_paths, *reference_relative_paths],
    )

    logging.info("Device: %s", device)
    logging.info("Real ISS queries: %d", len(query_relative_paths))
    logging.info("Reference tiles: %d", len(reference_relative_paths))
    logging.info(
        "Pipeline: EarthLoc top-%d -> LightGlue -> minimum %d inliers",
        args.top_k,
        args.min_inliers,
    )

    # Parse map metadata directly from the EarthLoc filenames.
    reference_tiles = [
        parse_geotile_filename(str(data_root / relative_path))
        for relative_path in reference_relative_paths
    ]

    # Build evaluation-only known-overlap IDs. These labels never reach the model.
    positive_tile_ids_by_query_name = {
        query_name: {
            parse_geotile_filename(str(data_root / relative_path)).tile_id
            for relative_path in positive_relative_paths
        }
        for query_name, positive_relative_paths in manifest[
            "positive_tiles_by_query"
        ].items()
    }

    retriever = EarthLocRetriever(str(checkpoint_path), device=device)
    database = ReferenceDatabase(
        retriever=retriever,
        index=FaissFlatIndex(retriever.descriptor_dim),
    )

    logging.info("Embedding %d reference tiles with rotation augmentation...", len(reference_tiles))
    database.build(reference_tiles)

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=1024,
        img_size=512,
        max_ransac_iters=3,
        min_inliers=args.min_inliers,
        device=device,
    )
    pipeline = LocalizationPipeline(
        db=database,
        matcher=matcher,
        georef=Georeferencer(),
        config=PipelineConfig(
            top_k=args.top_k,
            min_inliers=args.min_inliers,
            query_size=512,
            max_ransac_iters=3,
        ),
    )

    records: list[dict] = []
    started_at = time.time()

    for frame_index, query_relative_path in enumerate(query_relative_paths):
        query_path = data_root / query_relative_path
        query_tile = parse_geotile_filename(str(query_path))

        result = pipeline.localize(load_rgb_image(query_path))

        record = result_to_coordinate_record(
            frame_index=frame_index,
            query_relative_path=query_relative_path,
            query_tile_id=query_tile.tile_id,
            result=result,
            positive_tile_ids=positive_tile_ids_by_query_name[query_path.name],
        )
        records.append(record)

        logging.info(
            "%s | status=%s | tile=%s | inliers=%d | tie_points=%d",
            query_tile.tile_id,
            record["status"],
            record["matched_tile_id"],
            int(record["confidence_num_inliers"]),
            len(record["tie_points"]),
        )

    elapsed_s = time.time() - started_at
    fixes = sum(record["status"] == "fix" for record in records)
    overlap_top_k = sum(
        record["evaluation_only"]["known_overlap_retrieved_top_k"]
        for record in records
    )

    payload = {
        "experiment": "real_iss_earthloc_lightglue_coordinate_export",
        "data_source": "real ISS query images and EarthLoc map tiles",
        "important_limitations": [
            "No real synchronized spacecraft IMU data is used here.",
            "No flight camera calibration is available in this dataset.",
            "The filenames provide a date but not precise per-image time.",
            "Basilisk will supply synchronized time, attitude, and gyro data in the hybrid OD experiment.",
        ],
        "num_frames": len(records),
        "num_fixes": fixes,
        "fix_rate_percent": 100.0 * fixes / len(records) if records else 0.0,
        "evaluation_only_known_overlap_in_top_k": overlap_top_k,
        "evaluation_only_known_overlap_top_k_rate_percent": (
            100.0 * overlap_top_k / len(records) if records else 0.0
        ),
        "pipeline_config": {
            "device": device,
            "top_k": args.top_k,
            "min_inliers": args.min_inliers,
            "max_num_keypoints": 1024,
            "lightglue_image_size": 512,
            "max_ransac_iters": 3,
        },
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print("\n=== Real ISS EarthLoc + LightGlue Coordinate Export ===")
    print(f"Frames evaluated: {len(records)}")
    print(f"Verified geometric fixes: {fixes}/{len(records)}")
    print(f"Known-overlap tile retrieved in top-{args.top_k}: {overlap_top_k}/{len(records)}")
    print(f"Output: {args.output}")
    print(
        "NEXT: Use this JSON with Basilisk time/attitude/gyro data to build "
        "FSW-Payload landmark observations."
    )


if __name__ == "__main__":
    main()