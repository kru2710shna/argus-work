#!/usr/bin/env python3
"""
Run Unity/Cesium nadir-camera navigation evaluation end to end.

This script:
1. Reads a Unity navigation-episode package/manifest.
2. Keeps usable NADIR EARTH -Z images.
3. Retrieves EarthLoc map-tile candidates from an existing FAISS index.
4. Verifies each candidate with SuperPoint + LightGlue + RANSAC.
5. Evaluates candidates against Unity truth (evaluation only).
6. Writes verified candidates only. Rejected candidates never become fixes.

It does NOT rebuild the EarthLoc database index.
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-directory", type=Path, required=True)
    parser.add_argument("--earthloc-source", type=Path, required=True)
    parser.add_argument("--earthloc-checkpoint", type=Path, required=True)
    parser.add_argument("--earthloc-index", type=Path, required=True)
    parser.add_argument("--tile-paths", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--lightglue-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--minimum-inliers", type=int, default=30)
    parser.add_argument("--min-visible-fraction", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def center_square(image):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def visible_fraction(image_path):
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    return float((pixels.max(axis=2) > 5).mean())


def choose_evenly_spaced(frames, maximum):
    if len(frames) <= maximum:
        return frames

    indices = np.linspace(0, len(frames) - 1, maximum, dtype=int)
    return [frames[index] for index in indices]


def ecef_to_lat_lon_degrees(x_m, y_m, z_m):
    a = 6378137.0
    e2 = 6.69437999014e-3

    longitude = math.atan2(y_m, x_m)
    p = math.hypot(x_m, y_m)
    latitude = math.atan2(z_m, p * (1.0 - e2))

    for _ in range(8):
        radius = a / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
        latitude = math.atan2(
            z_m + e2 * radius * math.sin(latitude),
            p,
        )

    return math.degrees(latitude), math.degrees(longitude)


def tile_center_from_filename(tile_relative_path):
    values = [
        float(value)
        for value in re.findall(r"@(-?\d+(?:\.\d+)?)", tile_relative_path)
    ]

    if len(values) < 8:
        raise ValueError(f"Cannot parse tile corners: {tile_relative_path}")

    latitudes = values[0:8:2]
    longitudes = values[1:8:2]
    return sum(latitudes) / 4.0, sum(longitudes) / 4.0


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        (lat1, lon1, lat2, lon2),
    )

    d_lat = lat2 - lat1
    d_lon = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi

    h = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(d_lon / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(h))


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    manifest_path = (
        args.episode_directory / "unity_nadir_navigation_manifest.json"
    )
    image_root = args.episode_directory / "images"

    required_paths = [
        manifest_path,
        args.earthloc_checkpoint,
        args.earthloc_index,
        args.tile_paths,
        args.database_root,
        args.lightglue_source,
        args.earthloc_source,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(missing))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    prepared_root = args.output_directory / "prepared_queries"
    shutil.rmtree(prepared_root, ignore_errors=True)
    prepared_root.mkdir()

    sys.path.insert(0, str(args.earthloc_source))
    sys.path.insert(0, str(args.lightglue_source))

    from apl_models.apl_model import APLModel
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd

    manifest = load_json(manifest_path)

    usable_frames = []
    for frame in manifest["frames"]:
        if frame["camera_name"] != "NADIR EARTH  -Z":
            continue

        image_path = image_root / frame["image_file"]
        if not image_path.is_file():
            continue

        visibility = visible_fraction(image_path)
        if visibility >= args.min_visible_fraction:
            frame = dict(frame)
            frame["visible_fraction"] = visibility
            usable_frames.append(frame)

    selected_frames = choose_evenly_spaced(usable_frames, args.max_frames)

    if not selected_frames:
        raise RuntimeError("No usable nadir images passed the visibility filter.")

    print("Device:", device)
    print("Usable nadir frames:", len(usable_frames))
    print("Selected frames:", [frame["frame_id"] for frame in selected_frames])

    model = APLModel().to(device).eval()
    state_dict = torch.load(
        args.earthloc_checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    index = faiss.read_index(str(args.earthloc_index))
    tile_paths = load_json(args.tile_paths)

    if index.ntotal != len(tile_paths):
        raise RuntimeError(
            f"FAISS vectors ({index.ntotal}) and tile paths "
            f"({len(tile_paths)}) do not match."
        )

    earthloc_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    def prepare_query(frame, rotation_ccw):
        output_path = prepared_root / (
            f"{frame['frame_id']}_rot{rotation_ccw}.png"
        )

        source_path = image_root / frame["image_file"]
        image = Image.open(source_path).convert("RGB")
        image = center_square(image).rotate(rotation_ccw, expand=False)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image.save(output_path)
        return output_path

    retrieval_frames = []

    print("\nRetrieving EarthLoc candidates...")
    for frame in tqdm(selected_frames, desc="EarthLoc retrieval"):
        candidates = []

        for rotation_ccw in (0, 90, 180, 270):
            query_path = prepare_query(frame, rotation_ccw)
            image = Image.open(query_path).convert("RGB")
            tensor = earthloc_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                descriptor = model(tensor).cpu().numpy().astype("float32")

            distances, indices = index.search(descriptor, args.top_k)

            for distance, tile_index in zip(distances[0], indices[0]):
                candidates.append({
                    "tile_relative_path": tile_paths[int(tile_index)],
                    "rotation_ccw_degrees": rotation_ccw,
                    "l2_distance": float(distance),
                })

        candidates.sort(key=lambda item: item["l2_distance"])

        retrieval_frames.append({
            "frame_id": frame["frame_id"],
            "image_file": frame["image_file"],
            "visible_fraction": frame["visible_fraction"],
            "truth_position_ecef_m": frame["position_ecef_m_truth"],
            "top_candidates": candidates[:args.top_k],
        })

    retrieval_payload = {
        "experiment": "Unity nadir to EarthLoc global retrieval",
        "num_frames": len(retrieval_frames),
        "top_k": args.top_k,
        "frames": retrieval_frames,
    }
    write_json(args.output_directory / "retrieval.json", retrieval_payload)

    def match_and_count_inliers(query_path, tile_path):
        image0 = load_image(query_path).to(device)
        image1 = load_image(tile_path).to(device)

        with torch.no_grad():
            features0 = extractor.extract(image0)
            features1 = extractor.extract(image1)
            matches01 = matcher({"image0": features0, "image1": features1})

        features0, features1, matches01 = [
            rbd(item) for item in (features0, features1, matches01)
        ]

        matches = matches01["matches"]
        raw_matches = int(matches.shape[0])

        if raw_matches < 4:
            return raw_matches, 0

        points0 = features0["keypoints"][matches[:, 0]].cpu().numpy()
        points1 = features1["keypoints"][matches[:, 1]].cpu().numpy()

        _, mask = cv2.findHomography(
            points0,
            points1,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

        return raw_matches, 0 if mask is None else int(mask.ravel().sum())

    verification_frames = []

    candidate_pairs = [
        (frame, candidate)
        for frame in retrieval_frames
        for candidate in frame["top_candidates"]
    ]

    print("\nVerifying candidates with SuperPoint + LightGlue + RANSAC...")
    for frame, candidate in tqdm(
        candidate_pairs,
        desc="Geometric verification",
    ):
        tile_path = args.database_root / candidate["tile_relative_path"]
        query_path = prepare_query(
            frame,
            candidate["rotation_ccw_degrees"],
        )

        if not tile_path.is_file():
            continue

        raw_matches, inliers = match_and_count_inliers(query_path, tile_path)

        record = dict(candidate)
        record["raw_matches"] = raw_matches
        record["ransac_inliers"] = inliers

        result = next(
            (
                item for item in verification_frames
                if item["frame_id"] == frame["frame_id"]
            ),
            None,
        )

        if result is None:
            result = {
                "frame_id": frame["frame_id"],
                "image_file": frame["image_file"],
                "truth_position_ecef_m": frame["truth_position_ecef_m"],
                "candidates": [],
            }
            verification_frames.append(result)

        result["candidates"].append(record)

    truth_report = []
    verified_candidates = []

    for result in verification_frames:
        result["candidates"].sort(
            key=lambda item: (
                -item["ransac_inliers"],
                item["l2_distance"],
            )
        )

        best = result["candidates"][0]
        result["best_candidate"] = best
        result["status"] = (
            "verified_fix"
            if best["ransac_inliers"] >= args.minimum_inliers
            else "no_fix"
        )

        true_lat, true_lon = ecef_to_lat_lon_degrees(
            *result["truth_position_ecef_m"]
        )
        tile_lat, tile_lon = tile_center_from_filename(
            best["tile_relative_path"]
        )
        error_km = haversine_km(true_lat, true_lon, tile_lat, tile_lon)

        truth_row = {
            "frame_id": result["frame_id"],
            "status": result["status"],
            "unity_truth_subsatellite_lat_lon_degrees": [
                true_lat,
                true_lon,
            ],
            "retrieved_tile_center_lat_lon_degrees": [
                tile_lat,
                tile_lon,
            ],
            "candidate_error_km": error_km,
            "lightglue_ransac_inliers": best["ransac_inliers"],
            "tile_relative_path": best["tile_relative_path"],
        }
        truth_report.append(truth_row)

        if result["status"] == "verified_fix":
            verified_candidates.append({
                "frame_id": result["frame_id"],
                "image_file": result["image_file"],
                "tile_relative_path": best["tile_relative_path"],
                "tile_center_lat_lon_degrees": [tile_lat, tile_lon],
                "rotation_ccw_degrees": best["rotation_ccw_degrees"],
                "ransac_inliers": best["ransac_inliers"],
            })

    verification_payload = {
        "experiment": "Unity -> EarthLoc -> SuperPoint/LightGlue -> RANSAC",
        "minimum_inliers_for_fix": args.minimum_inliers,
        "frames": verification_frames,
    }
    write_json(
        args.output_directory / "geometric_verification.json",
        verification_payload,
    )

    errors = [row["candidate_error_km"] for row in truth_report]
    truth_payload = {
        "meaning": (
            "Evaluation against Unity simulator truth only. "
            "Rejected candidates are not navigation fixes."
        ),
        "mean_candidate_error_km": sum(errors) / len(errors),
        "minimum_candidate_error_km": min(errors),
        "maximum_candidate_error_km": max(errors),
        "frames": truth_report,
    }
    write_json(
        args.output_directory / "candidate_truth_error_report.json",
        truth_payload,
    )

    write_json(
        args.output_directory / "verified_visual_candidates.json",
        {
            "meaning": (
                "Only geometrically verified candidates. "
                "An empty list means no visual navigation measurement was accepted."
            ),
            "verified_candidates": verified_candidates,
        },
    )

    print("\n=== Final Summary ===")
    print("Frames evaluated:", len(verification_frames))
    print("Verified visual fixes:", len(verified_candidates))
    print(f"Mean retrieved-candidate error: {sum(errors) / len(errors):,.1f} km")
    print("Results:", args.output_directory)


if __name__ == "__main__":
    main()