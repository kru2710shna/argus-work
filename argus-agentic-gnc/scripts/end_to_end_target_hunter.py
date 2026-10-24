#!/usr/bin/env python3
"""
End-to-end, recommendation-only Target Hunter replay.

Current real replay inputs:
  - VINSat prerecorded pose/detection archive: time, landmark lon/lat, confidence
  - operator region/city-scale target area, auto-divided into a grid
  - replay resource state: battery and storage

Optional future tool inputs:
  - EarthLoc fix JSON
  - image-evidence JSON from a VLM/change detector

Safety:
  - No uplink, spacecraft command, or image capture is sent.
  - 17R images are inventoried but never used for 11R inference because they
    are geographically misaligned with the 11R replay.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vinsat-root", type=Path, required=True)
    parser.add_argument("--region", default="11R")
    parser.add_argument("--episode", default="00105")
    parser.add_argument("--output", type=Path, required=True)

    # Auto-generated operator test area: dense cluster of VINSat detections.
    parser.add_argument("--grid-side", type=int, default=5,
                        help="5 means 25 initial city-grid targets")
    parser.add_argument("--cell-deg", type=float, default=0.04,
                        help="Grid-cell width/height in degrees")
    parser.add_argument("--city-name", default=None)
    parser.add_argument("--city-bbox", nargs=4, type=float, default=None,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))

    parser.add_argument("--deadline-s", type=int, default=7200)
    parser.add_argument("--max-captures", type=int, default=5)
    parser.add_argument("--cooldown-s", type=int, default=120)
    parser.add_argument("--outage", default=None, metavar="START:END")

    parser.add_argument("--battery-pct", type=float, default=72.0)
    parser.add_argument("--storage-mb", type=float, default=1200.0)
    parser.add_argument("--min-battery-pct", type=float, default=35.0)
    parser.add_argument("--min-storage-mb", type=float, default=300.0)
    parser.add_argument("--battery-decay-per-s", type=float, default=0.0005)
    parser.add_argument("--battery-per-capture", type=float, default=0.2)
    parser.add_argument("--storage-per-capture-mb", type=float, default=5.0)

    parser.add_argument("--min-landmarks", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.50)

    parser.add_argument("--earthloc-fixes", type=Path, default=None)
    parser.add_argument("--require-earthloc", action="store_true")
    parser.add_argument("--image-evidence", type=Path, default=None)
    return parser.parse_args()


def parse_outage(value):
    if not value:
        return None
    try:
        start, end = (int(x) for x in value.split(":"))
    except ValueError as error:
        raise SystemExit("--outage must be START:END, for example 700:780") from error
    if start < 0 or end < start:
        raise SystemExit("Invalid outage range.")
    return [start, end]


def load_json_list(path: Path | None, label: str):
    if path is None:
        return []
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise SystemExit(f"{label} must contain a JSON list.")
    return value


def archive_paths(root: Path, episode: str):
    archive = root / "estimation" / "landmarks" / "camera_ready" / "dets_and_poses.zip"
    if not archive.exists():
        raise SystemExit(f"Missing VINSat archive: {archive}")

    with ZipFile(archive) as z:
        names = z.namelist()
        det_name = next(
            (n for n in names if n.endswith(f"{episode}_all_detections.npy")),
            None,
        )
        pose_name = next(
            (n for n in names if Path(n).name.startswith(f"{episode}_")
             and n.endswith("_orbit_ecef_zyxvecs.npy")),
            None,
        )
        if not det_name or not pose_name:
            raise SystemExit(f"Could not find replay episode {episode} in {archive}")
        detections = np.load(BytesIO(z.read(det_name)), allow_pickle=False)
        poses = np.load(BytesIO(z.read(pose_name)), allow_pickle=False)

    if detections.ndim != 2 or detections.shape[1] < 6:
        raise SystemExit(f"Unexpected detection schema: {detections.shape}")

    return detections.astype(float), poses.astype(float), archive


def image_inventory(root: Path, region: str):
    dataset_root = root / "Dataset"
    answer = {"dataset_root": str(dataset_root), "directories": []}

    if not dataset_root.exists():
        answer["warning"] = "VINSat/Dataset folder was not found."
        return answer

    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        tif_files = list(folder.glob("*.tif"))
        region_matches = [p.name for p in tif_files if region.lower() in p.name.lower()]
        answer["directories"].append({
            "name": folder.name,
            "tif_count": len(tif_files),
            "files_matching_replay_region": len(region_matches),
        })

    answer["aligned_images_available_for_this_replay"] = any(
        x["files_matching_replay_region"] > 0 for x in answer["directories"]
    )
    return answer


def mgrs_region_bbox(region):
    """Approximate longitude/latitude extent for an MGRS UTM zone + latitude band."""
    region = region.strip().upper()
    bands = "CDEFGHJKLMNPQRSTUVWX"

    if len(region) < 2 or region[-1] not in bands:
        raise ValueError(f"Expected an MGRS region such as 11R, got: {region!r}")

    zone = int(region[:-1])
    band = region[-1]
    if not 1 <= zone <= 60:
        raise ValueError(f"Invalid MGRS UTM zone: {zone}")

    band_index = bands.index(band)
    min_lat = -80 + 8 * band_index
    max_lat = min_lat + (12 if band == "X" else 8)
    min_lon = -180 + 6 * (zone - 1)
    max_lon = min_lon + 6
    return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


def build_frames(detections, deadline_s, min_confidence, region):
    """Frame[t] contains confident detections truly inside the requested MGRS region."""
    min_lon, min_lat, max_lon, max_lat = mgrs_region_bbox(region)
    frames = defaultdict(list)

    for row in detections:
        t, lon, lat, _, _, confidence = row[:6]
        second = int(round(t))

        if not (
            0 <= second <= deadline_s
            and np.isfinite([lon, lat, confidence]).all()
            and confidence >= min_confidence
            and min_lon <= lon <= max_lon
            and min_lat <= lat <= max_lat
        ):
            continue

        frames[second].append({
            "lon": float(lon),
            "lat": float(lat),
            "confidence": float(confidence),
        })

    return dict(sorted(frames.items()))

def auto_dense_bbox(frames, side, cell_deg):
    """
    Find the densest side x side geographic grid window.
    This creates a reproducible test AOI, not a real named city boundary.
    """
    counts = Counter()
    for observations in frames.values():
        for item in observations:
            x = math.floor(item["lon"] / cell_deg)
            y = math.floor(item["lat"] / cell_deg)
            counts[(x, y)] += 1

    if not counts:
        raise SystemExit("No usable landmark detections after confidence/deadline filtering.")

    candidate_origins = set()
    for x, y in counts:
        for dx in range(side):
            for dy in range(side):
                candidate_origins.add((x - dx, y - dy))

    def window_score(origin):
        ox, oy = origin
        total = 0
        occupied = 0
        for dx in range(side):
            for dy in range(side):
                value = counts[(ox + dx, oy + dy)]
                total += value
                occupied += value > 0
        return total, occupied

    origin = max(candidate_origins, key=lambda item: window_score(item))
    ox, oy = origin
    return [
        round(ox * cell_deg, 8),
        round(oy * cell_deg, 8),
        round((ox + side) * cell_deg, 8),
        round((oy + side) * cell_deg, 8),
    ]


def build_targets(city_name, bbox, side):
    min_lon, min_lat, max_lon, max_lat = bbox
    targets = []
    for row in range(side):
        for col in range(side):
            left = min_lon + (max_lon - min_lon) * col / side
            right = min_lon + (max_lon - min_lon) * (col + 1) / side
            bottom = min_lat + (max_lat - min_lat) * row / side
            top = min_lat + (max_lat - min_lat) * (row + 1) / side
            target_id = f"{city_name.lower().replace(' ', '_')}_r{row + 1}c{col + 1}"
            targets.append({
                "target_id": target_id,
                "level": "city_grid_cell",
                "bbox_lonlat": [left, bottom, right, top],
                "center_lonlat": [round((left + right) / 2, 6), round((bottom + top) / 2, 6)],
                "operator_priority": 1.0,
                "state": "PROPOSED",
                "source": "operator_grid_or_auto_cluster",
            })
    return targets


def inside(item, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= item["lon"] <= max_lon and min_lat <= item["lat"] <= max_lat


def nearest_earthloc_fix(fixes, t, max_gap=30):
    candidates = [
        x for x in fixes
        if abs(int(x.get("time_s", -10**9)) - t) <= max_gap
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(int(x["time_s"]) - t))


def nearest_image_evidence(records, target_id, t, max_gap=180):
    candidates = [
        x for x in records
        if x.get("target_id") == target_id
        and abs(int(x.get("time_s", -10**9)) - t) <= max_gap
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(int(x["time_s"]) - t))


def resource_state(args, capture_count, time_s):
    return {
        "battery_pct_before_capture": round(
            args.battery_pct - args.battery_decay_per_s * time_s
            - args.battery_per_capture * capture_count,
            3,
        ),
        "storage_mb_before_capture": round(
            args.storage_mb - args.storage_per_capture_mb * capture_count,
            3,
        ),
    }


def safe_resources(args, state):
    return (
        state["battery_pct_before_capture"] >= args.min_battery_pct
        and state["storage_mb_before_capture"] >= args.min_storage_mb
    )


def is_outage(outage, t):
    return outage is not None and outage[0] <= t <= outage[1]


def localization_at_time(args, observations, earthloc):
    if len(observations) < args.min_landmarks:
        return None

    vinsat_quality = sum(x["confidence"] for x in observations) / len(observations)
    if vinsat_quality < args.min_confidence:
        return None

    if args.require_earthloc:
        if not earthloc or not earthloc.get("usable", False):
            return None
        quality = min(vinsat_quality, float(earthloc.get("confidence", 0.0)))
        source = "VINSat_plus_EarthLoc"
    elif earthloc and earthloc.get("usable", False):
        quality = (vinsat_quality + float(earthloc.get("confidence", 0.0))) / 2
        source = "VINSat_plus_optional_EarthLoc"
    else:
        quality = vinsat_quality
        source = "prerecorded_VINSat_landmark_detections"

    return {
        "localization_usable": True,
        "quality_score": round(float(quality), 4),
        "source": source,
        "landmark_count": len(observations),
        "mean_landmark_confidence": round(float(vinsat_quality), 4),
        "earthloc": earthloc,
    }


def discover_neighbors(parent, region_bbox, directions, targets):
    """Create new adjacent targets only when image evidence explicitly requests it."""
    known = {target["target_id"] for target in targets}
    min_lon, min_lat, max_lon, max_lat = parent["bbox_lonlat"]
    width, height = max_lon - min_lon, max_lat - min_lat
    rmin_lon, rmin_lat, rmax_lon, rmax_lat = region_bbox
    created = []

    for direction in directions:
        if direction == "west":
            bbox = [min_lon - width, min_lat, min_lon, max_lat]
        elif direction == "east":
            bbox = [max_lon, min_lat, max_lon + width, max_lat]
        elif direction == "south":
            bbox = [min_lon, min_lat - height, max_lon, min_lat]
        elif direction == "north":
            bbox = [min_lon, max_lat, max_lon, max_lat + height]
        else:
            continue

        if bbox[0] < rmin_lon or bbox[1] < rmin_lat or bbox[2] > rmax_lon or bbox[3] > rmax_lat:
            continue

        target_id = f"{parent['target_id']}_discovered_{direction}"
        if target_id in known:
            continue

        neighbor = {
            "target_id": target_id,
            "level": "discovered_adjacent_cell",
            "bbox_lonlat": bbox,
            "center_lonlat": [round((bbox[0] + bbox[2]) / 2, 6),
                               round((bbox[1] + bbox[3]) / 2, 6)],
            "operator_priority": round(parent["operator_priority"] * 0.9, 3),
            "state": "PROPOSED",
            "source": "image_evidence_frontier_expansion",
            "parent_target_id": parent["target_id"],
        }
        targets.append(neighbor)
        known.add(target_id)
        created.append(target_id)

    return created


def all_opportunities(args, targets, frames, start_time, capture_count, outage, earthloc_fixes):
    candidates = []

    for t, observations in frames.items():
        if t < start_time or t > args.deadline_s or is_outage(outage, t):
            continue

        earthloc = nearest_earthloc_fix(earthloc_fixes, t)
        localization = localization_at_time(args, observations, earthloc)
        if localization is None:
            continue

        resources = resource_state(args, capture_count, t)
        if not safe_resources(args, resources):
            continue

        urgency = max(0.0, 1.0 - t / max(1, args.deadline_s))

        for target in targets:
            if target["state"] != "PROPOSED":
                continue
            hits = [x for x in observations if inside(x, target["bbox_lonlat"])]
            if not hits:
                continue

            score = (
                0.50 * target["operator_priority"]
                + 0.30 * localization["quality_score"]
                + 0.20 * urgency
            )
            candidates.append({
                "target_id": target["target_id"],
                "time_s": t,
                "localization": localization,
                "resources": resources,
                "target_landmark_hits": len(hits),
                "urgency_score": round(urgency, 4),
                "mission_score": round(score, 4),
            })

    return candidates


def fixed_priority_baseline(args, initial_targets, frames, outage, earthloc_fixes):
    """Baseline: visit targets in fixed ID order; it does not rank mission score."""
    targets = deepcopy(initial_targets)
    events = []
    next_time = 0

    for capture_index in range(args.max_captures):
        selected = None
        for target in sorted(targets, key=lambda x: x["target_id"]):
            if target["state"] != "PROPOSED":
                continue
            candidates = all_opportunities(
                args, [target], frames, next_time, len(events), outage, earthloc_fixes
            )
            if candidates:
                selected = min(candidates, key=lambda x: x["time_s"])
                break

        if selected is None:
            break

        target = next(x for x in targets if x["target_id"] == selected["target_id"])
        target["state"] = "CAPTURE_REPLAYED_PENDING_IMAGE_ANALYSIS"
        selected["action"] = "FIXED_PRIORITY_CAPTURE"
        events.append(selected)
        next_time = selected["time_s"] + args.cooldown_s

    return {
        "strategy": "fixed_target_ID_order",
        "capture_count": len(events),
        "events": events,
    }


def agent_replay(args, targets, frames, region_bbox, outage, earthloc_fixes, image_records):
    events = []
    next_time = 0
    discovered = []

    for step in range(1, args.max_captures + 1):
        candidates = all_opportunities(
            args, targets, frames, next_time, len(events), outage, earthloc_fixes
        )
        if not candidates:
            events.append({
                "step": step,
                "action": "WAIT_OR_SKIP",
                "reason": "No pending target was simultaneously visible, localization-usable, resource-safe, and outside outage windows.",
                "requires_operator_approval": True,
            })
            break

        # Time-aware best-first scheduling: not BFS/DFS.
        selected = max(
            candidates,
            key=lambda x: (
                x["mission_score"],
                x["target_landmark_hits"],
                -x["time_s"],
                x["target_id"],
            ),
        )
        target = next(x for x in targets if x["target_id"] == selected["target_id"])
        t = selected["time_s"]

        image = nearest_image_evidence(image_records, target["target_id"], t)
        after = {
            "battery_pct_after_capture": round(
                selected["resources"]["battery_pct_before_capture"] - args.battery_per_capture, 3
            ),
            "storage_mb_after_capture": round(
                selected["resources"]["storage_mb_before_capture"] - args.storage_per_capture_mb, 3
            ),
        }

        target["state"] = (
            "CAPTURE_REPLAYED_IMAGE_ANALYZED"
            if image else "CAPTURE_REPLAYED_PENDING_IMAGE_ANALYSIS"
        )

        new_targets = []
        if image and image.get("event_detected", False):
            new_targets = discover_neighbors(
                target,
                region_bbox,
                image.get("edge_directions", []),
                targets,
            )
            discovered.extend(new_targets)

        events.append({
            "step": step,
            "action": "PROPOSE_CAPTURE",
            "target_id": target["target_id"],
            "target_level": target["level"],
            "target_bbox_lonlat": target["bbox_lonlat"],
            "time_after_episode_start_s": t,
            "time_after_episode_start_hms": f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}",
            "operator_priority": target["operator_priority"],
            "localization": selected["localization"],
            "resource_replay": {**selected["resources"], **after},
            "target_landmark_hits": selected["target_landmark_hits"],
            "urgency_score": selected["urgency_score"],
            "mission_score": selected["mission_score"],
            "visibility_evidence": "One or more prerecorded VINSat landmarks fall inside this target.",
            "image_evidence": image or {
                "status": "not_available",
                "reason": "No geographically/time-aligned image-evidence record was supplied."
            },
            "new_targets_created": new_targets,
            "resulting_target_state": target["state"],
            "requires_operator_approval": True,
            "uplink_sent": False,
            "flight_command_generated": False,
        })
        next_time = t + args.cooldown_s

    return events, discovered


def main():
    args = parse_args()
    outage = parse_outage(args.outage)
    detections, poses, archive = archive_paths(args.vinsat_root, args.episode)
    replay_region_bbox = mgrs_region_bbox(args.region)
    frames = build_frames(
        detections, args.deadline_s, args.min_confidence, args.region
    )

    if args.city_bbox:
        city_bbox = args.city_bbox
        target_source = "operator_provided_bbox"
    else:
        city_bbox = auto_dense_bbox(frames, args.grid_side, args.cell_deg)
        target_source = "auto_selected_dense_VINSat_test_cluster"

    city_name = args.city_name or f"{args.region}_auto_cluster"
    region_bbox = replay_region_bbox  # Allow validated frontier expansion within the full replay region.
    targets = build_targets(city_name, city_bbox, args.grid_side)
    initial_targets = deepcopy(targets)

    earthloc_fixes = load_json_list(args.earthloc_fixes, "EarthLoc fixes")
    image_records = load_json_list(args.image_evidence, "Image evidence")

    baseline = fixed_priority_baseline(
        args, initial_targets, frames, outage, earthloc_fixes
    )
    events, discovered = agent_replay(
        args, targets, frames, region_bbox, outage, earthloc_fixes, image_records
    )

    primary = next(
        (event for event in events if event["action"] == "PROPOSE_CAPTURE"),
        events[0],
    )
    inventory = image_inventory(args.vinsat_root, args.region)

    result = {
        "mission": {
            "name": "End-to-end Target Hunter replay",
            "recommendation_only": True,
            "region": args.region,
            "replay_region_bbox_lonlat": replay_region_bbox,
            "episode": args.episode,
            "replay_duration_s": int(poses.shape[0] - 1),
            "decision_deadline_s": args.deadline_s,
            "city_name": city_name,
            "city_bbox_lonlat": city_bbox,
            "grid_shape": [args.grid_side, args.grid_side],
            "initial_target_count": len(initial_targets),
            "target_source": target_source,
        },
        "truthfulness": {
            "vinsat_pose_and_visibility_source": str(archive),
            "vinsat_model_available": str(args.vinsat_root / "models" / f"{args.region}.pt"),
            "vinsat_model_executed": False,
            "earthloc_executed": bool(earthloc_fixes),
            "image_model_executed": bool(image_records),
            "camera_footprint": "landmark-in-target proxy; not calibrated Argus camera footprint",
            "resource_source": "replay assumptions; not decoded Argus telemetry",
            "uplink_sent": False,
            "flight_command_generated": False,
        },
        "image_dataset_inventory": inventory,
        "policy": {
            "planner": "time_aware_best_first_priority_search",
            "not_used": "BFS/DFS are not the final policy; they are possible comparison baselines.",
            "score": "0.50*operator_priority + 0.30*localization_quality + 0.20*urgency",
            "min_landmarks": args.min_landmarks,
            "min_confidence": args.min_confidence,
            "min_battery_pct": args.min_battery_pct,
            "min_storage_mb": args.min_storage_mb,
            "outage_s": outage,
        },
        "targets": targets,
        "agent_events": events,
        "primary_recommendation": primary,
        "fixed_priority_baseline": baseline,
        "metrics": {
            "agent_capture_proposals": sum(x["action"] == "PROPOSE_CAPTURE" for x in events),
            "fixed_priority_captures": baseline["capture_count"],
            "discovered_neighbor_targets": discovered,
            "final_target_count": len(targets),
            "frame_count_with_confident_landmarks": len(frames),
        },
        "safety": {
            "operator_approval_required": True,
            "uplink_sent": False,
            "flight_command_generated": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("=== END-TO-END TARGET HUNTER REPLAY ===")
    print("Region:", args.region, "| episode:", args.episode)
    print("Targets:", len(initial_targets), "| grid:", f"{args.grid_side}x{args.grid_side}")
    print("City-scale AOI:", city_bbox)
    print("Confident replay frames:", len(frames))
    print("Agent captures:", result["metrics"]["agent_capture_proposals"])
    print("Fixed baseline captures:", baseline["capture_count"])
    print("\nPrimary recommendation:")
    print(json.dumps(primary, indent=2))
    print("\nImage alignment status:",
          inventory.get("aligned_images_available_for_this_replay", False))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
