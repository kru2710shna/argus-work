#!/usr/bin/env python3
"""
Recommendation-only VINSat Target Hunter replay.

Uses prerecorded VINSat landmark detections:
[time_index, landmark_longitude, landmark_latitude, pixel_x, pixel_y, confidence]

This is a planner integration test. It does not run 11R.pt, generate an
Argus flight command, transmit an uplink, or claim real Argus telemetry.
"""

from __future__ import annotations

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

DETECTION_CONFIDENCE_MIN = 0.50
LOCALIZATION_MEAN_CONFIDENCE_MIN = 0.60
LOCALIZATION_LANDMARK_COUNT_MIN = 3
DEMO_PRIORITIES = (1.00, 0.75, 0.50)


def load_cells(csv_path: Path, region: str) -> list[dict]:
    cells = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            left = float(row["Top-Left Longitude"])
            top = float(row["Top-Left Latitude"])
            right = float(row["Bottom-Right Longitude"])
            bottom = float(row["Bottom-Right Latitude"])
            cells.append(
                {
                    "target_id": f"{region}_cell_{index:03d}",
                    "min_lon": min(left, right),
                    "max_lon": max(left, right),
                    "min_lat": min(bottom, top),
                    "max_lat": max(bottom, top),
                    "center_lon": float(row["Centroid Longitude"]),
                    "center_lat": float(row["Centroid Latitude"]),
                }
            )
    return cells


def load_episode(archive: Path, episode: str) -> tuple[np.ndarray, np.ndarray]:
    pose_name = f"tmp_pose/{episode}_orbit_ecef_zyxvecs.npy"
    detection_name = f"tmp_dets/{episode}_all_detections.npy"

    with ZipFile(archive) as z:
        missing = [name for name in (pose_name, detection_name) if name not in z.namelist()]
        if missing:
            raise ValueError(f"Episode {episode} is unavailable: {missing}")

        poses = np.load(BytesIO(z.read(pose_name)), allow_pickle=True)
        detections = np.load(BytesIO(z.read(detection_name)), allow_pickle=True)

    if poses.ndim != 2 or poses.shape[1] != 12:
        raise ValueError(f"Unexpected pose format: {poses.shape}")
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError(f"Unexpected detection format: {detections.shape}")

    return poses, detections


def target_mask(detections: np.ndarray, cell: dict) -> np.ndarray:
    longitude = detections[:, 1]
    latitude = detections[:, 2]
    confidence = detections[:, 5]

    return (
        (longitude >= cell["min_lon"])
        & (longitude <= cell["max_lon"])
        & (latitude >= cell["min_lat"])
        & (latitude <= cell["max_lat"])
        & (confidence >= DETECTION_CONFIDENCE_MIN)
    )


def choose_demo_targets(cells: list[dict], detections: np.ndarray) -> list[dict]:
    """
    Calibration-only convenience for this first replay:
    choose cells with the strongest archived visual support.

    In deployment, replace this with operator-supplied AOIs and priorities.
    """
    ranked = []
    for cell in cells:
        mask = target_mask(detections, cell)
        if not mask.any():
            continue

        times = detections[mask, 0].astype(int)
        confidences = detections[mask, 5]
        ranked.append(
            {
                **cell,
                "calibration_unique_visible_seconds": int(np.unique(times).size),
                "calibration_landmark_detections": int(mask.sum()),
                "calibration_mean_detection_confidence": round(float(confidences.mean()), 4),
            }
        )

    ranked.sort(
        key=lambda row: (
            -row["calibration_unique_visible_seconds"],
            -row["calibration_mean_detection_confidence"],
            row["target_id"],
        )
    )

    if len(ranked) < 3:
        raise RuntimeError(
            f"Only {len(ranked)} 11R cells had detection support; expected at least 3."
        )

    selected = []
    for priority, row in zip(DEMO_PRIORITIES, ranked[:3]):
        selected.append({**row, "operator_priority": priority})

    return selected


def fixed_operator_targets(cells: list[dict], target_ids: list[str]) -> list[dict]:
    """Use operator-selected targets instead of replay-calibrated targets."""
    by_id = {cell["target_id"]: cell for cell in cells}
    missing = [target_id for target_id in target_ids if target_id not in by_id]
    if missing:
        raise ValueError(f"Unknown operator target IDs: {missing}")
    if len(target_ids) > len(DEMO_PRIORITIES):
        raise ValueError(
            f"At most {len(DEMO_PRIORITIES)} targets are supported in this demo"
        )

    return [
        {**by_id[target_id], "operator_priority": priority}
        for target_id, priority in zip(target_ids, DEMO_PRIORITIES)
    ]


def localization_by_second(detections: np.ndarray) -> dict[int, dict]:
    """
    Evidence is all confident landmarks in the camera view at a time step,
    not only landmarks inside the requested target cell.
    """
    valid = detections[detections[:, 5] >= DETECTION_CONFIDENCE_MIN]
    times = valid[:, 0].astype(int)

    quality = {}
    for second in np.unique(times):
        rows = valid[times == second]
        count = int(rows.shape[0])
        mean_confidence = float(rows[:, 5].mean())

        count_score = min(1.0, count / 8.0)
        confidence_score = min(
            1.0,
            max(
                0.0,
                (mean_confidence - DETECTION_CONFIDENCE_MIN)
                / (1.0 - DETECTION_CONFIDENCE_MIN),
            ),
        )
        quality_score = 0.5 * count_score + 0.5 * confidence_score

        quality[int(second)] = {
            "landmark_count": count,
            "mean_confidence": round(mean_confidence, 4),
            "quality_score": round(quality_score, 4),
            "localization_usable": (
                count >= LOCALIZATION_LANDMARK_COUNT_MIN
                and mean_confidence >= LOCALIZATION_MEAN_CONFIDENCE_MIN
            ),
        }

    return quality


def rank_opportunities(
    targets: list[dict],
    detections: np.ndarray,
    localization: dict[int, dict],
    deadline_s: int,
    initial_battery_pct: float,
    initial_storage_mb: float,
) -> list[dict]:
    opportunities = []

    # Explicit replay assumptions, not flight telemetry.
    battery_drain_pct_per_second = 0.0005
    capture_storage_cost_mb = 5.0

    for target in targets:
        mask = target_mask(detections, target)
        visible_seconds = sorted(set(detections[mask, 0].astype(int).tolist()))

        for second in visible_seconds:
            if second > deadline_s:
                continue

            loc = localization.get(second)
            if not loc or not loc["localization_usable"]:
                continue

            battery_before_capture = (
                initial_battery_pct - battery_drain_pct_per_second * second
            )
            storage_after_capture = initial_storage_mb - capture_storage_cost_mb

            if battery_before_capture < 35.0 or storage_after_capture < 300.0:
                continue

            urgency = max(0.0, 1.0 - second / deadline_s)
            score = (
                0.50 * target["operator_priority"]
                + 0.30 * loc["quality_score"]
                + 0.20 * urgency
            )

            opportunities.append(
                {
                    "target_id": target["target_id"],
                    "target_center_lonlat": [
                        round(target["center_lon"], 6),
                        round(target["center_lat"], 6),
                    ],
                    "pose_index": second,
                    "time_after_episode_start_s": second,
                    "time_after_episode_start_hms": (
                        f"{second // 3600:02d}:"
                        f"{(second % 3600) // 60:02d}:"
                        f"{second % 60:02d}"
                    ),
                    "operator_priority": target["operator_priority"],
                    "localization": loc,
                    "resource_replay": {
                        "battery_pct_before_capture": round(battery_before_capture, 3),
                        "storage_mb_after_capture": round(storage_after_capture, 3),
                    },
                    "urgency_score": round(urgency, 4),
                    "mission_score": round(score, 4),
                    "visibility_evidence": (
                        "One or more prerecorded VINSat landmarks fall inside "
                        "the target cell at this time."
                    ),
                }
            )

    return sorted(
        opportunities,
        key=lambda row: (
            -row["mission_score"],
            row["time_after_episode_start_s"],
            row["target_id"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vinsat-root",
        type=Path,
        default=Path.home() / "argus-work" / "VINSat",
    )
    parser.add_argument("--region", default="11R")
    parser.add_argument("--episode", default="00092")
    parser.add_argument(
        "--deadline-s",
        type=int,
        default=7200,
        help="Mission deadline measured from replay start; default is 2 hours.",
    )
    parser.add_argument("--battery-pct", type=float, default=72.0)
    parser.add_argument("--storage-mb", type=float, default=1200.0)
    parser.add_argument(
        "--target-ids",
        nargs="+",
        help="Fixed operator-selected target IDs; skips calibration-based target selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/vinsat_target_hunter_11R.json"),
    )
    args = parser.parse_args()

    vinsat = args.vinsat_root.expanduser().resolve()
    cells = load_cells(
        vinsat / "sim" / "landmark_csvs" / f"{args.region}_top_salient.csv",
        args.region,
    )
    poses, detections = load_episode(
        vinsat / "estimation" / "landmarks" / "camera_ready" / "dets_and_poses.zip",
        args.episode,
    )

    target_source = (
        "operator_fixed"
        if args.target_ids
        else "calibration_selected_for_integration_demo"
    )
    targets = (
        fixed_operator_targets(cells, args.target_ids)
        if args.target_ids
        else choose_demo_targets(cells, detections)
    )
    localization = localization_by_second(detections)
    opportunities = rank_opportunities(
        targets=targets,
        detections=detections,
        localization=localization,
        deadline_s=args.deadline_s,
        initial_battery_pct=args.battery_pct,
        initial_storage_mb=args.storage_mb,
    )

    result = {
        "mission": {
            "name": "VINSat 11R Target Hunter replay",
            "episode": args.episode,
            "duration_s": int(poses.shape[0] - 1),
            "sample_rate_hz": 1,
            "deadline_s": args.deadline_s,
            "recommendation_only": True,
        },
        "truthfulness": {
            "vision_source": "prerecorded VINSat landmark detections",
            "vision_model_executed_in_this_run": False,
            "target_source": target_source,
            "available_region_model": str(vinsat / "models" / f"{args.region}.pt"),
            "resource_source": "explicit replay assumptions; not Argus telemetry",
            "flight_command_generated": False,
            "uplink_sent": False,
        },
        "operator_targets": targets,
        "candidate_opportunity_count": len(opportunities),
        "top_opportunities": opportunities[:20],
        "recommendation": (
            {
                "action": "PROPOSE_CAPTURE",
                **opportunities[0],
                "requires_operator_approval": True,
            }
            if opportunities
            else {
                "action": "WAIT_OR_SKIP",
                "reason": (
                    "No target was simultaneously visible, localization-usable, "
                    "resource-safe, and before the deadline."
                ),
                "requires_operator_approval": True,
            }
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("=== VINSAT TARGET-HUNTER REPLAY ===")
    print("Region:", args.region)
    print("Episode:", args.episode)
    print("Replay duration:", poses.shape[0] - 1, "seconds")
    print("Demo targets:", [target["target_id"] for target in targets])
    print("Feasible opportunities:", len(opportunities))
    print("\nRecommendation:")
    print(json.dumps(result["recommendation"], indent=2))
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
