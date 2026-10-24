#!/usr/bin/env python3
"""
Offline, recommendation-only VINSat City Target Hunter.

Mission:
  Operator provides a broad region and a city-scale AOI.
  The script divides that AOI into smaller cells and uses prerecorded VINSat
  landmark detections to determine when a cell is visible and localization is
  usable. It then proposes capture opportunities subject to battery, storage,
  deadline, and optional simulated failures.

Truthfulness:
  - No Argus telemetry, live camera image, EarthLoc inference, uplink, or
    flight command is used here.
  - VINSat landmark detections are prerecorded replay evidence.
  - --replay-approval simulates an operator accepting a proposal only for
    offline state-machine testing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def require_valid_bbox(values: list[float], label: str) -> tuple[float, float, float, float]:
    west, south, east, north = values
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"Invalid {label}: expected WEST SOUTH EAST NORTH")
    return west, south, east, north


def point_in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def bbox_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def parse_outage(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        start, end = (int(x) for x in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--outage must be START_S:END_S") from error
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("Invalid outage range")
    return start, end


def zip_member(zip_file: ZipFile, suffix: str) -> str:
    matches = [name for name in zip_file.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one archive member ending in {suffix!r}; found {matches}"
        )
    return matches[0]


def load_vinsat_replay(vinsat_root: Path, episode: str) -> tuple[dict[int, list[tuple[float, float, float]]], int]:
    archive = vinsat_root / "estimation" / "landmarks" / "camera_ready" / "dets_and_poses.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing VINSat archive: {archive}")

    with ZipFile(archive) as zip_file:
        detections_name = zip_member(zip_file, f"{episode}_all_detections.npy")
        poses_name = zip_member(zip_file, f"{episode}_orbit_ecef_zyxvecs.npy")

        detections = np.load(BytesIO(zip_file.read(detections_name)), allow_pickle=False)
        poses = np.load(BytesIO(zip_file.read(poses_name)), allow_pickle=False)

    if detections.ndim != 2 or detections.shape[1] < 6:
        raise ValueError(f"Unexpected detection shape: {detections.shape}")
    if poses.ndim != 2 or poses.shape[0] < 2:
        raise ValueError(f"Unexpected pose shape: {poses.shape}")

    # Detection schema inferred from VINSat replay archive:
    # [time_index, landmark_lon, landmark_lat, pixel_x, pixel_y, confidence]
    detections_by_second: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    duration_s = int(poses.shape[0] - 1)

    for row in detections:
        if not np.all(np.isfinite(row[:6])):
            continue

        second = int(round(float(row[0])))
        lon, lat, confidence = float(row[1]), float(row[2]), float(row[5])

        if 0 <= second <= duration_s and -180 <= lon <= 180 and -90 <= lat <= 90:
            detections_by_second[second].append((lon, lat, confidence))

    return detections_by_second, duration_s


def load_earthloc_fixes(path: Path | None) -> dict[int, dict]:
    """
    Optional future adapter.

    Expected JSON:
    {
      "fixes": [
        {"time_s": 745, "lon": -115.82, "lat": 29.85,
         "confidence": 0.82, "status": "fix"}
      ]
    }

    No file is passed in the current replay, so EarthLoc is not executed.
    """
    if path is None:
        return {}

    raw = json.loads(path.read_text())
    rows = raw["fixes"] if isinstance(raw, dict) else raw
    fixes: dict[int, dict] = {}

    for row in rows:
        second = int(row["time_s"])
        fix = {
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "confidence": float(row["confidence"]),
            "status": str(row.get("status", "fix")).lower(),
        }
        if not (-180 <= fix["lon"] <= 180 and -90 <= fix["lat"] <= 90):
            raise ValueError(f"Invalid EarthLoc coordinate at t={second}")
        if not (0 <= fix["confidence"] <= 1):
            raise ValueError(f"Invalid EarthLoc confidence at t={second}")

        previous = fixes.get(second)
        if previous is None or fix["confidence"] > previous["confidence"]:
            fixes[second] = fix

    return fixes


def build_city_cells(
    city_name: str,
    city_bbox: tuple[float, float, float, float],
    rows: int,
    cols: int,
    city_priority: float,
    focus_cell: str | None,
) -> list[dict]:
    west, south, east, north = city_bbox
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in city_name).strip("_")
    cells: list[dict] = []

    for row in range(rows):
        for col in range(cols):
            cell_west = west + (east - west) * col / cols
            cell_east = west + (east - west) * (col + 1) / cols
            cell_south = south + (north - south) * row / rows
            cell_north = south + (north - south) * (row + 1) / rows
            short_id = f"r{row + 1}c{col + 1}"
            target_id = f"{slug}_{short_id}"

            cells.append(
                {
                    "target_id": target_id,
                    "short_id": short_id,
                    "level": "city_grid_cell",
                    "bbox_lonlat": [cell_west, cell_south, cell_east, cell_north],
                    "center_lonlat": [
                        round((cell_west + cell_east) / 2, 6),
                        round((cell_south + cell_north) / 2, 6),
                    ],
                    "operator_priority": city_priority,
                }
            )

    if focus_cell:
        matches = [
            cell
            for cell in cells
            if cell["short_id"] == focus_cell or cell["target_id"] == focus_cell
        ]
        if len(matches) != 1:
            known = ", ".join(cell["short_id"] for cell in cells)
            raise ValueError(f"Unknown --focus-cell {focus_cell!r}. Choose one of: {known}")

        for cell in cells:
            cell["operator_priority"] = city_priority * (1.0 if cell in matches else 0.55)

    return cells


def localization_evidence(
    detections: list[tuple[float, float, float]],
    earthloc_fix: dict | None,
    in_outage: bool,
    min_landmarks: int,
    min_landmark_confidence: float,
    min_earthloc_confidence: float,
    require_earthloc: bool,
) -> dict:
    if in_outage:
        return {
            "localization_usable": False,
            "quality_score": 0.0,
            "source": "injected_replay_outage",
            "landmark_count": 0,
            "mean_landmark_confidence": None,
            "earthloc": None,
        }

    confident = [confidence for _, _, confidence in detections if confidence >= min_landmark_confidence]
    vinsat_usable = len(confident) >= min_landmarks
    vinsat_quality = float(np.mean(confident)) if confident else 0.0

    earthloc_usable = False
    earthloc_quality = 0.0
    if earthloc_fix is not None:
        earthloc_usable = (
            earthloc_fix["status"] == "fix"
            and earthloc_fix["confidence"] >= min_earthloc_confidence
        )
        earthloc_quality = earthloc_fix["confidence"] if earthloc_usable else 0.0

    if require_earthloc:
        usable = vinsat_usable and earthloc_usable
        quality = min(vinsat_quality, earthloc_quality) if usable else 0.0
        source = "VINSat + EarthLoc required"
    elif earthloc_fix is not None:
        usable = vinsat_usable or earthloc_usable
        quality = max(vinsat_quality, earthloc_quality)
        source = "VINSat and/or EarthLoc"
    else:
        usable = vinsat_usable
        quality = vinsat_quality
        source = "prerecorded_VINSat_landmark_detections"

    return {
        "localization_usable": bool(usable),
        "quality_score": round(float(quality), 4),
        "source": source,
        "landmark_count": len(confident),
        "mean_landmark_confidence": round(vinsat_quality, 4) if confident else None,
        "earthloc": earthloc_fix,
    }


def target_visible(
    detections: list[tuple[float, float, float]],
    earthloc_fix: dict | None,
    target: dict,
) -> tuple[bool, str, int]:
    bbox = tuple(target["bbox_lonlat"])

    # If a future EarthLoc fix is supplied, it can provide direct point-in-AOI evidence.
    if earthloc_fix and earthloc_fix["status"] == "fix":
        if point_in_bbox(earthloc_fix["lon"], earthloc_fix["lat"], bbox):
            return True, "EarthLoc point-in-target proxy", 0

    landmark_hits = sum(
        point_in_bbox(lon, lat, bbox)
        for lon, lat, _confidence in detections
    )

    if landmark_hits:
        return (
            True,
            "One or more prerecorded VINSat landmarks fall inside this city cell.",
            landmark_hits,
        )

    return False, "No target-cell landmark evidence at this timestep.", 0


def best_opportunity(
    *,
    start_second: int,
    end_second: int,
    targets: list[dict],
    status: dict[str, str],
    detections_by_second: dict[int, list[tuple[float, float, float]]],
    earthloc_fixes: dict[int, dict],
    outage: tuple[int, int] | None,
    args: argparse.Namespace,
    completed_capture_count: int,
) -> tuple[dict | None, int]:
    best: dict | None = None
    eligible_count = 0

    for second in range(start_second, end_second + 1):
        in_outage = outage is not None and outage[0] <= second <= outage[1]
        detections = detections_by_second.get(second, [])
        earthloc_fix = earthloc_fixes.get(second)

        localization = localization_evidence(
            detections=detections,
            earthloc_fix=earthloc_fix,
            in_outage=in_outage,
            min_landmarks=args.min_landmarks,
            min_landmark_confidence=args.min_landmark_confidence,
            min_earthloc_confidence=args.min_earthloc_confidence,
            require_earthloc=args.require_earthloc,
        )
        if not localization["localization_usable"]:
            continue

        battery_before = (
            args.battery_pct
            - args.battery_drain_pct_per_s * second
            - args.capture_energy_pct * completed_capture_count
        )
        battery_after = battery_before - args.capture_energy_pct
        storage_after = args.storage_mb - args.capture_cost_mb * (completed_capture_count + 1)

        if battery_after < args.min_battery_pct or storage_after < args.min_storage_mb:
            continue

        urgency = max(0.0, 1.0 - second / max(1, args.deadline_s))

        for target in targets:
            if status[target["target_id"]] != "PROPOSED":
                continue

            visible, visibility_evidence, landmark_hits = target_visible(
                detections=detections,
                earthloc_fix=earthloc_fix,
                target=target,
            )
            if not visible:
                continue

            eligible_count += 1
            score = (
                0.50 * target["operator_priority"]
                + 0.30 * localization["quality_score"]
                + 0.20 * urgency
            )

            candidate = {
                "target": target,
                "time_s": second,
                "localization": localization,
                "battery_before": battery_before,
                "battery_after": battery_after,
                "storage_after": storage_after,
                "urgency": urgency,
                "mission_score": score,
                "visibility_evidence": visibility_evidence,
                "target_landmark_hits": landmark_hits,
            }

            if (
                best is None
                or candidate["mission_score"] > best["mission_score"]
                or (
                    math.isclose(candidate["mission_score"], best["mission_score"])
                    and candidate["time_s"] < best["time_s"]
                )
            ):
                best = candidate

    return best, eligible_count


def proposal_from(candidate: dict, region_label: str, city_name: str) -> dict:
    target = candidate["target"]
    return {
        "action": "PROPOSE_CAPTURE",
        "region": region_label,
        "city": city_name,
        "target_id": target["target_id"],
        "target_level": target["level"],
        "target_bbox_lonlat": target["bbox_lonlat"],
        "target_center_lonlat": target["center_lonlat"],
        "time_after_episode_start_s": candidate["time_s"],
        "time_after_episode_start_hms": hms(candidate["time_s"]),
        "operator_priority": round(target["operator_priority"], 4),
        "localization": candidate["localization"],
        "resource_replay": {
            "battery_pct_before_capture": round(candidate["battery_before"], 3),
            "battery_pct_after_capture": round(candidate["battery_after"], 3),
            "storage_mb_after_capture": round(candidate["storage_after"], 3),
        },
        "urgency_score": round(candidate["urgency"], 4),
        "mission_score": round(candidate["mission_score"], 4),
        "visibility_evidence": candidate["visibility_evidence"],
        "target_landmark_hits": candidate["target_landmark_hits"],
        "requires_operator_approval": True,
        "uplink_sent": False,
        "flight_command_generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vinsat-root", type=Path, required=True)
    parser.add_argument("--region", default="11R")
    parser.add_argument("--episode", default="00105")
    parser.add_argument("--region-label", required=True)
    parser.add_argument("--region-bbox", type=float, nargs=4)
    parser.add_argument("--city-name", required=True)
    parser.add_argument("--city-bbox", type=float, nargs=4, required=True)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--city-priority", type=float, default=1.0)
    parser.add_argument("--focus-cell", help="Optional: r1c1, r2c3, or full target ID")
    parser.add_argument("--deadline-s", type=int, default=7200)
    parser.add_argument("--battery-pct", type=float, default=72.0)
    parser.add_argument("--storage-mb", type=float, default=1200.0)
    parser.add_argument("--min-battery-pct", type=float, default=35.0)
    parser.add_argument("--min-storage-mb", type=float, default=300.0)
    parser.add_argument("--battery-drain-pct-per-s", type=float, default=0.0005)
    parser.add_argument("--capture-energy-pct", type=float, default=0.20)
    parser.add_argument("--capture-cost-mb", type=float, default=5.0)
    parser.add_argument("--capture-cooldown-s", type=int, default=30)
    parser.add_argument("--min-landmarks", type=int, default=3)
    parser.add_argument("--min-landmark-confidence", type=float, default=0.50)
    parser.add_argument("--earthloc-fixes", type=Path)
    parser.add_argument("--min-earthloc-confidence", type=float, default=0.60)
    parser.add_argument("--require-earthloc", action="store_true")
    parser.add_argument("--outage", help="Inject localization outage: START_S:END_S")
    parser.add_argument("--replay-approval", action="store_true")
    parser.add_argument("--max-captures", type=int, default=1)
    parser.add_argument(
        "--first-outcome",
        choices=["complete", "no_fix", "image_reject"],
        default="complete",
        help="Offline injected replay outcome after first approved proposal.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/city_target_hunter.json"))
    args = parser.parse_args()

    if args.grid_rows <= 0 or args.grid_cols <= 0 or args.max_captures <= 0:
        raise SystemExit("Grid dimensions and --max-captures must be positive.")
    if args.deadline_s <= 0:
        raise SystemExit("--deadline-s must be positive.")

    city_bbox = require_valid_bbox(args.city_bbox, "city bbox")
    region_bbox = require_valid_bbox(args.region_bbox, "region bbox") if args.region_bbox else None
    if region_bbox and not bbox_inside(city_bbox, region_bbox):
        raise SystemExit("City AOI must be inside the supplied region bounding box.")

    outage = parse_outage(args.outage)
    detections_by_second, replay_duration_s = load_vinsat_replay(args.vinsat_root, args.episode)
    earthloc_fixes = load_earthloc_fixes(args.earthloc_fixes)
    end_second = min(args.deadline_s, replay_duration_s)

    targets = build_city_cells(
        city_name=args.city_name,
        city_bbox=city_bbox,
        rows=args.grid_rows,
        cols=args.grid_cols,
        city_priority=args.city_priority,
        focus_cell=args.focus_cell,
    )
    status = {target["target_id"]: "PROPOSED" for target in targets}

    events: list[dict] = []
    primary_recommendation: dict | None = None
    cursor_s = 0
    approved_capture_count = 0

    for plan_step in range(1, args.max_captures + 1):
        candidate, candidate_count = best_opportunity(
            start_second=cursor_s,
            end_second=end_second,
            targets=targets,
            status=status,
            detections_by_second=detections_by_second,
            earthloc_fixes=earthloc_fixes,
            outage=outage,
            args=args,
            completed_capture_count=approved_capture_count,
        )

        if candidate is None:
            events.append(
                {
                    "step": plan_step,
                    "action": "WAIT_OR_SKIP",
                    "reason": (
                        "No pending city cell was simultaneously visible, "
                        "localization-usable, resource-safe, and before deadline."
                    ),
                    "requires_operator_approval": True,
                }
            )
            break

        proposal = proposal_from(candidate, args.region_label, args.city_name)
        proposal["step"] = plan_step
        proposal["candidate_opportunities_considered"] = candidate_count
        proposal["lifecycle_transition"] = "PROPOSED -> FEASIBILITY_CHECKED -> SCHEDULED"
        events.append(proposal)

        if primary_recommendation is None:
            primary_recommendation = proposal

        if not args.replay_approval:
            break

        target_id = proposal["target_id"]
        outcome = args.first_outcome if approved_capture_count == 0 else "complete"

        if outcome == "complete":
            status[target_id] = "CAPTURE_REPLAYED_PENDING_IMAGE_ANALYSIS"
            resulting_state = status[target_id]
        else:
            status[target_id] = f"DEFERRED_{outcome.upper()}"
            resulting_state = status[target_id]

        events.append(
            {
                "step": plan_step,
                "action": "REPLAY_CAPTURE_OUTCOME",
                "target_id": target_id,
                "operator_approval_simulated": True,
                "outcome": outcome,
                "resulting_target_state": resulting_state,
                "truthfulness": (
                    "Injected replay state transition only; no camera image, "
                    "uplink, or spacecraft capture occurred."
                ),
            }
        )

        approved_capture_count += 1
        cursor_s = candidate["time_s"] + args.capture_cooldown_s

    if primary_recommendation is None:
        primary_recommendation = {
            "action": "WAIT_OR_SKIP",
            "reason": "No safe city-cell opportunity was found.",
            "requires_operator_approval": True,
        }

    model_path = args.vinsat_root / "models" / f"{args.region}.pt"
    result = {
        "mission": {
            "name": "VINSat Region-to-City Target Hunter replay",
            "episode": args.episode,
            "replay_duration_s": replay_duration_s,
            "deadline_s": end_second,
            "recommendation_only": True,
            "operator_scope": {
                "region_label": args.region_label,
                "region_bbox_lonlat": list(region_bbox) if region_bbox else None,
                "city_name": args.city_name,
                "city_bbox_lonlat": list(city_bbox),
                "city_aoi_source": "operator-provided bounding box; not a municipal-boundary dataset",
                "city_grid_shape": [args.grid_rows, args.grid_cols],
            },
        },
        "truthfulness": {
            "visibility_and_localization_source": "prerecorded VINSat landmark detections",
            "vinsat_region_model_available": str(model_path) if model_path.is_file() else None,
            "vinsat_region_model_executed": False,
            "earthloc_executed": bool(args.earthloc_fixes),
            "earthloc_fix_file": str(args.earthloc_fixes) if args.earthloc_fixes else None,
            "resource_source": "explicit replay assumptions; not decoded Argus telemetry",
            "camera_footprint_intersection": (
                "landmark-in-city-cell proxy; not yet a calibrated Argus camera footprint"
            ),
            "uplink_sent": False,
            "flight_command_generated": False,
        },
        "policy": {
            "min_landmarks": args.min_landmarks,
            "min_landmark_confidence": args.min_landmark_confidence,
            "min_battery_pct": args.min_battery_pct,
            "min_storage_mb": args.min_storage_mb,
            "score": "0.50 * operator_priority + 0.30 * localization_quality + 0.20 * urgency",
            "injected_outage_s": list(outage) if outage else None,
        },
        "city_grid_targets": targets,
        "events": events,
        "target_states": status,
        "primary_recommendation": primary_recommendation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("=== VINSAT REGION-TO-CITY TARGET HUNTER ===")
    print("Region:", args.region_label)
    print("City AOI:", args.city_name)
    print("Grid cells:", len(targets))
    print("Replay duration:", replay_duration_s, "seconds")
    print("\nPrimary recommendation:")
    print(json.dumps(primary_recommendation, indent=2))
    print("\nTarget states:")
    print(json.dumps(status, indent=2))
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()