#!/usr/bin/env python3
"""
NASA Astronaut Photography Target-Hunter replay.

Run:
  python -m pip install Pillow
  python scripts/nasa_target_hunter.py --self-test
  python scripts/nasa_target_hunter.py --write-example configs/nasa_new_orleans_demo.json

  export NASA_EO_API_KEY="YOUR_FREE_NASA_KEY"
  python scripts/nasa_target_hunter.py \
    --mission configs/nasa_new_orleans_demo.json \
    --out artifacts/nasa_new_orleans_demo \
    --download-image

Do NOT commit:
  - NASA_EO_API_KEY
  - artifacts/ downloaded images/results

This is historical replay only:
  - no uplink
  - no FSW command
  - no capture command
  - all outputs require operator review

NASA API key instructions:
  https://eol.jsc.nasa.gov/SearchPhotos/PhotosDatabaseAPI/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageStat

NASA_API = (
    "https://eol.jsc.nasa.gov/SearchPhotos/PhotosDatabaseAPI/"
    "PhotosDatabaseAPI.pl"
)
NASA_IMAGE_ROOT = "https://eol.jsc.nasa.gov/DatabaseImages"

EXAMPLE_MISSION = {
    "mission_id": "nasa_new_orleans_target_hunter_demo",
    "objective": (
        "Find a usable historical astronaut image covering the New Orleans "
        "target region. This is an image-acquisition replay, not flood truth."
    ),
    "target_name": "New Orleans, Louisiana",
    "bbox_lonlat": [-90.14, 29.90, -90.05, 30.02],
    "start_utc": "2014-03-13T00:00:00Z",
    "end_utc": "2014-03-13T23:59:59Z",
    "policy": {
        "max_cloud_pct": 25.0,
        "min_sun_elevation_deg": 15.0,
        "allowed_camera_tilts": ["NV", "LO"],
        "max_numeric_tilt_deg": 40.0,
        "min_target_footprint_overlap": 0.20,
        "search_padding_degrees": 1.0,
        "urgency_decay_hours": 72.0,
        "urgency_weight": 0.55,
        "image_quality_weight": 0.45,
        "recommendation_only": True,
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"Mission file not found: {path}")
    except json.JSONDecodeError as error:
        fail(f"Invalid JSON in {path}: {error}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def field(row: dict[str, Any], table: str, name: str) -> Any:
    for key in (f"{table}.{name}", f"{table}_{name}", name):
        if key in row:
            return row[key]
    return None


def parse_utc(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        fail(f"Expected ISO UTC time, got: {value!r}")


def validate_bbox(bbox: Any) -> tuple[float, float, float, float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        fail("bbox_lonlat must be [west, south, east, north].")

    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        fail(f"Invalid bbox_lonlat: {bbox}")

    # This first implementation does not support a target crossing ±180°.
    if east - west > 180:
        fail("Target crossing the anti-meridian is not supported in this replay.")

    return west, south, east, north


def mission_policy(mission: dict[str, Any]) -> dict[str, Any]:
    policy = mission.get("policy")
    if not isinstance(policy, dict):
        fail("Mission must contain a policy object.")

    defaults = {
        "max_cloud_pct": 25.0,
        "min_sun_elevation_deg": 15.0,
        "allowed_camera_tilts": ["NV", "LO"],
        "max_numeric_tilt_deg": 35.0,
        "min_target_footprint_overlap": 0.20,
        "search_padding_degrees": 1.0,
        "urgency_decay_hours": 72.0,
        "urgency_weight": 0.55,
        "image_quality_weight": 0.45,
        "recommendation_only": True,
    }

    merged = {**defaults, **policy}
    if abs(
        float(merged["urgency_weight"])
        + float(merged["image_quality_weight"])
        - 1.0
    ) > 1e-9:
        fail("urgency_weight + image_quality_weight must equal 1.0.")

    return merged


def rectangle_polygon(
    west: float, south: float, east: float, north: float, cos_lat: float
) -> list[tuple[float, float]]:
    return [
        (west * cos_lat, south),
        (east * cos_lat, south),
        (east * cos_lat, north),
        (west * cos_lat, north),
    ]


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0

    total = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1

    return abs(total) * 0.5


def clip_half_plane(
    polygon: list[tuple[float, float]],
    axis: int,
    threshold: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman polygon clip against x/y <=/>= threshold."""
    if not polygon:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= threshold if keep_greater else point[axis] <= threshold

    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)

    for current in polygon:
        current_inside = inside(current)

        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(delta) > 1e-12:
                fraction = (threshold - previous[axis]) / delta
                intersection = (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
                output.append(intersection)

        if current_inside:
            output.append(current)

        previous = current
        previous_inside = current_inside

    return output


def target_overlap_fraction(
    footprint_lonlat: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
) -> float:
    """
    Returns fraction of the target rectangle covered by the image footprint.

    Uses a local equirectangular approximation, suitable for a compact target
    region. It is a replay geometry gate, not a precision geodesic calculation.
    """
    west, south, east, north = bbox
    center_lat = (south + north) * 0.5
    cos_lat = math.cos(math.radians(center_lat))

    subject = [(lon * cos_lat, lat) for lon, lat in footprint_lonlat]
    min_x, max_x = west * cos_lat, east * cos_lat
    min_y, max_y = south, north

    clipped = subject
    clipped = clip_half_plane(clipped, axis=0, threshold=min_x, keep_greater=True)
    clipped = clip_half_plane(clipped, axis=0, threshold=max_x, keep_greater=False)
    clipped = clip_half_plane(clipped, axis=1, threshold=min_y, keep_greater=True)
    clipped = clip_half_plane(clipped, axis=1, threshold=max_y, keep_greater=False)

    target = rectangle_polygon(west, south, east, north, cos_lat)
    target_area = polygon_area(target)
    if target_area <= 0:
        return 0.0

    return max(0.0, min(1.0, polygon_area(clipped) / target_area))


def frame_datetime(row: dict[str, Any]) -> datetime | None:
    date = str(field(row, "frames", "pdate") or "").strip()
    time = str(field(row, "frames", "ptime") or "").strip()

    if len(date) != 8 or not date.isdigit():
        return None

    time = "".join(character for character in time if character.isdigit())
    if len(time) < 6:
        return None

    try:
        return datetime.strptime(date + time[:6], "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def tilt_value(raw_tilt: Any) -> tuple[str | None, float | None]:
    text = str(raw_tilt or "").strip().upper()
    if not text:
        return None, None

    numeric = as_float(text)
    if numeric is not None:
        return "NUMERIC_DEG", numeric

    return text, None


def image_url(directory: str, filename: str) -> str:
    return f"{NASA_IMAGE_ROOT}/{directory.strip('/')}/{filename}"


def best_image_option(options: list[dict[str, str]]) -> dict[str, str] | None:
    if not options:
        return None

    def rank(option: dict[str, str]) -> tuple[int, str]:
        directory = option["directory"].lower()
        if "esc/large" in directory:
            return (0, directory)
        if "esc/medium" in directory:
            return (1, directory)
        if "esc/small" in directory:
            return (2, directory)
        return (3, directory)

    return sorted(options, key=rank)[0]


def nasa_rows(
    mission: dict[str, Any],
    api_key: str,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    west, south, east, north = validate_bbox(mission["bbox_lonlat"])
    start = parse_utc(mission["start_utc"])
    end = parse_utc(mission["end_utc"])

    if end < start:
        fail("end_utc must be later than start_utc.")

    pad = float(policy["search_padding_degrees"])
    qwest, qsouth = west - pad, south - pad
    qeast, qnorth = east + pad, north + pad

    # We require mlcoord fields, so returned candidates have corner coordinates.
    # NASA API inner-joins tables listed in "return".
    query_parts = [
        "frames", "body", "eq", "Earth",
        "frames", "pdate", "ge", start.strftime("%Y%m%d"),
        "frames", "pdate", "le", end.strftime("%Y%m%d"),
        "frames", "lat", "ge", f"{qsouth:.6f}",
        "frames", "lat", "le", f"{qnorth:.6f}",
        "frames", "lon", "ge", f"{qwest:.6f}",
        "frames", "lon", "le", f"{qeast:.6f}",
    ]

    return_parts = [
        "frames", "mission",
        "frames", "roll",
        "frames", "frame",
        "frames", "pdate",
        "frames", "ptime",
        "frames", "cldp",
        "frames", "elev",
        "frames", "tilt",
        "frames", "fclt",
        "frames", "lat",
        "frames", "lon",
        "frames", "camera",
        "frames", "geon",
        "mlcoord", "ul_lat",
        "mlcoord", "ul_lon",
        "mlcoord", "ur_lat",
        "mlcoord", "ur_lon",
        "mlcoord", "lr_lat",
        "mlcoord", "lr_lon",
        "mlcoord", "ll_lat",
        "mlcoord", "ll_lon",
        "images", "directory",
        "images", "filename",
        "images", "width",
        "images", "height",
        "images", "filesize",
    ]

    parameters = {
        "query": "|".join(query_parts),
        "return": "|".join(return_parts),
        "key": api_key,
    }

    # Never save or print the API key.
    safe_parameters = {**parameters, "key": "REDACTED"}
    safe_url = f"{NASA_API}?{urlencode(safe_parameters)}"

    request = Request(
        f"{NASA_API}?{urlencode(parameters)}",
        headers={"User-Agent": "ArgusTargetHunterResearch/0.1"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except HTTPError as error:
        fail(
            f"NASA API HTTP {error.code}. Check NASA_EO_API_KEY and mission query."
        )
    except URLError as error:
        fail(f"Could not reach NASA API: {error}")
    except TimeoutError:
        fail("NASA API timed out.")

    if not isinstance(payload, list):
        fail(f"NASA API returned unexpected JSON: {payload!r}")

    return payload, safe_url


def normalize_candidates(
    rows: list[dict[str, Any]],
    mission: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    bbox = validate_bbox(mission["bbox_lonlat"])
    start_time = parse_utc(mission["start_utc"])
    allowed_tilts = {str(value).upper() for value in policy["allowed_camera_tilts"]}

    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        mission_id = str(field(row, "frames", "mission") or "").strip()
        roll = str(field(row, "frames", "roll") or "").strip()
        frame = str(field(row, "frames", "frame") or "").strip()

        if not all((mission_id, roll, frame)):
            continue

        photo_id = f"{mission_id}-{roll}-{frame}"
        candidate = grouped.setdefault(
            photo_id,
            {
                "photo_id": photo_id,
                "mission": mission_id,
                "roll": roll,
                "frame": frame,
                "datetime_utc": None,
                "cloud_pct": None,
                "sun_elevation_deg": None,
                "camera_tilt_raw": None,
                "focal_length_mm": None,
                "photo_center_lat": None,
                "photo_center_lon": None,
                "camera": None,
                "geographic_name": None,
                "footprint_lonlat": None,
                "image_options": [],
            },
        )

        timestamp = frame_datetime(row)
        if timestamp:
            candidate["datetime_utc"] = timestamp.isoformat().replace("+00:00", "Z")

        candidate["cloud_pct"] = as_float(field(row, "frames", "cldp"))
        candidate["sun_elevation_deg"] = as_float(field(row, "frames", "elev"))
        candidate["camera_tilt_raw"] = field(row, "frames", "tilt")
        candidate["focal_length_mm"] = as_float(field(row, "frames", "fclt"))
        candidate["photo_center_lat"] = as_float(field(row, "frames", "lat"))
        candidate["photo_center_lon"] = as_float(field(row, "frames", "lon"))
        candidate["camera"] = field(row, "frames", "camera")
        candidate["geographic_name"] = field(row, "frames", "geon")

        corners = [
            (
                as_float(field(row, "mlcoord", "ul_lon")),
                as_float(field(row, "mlcoord", "ul_lat")),
            ),
            (
                as_float(field(row, "mlcoord", "ur_lon")),
                as_float(field(row, "mlcoord", "ur_lat")),
            ),
            (
                as_float(field(row, "mlcoord", "lr_lon")),
                as_float(field(row, "mlcoord", "lr_lat")),
            ),
            (
                as_float(field(row, "mlcoord", "ll_lon")),
                as_float(field(row, "mlcoord", "ll_lat")),
            ),
        ]

        if all(lon is not None and lat is not None for lon, lat in corners):
            candidate["footprint_lonlat"] = corners

        directory = str(field(row, "images", "directory") or "").strip()
        filename = str(field(row, "images", "filename") or "").strip()
        if directory and filename:
            candidate["image_options"].append(
                {
                    "directory": directory,
                    "filename": filename,
                    "url": image_url(directory, filename),
                    "width": as_float(field(row, "images", "width")),
                    "height": as_float(field(row, "images", "height")),
                    "filesize_bytes": as_float(field(row, "images", "filesize")),
                }
            )

    candidates = list(grouped.values())

    for candidate in candidates:
        rejections: list[str] = []
        footprint = candidate["footprint_lonlat"]

        if not footprint:
            overlap = 0.0
            rejections.append("missing verified four-corner image footprint")
        else:
            overlap = target_overlap_fraction(footprint, bbox)

        candidate["target_footprint_overlap_fraction"] = round(overlap, 6)

        if overlap < float(policy["min_target_footprint_overlap"]):
            rejections.append("insufficient target-footprint overlap")

        cloud = candidate["cloud_pct"]
        if cloud is None:
            rejections.append("missing cloud estimate")
        elif cloud > float(policy["max_cloud_pct"]):
            rejections.append("cloud above policy")

        sun = candidate["sun_elevation_deg"]
        if sun is None:
            rejections.append("missing sun elevation")
        elif sun < float(policy["min_sun_elevation_deg"]):
            rejections.append("sun below policy")

        tilt_kind, numeric_tilt = tilt_value(candidate["camera_tilt_raw"])
        candidate["camera_tilt_kind"] = tilt_kind
        candidate["camera_tilt_deg"] = numeric_tilt

        tilt_ok = (
            tilt_kind in allowed_tilts
            or (
                tilt_kind == "NUMERIC_DEG"
                and numeric_tilt is not None
                and numeric_tilt <= float(policy["max_numeric_tilt_deg"])
            )
        )
        if not tilt_ok:
            rejections.append("camera tilt outside policy")

        option = best_image_option(candidate["image_options"])
        if option is None:
            rejections.append("no downloadable image file")
        else:
            candidate["selected_image"] = option

        timestamp = (
            parse_utc(candidate["datetime_utc"])
            if candidate["datetime_utc"]
            else None
        )
        if timestamp is None:
            rejections.append("missing valid capture timestamp")

        candidate["policy_eligible"] = not rejections
        candidate["rejections"] = rejections

        if candidate["policy_eligible"] and timestamp is not None:
            elapsed_hours = max(0.0, (timestamp - start_time).total_seconds() / 3600.0)
            urgency = math.exp(
                -elapsed_hours / float(policy["urgency_decay_hours"])
            )
            cloud_quality = max(0.0, 1.0 - (cloud or 100.0) / 100.0)
            sun_quality = min(1.0, max(0.0, (sun or 0.0) / 60.0))
            geometry_quality = overlap

            image_quality = (
                0.45 * cloud_quality
                + 0.25 * sun_quality
                + 0.30 * geometry_quality
            )
            mission_score = (
                float(policy["urgency_weight"]) * urgency
                + float(policy["image_quality_weight"]) * image_quality
            )

            candidate["hours_after_mission_start"] = round(elapsed_hours, 3)
            candidate["urgency"] = round(urgency, 6)
            candidate["image_quality"] = round(image_quality, 6)
            candidate["mission_score"] = round(mission_score, 6)

    return sorted(
        candidates,
        key=lambda item: (
            not item["policy_eligible"],
            -(item.get("mission_score") or -1.0),
            item["photo_id"],
        ),
    )


def download_and_inspect(url: str, out_path: Path) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "ArgusTargetHunterResearch/0.1"})

    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        return {"download_status": "failed", "error": str(error)}

    out_path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()

    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        preview_path = out_path.with_name(out_path.stem + "_preview.jpg")
        image.save(preview_path, quality=92)

        stats = ImageStat.Stat(image)
        brightness = sum(stats.mean) / (3 * 255.0)
        contrast = sum(stats.stddev) / (3 * 255.0)

        return {
            "download_status": "downloaded",
            "file": str(out_path),
            "preview_file": str(preview_path),
            "sha256": sha256,
            "bytes": len(payload),
            "image_size_px": list(image.size),
            "brightness_proxy_0_to_1": round(brightness, 4),
            "contrast_proxy_0_to_1": round(contrast, 4),
            "image_quality_proxy_pass": (
                0.10 < brightness < 0.90 and contrast > 0.04
            ),
            "warning": (
                "Brightness/contrast are simple image-health proxies, not "
                "cloud detection, flood detection, or localization."
            ),
        }
    except Exception as error:
        return {
            "download_status": "downloaded_but_not_decoded",
            "file": str(out_path),
            "sha256": sha256,
            "bytes": len(payload),
            "error": str(error),
        }


def run(mission: dict[str, Any], api_key: str, out_dir: Path, download: bool) -> dict[str, Any]:
    required = ["mission_id", "objective", "target_name", "bbox_lonlat", "start_utc", "end_utc"]
    missing = [name for name in required if name not in mission]
    if missing:
        fail(f"Mission missing required fields: {missing}")

    policy = mission_policy(mission)
    rows, safe_api_url = nasa_rows(mission, api_key, policy)
    candidates = normalize_candidates(rows, mission, policy)
    eligible = [candidate for candidate in candidates if candidate["policy_eligible"]]
    selected = eligible[0] if eligible else None

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "mission.json", mission)
    write_json(out_dir / "candidates.json", candidates)

    image_result = None
    if selected and download:
        image_info = selected["selected_image"]
        extension = Path(image_info["filename"]).suffix or ".jpg"
        output_image = out_dir / f"selected_{selected['photo_id']}{extension.lower()}"
        image_result = download_and_inspect(image_info["url"], output_image)

    report = {
        "generated_utc": utc_now(),
        "source": {
            "name": "NASA Gateway to Astronaut Photography of Earth",
            "api_query_url_key_redacted": safe_api_url,
            "raw_api_rows": len(rows),
            "unique_photo_candidates": len(candidates),
            "geometry_requirement": (
                "Only records with four image-corner coordinates were accepted "
                "for target coverage scoring."
            ),
        },
        "mission": mission,
        "policy": policy,
        "ranked_candidates": candidates,
        "selected_candidate": selected,
        "selected_image_download": image_result,
        "agent_action": (
            {
                "action": "HISTORICAL_IMAGE_FOUND_FOR_REPLAY",
                "reason": (
                    "Selected the highest-scoring NASA image that passed "
                    "footprint overlap, cloud, lighting, tilt, and image-file gates."
                ),
                "requires_operator_approval": True,
                "uplink_sent": False,
                "flight_command_generated": False,
            }
            if selected
            else {
                "action": "NO_USABLE_HISTORICAL_IMAGE",
                "reason": (
                    "No NASA catalogued image passed all geometry and quality "
                    "gates for this target/time window. This is a safe skip."
                ),
                "requires_operator_approval": True,
                "uplink_sent": False,
                "flight_command_generated": False,
            }
        ),
        "limitations": [
            "This is historical astronaut-photography retrieval, not a future Argus pass predictor.",
            "NASA camera geometry and image quality are not Argus camera geometry or quality.",
            "No flood/change conclusion is made by this script.",
            "Future deployment replaces NASA metadata with Argus orbit, ADCS, camera, payload, EPS, and storage telemetry.",
        ],
    }

    write_json(out_dir / "recommendation.json", report)
    return report


def self_test() -> None:
    target = (-90.10, 29.90, -90.00, 30.00)

    fully_covering = [
        (-90.20, 29.80),
        (-89.90, 29.80),
        (-89.90, 30.10),
        (-90.20, 30.10),
    ]
    partial = [
        (-90.20, 29.80),
        (-90.05, 29.80),
        (-90.05, 30.10),
        (-90.20, 30.10),
    ]

    assert abs(target_overlap_fraction(fully_covering, target) - 1.0) < 1e-9
    assert 0.45 < target_overlap_fraction(partial, target) < 0.55
    assert tilt_value("NV") == ("NV", None)
    assert tilt_value("37") == ("NUMERIC_DEG", 37.0)

    print("PASS: geometry, overlap, and camera-tilt safety tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, help="Mission JSON input")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/nasa_target_hunter"),
        help="Output directory; do not commit this directory.",
    )
    parser.add_argument(
        "--download-image",
        action="store_true",
        help="Download the selected historical image and produce a preview.",
    )
    parser.add_argument(
        "--write-example",
        type=Path,
        help="Write a New Orleans historical target-query example and exit.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline geometry tests; requires no API key.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.write_example:
        write_json(args.write_example, EXAMPLE_MISSION)
        print(f"Wrote example mission: {args.write_example}")
        return

    if not args.mission:
        parser.error("--mission is required unless --self-test or --write-example is used.")

    api_key = os.environ.get("NASA_EO_API_KEY", "").strip()
    if not api_key:
        fail(
            "NASA_EO_API_KEY is not set. Request a free key from "
            "jsc-earthweb@mail.nasa.gov, then run:\n"
            '  export NASA_EO_API_KEY="YOUR_KEY"\n'
            "Do not put the key in JSON or commit it."
        )

    report = run(read_json(args.mission), api_key, args.out, args.download_image)

    print("\n=== NASA TARGET-HUNTER RESULT ===")
    print(
        json.dumps(
            {
                "action": report["agent_action"]["action"],
                "candidate_count": report["source"]["unique_photo_candidates"],
                "selected_photo": (
                    report["selected_candidate"]["photo_id"]
                    if report["selected_candidate"]
                    else None
                ),
                "selected_score": (
                    report["selected_candidate"]["mission_score"]
                    if report["selected_candidate"]
                    else None
                ),
                "result": str(args.out / "recommendation.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()