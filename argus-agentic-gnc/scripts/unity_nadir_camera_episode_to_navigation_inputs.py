"""Prepare Unity nadir-camera exports for the Python navigation pipeline.

WHAT THIS FILE DOES
-------------------
Unity exports five camera streams plus one JSONL metadata file. This script
selects only the Earth-facing camera:

    "NADIR EARTH  -Z"

and writes one clean JSON manifest for later EarthLoc, LightGlue, coordinate
conversion, gyro assembly, and FSW orbit determination.

INPUT
-----
Unity NavigationEpisode directory:

    *.png
    navigation_metadata.jsonl

OUTPUT
------
A JSON manifest containing absolute PNG paths plus the synchronized Unity
metadata needed by the navigation pipeline.

IMPORTANT
---------
This script deliberately preserves Unity coordinate data exactly as exported.
It does NOT yet claim Unity coordinates are the same as the Python/FSW frame
convention. The explicit Unity-to-navigation-frame conversion is the next
layer, where it can be documented and tested safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NADIR_CAMERA_NAME = "NADIR EARTH  -Z"


def require_list(value: object, name: str, length: int) -> list[float]:
    """Validate one fixed-size finite numeric list from Unity JSON."""
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a JSON list with {length} values")

    result: list[float] = []

    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"{name} values must be numeric")
        result.append(float(item))

    return result


def require_positive(value: object, name: str) -> float:
    """Validate one positive numeric calibration value."""
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def load_nadir_records(episode_directory: Path) -> list[dict[str, Any]]:
    """Read Unity JSONL and return validated Earth-facing camera records."""
    metadata_path = episode_directory / "navigation_metadata.jsonl"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Unity metadata file does not exist: {metadata_path}")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        metadata_path.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on metadata line {line_number}"
            ) from error

        if raw_record.get("camera_name") != NADIR_CAMERA_NAME:
            continue

        image_file = raw_record.get("image_file")
        if not isinstance(image_file, str) or not image_file:
            raise ValueError(
                f"Line {line_number}: image_file must be a non-empty string"
            )

        image_path = episode_directory / image_file
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Line {line_number}: exported PNG is missing: {image_path}"
            )

        timestamp_utc = raw_record.get("timestamp_utc")
        if not isinstance(timestamp_utc, str) or not timestamp_utc:
            raise ValueError(
                f"Line {line_number}: timestamp_utc must be a non-empty string"
            )

        sequence = raw_record.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError(
                f"Line {line_number}: sequence must be a non-negative integer"
            )

        image_width_px = raw_record.get("image_width_px")
        image_height_px = raw_record.get("image_height_px")

        if (
            not isinstance(image_width_px, int)
            or image_width_px <= 0
            or not isinstance(image_height_px, int)
            or image_height_px <= 0
        ):
            raise ValueError(
                f"Line {line_number}: image width and height must be positive"
            )

        records.append(
            {
                "frame_index": len(records),
                "frame_id": f"unity_sequence_{sequence:06d}",
                "unity_sequence": sequence,
                "timestamp_utc": timestamp_utc,
                "simulation_time_seconds": float(
                    raw_record["simulation_time_seconds"]
                ),
                "image_path": str(image_path.resolve()),
                "image_file": image_file,
                "camera_name": NADIR_CAMERA_NAME,
                "image_width_px": image_width_px,
                "image_height_px": image_height_px,
                "camera_intrinsics_px": {
                    "fx_px": require_positive(raw_record.get("fx_px"), "fx_px"),
                    "fy_px": require_positive(raw_record.get("fy_px"), "fy_px"),
                    "cx_px": float(raw_record["cx_px"]),
                    "cy_px": float(raw_record["cy_px"]),
                },
                # Preserved unchanged. Convert Unity conventions explicitly later.
                "rotation_camera_to_body_unity_xyzw": require_list(
                    raw_record.get("rotation_camera_to_body_unity_xyzw"),
                    "rotation_camera_to_body_unity_xyzw",
                    4,
                ),
                "position_ecef_m_truth": require_list(
                    raw_record.get("position_ecef_m"),
                    "position_ecef_m",
                    3,
                ),
                "velocity_ecef_m_per_s_truth": require_list(
                    raw_record.get("velocity_ecef_m_per_s"),
                    "velocity_ecef_m_per_s",
                    3,
                ),
                "rotation_body_to_ecef_unity_xyzw_truth": require_list(
                    raw_record.get("rotation_body_to_ecef_xyzw"),
                    "rotation_body_to_ecef_xyzw",
                    4,
                ),
                "angular_velocity_body_truth_rad_per_s": require_list(
                    raw_record.get("angular_velocity_body_truth_rad_per_s"),
                    "angular_velocity_body_truth_rad_per_s",
                    3,
                ),
            }
        )

    if not records:
        raise ValueError(
            f"No {NADIR_CAMERA_NAME!r} records found in {metadata_path}"
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Unity nadir images and create a navigation manifest."
    )
    parser.add_argument(
        "--episode-directory",
        type=Path,
        required=True,
        help="Unity NavigationEpisodes/<episode> directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON manifest path.",
    )
    args = parser.parse_args()

    episode_directory = args.episode_directory.resolve()
    records = load_nadir_records(episode_directory)

    payload = {
        "experiment": "unity_nadir_camera_navigation_input_export",
        "source": {
            "simulator": "Unity + Cesium",
            "episode_directory": str(episode_directory),
            "selected_camera_name": NADIR_CAMERA_NAME,
        },
        "coordinate_convention_status": (
            "Unity quaternion and coordinate conventions are preserved, "
            "not yet converted to the Python/FSW navigation convention."
        ),
        "num_frames": len(records),
        "frames": records,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("=== Unity Nadir Navigation Input Export ===")
    print(f"Nadir frames selected: {len(records)}")
    print(f"Manifest: {output_path}")
    print("NEXT: EarthLoc/LightGlue reads image_path from this manifest.")


if __name__ == "__main__":
    main()